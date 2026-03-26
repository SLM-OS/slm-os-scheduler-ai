# SLM-OS AI Scheduler: Simulator, Dataset, and Model Training Plan
## Architecture and Implementation Roadmap
### Version 1.0 — March 2026

---

## Executive Summary

This document defines the complete architecture and implementation plan for building an AI-driven process scheduler for SLM-OS. The system comprises four major deliverables: a discrete-event scheduling simulator that faithfully models SLM-OS's heterogeneous multi-core environment, a synthetic training dataset generated from expert scheduling policies and realistic industrial IoT workloads, three trained scheduling models (feedforward MLP, gradient-boosted decision tree, and RL policy network), and an integration path back into SLM-OS's Rust runtime layer.

The AI scheduler replaces the hand-tuned heuristics in `runtime/src/sched/deadline.rs` with a learned policy that optimizes deadline compliance, inference latency, power efficiency, and core utilization simultaneously.

---

## 1. Simulator Architecture

### 1.1 Design Philosophy

The simulator is a **discrete-event simulation** (DES) written in Python. DES is the right choice over cycle-accurate simulation because scheduling decisions happen at millisecond granularity (not nanosecond), and we need to generate millions of decision samples efficiently. The simulator must be fast enough to serve as an RL training environment running thousands of episodes per hour.

The simulator models SLM-OS at the scheduling decision boundary — it does not simulate cache lines or pipeline stages, but it does model core heterogeneity, task deadlines, memory pressure, and the interaction patterns your Phase 3 scheduler already handles.

### 1.2 Core Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SimulatorEngine                              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
│  │  EventQueue   │  │  Clock       │  │  MetricsCollector         │ │
│  │  (min-heap)   │  │  (ns ticks)  │  │  (per-episode stats)      │ │
│  └──────────────┘  └──────────────┘  └───────────────────────────┘ │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    PlatformModel                              │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐     │   │
│  │  │ Core 0 │ │ Core 1 │ │ Core 2 │ │ Core 3 │ │ Core 4 │ ... │   │
│  │  │ (perf) │ │ (perf) │ │ (eff)  │ │ (eff)  │ │ (perf) │     │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘     │   │
│  │  ┌────────────────────┐  ┌──────────────────────────────┐    │   │
│  │  │ MemorySubsystem    │  │ GPU/AcceleratorModel         │    │   │
│  │  │ (pool tracking)    │  │ (queue + latency model)      │    │   │
│  │  └────────────────────┘  └──────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   WorkloadGenerator                           │   │
│  │  ┌──────────────┐ ┌───────────────┐ ┌──────────────────┐    │   │
│  │  │ AnomalyDet   │ │ PredMaint     │ │ SecurityMonitor  │    │   │
│  │  │ (high-freq)  │ │ (bursty)      │ │ (background)     │    │   │
│  │  └──────────────┘ └───────────────┘ └──────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   SchedulingAgent (swappable)                 │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐ │   │
│  │  │ Heuristic│ │ MLP      │ │ XGBoost  │ │ RL Policy Net  │ │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Platform Model

The platform model captures the two target hardware configurations. Each is a named preset.

**Jetson Orin Nano Profile:**

| Property | Value |
|----------|-------|
| CPU Cores | 6 × Cortex-A78AE |
| Core Type | All performance (homogeneous) |
| Max Frequency | 1.5 GHz |
| L1 D-Cache | 64 KB per core |
| L2 Cache | 256 KB per core |
| L3 Cache | 4 MB shared |
| Total RAM | 8 GB LPDDR5 |
| GPU | 1024-core Ampere, 32 Tensor Cores |
| Power Weight (per core) | 1.0 |
| Context Switch Cost | ~2–5 µs modeled |

**Raspberry Pi 5 Profile:**

| Property | Value |
|----------|-------|
| CPU Cores | 4 × Cortex-A76 |
| Core Type | All performance (homogeneous) |
| Max Frequency | 2.4 GHz |
| L1 Cache | 64 KB per core |
| L2 Cache | 512 KB per core |
| Total RAM | 4 GB LPDDR4X |
| GPU | VideoCore VII (not usable for inference) |
| Power Weight (per core) | 0.7 (lower TDP than Jetson) |
| Context Switch Cost | ~3–6 µs modeled |

**Hypothetical big.LITTLE Profile (for future/research):**

| Property | Value |
|----------|-------|
| CPU Cores | 2 × A78 (performance) + 4 × A55 (efficiency) |
| Perf Core Freq | 2.0 GHz |
| Eff Core Freq | 1.0 GHz |
| Perf Power Weight | 1.0 |
| Eff Power Weight | 0.3 |

Each core in the simulator tracks: current task (or idle), time remaining on current quantum, cumulative utilization over a sliding window, run queue depth, and an estimated cache pressure metric (sum of working set sizes of recently-run tasks / cache size).

### 1.4 Core State Model

```python
@dataclass
class CoreState:
    core_id: int
    core_type: CoreType           # PERFORMANCE | EFFICIENCY
    max_freq_mhz: int
    current_task: Optional[TaskID]
    quantum_remaining_ns: int
    isolated: bool
    run_queue: List[TaskID]       # ordered by effective_priority
    utilization_pct: float        # sliding window, 0.0–1.0
    cache_pressure: float         # working_set_sum / cache_size, 0.0–∞
    power_weight: float           # relative power consumption
    idle_time_ns: int             # cumulative since boot
    context_switches: int         # cumulative since boot
```

### 1.5 Task Model

Tasks in the simulator mirror `struct slm_task` from SLM-OS plus the SLM-specific fields from `SlmTaskInfo` in the Rust runtime.

```python
@dataclass
class SimTask:
    task_id: int
    name: str
    state: TaskState              # READY | RUNNING | BLOCKED | TERMINATED

    # Standard scheduling fields
    priority: int                 # 0–7 (IDLE=0, LOW=1..2, NORMAL=3..4, HIGH=5..6, CRITICAL=7)
    effective_priority: int       # after deadline boost
    assigned_cpu: Optional[int]
    cpu_affinity: int             # CPU_AFFINITY_ANY = -1, else specific core

    # SLM-specific fields (from Phase 3 scheduler)
    model_handle: Optional[int]
    model_size_mb: float          # weight footprint
    working_set_mb: float         # active inference workspace
    ops_per_inference: float      # GFLOPS per inference call
    inference_duration_ns: int    # estimated single-inference time on perf core
    can_use_gpu: bool

    # Deadline fields
    deadline_ns: int              # absolute deadline (0 = no deadline)
    arrival_time_ns: int          # when task entered ready queue
    remaining_work_ns: int        # estimated remaining compute time

    # Component info
    component_type: ComponentType # ANOMALY_DETECTOR | PRED_MAINT | SECURITY_MON | SYSTEM | OTHER
```

### 1.6 Event System

The simulator is driven by a priority queue of events ordered by timestamp.

**Event Types:**
- `TASK_ARRIVAL` — new task enters the system (from workload generator)
- `TASK_COMPLETION` — running task finishes its inference
- `QUANTUM_EXPIRED` — preemption timer fires on a core
- `DEADLINE_CHECK` — periodic sweep to update deadline boosts (every 1 ms simulated)
- `SCHEDULING_DECISION` — the agent must choose what to do (triggered after arrival, completion, or preemption)
- `GPU_COMPLETION` — GPU inference finishes, results available
- `MIGRATION_CHECK` — periodic load balance evaluation (every 10 ms simulated)

The simulator loop:
1. Pop next event from the min-heap
2. Advance the clock to event timestamp
3. Process the event (update core/task state)
4. If event triggers a scheduling decision, extract state → call agent → apply action
5. Log the (state, action, next_state, reward) tuple
6. Repeat until episode ends

### 1.7 Memory Subsystem Model

Mirrors the Phase 3 model memory allocator:

```python
@dataclass
class MemorySubsystem:
    total_ram_mb: int
    weight_pool_mb: int           # read-only model weights (16 MB on QEMU, scales with RAM)
    workspace_pool_mb: int        # inference workspace (8 MB on QEMU, scales with RAM)
    weight_pool_used_mb: float
    workspace_pool_used_mb: float
    loaded_models: Dict[int, ModelInfo]  # model_handle → ModelInfo
    model_refcounts: Dict[int, int]      # for zero-copy sharing tracking
```

The simulator checks memory constraints before allowing model loads and tracks pool pressure as a feature for the scheduling models.

### 1.8 GPU Model (Simplified)

For the Jetson platform, the GPU is modeled as a queue with a throughput model. Tasks submitted to the GPU have a latency based on their `ops_per_inference` divided by 8 TFLOPS (Jetson's rated performance), plus a fixed overhead for DMA transfer (~50 µs) and cache flush (~10 µs). The GPU processes one inference at a time (cooperative scheduling, per design doc Section 7.3).

```python
@dataclass
class GPUModel:
    available: bool                # False for Pi 5
    throughput_tflops: float       # 8.0 for Jetson
    queue: List[GPURequest]
    current_job: Optional[GPURequest]
    dma_overhead_ns: int           # 50_000
    cache_flush_overhead_ns: int   # 10_000
```

---

## 2. State, Action, and Reward Design

This section defines the interface between the simulator and the scheduling agents. All three models share the same state/action/reward specification, ensuring fair comparison.

### 2.1 Observation Space (State Vector)

The state is extracted at every `SCHEDULING_DECISION` event. It is a fixed-size numerical vector to be compatible with all three model types.

**Per-Core Features (6 features × N_CORES):**

| # | Feature | Range | Description |
|---|---------|-------|-------------|
| 1 | `utilization` | [0, 1] | Sliding window CPU utilization |
| 2 | `run_queue_depth` | [0, 32] normalized | Number of tasks waiting on this core |
| 3 | `cache_pressure` | [0, 5] clipped | Working set sum / cache size |
| 4 | `core_type` | {0, 1} | 0 = efficiency, 1 = performance |
| 5 | `isolated` | {0, 1} | Whether core is isolated for RT |
| 6 | `current_task_priority` | [0, 1] | Current task's priority / 7, or 0 if idle |

For 6 cores: **36 features**.

**Per-Pending-Task Features (8 features × K tasks, K = top-8 by priority):**

| # | Feature | Range | Description |
|---|---------|-------|-------------|
| 1 | `priority_norm` | [0, 1] | priority / 7 |
| 2 | `deadline_urgency` | [0, 1] | 1 − (time_to_deadline / max_deadline), clipped; 1.0 = overdue |
| 3 | `working_set_norm` | [0, 1] | working_set_mb / max_working_set |
| 4 | `model_size_norm` | [0, 1] | model_size_mb / max_model_size |
| 5 | `inference_duration_norm` | [0, 1] | inference_duration_ns / max_inference_duration |
| 6 | `can_use_gpu` | {0, 1} | GPU eligible flag |
| 7 | `wait_time_norm` | [0, 1] | (now − arrival_time) / max_wait_time |
| 8 | `component_type_onehot` | 4 bits | One-hot: anomaly/predmaint/security/other |

Wait — using one-hot for component type changes the size. Let's simplify: encode component type as a single normalized value (0.0, 0.33, 0.67, 1.0).

So 8 features × 8 tasks = **64 features**. Tasks are sorted by effective_priority descending. If fewer than 8 tasks are pending, pad with zeros.

**Global Features (8 features):**

| # | Feature | Range | Description |
|---|---------|-------|-------------|
| 1 | `total_ready_count_norm` | [0, 1] | total_ready / max_ready |
| 2 | `deadline_miss_rate` | [0, 1] | misses / total_completed over last 100 tasks |
| 3 | `avg_latency_norm` | [0, 1] | avg inference latency / target latency |
| 4 | `weight_pool_pressure` | [0, 1] | weight_pool_used / weight_pool_total |
| 5 | `workspace_pool_pressure` | [0, 1] | workspace_pool_used / workspace_pool_total |
| 6 | `gpu_queue_depth_norm` | [0, 1] | gpu_queue_depth / max_gpu_queue |
| 7 | `system_load_imbalance` | [0, 1] | std_dev(core_utilizations) / mean(core_utilizations) |
| 8 | `time_in_episode_norm` | [0, 1] | current_time / episode_duration |

**Total State Vector: 36 + 64 + 8 = 108 features.**

All features are normalized to [0, 1] to ensure numerical stability across models.

### 2.2 Action Space

The scheduling decision is decomposed into three sub-actions applied to the highest-priority pending task:

**Action 1: Core Assignment** (discrete, N_CORES + 1 options)
- Values 0 through N_CORES−1: assign to specific core
- Value N_CORES: assign to GPU (if available and task is GPU-eligible; else no-op, falls back to least-loaded core)

**Action 2: Priority Adjustment** (discrete, 3 options)
- 0: lower effective_priority by 1 (min 0)
- 1: keep current effective_priority
- 2: raise effective_priority by 1 (max 7)

**Action 3: Preempt** (binary)
- 0: enqueue at back of target core's run queue
- 1: preempt current task on target core (if lower priority)

Combined action space: (N_CORES + 1) × 3 × 2 = **42 discrete actions** on Jetson (7 × 3 × 2) or **30 on Pi 5** (5 × 3 × 2).

For the MLP and RL models, this is a single discrete output. For XGBoost, we decompose into three separate classifiers (see Section 4).

### 2.3 Reward Function

The reward is computed after each scheduling decision takes effect and the next state is observed. It is a weighted sum of four components:

```
R(s, a, s') = w_d · R_deadline + w_l · R_latency + w_b · R_balance + w_p · R_power
```

**Default weights:** `w_d = 0.50`, `w_l = 0.25`, `w_b = 0.15`, `w_p = 0.10`

**R_deadline (Deadline Compliance):**
```
For each task completed since last step:
    if completed before deadline:  +1.0
    if completed after deadline:   −2.0 × (overshoot_ms / deadline_ms)  (clipped to −2.0)
    if no deadline:                +0.1  (small bonus for completing any work)
Average over tasks completed.
```

**R_latency (Inference Latency):**
```
R_latency = 1.0 − (actual_latency / target_latency), clipped to [−1.0, 1.0]
target_latency is per-component:
    anomaly_detector:       5 ms
    predictive_maintenance: 50 ms
    security_monitor:       20 ms
    system:                 10 ms
```

**R_balance (Core Utilization Balance):**
```
R_balance = 1.0 − coefficient_of_variation(core_utilizations)
         = 1.0 − (std_dev / mean), clipped to [0.0, 1.0]
Idle cores count as 0.0 utilization.
```

**R_power (Power Efficiency):**
```
R_power = 1.0 − (weighted_active_power / max_possible_power)
where weighted_active_power = sum(core_utilization[i] × power_weight[i])
Only meaningful on big.LITTLE; on homogeneous platforms, this rewards lower overall utilization for the same throughput.
```

### 2.4 Episode Structure

Each episode represents a fixed time window of simulated scheduling:
- **Duration:** 10 seconds simulated time (configurable)
- **Warmup:** First 1 second of each episode is warmup — metrics are collected but not included in reward
- **Termination:** Episode ends at duration limit or if the system enters an unrecoverable state (all cores deadlocked, which shouldn't happen)
- **Reset:** At episode start, cores are idle, memory pools are empty, and a fresh workload sequence begins

---

## 3. Workload Generator

### 3.1 Component Workload Profiles

Each workload profile generates a stream of `TASK_ARRIVAL` events with realistic timing and resource characteristics.

**Anomaly Detector (High-Frequency, Small Model):**

| Parameter | Distribution | Notes |
|-----------|-------------|-------|
| Arrival interval | Exponential(λ=100 Hz) | ~10 ms between requests |
| Burst mode | Poisson bursts of 5–20 requests | 5% of the time |
| Model size | Fixed 12 MB | Small vibration model |
| Working set | Fixed 4 MB | Fits in L2 |
| Inference duration | Normal(µ=1 ms, σ=0.2 ms) on perf core | Scales with core speed |
| Ops per inference | 1.2 GFLOPS | From component spec |
| Priority | NORMAL (3) | Steady-state default |
| Deadline | 5 ms (soft) | Alert latency target |
| GPU eligible | No | Too small to benefit |

**Predictive Maintenance (Bursty, Medium Model):**

| Parameter | Distribution | Notes |
|-----------|-------------|-------|
| Arrival interval | Periodic(1 Hz) + burst(10 Hz for 500 ms) every 30s | Sensor aggregation cycle |
| Model size | Fixed 45 MB | Larger temporal model |
| Working set | Fixed 16 MB | Needs L3 |
| Inference duration | Normal(µ=25 ms, σ=5 ms) on perf core | |
| Ops per inference | 4.8 GFLOPS | |
| Priority | HIGH (5) | Equipment safety |
| Deadline | 50 ms (hard) | Control loop deadline |
| GPU eligible | Yes | Benefits from Tensor Cores |

**Security Monitor (Background, Continuous):**

| Parameter | Distribution | Notes |
|-----------|-------------|-------|
| Arrival interval | Periodic(10 Hz) | Network packet batch analysis |
| Model size | Fixed 8 MB | Lightweight classifier |
| Working set | Fixed 3 MB | Small |
| Inference duration | Normal(µ=3 ms, σ=1 ms) on perf core | |
| Ops per inference | 0.8 GFLOPS | |
| Priority | LOW (2) | Background, can be preempted |
| Deadline | 20 ms (soft) | Can tolerate delay |
| GPU eligible | No | |

**System Tasks (OS Overhead):**

| Parameter | Distribution | Notes |
|-----------|-------------|-------|
| Arrival interval | Periodic(100 Hz) | Timer tick, bookkeeping |
| Model size | 0 | Not an SLM task |
| Working set | Fixed 0.5 MB | |
| Inference duration | Normal(µ=50 µs, σ=10 µs) | Very short |
| Priority | HIGH (6) | OS tasks are high priority |
| Deadline | 0 (none) | Best-effort but fast |
| GPU eligible | No | |

### 3.2 Scenario Compositions

Training data is generated from a variety of scenario mixes. Each scenario defines which component workloads are active and at what intensity.

| Scenario | Components Active | Total Load | Purpose |
|----------|------------------|------------|---------|
| `light_single` | Anomaly only | ~10% | Baseline easy case |
| `light_mixed` | Anomaly + Security | ~25% | Two-component interaction |
| `medium_mixed` | All three + System | ~50% | Typical operating point |
| `heavy_inference` | All three at 2× rate + System | ~80% | Stress test — forces tradeoffs |
| `burst_storm` | Normal + periodic burst of all three | Spikes to ~95% | Transient overload recovery |
| `deadline_pressure` | PredMaint at 3× rate | ~60%, heavy deadlines | Tests deadline compliance |
| `memory_pressure` | 4 PredMaint instances | Pool near-exhaustion | Tests memory-aware scheduling |
| `asymmetric` | Anomaly on cores 0-1, PredMaint on 4-5 | ~40% unbalanced | Tests migration/balancing |

During training data generation, each scenario runs for thousands of episodes with different random seeds.

### 3.3 Workload Variability

To prevent overfitting, the generator adds noise and variation:
- Inference duration varies per-call (Normal distribution as specified)
- Arrival times have jitter (±10% of base interval)
- Every 100th episode, model sizes are perturbed (±20%) to simulate model updates
- Occasionally (5% of episodes), a component "crashes" and stops generating tasks for 2 seconds, then resumes (testing recovery)

---

## 4. Model Architectures

### 4.1 Model A: Feedforward Neural Network (MLP)

**Role:** Fast, differentiable scheduler suitable for direct deployment in SLM-OS's kernel hot path.

**Architecture:**

```
Input (108) → Dense(256, ReLU) → BatchNorm → Dropout(0.1)
           → Dense(256, ReLU) → BatchNorm → Dropout(0.1)
           → Dense(128, ReLU) → BatchNorm
           → Dense(N_ACTIONS, Softmax)

N_ACTIONS = (N_CORES + 1) × 3 × 2
  Jetson: 42
  Pi 5:   30
```

**Parameter Count:** ~108 × 256 + 256 × 256 + 256 × 128 + 128 × 42 ≈ **131K parameters**

**Inference Latency (estimated):**
- FP32 on Cortex-A78 @ 1.5 GHz: ~15–30 µs (within 5% overhead budget for 10 ms quantum)
- INT8 quantized: ~5–10 µs

**Training Approach:** Supervised learning (imitation learning from expert policies), with optional fine-tuning via DAgger (Dataset Aggregation) where the trained model's own decisions are corrected by the expert and added back to the training set.

**Loss Function:** Cross-entropy loss over the action space, with per-sample weighting by reward magnitude (higher-reward expert decisions get more weight).

**Framework:** PyTorch (training) → ONNX export → quantize to INT8 → deploy as C inference in SLM-OS kernel.

### 4.2 Model B: Gradient-Boosted Decision Trees (XGBoost)

**Role:** Highly interpretable baseline that provides fast inference and feature importance analysis. Helps identify which state features matter most for scheduling quality.

**Architecture:** Three separate XGBoost classifiers, one per sub-action:

```
Classifier 1: Core Assignment
    Input:  108 features (state vector)
    Output: N_CORES + 1 classes (multi-class classification)
    Trees:  200 trees, max_depth=8

Classifier 2: Priority Adjustment
    Input:  108 features + predicted core assignment (one-hot)
    Output: 3 classes (lower / keep / raise)
    Trees:  100 trees, max_depth=6

Classifier 3: Preempt Decision
    Input:  108 features + predicted core + predicted priority adjustment
    Output: 2 classes (no preempt / preempt)
    Trees:  100 trees, max_depth=6
```

**Why decomposed:** XGBoost handles multi-class well, but the combined 42-class action space would be sparse. Decomposition lets each classifier focus on its sub-problem and produces more interpretable feature importances per decision type.

**Inference Latency (estimated):**
- 200 trees × depth 8: ~5–15 µs on A78 (tree traversal is branch-heavy but small working set)
- No quantization needed — integer comparisons natively

**Training Approach:** Supervised learning on expert-labeled data. Each training sample is (state_vector, expert_core_choice, expert_priority_adj, expert_preempt).

**Framework:** XGBoost Python for training → export to JSON or custom C predictor for SLM-OS deployment.

### 4.3 Model C: RL Policy Network (PPO)

**Role:** The most capable model — learns scheduling policies that may exceed expert heuristics by exploring novel strategies. This is the primary research contribution.

**Architecture: Actor-Critic with shared backbone.**

```
Shared Backbone:
    Input (108) → Dense(256, ReLU) → Dense(256, ReLU)

Actor Head (policy):
    Backbone output → Dense(128, ReLU) → Dense(N_ACTIONS, Softmax)
    Outputs: π(a|s) — probability distribution over actions

Critic Head (value function):
    Backbone output → Dense(128, ReLU) → Dense(1, Linear)
    Outputs: V(s) — estimated cumulative future reward
```

**Parameter Count:** Shared backbone: ~93K; Actor head: ~38K; Critic head: ~33K. **Total: ~164K parameters.**

**Training Algorithm: Proximal Policy Optimization (PPO)**

PPO is chosen over alternatives because:
- Stable training (clipped objective prevents catastrophic policy updates)
- Works well with discrete action spaces
- Proven effective for resource management problems (Decima, Park)
- Relatively sample-efficient compared to vanilla policy gradient

**Key Hyperparameters (starting points, will tune):**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 3e-4 | Adam optimizer |
| Clip ratio (ε) | 0.2 | Standard PPO clip |
| Discount (γ) | 0.99 | Long-horizon — scheduling decisions have lasting effects |
| GAE lambda (λ) | 0.95 | Generalized Advantage Estimation |
| Entropy coefficient | 0.01 | Encourage exploration early |
| Value loss coefficient | 0.5 | Balance actor/critic learning |
| Batch size | 2048 transitions | Collected across parallel envs |
| Mini-batch size | 256 | SGD mini-batches within PPO update |
| PPO epochs per update | 4 | Reuse collected batch |
| Parallel environments | 16 | Vectorized simulation |
| Total training steps | 5M–10M | ~2500–5000 episodes |

**Curriculum Learning:** Training proceeds through increasingly difficult scenarios:
1. **Phase A (steps 0–1M):** `light_single` and `light_mixed` only
2. **Phase B (steps 1M–3M):** Add `medium_mixed` and `deadline_pressure`
3. **Phase C (steps 3M–5M):** Full scenario distribution
4. **Phase D (steps 5M–10M):** Weighted toward `heavy_inference` and `burst_storm` (the hard cases)

**Framework:** Stable Baselines 3 (SB3) with a custom Gymnasium environment wrapping the simulator. SB3 provides the PPO implementation; the custom env implements `reset()`, `step()`, `observation_space`, and `action_space`.

**Deployment:** Export actor network (no critic needed at inference) → ONNX → INT8 quantization → C inference in SLM-OS.

---

## 5. Expert Scheduling Policies (for Dataset Generation)

### 5.1 Expert Policy 1: SLM-OS Hybrid (Baseline)

This replicates the Phase 3 scheduler logic exactly:
- Priority queue ordered by effective_priority
- Deadline boost: <100 ms → +1; <50 ms → HIGH(6); <10 ms → CRITICAL(7)
- Working set < 8 MB → any core; ≥ 8 MB → performance core
- Deadline tasks pinned to non-boot CPUs
- `find_target_cpu()` picks least-loaded non-isolated core
- No preemption decision (always enqueue)

### 5.2 Expert Policy 2: Earliest Deadline First (EDF)

- Strict deadline ordering: task with nearest deadline runs first
- Ties broken by priority, then FIFO
- Core assignment: least-loaded core
- Preempts current task if new task has nearer deadline
- No working set awareness

### 5.3 Expert Policy 3: Weighted Multi-Objective

A richer heuristic that considers multiple factors simultaneously:
```
score(task, core) = w1 × deadline_urgency(task)
                  + w2 × priority_norm(task)
                  + w3 × cache_affinity(task, core)
                  + w4 × (1 − core_utilization(core))
                  + w5 × power_efficiency(core)

where:
    cache_affinity = 1.0 if task.working_set ≤ core.cache_remaining, else 0.0
    w1=0.35, w2=0.25, w3=0.20, w4=0.15, w5=0.05
```
Assign each pending task to the core that maximizes its score. Preempt if score exceeds current task's score by > 0.3.

### 5.4 Expert Policy 4: Random (Negative Examples)

- Random core assignment (uniform)
- Random priority adjustment
- Random preempt decision (50/50)
- Used to generate negative examples for supervised training and as an RL baseline

### 5.5 Oracle Policy (Upper Bound, Offline Only)

Runs the full episode, then backtracks to find the scheduling decisions that would have minimized deadline misses. This is computationally expensive (runs each episode multiple times with different strategies via beam search) and serves only as a theoretical performance ceiling, not as a training signal.

---

## 6. Dataset Specification

### 6.1 Dataset Format

Each sample in the training dataset is a transition tuple:

```
(state, action, reward, next_state, done, metadata)
```

Stored as Apache Parquet files for efficient columnar access. One file per scenario per expert policy.

**Schema:**

| Column | Type | Shape | Description |
|--------|------|-------|-------------|
| `state` | float32 | (108,) | Normalized state vector |
| `action` | int32 | scalar | Combined action index (0 to N_ACTIONS−1) |
| `action_core` | int32 | scalar | Decomposed: core assignment |
| `action_priority` | int32 | scalar | Decomposed: priority adjustment |
| `action_preempt` | int32 | scalar | Decomposed: preempt flag |
| `reward` | float32 | scalar | Immediate reward |
| `reward_deadline` | float32 | scalar | Deadline component |
| `reward_latency` | float32 | scalar | Latency component |
| `reward_balance` | float32 | scalar | Balance component |
| `reward_power` | float32 | scalar | Power component |
| `next_state` | float32 | (108,) | State after action |
| `done` | bool | scalar | Episode terminated |
| `expert_policy` | string | — | Which expert generated this |
| `scenario` | string | — | Scenario name |
| `platform` | string | — | jetson / pi5 / biglittle |
| `episode_id` | int64 | — | Unique episode identifier |
| `step_in_episode` | int32 | — | Step number within episode |
| `sim_time_ns` | int64 | — | Simulation timestamp |

### 6.2 Dataset Size Targets

| Expert Policy | Episodes per Scenario | Scenarios | Steps per Episode | Total Samples |
|---------------|----------------------|-----------|-------------------|---------------|
| SLM-OS Hybrid | 500 | 8 | ~1000 | ~4M |
| EDF | 500 | 8 | ~1000 | ~4M |
| Weighted Multi-Obj | 500 | 8 | ~1000 | ~4M |
| Random | 200 | 8 | ~1000 | ~1.6M |
| **Total** | | | | **~13.6M samples** |

At ~108 × 4 bytes × 2 (state + next_state) + overhead ≈ 1 KB per sample, total dataset: **~14 GB** in Parquet (with compression, likely ~4–6 GB).

Generation time estimate: 1000 steps/episode × 1 µs/step simulation ≈ 1 ms/episode wall clock (excluding logging overhead). 6800 episodes × 3 platforms = 20,400 episodes total. Even at 100 ms per episode including I/O, that's ~34 minutes total generation time. Very feasible.

### 6.3 Train/Validation/Test Split

- **Train:** 70% (random episode-level split, not step-level — prevents temporal leakage)
- **Validation:** 15%
- **Test:** 15%

Additionally, hold out the `burst_storm` scenario entirely from training for the MLP and XGBoost as an out-of-distribution test. The RL model sees it during curriculum Phase D.

---

## 7. Training Pipelines

### 7.1 MLP Training Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Load Parquet │────>│ Filter best  │────>│ Normalize    │
│ (train split)│     │ experts only │     │ (z-score or  │
│              │     │ (top-3, not  │     │  min-max per │
│              │     │  random)     │     │  feature)    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                │
                                                v
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Export ONNX  │<────│ Train MLP    │<────│ Weight by    │
│ + quantize   │     │ (PyTorch)    │     │ reward (high │
│ to INT8      │     │ 50 epochs    │     │ reward = more│
│              │     │ lr schedule  │     │ weight)      │
└──────────────┘     └──────────────┘     └──────────────┘
        │
        v
┌──────────────┐     ┌──────────────┐
│ DAgger Loop  │────>│ Retrain with │ (optional, 3 iterations)
│ (run model in│     │ corrected    │
│  simulator,  │     │ data added)  │
│  expert      │     │              │
│  corrects)   │     │              │
└──────────────┘     └──────────────┘
```

**Key Details:**
- Optimizer: AdamW, lr=1e-3 with cosine annealing to 1e-5 over 50 epochs
- Batch size: 1024
- Loss: Cross-entropy with sample weights = |reward| (normalized)
- Early stopping: patience=5 on validation loss
- DAgger iterations: 3 rounds of 1000 episodes each, where the MLP makes decisions but the expert provides the "correct" action. New data is mixed 50/50 with original training data.

### 7.2 XGBoost Training Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Load Parquet │────>│ Filter best  │────>│ Feature      │
│ (train split)│     │ experts only │     │ engineering  │
│              │     │              │     │ (add derived │
│              │     │              │     │  features)   │
└──────────────┘     └──────────────┘     └──────────────┘
                                                │
                         ┌──────────────────────┤
                         v                      v
                  ┌──────────────┐     ┌──────────────┐
                  │ Train Core   │     │ Train Priority│
                  │ Assignment   │     │ Adjustment    │
                  │ Classifier   │     │ Classifier    │
                  │ (XGBoost     │     │ (XGBoost      │
                  │  multi-class)│     │  multi-class) │
                  └──────────────┘     └──────────────┘
                         │                      │
                         v                      v
                  ┌──────────────┐     ┌──────────────┐
                  │ Train Preempt│     │ Export models │
                  │ Classifier   │────>│ (JSON + C     │
                  │ (XGBoost     │     │  predictor)   │
                  │  binary)     │     │              │
                  └──────────────┘     └──────────────┘
```

**Derived Features (added for XGBoost):**
- `max_deadline_urgency` — urgency of the most urgent pending task
- `core_util_std` — standard deviation of core utilizations
- `deadline_task_count` — number of pending tasks with deadlines
- `gpu_should_use` — 1 if GPU available AND most urgent task is GPU-eligible AND GPU queue < 2
- `best_cache_fit_core` — core ID where task's working set fits best

These hand-engineered features help XGBoost since it doesn't learn feature interactions as naturally as neural networks.

**Hyperparameter Tuning:** 5-fold cross-validation with Optuna (100 trials):
- `n_estimators`: [50, 500]
- `max_depth`: [4, 12]
- `learning_rate`: [0.01, 0.3]
- `subsample`: [0.6, 1.0]
- `colsample_bytree`: [0.6, 1.0]
- `min_child_weight`: [1, 10]

### 7.3 RL (PPO) Training Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    Training Loop (PPO)                        │
│                                                              │
│  ┌────────────────────┐         ┌─────────────────────────┐ │
│  │ 16 Parallel Envs   │────────>│ Rollout Buffer          │ │
│  │ (vectorized sim)   │ collect │ (2048 transitions)      │ │
│  │                    │ steps   │                          │ │
│  └────────────────────┘         └─────────────────────────┘ │
│            ^                              │                  │
│            │ updated                      │ compute          │
│            │ policy                       │ advantages       │
│            │                              v                  │
│  ┌────────────────────┐         ┌─────────────────────────┐ │
│  │ Actor-Critic       │<────────│ PPO Update              │ │
│  │ Network            │ gradient│ (4 epochs, 256 batch)   │ │
│  │                    │ update  │ clip ratio = 0.2        │ │
│  └────────────────────┘         └─────────────────────────┘ │
│            │                                                 │
│            │ every 10K steps                                 │
│            v                                                 │
│  ┌────────────────────┐         ┌─────────────────────────┐ │
│  │ Evaluation Run     │────────>│ Metrics & Checkpoints   │ │
│  │ (100 episodes,     │         │ (TensorBoard, W&B)      │ │
│  │  deterministic)    │         │                          │ │
│  └────────────────────┘         └─────────────────────────┘ │
│                                                              │
│  Curriculum: Phase A → B → C → D (see Section 4.3)          │
└──────────────────────────────────────────────────────────────┘
```

**Pre-training (optional but recommended):**
Before PPO training, initialize the actor network weights using behavioral cloning from the expert dataset (same as MLP training, 10 epochs). This gives PPO a warm start rather than random exploration, dramatically accelerating convergence.

**Reward Shaping Additions for RL Only:**
- Small step penalty: −0.001 per step (encourages faster decisions)
- Exploration bonus: +0.05 for choosing a core that hasn't been used in last 5 decisions (encourages trying all cores)
- These are removed in the final evaluation to ensure fair comparison

---

## 8. Evaluation Framework

### 8.1 Metrics

All three models (plus the expert policies and random baseline) are evaluated on the same held-out test episodes using identical random seeds.

**Primary Metrics:**

| Metric | Target | Description |
|--------|--------|-------------|
| Deadline Compliance Rate (DCR) | ≥ 95% | % of deadline tasks completed on time |
| Mean Inference Latency | < component target | Average time from arrival to completion |
| P99 Inference Latency | < 2× component target | Tail latency |
| Scheduling Overhead | < 5% | Model inference time / quantum (10 ms) |

**Secondary Metrics:**

| Metric | Description |
|--------|-------------|
| Core Utilization Balance | Coefficient of variation across cores |
| Throughput | Tasks completed per second |
| Power Efficiency Score | Weighted utilization (lower is better for same throughput) |
| Deadline Miss Severity | Mean overshoot when deadlines are missed |
| Starvation Count | Tasks waiting > 10× expected latency |
| Priority Inversion Count | Higher-priority task waited behind lower |

### 8.2 Evaluation Scenarios

| Scenario Set | Purpose |
|-------------|---------|
| Test split of all training scenarios | In-distribution performance |
| `burst_storm` (held out for MLP/XGBoost) | Out-of-distribution generalization |
| Novel `cascading_failure` scenario | Component crashes trigger load spikes on remaining components |
| Novel `model_swap` scenario | Hot-swap of a component mid-episode (model size changes) |

### 8.3 Statistical Significance

Each evaluation runs 200 episodes per scenario per model. Report mean ± 95% confidence interval. Use paired t-test to compare models on the same episodes (same random seeds).

### 8.4 Ablation Studies

- **State features:** Remove feature groups (per-core, per-task, global) and measure impact
- **Reward weights:** Vary w_d, w_l, w_b, w_p and show Pareto frontier
- **Model size:** Test MLP variants (64, 128, 256, 512 neurons) for accuracy vs latency tradeoff
- **Action space:** Test flat (42 classes) vs decomposed (3 classifiers) for MLP

---

## 9. SLM-OS Integration Path

### 9.1 Deployment Architecture

The trained model runs inside the Rust scheduling policy layer (`runtime/src/sched/`), called from the C kernel at each scheduling decision point.

```
C Kernel (sched/core.c)
    │
    │  slm_schedule_next_task() — called on quantum expiry / task completion
    │
    ├──> C: Extract raw state from kernel data structures
    ├──> C: Pack into SchedulerState struct
    ├──> FFI call → Rust: ai_schedule(state: &SchedulerState) → SchedulerAction
    │       │
    │       ├──> Rust: Normalize state features
    │       ├──> Rust: Run model inference (ONNX Runtime Lite or custom C)
    │       ├──> Rust: Decode action → (core, priority_adj, preempt)
    │       └──> Rust: Return SchedulerAction
    │
    ├──> C: Apply core assignment
    ├──> C: Apply priority adjustment
    └──> C: Preempt if requested
```

### 9.2 Model Format for Deployment

- **MLP:** Export PyTorch → ONNX → convert to flat weight arrays + hand-written C inference (matrix multiply + ReLU). No runtime dependency. Weights stored as `const` arrays in a C source file.
- **XGBoost:** Export to JSON → convert to C if-else chains or lookup tables using `treelite` compiler. No runtime dependency.
- **RL Policy:** Same as MLP (only the actor network is deployed, not the critic).

### 9.3 Fallback Behavior

If the AI model produces an invalid action (e.g., assigns to GPU when unavailable, or assigns to an offline core), the system falls back to the Phase 3 heuristic scheduler. The fallback rate is tracked as a metric. If fallback rate exceeds 5%, the model should be retrained.

---

## 10. Implementation Phases and Milestones

### Phase S1: Simulator Core (2 weeks)

**Goal:** Working discrete-event simulator with platform models, basic workload generation, and heuristic expert policy.

| # | Task | Deliverable | Days |
|---|------|-------------|------|
| 1.1 | Project setup: Python package, dependencies (numpy, gymnasium, dataclasses) | `slm_sim/` package structure | 0.5 |
| 1.2 | Implement `CoreState`, `SimTask`, `MemorySubsystem`, `GPUModel` data classes | `slm_sim/models.py` | 1 |
| 1.3 | Implement event queue (min-heap) and `SimulatorEngine` main loop | `slm_sim/engine.py` | 2 |
| 1.4 | Implement Jetson and Pi 5 platform profiles | `slm_sim/platforms.py` | 0.5 |
| 1.5 | Implement state vector extraction (108 features) | `slm_sim/observation.py` | 1 |
| 1.6 | Implement action application logic | `slm_sim/actions.py` | 1 |
| 1.7 | Implement reward computation (4 components) | `slm_sim/reward.py` | 1 |
| 1.8 | Implement SLM-OS Hybrid expert policy | `slm_sim/experts/hybrid.py` | 1 |
| 1.9 | Smoke test: run 100 episodes with hybrid policy, verify metrics are reasonable | Test script + log output | 1 |
| 1.10 | Gymnasium `Env` wrapper (`reset`, `step`, observation/action spaces) | `slm_sim/gym_env.py` | 1 |

**Milestone Gate:** Simulator runs 1000 episodes/minute. Expert policy achieves >90% DCR on `light_single`.

### Phase S2: Workload Generator and Expert Policies (1.5 weeks)

**Goal:** All four component workload profiles, all scenario compositions, and all expert policies implemented and validated.

| # | Task | Deliverable | Days |
|---|------|-------------|------|
| 2.1 | Implement Anomaly Detector workload profile | `slm_sim/workloads/anomaly.py` | 0.5 |
| 2.2 | Implement Predictive Maintenance workload profile | `slm_sim/workloads/predmaint.py` | 0.5 |
| 2.3 | Implement Security Monitor workload profile | `slm_sim/workloads/security.py` | 0.5 |
| 2.4 | Implement System Tasks workload profile | `slm_sim/workloads/system.py` | 0.5 |
| 2.5 | Implement scenario composer (combines profiles at specified intensities) | `slm_sim/workloads/scenarios.py` | 1 |
| 2.6 | Implement EDF expert policy | `slm_sim/experts/edf.py` | 0.5 |
| 2.7 | Implement Weighted Multi-Objective expert policy | `slm_sim/experts/weighted.py` | 1 |
| 2.8 | Implement Random baseline policy | `slm_sim/experts/random_policy.py` | 0.25 |
| 2.9 | Implement Oracle policy (offline beam search) | `slm_sim/experts/oracle.py` | 2 |
| 2.10 | Validate all experts across all scenarios, collect baseline metrics | `results/expert_baselines.csv` | 1 |
| 2.11 | Add workload variability (noise, jitter, crashes) | Update workload modules | 0.5 |

**Milestone Gate:** All expert policies produce valid scheduling traces. Hybrid and Weighted policies achieve >90% DCR on `medium_mixed`. Random achieves <60% DCR (confirming the problem is non-trivial).

### Phase S3: Dataset Generation (1 week)

**Goal:** Generate, validate, and split the complete training dataset.

| # | Task | Deliverable | Days |
|---|------|-------------|------|
| 3.1 | Implement Parquet logging of transition tuples | `slm_sim/logging.py` | 1 |
| 3.2 | Implement parallel episode runner (multiprocessing) | `slm_sim/runner.py` | 1 |
| 3.3 | Generate full dataset: 3 experts × 8 scenarios × 500 episodes × 3 platforms | `data/raw/` (~4–6 GB) | 1 |
| 3.4 | Generate random baseline dataset: 8 scenarios × 200 episodes × 3 platforms | `data/raw/random/` | 0.25 |
| 3.5 | Dataset validation: check for NaN, out-of-range, verify reward distributions | `scripts/validate_dataset.py` | 0.5 |
| 3.6 | Train/val/test split at episode level (70/15/15) | `data/splits/` | 0.25 |
| 3.7 | Compute and save normalization statistics (mean, std, min, max per feature) | `data/normalization.json` | 0.25 |
| 3.8 | Dataset analysis: feature distributions, reward distributions, expert comparison | `notebooks/dataset_analysis.ipynb` | 1 |

**Milestone Gate:** Dataset passes all validation checks. Feature distributions are sensible. Expert policies show clear performance stratification (Weighted > Hybrid ≈ EDF >> Random).

### Phase S4: MLP Training (1 week)

**Goal:** Trained MLP model that matches or exceeds the best expert policy.

| # | Task | Deliverable | Days |
|---|------|-------------|------|
| 4.1 | Implement PyTorch dataset loader from Parquet files | `training/mlp/dataset.py` | 0.5 |
| 4.2 | Implement MLP architecture (Section 4.1) | `training/mlp/model.py` | 0.5 |
| 4.3 | Implement training loop with reward-weighted cross-entropy | `training/mlp/train.py` | 1 |
| 4.4 | Train initial model (50 epochs, early stopping) | `models/mlp/mlp_v1.pt` | 0.5 |
| 4.5 | Evaluate on test set, compare to expert baselines | `results/mlp_v1_eval.csv` | 0.5 |
| 4.6 | Implement DAgger loop (3 iterations) | `training/mlp/dagger.py` | 1 |
| 4.7 | Retrain with DAgger data | `models/mlp/mlp_v2_dagger.pt` | 0.5 |
| 4.8 | Export to ONNX, quantize to INT8, benchmark inference latency | `models/mlp/mlp_v2.onnx` | 1 |
| 4.9 | Hyperparameter sensitivity analysis (hidden size, depth, dropout) | `results/mlp_hparam_sweep.csv` | 1 |

**Milestone Gate:** MLP achieves DCR ≥ best expert on `medium_mixed`. ONNX inference < 30 µs on x86 (proxy for ARM).

### Phase S5: XGBoost Training (1 week)

**Goal:** Trained XGBoost ensemble with feature importance analysis.

| # | Task | Deliverable | Days |
|---|------|-------------|------|
| 5.1 | Implement feature engineering (derived features) | `training/xgboost/features.py` | 0.5 |
| 5.2 | Implement three-classifier training pipeline | `training/xgboost/train.py` | 1 |
| 5.3 | Run Optuna hyperparameter search (100 trials per classifier) | `models/xgboost/optuna_results.json` | 1 |
| 5.4 | Train final classifiers with best hyperparameters | `models/xgboost/core_clf.json` etc. | 0.5 |
| 5.5 | Extract and visualize feature importances | `results/xgboost_feature_importance.png` | 0.5 |
| 5.6 | Evaluate on test set, compare to MLP and experts | `results/xgboost_eval.csv` | 0.5 |
| 5.7 | Export to C predictor using treelite | `models/xgboost/predictor.c` | 1 |
| 5.8 | Benchmark inference latency of C predictor | `results/xgboost_latency.csv` | 0.5 |
| 5.9 | SHAP analysis: per-sample decision explanations | `notebooks/xgboost_shap.ipynb` | 1 |

**Milestone Gate:** XGBoost achieves DCR within 5% of MLP. Feature importance reveals interpretable scheduling logic. C predictor inference < 20 µs.

### Phase S6: RL (PPO) Training (2 weeks)

**Goal:** RL policy that discovers scheduling strategies exceeding expert heuristics.

| # | Task | Deliverable | Days |
|---|------|-------------|------|
| 6.1 | Implement Gymnasium-compatible vectorized environment | `training/rl/vec_env.py` | 1 |
| 6.2 | Implement custom actor-critic network for SB3 | `training/rl/network.py` | 1 |
| 6.3 | Implement curriculum schedule (Phases A–D) | `training/rl/curriculum.py` | 0.5 |
| 6.4 | Pre-train actor via behavioral cloning (10 epochs on expert data) | `models/rl/pretrained_actor.pt` | 0.5 |
| 6.5 | PPO training Phase A: light scenarios (1M steps) | Checkpoint + TensorBoard logs | 1 |
| 6.6 | PPO training Phase B: add medium + deadline (2M steps) | Checkpoint | 1 |
| 6.7 | PPO training Phase C: full distribution (2M steps) | Checkpoint | 1 |
| 6.8 | PPO training Phase D: hard scenarios (2M–5M steps) | Checkpoint | 2 |
| 6.9 | Evaluate on test set and out-of-distribution scenarios | `results/rl_eval.csv` | 0.5 |
| 6.10 | Ablation: reward weight sensitivity analysis | `results/rl_reward_ablation.csv` | 1 |
| 6.11 | Export actor to ONNX, quantize to INT8 | `models/rl/policy_v1.onnx` | 0.5 |
| 6.12 | Policy visualization: what does the RL agent do differently from experts? | `notebooks/rl_policy_analysis.ipynb` | 1 |

**Milestone Gate:** RL policy achieves DCR ≥ 97% on `medium_mixed` (exceeding experts). Shows meaningful improvement on `burst_storm` and `deadline_pressure` scenarios. Training curves show stable convergence.

### Phase S7: Comparative Evaluation and Documentation (1 week)

**Goal:** Complete head-to-head comparison, statistical analysis, and integration spec.

| # | Task | Deliverable | Days |
|---|------|-------------|------|
| 7.1 | Run all models + experts on identical test episodes (200 × 10 scenarios × 3 platforms) | `results/full_comparison.csv` | 0.5 |
| 7.2 | Statistical significance testing (paired t-tests, confidence intervals) | `results/significance.csv` | 0.5 |
| 7.3 | Generate comparison tables and charts | `results/charts/` | 1 |
| 7.4 | Ablation study: state feature groups | `results/ablation_features.csv` | 1 |
| 7.5 | Write integration specification for SLM-OS deployment | `docs/ai_scheduler_integration.md` | 1 |
| 7.6 | Write final analysis report | `docs/ai_scheduler_results.md` | 1.5 |
| 7.7 | Package models and deployment artifacts | `deploy/` directory | 0.5 |

**Milestone Gate:** Clear winner identified (or Pareto analysis if tradeoffs exist). Integration path documented. All results reproducible from scripts.

---

## 11. Timeline Summary

```
Week 1–2:   Phase S1 (Simulator Core)
Week 3:     Phase S2 (Workload + Experts) — first half
Week 4:     Phase S2 (finish) + Phase S3 (Dataset Generation)
Week 5:     Phase S4 (MLP Training)
Week 6:     Phase S5 (XGBoost Training)
Week 7–8:   Phase S6 (RL/PPO Training)
Week 9:     Phase S7 (Evaluation + Documentation)
```

**Total: ~9 weeks** (fits within a single academic semester alongside other project work).

---

## 12. Repository Structure

```
slm-os-scheduler-ai/
├── slm_sim/                          # Simulator package
│   ├── __init__.py
│   ├── engine.py                     # Discrete-event simulation engine
│   ├── models.py                     # CoreState, SimTask, etc.
│   ├── platforms.py                  # Jetson, Pi5, big.LITTLE profiles
│   ├── observation.py                # State vector extraction
│   ├── actions.py                    # Action application logic
│   ├── reward.py                     # Reward computation
│   ├── gym_env.py                    # Gymnasium Env wrapper
│   ├── workloads/
│   │   ├── __init__.py
│   │   ├── anomaly.py
│   │   ├── predmaint.py
│   │   ├── security.py
│   │   ├── system.py
│   │   └── scenarios.py              # Scenario compositions
│   └── experts/
│       ├── __init__.py
│       ├── hybrid.py                 # SLM-OS Phase 3 heuristic
│       ├── edf.py                    # Earliest Deadline First
│       ├── weighted.py               # Weighted multi-objective
│       ├── random_policy.py
│       └── oracle.py                 # Offline optimal (upper bound)
│
├── training/                         # Model training code
│   ├── mlp/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── dagger.py
│   ├── xgboost/
│   │   ├── features.py
│   │   └── train.py
│   └── rl/
│       ├── vec_env.py
│       ├── network.py
│       ├── curriculum.py
│       └── train.py
│
├── evaluation/
│   ├── run_eval.py                   # Run all models on test episodes
│   ├── metrics.py                    # Metric computation
│   ├── statistical_tests.py          # Significance testing
│   └── visualization.py             # Charts and tables
│
├── scripts/
│   ├── generate_dataset.py
│   ├── validate_dataset.py
│   └── export_models.py              # ONNX + C export
│
├── notebooks/
│   ├── dataset_analysis.ipynb
│   ├── xgboost_shap.ipynb
│   └── rl_policy_analysis.ipynb
│
├── data/                             # Generated data (gitignored)
│   ├── raw/
│   ├── splits/
│   └── normalization.json
│
├── models/                           # Trained models (gitignored)
│   ├── mlp/
│   ├── xgboost/
│   └── rl/
│
├── results/                          # Evaluation results
│   ├── charts/
│   └── *.csv
│
├── deploy/                           # SLM-OS integration artifacts
│   ├── mlp_weights.c                 # Flat C arrays for MLP
│   ├── xgboost_predictor.c           # Compiled decision trees
│   ├── rl_policy_weights.c           # Actor network weights
│   └── ai_scheduler.h                # C API header
│
├── docs/
│   ├── ai_scheduler_integration.md
│   └── ai_scheduler_results.md
│
├── requirements.txt                  # Python dependencies
├── pyproject.toml
└── README.md
```

---

## 13. Dependencies

**Python Packages:**
- `numpy` — numerical computation
- `gymnasium` — RL environment interface
- `torch` (PyTorch) — MLP training
- `xgboost` — gradient-boosted trees
- `stable-baselines3` — PPO implementation
- `optuna` — hyperparameter optimization
- `pyarrow` — Parquet I/O
- `pandas` — data analysis
- `matplotlib`, `seaborn` — visualization
- `shap` — XGBoost explainability
- `onnx`, `onnxruntime` — model export and benchmark
- `scipy` — statistical tests
- `tensorboard` or `wandb` — training monitoring

**Hardware Requirements:**
- Training: Linux PC with RTX 3050 6GB (for RL training acceleration via PyTorch CUDA)
- Simulation: CPU-only (no GPU needed for the simulator itself)
- Estimated training time: MLP ~1 hour, XGBoost ~2 hours (with Optuna), RL ~8–24 hours

---

## 14. Risk Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Simulator doesn't match real SLM-OS behavior | Models don't transfer | Medium | Validate simulator against QEMU scheduler benchmarks from Phase 3. Tune timing parameters to match measured context switch and queue operation latencies. |
| RL training fails to converge | No RL model | Medium | Pre-training with behavioral cloning gives a strong starting point. Curriculum learning reduces initial difficulty. Fall back to DAgger-trained MLP if needed. |
| 108-feature state vector is too large for kernel inference | Deployment fails latency budget | Low | Ablation study (Phase S7) identifies which features can be dropped. Minimum viable state vector likely ~40 features (per-core + top-2 tasks + global). |
| XGBoost trees too large for embedded deployment | Can't deploy XGBoost | Low | Treelite compiles to optimized C. Alternatively, use pruned trees (fewer/shallower) trading slight accuracy. |
| Expert policies are suboptimal | Ceiling on supervised learning | Medium | Oracle policy provides upper bound. RL can exceed experts. DAgger helps MLP learn from its own mistakes, not just expert demonstrations. |
| Dataset too small / not diverse enough | Models overfit | Low | 13.6M samples is substantial. Workload variability adds noise. Held-out scenarios test generalization. |

---

## 15. Success Criteria

The AI scheduler project is successful if:

1. **At least one model exceeds the Phase 3 heuristic scheduler** on deadline compliance rate by ≥ 2 percentage points on the `medium_mixed` scenario
2. **Inference latency stays within the 5% overhead budget** (< 500 µs per decision for 10 ms quantum)
3. **The simulator is validated** against at least three real SLM-OS scheduling benchmarks from Phase 3 (context switch time, priority ordering, deadline boost behavior)
4. **Feature importance analysis** reveals which scheduling state features matter most (academic contribution)
5. **Results are reproducible** from the provided scripts and random seeds

---

*This is a living document that will be updated as implementation progresses.*
*Created: March 2026*
