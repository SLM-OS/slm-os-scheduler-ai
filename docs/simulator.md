# Simulator

The discrete-event simulator (`slm_sim/`) models SLM-OS's multi-core scheduling behavior at the decision granularity needed for training ML models.

## Engine (`engine.py`)

The `SimulatorEngine` is a min-heap event queue simulator. Each episode:

1. `reset()` — clears state, creates cores from platform profile, seeds periodic events (deadline check at 1ms, migration check at 10ms)
2. Workloads inject `TASK_ARRIVAL` events into the queue
3. Main loop: pop event → advance clock → process event → if scheduling decision needed, agent decides → apply action → log transition
4. Episode ends when clock exceeds `episode_duration_ns` (default 10s)

### Event Types

| Event | Trigger | Handler |
|-------|---------|---------|
| `TASK_ARRIVAL` | Workload generator | Creates SimTask, adds to ready pool |
| `TASK_COMPLETION` | Task finishes on core | Frees core, triggers next scheduling decision |
| `QUANTUM_EXPIRED` | 10ms timer | Preempts running task, reschedules |
| `DEADLINE_CHECK` | Every 1ms | Updates deadline boosts on all tasks |
| `GPU_COMPLETION` | GPU finishes inference | Returns task to CPU scheduling |
| `MIGRATION_CHECK` | Every 10ms | Checks for load imbalance |
| `SCHEDULING_DECISION` | Various | Agent selects action for a ready task |

### Episode Metrics

Each episode produces an `EpisodeMetrics` dataclass:
- `total_tasks`, `completed_tasks`
- `deadline_misses`, `deadline_met`
- `total_latency_ns`, `max_latency_ns`
- Per-core utilization array
- `gpu_tasks_completed`
- `context_switches`

**Performance:** >1,600 episodes/minute (optimized by replacing numpy scalar operations with pure Python math).

## Platforms (`platforms.py`)

Three hardware profiles modeled as `PlatformProfile` dataclasses:

### Jetson Orin Nano
- 6 Cortex-A78AE cores @ 1.5 GHz (all PERFORMANCE type)
- 8 GB LPDDR5, 256 KB L2/core
- GPU: 1024-core Ampere, 8 TFLOPS, available for inference offload
- Weight pool: 512 MB, workspace pool: 256 MB
- Context switch: 2-5 us
- **Action space: 42** (7 targets x 3 priority x 2 preempt)

### Raspberry Pi 5
- 4 Cortex-A76 cores @ 2.4 GHz (all PERFORMANCE type)
- 4 GB LPDDR4X, 512 KB L2/core
- No GPU
- Weight pool: 256 MB, workspace pool: 128 MB
- Context switch: 3-6 us
- **Action space: 24** (4 targets x 3 x 2)

### big.LITTLE (Hypothetical)
- 2 Cortex-A78 (PERFORMANCE, 2.0 GHz, power_weight=1.0) + 4 Cortex-A55 (EFFICIENCY, 1.0 GHz, power_weight=0.3)
- 4 GB RAM, L2: 256 KB (perf) / 128 KB (eff)
- No GPU
- **Action space: 36** (6 targets x 3 x 2)

Platforms are registered in `PLATFORM_REGISTRY` and retrieved via `get_platform(name)`.

## Workloads (`workloads/`)

Four workload profiles model real SLM-OS inference tasks:

### Anomaly Detector (`anomaly.py`)
- **Rate:** 100 Hz exponential arrivals, 5% chance of burst (5-20 requests)
- **Model:** 12 MB, 4 MB working set
- **Inference:** ~600 us
- **Deadline:** 5 ms (soft)
- **Priority:** NORMAL (4)

### Predictive Maintenance (`predmaint.py`)
- **Rate:** 1 Hz periodic + 10 Hz burst every 30s
- **Model:** 45 MB, 16 MB working set
- **Inference:** ~2 ms
- **Deadline:** 50 ms (hard)
- **Priority:** HIGH (6)
- **GPU-eligible:** Yes

### Security Monitor (`security.py`)
- **Rate:** 10 Hz periodic
- **Model:** 8 MB, 3 MB working set
- **Inference:** ~800 us
- **Deadline:** 20 ms (soft)
- **Priority:** LOW (2)

### System Tasks (`system.py`)
- **Rate:** 100 Hz periodic
- **No model** (OS overhead)
- **Duration:** ~50 us
- **No deadline**
- **Priority:** HIGH (6)

### Scenarios (`scenarios.py`)

Eight `ScenarioConfig` definitions combine workloads at varying intensities:

| Scenario | Workloads | Approx. Load |
|----------|-----------|-------------|
| `light_single` | Anomaly only | ~10% |
| `light_mixed` | Anomaly + Security | ~25% |
| `medium_mixed` | All four components | ~50% |
| `heavy_inference` | All at 2x rate | ~80% |
| `burst_storm` | Normal + periodic bursts | spikes ~95% |
| `deadline_pressure` | PredMaint at 3x rate | ~60% |
| `memory_pressure` | 4x PredMaint | pool exhaustion |
| `asymmetric` | Pinned to specific cores | ~40%, unbalanced |

`ScenarioComposer` creates workload instances from a scenario name, with optional `burst_overlay` and `affinity_map` for per-workload core pinning.

## Expert Policies (`experts/`)

Five `ExpertPolicy` implementations provide training data:

### Hybrid (`hybrid.py`)
Exact replica of SLM-OS Phase 3 scheduler:
- Priority queue ordering
- Deadline boost thresholds: <10ms → CRITICAL, <50ms → HIGH, <100ms → priority+1
- Working set affinity: if task WS < 8MB, prefer core with matching cache contents
- `_find_performance_cpu()`: penalizes core 0 (reserved for OS), scores by queue depth
- ~100% DCR on medium_mixed

### EDF (`edf.py`)
Earliest Deadline First:
- Assigns to least-loaded core
- Preempts if new task has nearer deadline than running task
- Tasks without deadlines get lowest priority

### Weighted Multi-Objective (`weighted.py`)
5-factor scoring for each (task, core) pair:
- Deadline urgency: 0.35
- Priority: 0.25
- Cache affinity: 0.20
- Utilization: 0.15
- Power efficiency: 0.05
- Preempts if score gap > 0.3

### Random (`random_policy.py`)
Uniform random core, priority adjustment, and preemption. Provides negative training examples.

### Oracle (`oracle.py`)
Offline beam search:
1. Runs hybrid policy as baseline
2. Greedy forward search with heuristic action scoring
3. Returns theoretically-optimal decisions
- Computationally expensive (not used for bulk data generation)
