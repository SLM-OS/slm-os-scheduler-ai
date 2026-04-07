# State, Action, and Reward

## State Vector (108 dimensions)

All features are normalized to [0, 1]. Extracted by `slm_sim/observation.py:extract_observation()`.

### Per-Core Features (6 features x 6 cores = 36)

Cores are padded to `MAX_CORES=6` across all platforms (Pi 5 has 4 real cores + 2 zero-padded).

| Offset | Feature | Normalization | Source |
|--------|---------|---------------|--------|
| c*6+0 | Utilization | Already [0, 1] | Sliding window busy time |
| c*6+1 | Run queue depth | depth / 32 | `len(core.run_queue)` |
| c*6+2 | Cache pressure | working_set_sum / (cache_size * 5) | Sum of task WS on core |
| c*6+3 | Core type | 0.0=efficiency, 1.0=performance | `CoreType` enum |
| c*6+4 | Isolated | 0.0 or 1.0 | `core.isolated` flag |
| c*6+5 | Current task priority | priority / 7 | Running task's `effective_priority` |

### Per-Task Features (8 features x 8 tasks = 64)

Top 8 ready tasks sorted by `effective_priority` descending. Padded with zeros if fewer than 8.

| Offset | Feature | Normalization | Source |
|--------|---------|---------------|--------|
| 36+t*8+0 | Priority | priority / 7 | `task.effective_priority` |
| 36+t*8+1 | Deadline urgency | 1 - (time_remaining / 1e9) | Clamped [0,1]; 1.0 if overdue |
| 36+t*8+2 | Working set | MB / 64 | `task.working_set_mb` |
| 36+t*8+3 | Model size | MB / 128 | `task.model_size_mb` |
| 36+t*8+4 | Inference duration | ns / 100,000,000 | `task.inference_duration_ns` |
| 36+t*8+5 | GPU eligible | 0.0 or 1.0 | `task.can_use_gpu` |
| 36+t*8+6 | Wait time | (now - arrival) / 1e9 | Time since task arrived |
| 36+t*8+7 | Component type | type_enum / 4 | 0=anomaly, 1=predmaint, 2=security, 3=system |

### Global Features (8)

| Offset | Feature | Normalization |
|--------|---------|---------------|
| 100 | Total ready count | count / 64 |
| 101 | Deadline miss rate | misses / completions (last 100) |
| 102 | Average latency | ns / 10,000,000 |
| 103 | Weight pool pressure | used / total |
| 104 | Workspace pool pressure | used / total |
| 105 | GPU queue depth | (queued + running) / 8 |
| 106 | Load imbalance | std(utils) / mean(utils), clamped [0,1] |
| 107 | Episode time progress | clock / duration |

### Normalization Constants

```python
MAX_RUN_QUEUE_DEPTH     = 32
MAX_CACHE_PRESSURE      = 5.0
MAX_WORKING_SET_MB      = 64.0
MAX_MODEL_SIZE_MB       = 128.0
MAX_INFERENCE_DURATION_NS = 100_000_000   # 100 ms
MAX_WAIT_TIME_NS        = 1_000_000_000   # 1 second
MAX_READY_COUNT         = 64
MAX_GPU_QUEUE           = 8
MAX_DEADLINE_NS         = 1_000_000_000   # 1 second
```

Optional z-score normalization can be applied using stats from `split_dataset.py` (`data/normalization.json`).

---

## Action Space

Defined in `slm_sim/actions.py`. Each scheduling decision is decomposed into three sub-actions:

| Sub-action | Values | Meaning |
|------------|--------|---------|
| Core assignment | 0 to N_CORES-1, or N_CORES for GPU | Which core (or GPU) to assign the task to |
| Priority adjustment | 0=lower, 1=keep, 2=raise | Move to prev/same/next kernel priority level |
| Preempt | 0=no, 1=yes | Preempt current task on target core |

### Encoding

Flattened to a single integer for neural network output:

```python
# Encode
idx = core_assignment * 3 * 2 + priority_adj * 2 + preempt

# Decode
preempt        = idx % 2;  idx //= 2
priority_adj   = idx % 3;  idx //= 3
core_assignment = idx
```

### Action Space Size Per Platform

| Platform | Cores | GPU | Targets | Total Actions |
|----------|-------|-----|---------|---------------|
| Jetson Orin Nano | 6 | Yes | 7 | 42 |
| Raspberry Pi 5 | 4 | No | 4 | 24 |
| big.LITTLE | 6 | No | 6 | 36 |

### Priority Levels

Priority adjustment uses kernel-compatible levels with gaps: `[0, 2, 4, 6, 7]`

- "Lower" moves to the previous level (e.g., NORMAL(4) → LOW(2))
- "Raise" moves to the next level (e.g., NORMAL(4) → HIGH(6))
- Cannot go below IDLE(0) or above CRITICAL(7)

### Action Application

`apply_action()` handles edge cases:
- If target core is invalid or isolated, falls back to least-loaded non-isolated core
- If GPU target but GPU unavailable, falls back to CPU
- Preemption only occurs if running task has lower priority than arriving task

---

## Reward Function

Defined in `slm_sim/reward.py`. Four weighted components:

```
R = 0.50 * R_deadline + 0.25 * R_latency + 0.15 * R_balance + 0.10 * R_power
```

### R_deadline (weight: 0.50)

Dominant component. Measures deadline compliance for a completed task.

| Condition | Value |
|-----------|-------|
| Task met deadline | +1.0 |
| Task missed deadline | -2.0 * (overshoot / deadline), clamped to [-2.0, 0] |
| No deadline | +0.1 (small positive to avoid penalizing system tasks) |

### R_latency (weight: 0.25)

Measures how close actual latency is to target:

```
R_latency = 1.0 - (actual_latency / target_latency)
```

Clamped to [-1, 1]. Per-component targets:

| Component | Target |
|-----------|--------|
| Anomaly Detector | 5 ms |
| Predictive Maintenance | 50 ms |
| Security Monitor | 20 ms |
| System Tasks | 10 ms |

### R_balance (weight: 0.15)

Measures core utilization balance using coefficient of variation:

```
R_balance = 1.0 - (std_dev(core_utilizations) / mean(core_utilizations))
```

Clamped to [0, 1]. Perfectly balanced cores yield 1.0.

### R_power (weight: 0.10)

Measures power efficiency:

```
R_power = 1.0 - (weighted_active_power / max_possible_power)
weighted_active = sum(utilization[i] * power_weight[i])
```

Lower utilization of high-power cores yields higher reward. Relevant for big.LITTLE where efficiency cores have `power_weight=0.3` vs performance cores at `1.0`.

### Configuration

Weights are configurable via `RewardConfig`:

```python
@dataclass
class RewardConfig:
    w_deadline: float = 0.50
    w_latency:  float = 0.25
    w_balance:  float = 0.15
    w_power:    float = 0.10
```
