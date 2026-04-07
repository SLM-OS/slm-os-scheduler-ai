# Dataset Pipeline

## Generation (`scripts/generate_dataset.py`)

Runs expert policies in the simulator to collect (state, action, reward) transition tuples.

### Full Dataset

- **3 expert policies** (hybrid, EDF, weighted) x **8 scenarios** x **3 platforms** x **500 episodes** = 36,000 expert episodes
- **1 random policy** x 8 scenarios x 3 platforms x 200 episodes = 4,800 random episodes
- **Episode duration:** 10 seconds simulated time
- **Total:** ~80 million transitions, ~15 GB compressed Parquet

### Small Dataset (for testing)

`--small` flag: 2 scenarios, 1 platform, 5 episodes per expert.

### Usage

```bash
# Full dataset (~hours, use nohup)
nohup python scripts/generate_dataset.py --workers 6 > dataset_gen.log 2>&1 &

# Small test dataset (~minutes)
python scripts/generate_dataset.py --small
```

`--workers N` controls parallelism. Recommended: CPU cores minus 2.

### Output

One Parquet file per (expert, scenario, platform) combination:
```
data/raw/
  slm_os_hybrid__light_single__jetson_orin_nano.parquet
  slm_os_hybrid__light_single__raspberry_pi5.parquet
  edf__medium_mixed__big_little.parquet
  random__burst_storm__jetson_orin_nano.parquet
  ...
```

## Parquet Schema

Each row is one transition (scheduling decision):

| Column | Type | Description |
|--------|------|-------------|
| `state_000` - `state_107` | float32 | 108-dim normalized state vector |
| `action` | int32 | Encoded action index |
| `action_core` | int32 | Core assignment (decoded) |
| `action_priority` | int32 | Priority adjustment (decoded) |
| `action_preempt` | int32 | Preempt flag (decoded) |
| `reward` | float32 | Total reward |
| `reward_deadline` | float32 | Deadline component |
| `reward_latency` | float32 | Latency component |
| `reward_balance` | float32 | Balance component |
| `reward_power` | float32 | Power component |
| `next_state_000` - `next_state_107` | float32 | Next state (for RL) |
| `done` | bool | Episode termination flag |
| `expert_policy` | string | Which expert generated this |
| `scenario` | string | Scenario name |
| `platform` | string | Platform name |
| `episode_id` | int64 | Episode number |
| `step_in_episode` | int32 | Step within episode |
| `sim_time_ns` | int64 | Simulated timestamp |

Total: ~232 columns per row.

## Validation (`scripts/validate_dataset.py`)

Checks:
- No NaN values in state or reward columns
- State features in [0, 1] range
- Rewards in reasonable range
- Episode completeness (no truncated episodes)
- All expected (expert, scenario, platform) combinations present

## Splitting (`scripts/split_dataset.py`)

**Episode-level 70/15/15 split** — ensures all transitions from the same episode stay in the same split (prevents temporal data leakage).

**Memory-efficient:** Three-pass design:
1. **Pass 1:** Scan metadata columns only to discover unique episode keys
2. **Pass 2:** Read each raw file one at a time, split rows, append to output `ParquetWriter` streams
3. **Pass 3:** Compute normalization stats from training set using Welford's online algorithm (row-group streaming)

**Output:**
```
data/splits/
  train.parquet   (~11 GB, 70% of episodes)
  val.parquet     (~2.4 GB, 15%)
  test.parquet    (~2.4 GB, 15%)

data/normalization.json   (per-feature mean, std, min, max from training set)
```

## Data Loading

Both MLP and XGBoost loaders support memory-efficient operation:

- **Streaming row-group reads:** Parquet files are read one row group at a time, never loading the full table into memory alongside the numpy arrays
- **`max_rows` subsampling:** When the full dataset exceeds RAM, randomly subsample to a specified limit (default 10M rows ~= 4 GB)
- **Column selection:** Only needed columns are read (not all 232)

### MLP Dataset (`training/mlp/dataset.py`)

`SchedulerDataset(parquet_path, max_rows=10_000_000)`:
- Filters to expert policies (excludes random)
- Loads `state_000`-`state_107`, `action`, `reward`, `expert_policy`
- Computes sample weights: `|reward| / mean(|reward|)`
- Optional z-score normalization from `normalization.json`
- Returns `(state, action, weight)` tuples for PyTorch DataLoader

### XGBoost Features (`training/xgboost/features.py`)

`load_features_and_labels(parquet_path, max_rows=10_000_000)`:
- Same streaming approach
- Adds 5 derived features (113 total)
- Returns `(X, y_core, y_priority, y_preempt)` arrays

## Memory Considerations

The full training set is ~11 GB compressed, ~32 GB as float32 numpy arrays. On a 54 GB RAM machine:

- **Don't** load the full dataset into memory
- **Do** use `max_rows=10_000_000` (10M rows ~= 4 GB for states)
- Training scripts (`scripts/train_all.py`) run each model in a **separate subprocess** so memory is fully reclaimed between stages
