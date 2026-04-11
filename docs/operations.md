# Operations Guide

## Setup

```bash
cd ~/projects/slm-os-scheduler-ai
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.12+.

## Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

204 tests, ~2 minutes. All should pass before running any pipeline step.

## End-to-End Pipeline

### Step 1: Generate Dataset

```bash
nohup python scripts/generate_dataset.py --workers 6 > dataset_gen.log 2>&1 &
tail -f dataset_gen.log
```

- **Time:** Several hours depending on hardware
- **Output:** 96 Parquet files in `data/raw/` (~15 GB)
- **Workers:** Use CPU cores minus 2 to leave headroom
- **Safe to run while machine is in use** — pure CPU computation, no special hardware access
- Simulated time, not wall-clock — host load does not affect results

### Step 2: Validate Dataset

```bash
python scripts/validate_dataset.py
```

Checks for NaN, out-of-range values, and episode completeness.

### Step 3: Split Dataset

```bash
python scripts/split_dataset.py
```

- **Output:** `data/splits/{train,val,test}.parquet` + `data/normalization.json`
- **Memory-efficient:** Streams row groups, never loads full dataset into RAM

### Step 4: Train Models

```bash
nohup python scripts/train_all.py > train_all.log 2>&1 &
tail -f train_all.log
```

Runs three stages in **separate subprocesses** (memory isolation):
1. MLP (~minutes): timestamped epoch-by-epoch progress
2. XGBoost (~minutes): per-classifier progress
3. PPO (~hours): progress every 100K steps with ETA

Each stage uses at most ~10M rows (subsampled from ~80M total) to fit in RAM.

For platform-specific MLP (required for deployment):
```bash
python scripts/_train_mlp.py --platform jetson_orin_nano
```

**Models saved to:**
- `models/mlp/best.pt` (mixed-platform) or `models/mlp/best_{platform}.pt`
- `models/xgboost/{core,priority,preempt}_clf.json` + `meta.json`
- `models/ppo/best_model.zip` + `models/ppo/checkpoints/`

### Step 5: Export to C

```bash
python scripts/export_models.py --model mlp --platform jetson_orin_nano
python scripts/export_models.py --model ppo --platform jetson_orin_nano
```

**Output:** `deploy/generated/` with C weight arrays and verification tests.

### Step 6: Verify C Inference

```bash
cd deploy/generated
gcc -O2 -o verify_mlp verify_inference_mlp.c ai_weights_mlp.c -lm
./verify_mlp
# Expected: Results: 1000/1000 passed
```

### Step 7: Evaluate (Optional)

```bash
python -c "
from evaluation.run_eval import run_full_evaluation
results = run_full_evaluation()
"
```

## Hardware Recommendations

### Dataset Generation

| Machine | Workers | Approx. Time |
|---------|---------|-------------|
| 8-core desktop (i7-6700) | 6 | ~8 hours |
| 16-core workstation | 12 | ~4 hours |
| 4-core laptop | 2 | ~16 hours |

### Training

| Stage | RAM Needed | Time |
|-------|-----------|------|
| MLP | ~8 GB (10M rows) | ~10 min |
| XGBoost | ~10 GB (10M rows + trees) | ~20 min |
| PPO | ~500 MB (simulator only) | ~8 hours (5M steps) |

**Total RAM:** At least 16 GB. Training scripts use subprocess isolation so only one stage's data is in memory at a time.

## Troubleshooting

### Out of Memory during split_dataset.py

Fixed in current version. The script now uses streaming row-group reads and Welford's online algorithm. If you have an old version, update the script.

### Out of Memory during training

The `max_rows` parameter (default 10M) limits how much data is loaded. Reduce it if needed:

```python
train_ds = SchedulerDataset("data/splits/train.parquet", max_rows=5_000_000)
```

### Model trained with wrong action count

The training data mixes all 3 platforms, so `n_actions` is determined by the max action index in the subsampled data. If you need a specific platform's action space, filter the training data to that platform only.

### PPO training takes too long

Reduce `total_timesteps` in `PPOConfig`. 2M steps gives reasonable results; 5M is for full training. Checkpoints are saved every 500K steps, so you can stop early and use the best checkpoint.

### Hex float warnings in C compilation

Some older compilers may warn about hex float literals. Use `-std=c99` or later.

## File Size Reference

| Artifact | Approximate Size |
|----------|-----------------|
| Raw dataset (data/raw/) | ~15 GB |
| Train split | ~11 GB |
| Val/Test splits | ~2.4 GB each |
| MLP checkpoint | ~540 KB |
| XGBoost model (JSON) | ~27 MB |
| PPO checkpoint | ~2 MB |
| Exported MLP C weights | ~3 MB (source), ~515 KB (compiled) |
| Exported PPO C weights | ~3 MB (source), ~515 KB (compiled) |
| Exported XGBoost C trees | ~23 MB (source), ~6.3 MB (compiled) |
