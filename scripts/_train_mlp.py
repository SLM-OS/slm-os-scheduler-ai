#!/usr/bin/env python3
"""Train MLP model. Called by train_all.py in a subprocess."""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.mlp.dataset import SchedulerDataset
from training.mlp.train import train_mlp, TrainConfig

TRAIN_PARQUET = Path("data/splits/train.parquet")
VAL_PARQUET = Path("data/splits/val.parquet")
MAX_ROWS = 10_000_000

ts = lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"[{ts()}] Loading training data (max {MAX_ROWS:,} rows)...", flush=True)
t0 = time.time()
train_ds = SchedulerDataset(TRAIN_PARQUET, max_rows=MAX_ROWS)
print(f"[{ts()}] Train loaded: {len(train_ds):,} rows ({time.time() - t0:.1f}s)", flush=True)

print(f"[{ts()}] Loading validation data...", flush=True)
t0 = time.time()
val_ds = SchedulerDataset(VAL_PARQUET, max_rows=MAX_ROWS // 4)
print(f"[{ts()}] Val loaded: {len(val_ds):,} rows ({time.time() - t0:.1f}s)", flush=True)

model, result = train_mlp(train_ds, val_ds, TrainConfig(n_epochs=50),
                           save_path=Path("models/mlp/best.pt"))

print(f"[{ts()}] Best epoch {result.best_epoch}, "
      f"val loss {result.best_val_loss:.4f}, "
      f"val acc {result.best_val_acc:.4f}")
