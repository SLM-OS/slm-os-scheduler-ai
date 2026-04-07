#!/usr/bin/env python3
"""Train XGBoost model. Called by train_all.py in a subprocess."""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.xgboost.train import train_triple_classifier

TRAIN_PARQUET = Path("data/splits/train.parquet")
VAL_PARQUET = Path("data/splits/val.parquet")
MAX_ROWS = 10_000_000

ts = lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"[{ts()}] Training XGBoost (max {MAX_ROWS:,} rows)...", flush=True)
t0 = time.time()
triple, metrics = train_triple_classifier(
    TRAIN_PARQUET, VAL_PARQUET,
    max_rows=MAX_ROWS,
)
triple.save(Path("models/xgboost"))
print(f"[{ts()}] XGBoost saved to models/xgboost/ ({time.time() - t0:.1f}s total)", flush=True)
