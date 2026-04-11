#!/usr/bin/env python3
"""Train MLP model. Called by train_all.py in a subprocess.

Usage:
    python scripts/_train_mlp.py                              # mixed-platform
    python scripts/_train_mlp.py --platform jetson_orin_nano  # Jetson-only
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.mlp.dataset import SchedulerDataset
from training.mlp.train import train_mlp, TrainConfig

parser = argparse.ArgumentParser(description="Train MLP scheduler model")
parser.add_argument("--platform", type=str, default=None,
                    help="Filter to specific platform (e.g., 'jetson_orin_nano')")
args = parser.parse_args()

TRAIN_PARQUET = Path("data/splits/train.parquet")
VAL_PARQUET = Path("data/splits/val.parquet")
MAX_ROWS = 10_000_000

platform_label = args.platform or "all-platforms"
save_path = Path(f"models/mlp/best_{args.platform}.pt") if args.platform else Path("models/mlp/best.pt")

# Compute platform action space (experts may not use all actions,
# but the model must cover the full space for deployment)
platform_n_actions = None
if args.platform:
    from slm_sim.actions import action_space_size
    from slm_sim.platforms import get_platform
    plat = get_platform(args.platform)
    platform_n_actions = action_space_size(plat.num_cores, plat.gpu.available)

ts = lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"[{ts()}] Platform filter: {platform_label}", flush=True)
if platform_n_actions:
    print(f"[{ts()}] Platform action space: {platform_n_actions}", flush=True)
print(f"[{ts()}] Loading training data (max {MAX_ROWS:,} rows)...", flush=True)
t0 = time.time()
train_ds = SchedulerDataset(TRAIN_PARQUET, platform=args.platform, max_rows=MAX_ROWS)
print(f"[{ts()}] Train loaded: {len(train_ds):,} rows, "
      f"n_actions_in_data={train_ds.n_actions} ({time.time() - t0:.1f}s)", flush=True)

print(f"[{ts()}] Loading validation data...", flush=True)
t0 = time.time()
val_ds = SchedulerDataset(VAL_PARQUET, platform=args.platform, max_rows=MAX_ROWS // 4)
print(f"[{ts()}] Val loaded: {len(val_ds):,} rows ({time.time() - t0:.1f}s)", flush=True)

model, result = train_mlp(train_ds, val_ds, TrainConfig(n_epochs=50),
                           save_path=save_path, n_actions=platform_n_actions)

print(f"[{ts()}] Best epoch {result.best_epoch}, "
      f"val loss {result.best_val_loss:.4f}, "
      f"val acc {result.best_val_acc:.4f}")
