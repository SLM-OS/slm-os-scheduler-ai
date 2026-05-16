#!/usr/bin/env python3
"""Train MLP model. Called by train_all.py in a subprocess.

Usage:
    # Default: train from scratch on the simulator's expert demos
    python scripts/_train_mlp.py                              # mixed-platform
    python scripts/_train_mlp.py --platform jetson_orin_nano  # Jetson-only

    # Fine-tune (#879): start from an existing checkpoint and adapt
    # on captured SLM-OS scheduling traces ingested via
    # data/slmos_traces.py.
    python scripts/_train_mlp.py \\
        --source slmos-traces \\
        --input data/slmos_traces.parquet \\
        --platform jetson_orin_nano \\
        --output models/mlp/best_jetson_orin_nano_real.pt

The slmos-traces flow uses a deliberately small learning rate and
few epochs — the synthetic baseline already generalizes well, so the
fine-tune nudges it toward SLM-OS-specific scheduling behaviour
without overwriting the broader knowledge. From-scratch training on
the ~hundreds-to-low-thousands of decisions a typical SLM-OS capture
yields would overfit dramatically; fine-tuning from a pretrained
checkpoint is the standard embedded-ML adaptation shape.
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

import torch

from data.slmos_traces import SLMOS_TRACE_EXPERT_LABEL

parser = argparse.ArgumentParser(description="Train MLP scheduler model")
parser.add_argument("--platform", type=str, default=None,
                    help="Filter to specific platform (e.g., 'jetson_orin_nano')")
parser.add_argument("--source", type=str, default="simulator",
                    choices=["simulator", "slmos-traces"],
                    help="Training data source. 'simulator' (default) reads "
                         "the synthetic expert dataset under data/splits/. "
                         "'slmos-traces' reads a SLM-OS-captured trace "
                         "Parquet (emitted by data/slmos_traces.py) and "
                         "fine-tunes from an existing checkpoint.")
parser.add_argument("--input", type=Path, default=None,
                    help="Parquet path. Required for --source slmos-traces; "
                         "ignored for the simulator path.")
parser.add_argument("--output", type=Path, default=None,
                    help="Where to save the trained model. Defaults to "
                         "models/mlp/best[_<platform>].pt for simulator and "
                         "models/mlp/best[_<platform>]_real.pt for "
                         "slmos-traces.")
parser.add_argument("--init-from", type=Path, default=None,
                    help="Existing .pt checkpoint to load weights from "
                         "before training. For --source slmos-traces this "
                         "is the pretrained synthetic-baseline model; "
                         "defaults to models/mlp/best[_<platform>].pt.")
args = parser.parse_args()

MAX_ROWS = 10_000_000

ts = lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")

platform_label = args.platform or "all-platforms"

# Compute platform action space (the model output dim is platform-
# specific so deployment-target dimensionality matches at export time).
platform_n_actions = None
if args.platform:
    from slm_sim.actions import action_space_size
    from slm_sim.platforms import get_platform
    plat = get_platform(args.platform)
    platform_n_actions = action_space_size(plat.num_cores, plat.gpu.available)


def _default_save_path() -> Path:
    suffix = "_real" if args.source == "slmos-traces" else ""
    name = f"best_{args.platform}{suffix}.pt" if args.platform else f"best{suffix}.pt"
    return Path("models/mlp") / name


def _default_init_path() -> Path | None:
    # Only the slmos-traces flow needs a pretrained init; the
    # simulator flow trains from scratch.
    if args.source != "slmos-traces":
        return None
    name = f"best_{args.platform}.pt" if args.platform else "best.pt"
    return Path("models/mlp") / name


save_path = args.output or _default_save_path()
init_path = args.init_from or _default_init_path()

print(f"[{ts()}] Platform filter: {platform_label}", flush=True)
print(f"[{ts()}] Source: {args.source}", flush=True)
if platform_n_actions:
    print(f"[{ts()}] Platform action space: {platform_n_actions}", flush=True)
if init_path:
    print(f"[{ts()}] Init from: {init_path}", flush=True)
print(f"[{ts()}] Save to: {save_path}", flush=True)

# --- Dataset selection -------------------------------------------------

class _SchedulerSubset(torch.utils.data.Subset):
    """torch.utils.data.Subset doesn't expose `n_features` /
    `n_actions` — attributes `train_mlp` reads off the dataset for
    model construction. This wrapper forwards them from the parent
    SchedulerDataset so a `random_split` slice still type-checks at
    the trainer's call sites."""
    @property
    def n_features(self):
        return self.dataset.n_features

    @property
    def n_actions(self):
        return self.dataset.n_actions


if args.source == "slmos-traces":
    if args.input is None:
        sys.exit("--source slmos-traces requires --input <trace.parquet>")
    if not args.input.exists():
        sys.exit(f"slmos-traces input not found: {args.input}")
    print(f"[{ts()}] Loading SLM-OS trace data from {args.input}...", flush=True)
    t0 = time.time()
    # The ingester labels every row with SLMOS_TRACE_EXPERT_LABEL; the
    # dataset's default expert filter would discard them, so we override.
    full_ds = SchedulerDataset(
        args.input,
        experts={SLMOS_TRACE_EXPERT_LABEL},
        platform=args.platform,
        max_rows=MAX_ROWS,
    )
    if len(full_ds) == 0:
        sys.exit(
            f"slmos-traces Parquet at {args.input} contains zero rows for "
            f"expert_policy={SLMOS_TRACE_EXPERT_LABEL!r}, "
            f"platform={args.platform!r}. Check the ingester output."
        )
    # Disjoint train/val split via random_split. Earlier two-load-with-
    # different-seed approach produced overlapping subsamples (both
    # drew from the same parent set), so val_loss was a noisy mirror
    # of train_loss rather than a generalization signal.
    val_size = max(1, len(full_ds) // 10)
    train_size = len(full_ds) - val_size
    rng = torch.Generator().manual_seed(42)
    raw_train, raw_val = torch.utils.data.random_split(
        full_ds, [train_size, val_size], generator=rng,
    )
    train_ds = _SchedulerSubset(full_ds, raw_train.indices)
    val_ds = _SchedulerSubset(full_ds, raw_val.indices)
    print(f"[{ts()}] Trace load: {len(train_ds)} train / {len(val_ds)} val "
          f"rows ({time.time() - t0:.1f}s)", flush=True)
    # Fine-tune hyperparameters: smaller LR (preserve pretrained
    # weights), fewer epochs (small dataset overfits fast), tighter
    # patience.
    config = TrainConfig(
        n_epochs=10,
        lr=1e-4,         # 10× lower than the 1e-3 from-scratch default
        lr_min=1e-6,
        patience=3,
    )
else:
    TRAIN_PARQUET = Path("data/splits/train.parquet")
    VAL_PARQUET = Path("data/splits/val.parquet")
    print(f"[{ts()}] Loading training data (max {MAX_ROWS:,} rows)...", flush=True)
    t0 = time.time()
    train_ds = SchedulerDataset(TRAIN_PARQUET, platform=args.platform,
                                 max_rows=MAX_ROWS)
    print(f"[{ts()}] Train loaded: {len(train_ds):,} rows, "
          f"n_actions_in_data={train_ds.n_actions} "
          f"({time.time() - t0:.1f}s)", flush=True)

    print(f"[{ts()}] Loading validation data...", flush=True)
    t0 = time.time()
    val_ds = SchedulerDataset(VAL_PARQUET, platform=args.platform,
                              max_rows=MAX_ROWS // 4)
    print(f"[{ts()}] Val loaded: {len(val_ds):,} rows "
          f"({time.time() - t0:.1f}s)", flush=True)
    config = TrainConfig(n_epochs=50)

# --- Train -------------------------------------------------------------

pretrained_state = None
if init_path is not None:
    if not init_path.exists():
        sys.exit(f"init checkpoint not found: {init_path}")
    print(f"[{ts()}] Loading pretrained weights from {init_path}...", flush=True)
    pretrained_state = torch.load(
        init_path, map_location="cpu", weights_only=True
    )

save_path.parent.mkdir(parents=True, exist_ok=True)
model, result = train_mlp(
    train_ds, val_ds, config,
    save_path=save_path, n_actions=platform_n_actions,
    pretrained_state=pretrained_state,
)

print(f"[{ts()}] Best epoch {result.best_epoch}, "
      f"val loss {result.best_val_loss:.4f}, "
      f"val acc {result.best_val_acc:.4f}")
