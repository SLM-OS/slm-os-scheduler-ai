#!/usr/bin/env python3
"""Split dataset into train/val/test and compute normalization stats.

Episode-level 70/15/15 split. Computes per-feature normalization
statistics (mean, std, min, max) from the training set only.

Usage:
    python scripts/split_dataset.py [--raw-dir DIR] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slm_sim.observation import TOTAL_FEATURES

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def split_dataset(raw_dir: Path, output_dir: Path, seed: int = 42) -> dict:
    """Split all Parquet files into train/val/test at episode level.

    Returns dict with split statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all files
    parquet_files = sorted(raw_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No Parquet files found in {raw_dir}")
        return {}

    print(f"Loading {len(parquet_files)} files...")
    tables = [pq.read_table(f) for f in parquet_files]
    combined = pa.concat_tables(tables, promote_options="default")
    total_rows = len(combined)
    print(f"Total rows: {total_rows:,}")

    # Get unique episode IDs
    episode_ids = combined.column("episode_id").to_numpy()
    # Create unique episode key from (expert, scenario, platform, episode_id)
    experts = combined.column("expert_policy").to_pylist()
    scenarios = combined.column("scenario").to_pylist()
    platforms = combined.column("platform").to_pylist()

    # Build unique episode keys
    episode_keys = set()
    row_to_key = []
    for i in range(total_rows):
        key = (experts[i], scenarios[i], platforms[i], int(episode_ids[i]))
        episode_keys.add(key)
        row_to_key.append(key)

    unique_keys = sorted(episode_keys)
    n_episodes = len(unique_keys)
    print(f"Unique episodes: {n_episodes}")

    # Shuffle and split
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_episodes)

    n_train = int(n_episodes * TRAIN_RATIO)
    n_val = int(n_episodes * VAL_RATIO)

    train_keys = set(unique_keys[i] for i in indices[:n_train])
    val_keys = set(unique_keys[i] for i in indices[n_train:n_train + n_val])
    test_keys = set(unique_keys[i] for i in indices[n_train + n_val:])

    # Assign rows to splits
    train_mask = np.array([k in train_keys for k in row_to_key])
    val_mask = np.array([k in val_keys for k in row_to_key])
    test_mask = np.array([k in test_keys for k in row_to_key])

    train_table = combined.filter(pa.array(train_mask))
    val_table = combined.filter(pa.array(val_mask))
    test_table = combined.filter(pa.array(test_mask))

    # Write splits
    pq.write_table(train_table, output_dir / "train.parquet", compression="snappy")
    pq.write_table(val_table, output_dir / "val.parquet", compression="snappy")
    pq.write_table(test_table, output_dir / "test.parquet", compression="snappy")

    print(f"Train: {len(train_table):,} rows ({len(train_keys)} episodes)")
    print(f"Val:   {len(val_table):,} rows ({len(val_keys)} episodes)")
    print(f"Test:  {len(test_table):,} rows ({len(test_keys)} episodes)")

    # Compute normalization stats from training set only
    stats = compute_normalization_stats(train_table)
    stats_path = output_dir.parent / "normalization.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Normalization stats written to {stats_path}")

    return {
        "total_rows": total_rows,
        "train_rows": len(train_table),
        "val_rows": len(val_table),
        "test_rows": len(test_table),
        "train_episodes": len(train_keys),
        "val_episodes": len(val_keys),
        "test_episodes": len(test_keys),
    }


def compute_normalization_stats(table: pa.Table) -> dict:
    """Compute per-feature normalization statistics from a table.

    Returns dict with keys: mean, std, min, max — each a list of
    TOTAL_FEATURES floats.
    """
    means = []
    stds = []
    mins = []
    maxs = []

    for j in range(TOTAL_FEATURES):
        col_name = f"state_{j:03d}"
        if col_name in table.column_names:
            arr = table.column(col_name).to_numpy().astype(np.float64)
            means.append(float(np.mean(arr)))
            stds.append(float(np.std(arr)))
            mins.append(float(np.min(arr)))
            maxs.append(float(np.max(arr)))
        else:
            means.append(0.0)
            stds.append(1.0)
            mins.append(0.0)
            maxs.append(1.0)

    return {
        "n_features": TOTAL_FEATURES,
        "mean": means,
        "std": stds,
        "min": mins,
        "max": maxs,
    }


def main():
    parser = argparse.ArgumentParser(description="Split dataset and compute normalization stats")
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    raw_dir = Path(args.raw_dir) if args.raw_dir else project_root / "data" / "raw"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "data" / "splits"

    split_dataset(raw_dir, output_dir)


if __name__ == "__main__":
    main()
