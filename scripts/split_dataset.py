#!/usr/bin/env python3
"""Split dataset into train/val/test and compute normalization stats.

Episode-level 70/15/15 split. Computes per-feature normalization
statistics (mean, std, min, max) from the training set only.

Memory-efficient: scans files twice (once for episode keys, once to
write splits) instead of loading everything into RAM.

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


def _collect_episode_keys(parquet_files: list[Path]) -> list[tuple]:
    """Pass 1: scan metadata columns only to discover unique episode keys."""
    episode_keys = set()
    meta_columns = ["expert_policy", "scenario", "platform", "episode_id"]

    for f in parquet_files:
        table = pq.read_table(f, columns=meta_columns)
        experts = table.column("expert_policy").to_pylist()
        scenarios = table.column("scenario").to_pylist()
        platforms = table.column("platform").to_pylist()
        episode_ids = table.column("episode_id").to_pylist()

        for i in range(len(table)):
            episode_keys.add((experts[i], scenarios[i], platforms[i], int(episode_ids[i])))

        del table  # free immediately

    return sorted(episode_keys)


def _assign_splits(
    unique_keys: list[tuple], seed: int = 42
) -> tuple[set[tuple], set[tuple], set[tuple]]:
    """Shuffle episode keys and split 70/15/15."""
    n = len(unique_keys)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_keys = set(unique_keys[i] for i in indices[:n_train])
    val_keys = set(unique_keys[i] for i in indices[n_train:n_train + n_val])
    test_keys = set(unique_keys[i] for i in indices[n_train + n_val:])

    return train_keys, val_keys, test_keys


def _split_table(table: pa.Table, train_keys, val_keys, test_keys):
    """Split a single table's rows into three lists by episode key membership."""
    experts = table.column("expert_policy").to_pylist()
    scenarios = table.column("scenario").to_pylist()
    platforms = table.column("platform").to_pylist()
    episode_ids = table.column("episode_id").to_pylist()

    train_idx = []
    val_idx = []
    test_idx = []

    for i in range(len(table)):
        key = (experts[i], scenarios[i], platforms[i], int(episode_ids[i]))
        if key in train_keys:
            train_idx.append(i)
        elif key in val_keys:
            val_idx.append(i)
        elif key in test_keys:
            test_idx.append(i)

    return train_idx, val_idx, test_idx


def split_dataset(raw_dir: Path, output_dir: Path, seed: int = 42) -> dict:
    """Split all Parquet files into train/val/test at episode level.

    Processes one file at a time to keep memory usage bounded.
    Returns dict with split statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(raw_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No Parquet files found in {raw_dir}")
        return {}

    # --- Pass 1: collect episode keys (metadata columns only) ---
    print(f"Pass 1: scanning {len(parquet_files)} files for episode keys...")
    unique_keys = _collect_episode_keys(parquet_files)
    n_episodes = len(unique_keys)
    print(f"Unique episodes: {n_episodes}")

    train_keys, val_keys, test_keys = _assign_splits(unique_keys, seed)
    del unique_keys  # no longer needed

    # --- Pass 2: read each file, split rows, append to output writers ---
    print("Pass 2: splitting rows and writing output files...")

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    train_writer = None
    val_writer = None
    test_writer = None
    train_count = 0
    val_count = 0
    test_count = 0
    total_rows = 0

    try:
        for idx, f in enumerate(parquet_files):
            print(f"  [{idx + 1}/{len(parquet_files)}] {f.name}", flush=True)
            table = pq.read_table(f)
            total_rows += len(table)

            train_idx, val_idx, test_idx = _split_table(
                table, train_keys, val_keys, test_keys
            )

            if train_idx:
                batch = table.take(train_idx)
                if train_writer is None:
                    train_writer = pq.ParquetWriter(train_path, batch.schema, compression="snappy")
                train_writer.write_table(batch)
                train_count += len(train_idx)
                del batch

            if val_idx:
                batch = table.take(val_idx)
                if val_writer is None:
                    val_writer = pq.ParquetWriter(val_path, batch.schema, compression="snappy")
                val_writer.write_table(batch)
                val_count += len(val_idx)
                del batch

            if test_idx:
                batch = table.take(test_idx)
                if test_writer is None:
                    test_writer = pq.ParquetWriter(test_path, batch.schema, compression="snappy")
                test_writer.write_table(batch)
                test_count += len(test_idx)
                del batch

            del table
    finally:
        if train_writer:
            train_writer.close()
        if val_writer:
            val_writer.close()
        if test_writer:
            test_writer.close()

    print(f"\nTotal rows: {total_rows:,}")
    print(f"Train: {train_count:,} rows ({len(train_keys)} episodes)")
    print(f"Val:   {val_count:,} rows ({len(val_keys)} episodes)")
    print(f"Test:  {test_count:,} rows ({len(test_keys)} episodes)")

    # --- Pass 3: compute normalization stats from training set ---
    print("Pass 3: computing normalization stats from training set...")
    stats = compute_normalization_stats(train_path)
    stats_path = output_dir.parent / "normalization.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Normalization stats written to {stats_path}")

    return {
        "total_rows": total_rows,
        "train_rows": train_count,
        "val_rows": val_count,
        "test_rows": test_count,
        "train_episodes": len(train_keys),
        "val_episodes": len(val_keys),
        "test_episodes": len(test_keys),
    }


def compute_normalization_stats(train_path: Path) -> dict:
    """Compute per-feature normalization statistics using streaming reads.

    Uses Welford's online algorithm so only one row group is in memory at a time.
    """
    parquet_file = pq.ParquetFile(train_path)
    state_columns = [f"state_{j:03d}" for j in range(TOTAL_FEATURES)]

    # Running stats (Welford's online algorithm)
    n = 0
    mean = np.zeros(TOTAL_FEATURES, dtype=np.float64)
    m2 = np.zeros(TOTAL_FEATURES, dtype=np.float64)
    feat_min = np.full(TOTAL_FEATURES, np.inf, dtype=np.float64)
    feat_max = np.full(TOTAL_FEATURES, -np.inf, dtype=np.float64)

    for batch in parquet_file.iter_batches(columns=state_columns, batch_size=100_000):
        for j in range(TOTAL_FEATURES):
            arr = batch.column(state_columns[j]).to_numpy().astype(np.float64)
            batch_n = len(arr)
            batch_mean = arr.mean()
            batch_var = arr.var()

            # Combine running stats with this batch
            if n == 0:
                mean[j] = batch_mean
                m2[j] = batch_var * batch_n
            else:
                total_n = n + batch_n
                delta = batch_mean - mean[j]
                mean[j] = (n * mean[j] + batch_n * batch_mean) / total_n
                m2[j] += batch_var * batch_n + delta ** 2 * n * batch_n / total_n

            feat_min[j] = min(feat_min[j], arr.min())
            feat_max[j] = max(feat_max[j], arr.max())

        n += len(batch)

    std = np.sqrt(m2 / max(n, 1))

    return {
        "n_features": TOTAL_FEATURES,
        "n_samples": int(n),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": feat_min.tolist(),
        "max": feat_max.tolist(),
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
