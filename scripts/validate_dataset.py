#!/usr/bin/env python3
"""Validate the generated dataset.

Checks for NaN, out-of-range values, verifies reward distributions,
and ensures episode completeness. See plan Phase S3 task 3.5.

Usage:
    python scripts/validate_dataset.py [data_dir]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slm_sim.observation import TOTAL_FEATURES


def validate_file(path: Path) -> dict:
    """Validate a single Parquet file. Returns dict of issues found."""
    issues = []
    table = pq.read_table(path)
    n_rows = len(table)

    if n_rows == 0:
        return {"file": str(path), "rows": 0, "issues": ["Empty file"]}

    # Check state features are in [0, 1]
    for j in range(TOTAL_FEATURES):
        col_name = f"state_{j:03d}"
        if col_name in table.column_names:
            arr = table.column(col_name).to_numpy()
            if np.any(np.isnan(arr)):
                issues.append(f"{col_name}: contains NaN")
            if np.any(arr < -0.01):
                issues.append(f"{col_name}: values below 0 (min={arr.min():.4f})")
            if np.any(arr > 1.01):
                issues.append(f"{col_name}: values above 1 (max={arr.max():.4f})")

    # Check next_state features
    for j in range(TOTAL_FEATURES):
        col_name = f"next_state_{j:03d}"
        if col_name in table.column_names:
            arr = table.column(col_name).to_numpy()
            if np.any(np.isnan(arr)):
                issues.append(f"{col_name}: contains NaN")

    # Check reward is finite
    if "reward" in table.column_names:
        rewards = table.column("reward").to_numpy()
        if np.any(np.isnan(rewards)):
            issues.append("reward: contains NaN")
        if np.any(np.isinf(rewards)):
            issues.append("reward: contains Inf")

    # Check action is valid (non-negative integer)
    if "action" in table.column_names:
        actions = table.column("action").to_numpy()
        if np.any(actions < 0):
            issues.append(f"action: negative values found (min={actions.min()})")

    # Check episodes have done=True at the end
    if "episode_id" in table.column_names and "done" in table.column_names:
        episodes = table.column("episode_id").to_numpy()
        dones = table.column("done").to_numpy()
        unique_eps = np.unique(episodes)
        for ep_id in unique_eps:
            mask = episodes == ep_id
            ep_dones = dones[mask]
            if not ep_dones[-1]:
                issues.append(f"episode {ep_id}: missing terminal done=True")

    # Check metadata columns exist
    for col in ["expert_policy", "scenario", "platform"]:
        if col not in table.column_names:
            issues.append(f"Missing column: {col}")

    return {
        "file": path.name,
        "rows": n_rows,
        "issues": issues,
        "reward_mean": float(table.column("reward").to_numpy().mean()) if "reward" in table.column_names else None,
        "reward_std": float(table.column("reward").to_numpy().std()) if "reward" in table.column_names else None,
    }


def validate_dataset(data_dir: Path) -> bool:
    """Validate all Parquet files in a directory.

    Returns True if all files pass validation.
    """
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No Parquet files found in {data_dir}")
        return False

    total_rows = 0
    total_issues = 0
    all_passed = True

    print(f"Validating {len(parquet_files)} files in {data_dir}\n")

    for path in parquet_files:
        result = validate_file(path)
        total_rows += result["rows"]

        status = "PASS" if not result["issues"] else "FAIL"
        reward_info = ""
        if result["reward_mean"] is not None:
            reward_info = f" reward={result['reward_mean']:.3f}+/-{result['reward_std']:.3f}"

        print(f"  [{status}] {result['file']}: {result['rows']:,} rows{reward_info}")

        if result["issues"]:
            all_passed = False
            for issue in result["issues"]:
                print(f"    - {issue}")
                total_issues += 1

    print(f"\nTotal: {total_rows:,} rows, {total_issues} issues")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Validate SLM-OS scheduler dataset")
    parser.add_argument("data_dir", nargs="?", default=None,
                        help="Directory containing Parquet files")
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"

    passed = validate_dataset(data_dir)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
