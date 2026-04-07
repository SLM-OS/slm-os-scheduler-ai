#!/usr/bin/env python3
"""Train all three models: MLP, XGBoost, PPO.

Each stage runs in a separate subprocess so memory is fully
reclaimed between them. The training dataset is ~11GB on disk
and 30-40GB in memory, so stages cannot share a process.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

STAGES = [
    ("MLP",     "scripts/_train_mlp.py"),
    ("XGBoost", "scripts/_train_xgboost.py"),
    ("PPO",     "scripts/_train_ppo.py"),
]


def main():
    for name, script in STAGES:
        print("=" * 60)
        print(f"Training {name}...")
        print("=" * 60, flush=True)

        result = subprocess.run(
            [PYTHON, str(PROJECT_ROOT / script)],
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            print(f"\n{name} training failed (exit code {result.returncode})")
            sys.exit(result.returncode)

        print(f"{name} done.\n")

    print("All models trained.")


if __name__ == "__main__":
    main()
