"""PyTorch dataset loader for Parquet training data.

Loads state-action pairs from the generated Parquet dataset, applies
reward-based sample weighting, and provides batched iteration.

Supports a max_rows parameter to subsample when the full dataset
exceeds available RAM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from slm_sim.observation import TOTAL_FEATURES

# Expert policies used for supervised training (not random)
TRAINING_EXPERTS = {"slm_os_hybrid", "edf", "weighted_multi_objective"}


class SchedulerDataset(Dataset):
    """PyTorch dataset for supervised scheduler training.

    Loads state vectors, action labels, and sample weights from Parquet files.
    Filters to only include expert policies (excludes random).
    Sample weights are proportional to |reward| for reward-weighted CE loss.
    """

    def __init__(
        self,
        parquet_path: Path | str,
        experts: Optional[set[str]] = None,
        normalization_path: Optional[Path | str] = None,
        max_rows: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Args:
            parquet_path: Path to a single Parquet file (e.g., train.parquet).
            experts: Set of expert names to include. None = TRAINING_EXPERTS.
            normalization_path: Path to normalization.json for z-score normalization.
                If None, states are used as-is (already in [0,1]).
            max_rows: If set, randomly subsample to at most this many rows.
                Useful when the full dataset exceeds available RAM.
            seed: Random seed for subsampling.
        """
        parquet_path = Path(parquet_path)
        filter_experts = experts or TRAINING_EXPERTS

        # Streaming load: read row groups one at a time to avoid
        # holding the full PyArrow table in memory alongside numpy arrays.
        pf = pq.ParquetFile(parquet_path)
        needed_cols = (
            [f"state_{j:03d}" for j in range(TOTAL_FEATURES)]
            + ["action", "reward", "expert_policy"]
        )

        # Pass 1: count rows per expert to know final array size
        total_expert_rows = 0
        for i in range(pf.metadata.num_row_groups):
            rg = pf.read_row_group(i, columns=["expert_policy"])
            experts_in_group = rg.column("expert_policy").to_pylist()
            total_expert_rows += sum(1 for e in experts_in_group if e in filter_experts)
            del rg

        if total_expert_rows == 0:
            self.states = np.zeros((0, TOTAL_FEATURES), dtype=np.float32)
            self.actions = np.zeros(0, dtype=np.int64)
            self.weights = np.zeros(0, dtype=np.float32)
            return

        # Determine actual size (subsample if needed)
        if max_rows and total_expert_rows > max_rows:
            n = max_rows
            sample_ratio = max_rows / total_expert_rows
        else:
            n = total_expert_rows
            sample_ratio = None

        rng = np.random.default_rng(seed)

        # Allocate output arrays
        self.states = np.zeros((n, TOTAL_FEATURES), dtype=np.float32)
        self.actions = np.zeros(n, dtype=np.int64)
        rewards = np.zeros(n, dtype=np.float32)
        write_pos = 0

        # Pass 2: stream row groups and fill arrays
        for i in range(pf.metadata.num_row_groups):
            rg = pf.read_row_group(i, columns=needed_cols)
            expert_col = rg.column("expert_policy").to_pylist()
            mask = np.array([e in filter_experts for e in expert_col])

            if mask.sum() == 0:
                del rg
                continue

            rg = rg.filter(mask)
            rg_n = len(rg)

            # Subsample this row group proportionally
            if sample_ratio is not None:
                keep = int(round(rg_n * sample_ratio))
                if keep == 0:
                    del rg
                    continue
                keep = min(keep, n - write_pos)
                idx = rng.choice(rg_n, size=keep, replace=False)
                idx.sort()
                for j in range(TOTAL_FEATURES):
                    self.states[write_pos:write_pos + keep, j] = (
                        rg.column(f"state_{j:03d}").to_numpy()[idx]
                    )
                self.actions[write_pos:write_pos + keep] = (
                    rg.column("action").to_numpy().astype(np.int64)[idx]
                )
                rewards[write_pos:write_pos + keep] = (
                    rg.column("reward").to_numpy().astype(np.float32)[idx]
                )
                write_pos += keep
            else:
                chunk = min(rg_n, n - write_pos)
                for j in range(TOTAL_FEATURES):
                    self.states[write_pos:write_pos + chunk, j] = (
                        rg.column(f"state_{j:03d}").to_numpy()[:chunk]
                    )
                self.actions[write_pos:write_pos + chunk] = (
                    rg.column("action").to_numpy().astype(np.int64)[:chunk]
                )
                rewards[write_pos:write_pos + chunk] = (
                    rg.column("reward").to_numpy().astype(np.float32)[:chunk]
                )
                write_pos += chunk

            del rg
            if write_pos >= n:
                break

        # Trim if we ended up with fewer rows than allocated
        if write_pos < n:
            self.states = self.states[:write_pos]
            self.actions = self.actions[:write_pos]
            rewards = rewards[:write_pos]

        # Compute sample weights from |reward|
        abs_rewards = np.abs(rewards)
        mean_abs = abs_rewards.mean()
        if mean_abs > 1e-8:
            self.weights = abs_rewards / mean_abs
        else:
            self.weights = np.ones(len(self.actions), dtype=np.float32)
        del rewards

        # Optional z-score normalization
        if normalization_path is not None:
            self._apply_normalization(Path(normalization_path))

    def _apply_normalization(self, path: Path) -> None:
        """Apply z-score normalization using saved statistics."""
        with open(path) as f:
            stats = json.load(f)
        mean = np.array(stats["mean"], dtype=np.float32)
        std = np.array(stats["std"], dtype=np.float32)
        # Avoid division by zero
        std = np.where(std < 1e-8, 1.0, std)
        self.states = (self.states - mean) / std

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (state, action, weight)."""
        return (
            torch.from_numpy(self.states[idx]),
            torch.tensor(self.actions[idx], dtype=torch.long),
            torch.tensor(self.weights[idx], dtype=torch.float32),
        )

    @property
    def n_features(self) -> int:
        return TOTAL_FEATURES

    @property
    def n_actions(self) -> int:
        if len(self.actions) == 0:
            return 0
        return int(self.actions.max()) + 1
