"""PyTorch dataset loader for Parquet training data.

Loads state-action pairs from the generated Parquet dataset, applies
reward-based sample weighting, and provides batched iteration.
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
    ):
        """
        Args:
            parquet_path: Path to a single Parquet file (e.g., train.parquet).
            experts: Set of expert names to include. None = TRAINING_EXPERTS.
            normalization_path: Path to normalization.json for z-score normalization.
                If None, states are used as-is (already in [0,1]).
        """
        table = pq.read_table(Path(parquet_path))

        # Filter by expert policy
        filter_experts = experts or TRAINING_EXPERTS
        expert_col = table.column("expert_policy").to_pylist()
        mask = np.array([e in filter_experts for e in expert_col])

        if mask.sum() == 0:
            self.states = np.zeros((0, TOTAL_FEATURES), dtype=np.float32)
            self.actions = np.zeros(0, dtype=np.int64)
            self.weights = np.zeros(0, dtype=np.float32)
            return

        table = table.filter(mask)
        n = len(table)

        # Extract state features
        self.states = np.zeros((n, TOTAL_FEATURES), dtype=np.float32)
        for j in range(TOTAL_FEATURES):
            col = f"state_{j:03d}"
            self.states[:, j] = table.column(col).to_numpy()

        # Extract actions
        self.actions = table.column("action").to_numpy().astype(np.int64)

        # Compute sample weights from |reward|
        rewards = table.column("reward").to_numpy().astype(np.float32)
        abs_rewards = np.abs(rewards)
        # Normalize so mean weight = 1.0
        mean_abs = abs_rewards.mean()
        if mean_abs > 1e-8:
            self.weights = abs_rewards / mean_abs
        else:
            self.weights = np.ones(n, dtype=np.float32)

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
