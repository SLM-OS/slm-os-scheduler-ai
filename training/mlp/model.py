"""MLP architecture for the scheduling policy.

Input(108) -> Dense(256, ReLU) -> BN -> Dropout(0.1)
           -> Dense(256, ReLU) -> BN -> Dropout(0.1)
           -> Dense(128, ReLU) -> BN
           -> Dense(N_ACTIONS)

~131K parameters. See plan Section 4.1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from slm_sim.observation import TOTAL_FEATURES


class SchedulerMLP(nn.Module):
    """Feedforward neural network for scheduling policy.

    Takes a 108-dim state vector and outputs logits over the action space.
    Architecture matches plan Section 4.1.
    """

    def __init__(
        self,
        n_features: int = TOTAL_FEATURES,
        n_actions: int = 42,
        hidden1: int = 256,
        hidden2: int = 256,
        hidden3: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions

        self.network = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(dropout),

            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden3),

            nn.Linear(hidden3, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: State tensor of shape (batch, n_features).

        Returns:
            Logits tensor of shape (batch, n_actions).
        """
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get action predictions (argmax of logits).

        Args:
            x: State tensor of shape (batch, n_features).

        Returns:
            Action indices of shape (batch,).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=-1)

    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
