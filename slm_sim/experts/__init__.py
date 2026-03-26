"""Expert scheduling policies for training data generation.

Each expert implements decide(state) -> action, producing scheduling
decisions that serve as training labels for supervised learning models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from slm_sim.actions import SchedulingAction
    from slm_sim.engine import SimulatorEngine


class ExpertPolicy(ABC):
    """Base class for expert scheduling policies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Policy name for logging and dataset metadata."""
        ...

    @abstractmethod
    def decide(self, engine: SimulatorEngine, task_id: int) -> SchedulingAction:
        """Make a scheduling decision for the given pending task.

        Args:
            engine: Current simulator state.
            task_id: The task that needs to be scheduled.

        Returns:
            A SchedulingAction (core, priority_adj, preempt).
        """
        ...
