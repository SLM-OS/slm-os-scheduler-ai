"""Base class for workload generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from slm_sim.engine import SimulatorEngine


class WorkloadProfile(ABC):
    """Base class for component workload profiles.

    Each profile generates a stream of TASK_ARRIVAL events injected
    into the simulator's event queue.
    """

    def __init__(self, rng: np.random.Generator, intensity: float = 1.0):
        """
        Args:
            rng: Numpy random generator for reproducibility.
            intensity: Multiplier on arrival rate (1.0 = normal, 2.0 = double rate).
        """
        self.rng = rng
        self.intensity = intensity

    @abstractmethod
    def seed_events(self, engine: SimulatorEngine, start_ns: int,
                    end_ns: int) -> None:
        """Generate and push all TASK_ARRIVAL events for the episode window.

        Args:
            engine: Simulator engine to push events into.
            start_ns: Episode start time.
            end_ns: Episode end time.
        """
        ...

    @property
    @abstractmethod
    def component_name(self) -> str:
        """Human-readable name for this workload component."""
        ...
