"""Random baseline expert policy.

Generates negative training examples by making uniformly random
scheduling decisions. See plan Section 5.4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from slm_sim.actions import SchedulingAction
from slm_sim.experts import ExpertPolicy

if TYPE_CHECKING:
    from slm_sim.engine import SimulatorEngine


class RandomExpertPolicy(ExpertPolicy):
    """Uniformly random scheduling decisions."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng if rng is not None else np.random.default_rng()

    @property
    def name(self) -> str:
        return "random"

    def decide(self, engine: SimulatorEngine, task_id: int) -> SchedulingAction:
        """Random core, random priority adjustment, 50/50 preempt."""
        num_cores = len(engine.cores)
        has_gpu = engine.gpu.available

        # Random core (including GPU slot if available)
        max_core = num_cores + (1 if has_gpu else 0)
        core_assignment = int(self.rng.integers(0, max_core))

        # Random priority adjustment: 0=lower, 1=keep, 2=raise
        priority_adj = int(self.rng.integers(0, 3))

        # 50/50 preempt
        preempt = bool(self.rng.random() < 0.5)

        return SchedulingAction(
            core_assignment=core_assignment,
            priority_adj=priority_adj,
            preempt=preempt,
        )
