"""Curriculum learning schedule for PPO training.

Phases A-D progressively increase scenario difficulty.
See plan Section 4.3.

Phase A (0-1M steps):   light_single, light_mixed
Phase B (1M-3M):        + medium_mixed, deadline_pressure
Phase C (3M-5M):        full scenario distribution
Phase D (5M-10M):       weighted toward heavy_inference, burst_storm
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class CurriculumPhase:
    """A single phase of the curriculum."""
    name: str
    start_step: int
    scenarios: list[str]
    weights: Optional[list[float]] = None  # sampling weights, uniform if None


# Default curriculum phases (Section 4.3)
DEFAULT_PHASES = [
    CurriculumPhase(
        name="A",
        start_step=0,
        scenarios=["light_single", "light_mixed"],
    ),
    CurriculumPhase(
        name="B",
        start_step=1_000_000,
        scenarios=["light_single", "light_mixed", "medium_mixed", "deadline_pressure"],
    ),
    CurriculumPhase(
        name="C",
        start_step=3_000_000,
        scenarios=[
            "light_single", "light_mixed", "medium_mixed",
            "heavy_inference", "burst_storm", "deadline_pressure",
            "memory_pressure", "asymmetric",
        ],
    ),
    CurriculumPhase(
        name="D",
        start_step=5_000_000,
        scenarios=[
            "light_single", "light_mixed", "medium_mixed",
            "heavy_inference", "burst_storm", "deadline_pressure",
            "memory_pressure", "asymmetric",
        ],
        weights=[0.05, 0.05, 0.10, 0.30, 0.25, 0.10, 0.10, 0.05],
    ),
]


class CurriculumSchedule:
    """Manages scenario progression during training."""

    def __init__(self, phases: list[CurriculumPhase] | None = None,
                 seed: int = 42):
        self.phases = phases or DEFAULT_PHASES
        self.rng = np.random.default_rng(seed)
        self._current_phase_idx = 0

    @property
    def current_phase(self) -> CurriculumPhase:
        return self.phases[self._current_phase_idx]

    def update(self, total_steps: int) -> bool:
        """Update curriculum phase based on total training steps.

        Returns True if the phase changed.
        """
        new_idx = self._current_phase_idx
        for i, phase in enumerate(self.phases):
            if total_steps >= phase.start_step:
                new_idx = i
        changed = new_idx != self._current_phase_idx
        self._current_phase_idx = new_idx
        return changed

    def sample_scenario(self) -> str:
        """Sample a scenario from the current phase distribution."""
        phase = self.current_phase
        if phase.weights:
            return self.rng.choice(phase.scenarios, p=phase.weights)
        else:
            return self.rng.choice(phase.scenarios)


class CurriculumCallback(BaseCallback):
    """SB3 callback that updates the curriculum during training.

    At each rollout end, checks if the phase should advance and
    updates the environment's scenario accordingly.
    """

    def __init__(self, schedule: CurriculumSchedule, verbose: int = 0):
        super().__init__(verbose)
        self.schedule = schedule

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        total_steps = self.num_timesteps
        changed = self.schedule.update(total_steps)
        if changed and self.verbose > 0:
            phase = self.schedule.current_phase
            print(f"Curriculum: switched to Phase {phase.name} "
                  f"at step {total_steps} — scenarios: {phase.scenarios}")

        # Update scenario for each env in the vec_env
        new_scenario = self.schedule.sample_scenario()
        if hasattr(self.training_env, "env_method"):
            try:
                self.training_env.env_method("set_scenario", new_scenario)
            except AttributeError:
                pass  # env doesn't support dynamic scenario switching
