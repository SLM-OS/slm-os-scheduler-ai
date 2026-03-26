"""Earliest Deadline First (EDF) expert policy.

Strict deadline ordering: nearest deadline runs first. Ties broken by
priority then FIFO. See plan Section 5.2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from slm_sim.actions import SchedulingAction
from slm_sim.experts import ExpertPolicy
from slm_sim.models import TaskState

if TYPE_CHECKING:
    from slm_sim.engine import SimulatorEngine


class EDFExpertPolicy(ExpertPolicy):
    """Earliest Deadline First scheduling policy."""

    @property
    def name(self) -> str:
        return "edf"

    def decide(self, engine: SimulatorEngine, task_id: int) -> SchedulingAction:
        """Schedule by nearest deadline, least-loaded core, with preemption.

        - Tasks with deadlines are ordered by deadline (nearest first).
        - Tasks without deadlines sorted after all deadline tasks.
        - Ties broken by priority (descending), then arrival_time (FIFO).
        - Core assignment: least-loaded non-isolated core.
        - Preempts if new task has a nearer deadline than current task on target core.
        """
        task = engine.tasks[task_id]
        target_core = self._find_least_loaded(engine)
        preempt = self._should_preempt(engine, task, target_core)

        # Priority adjustment: keep (EDF doesn't modify priorities)
        return SchedulingAction(
            core_assignment=target_core,
            priority_adj=1,  # keep
            preempt=preempt,
        )

    def _find_least_loaded(self, engine: SimulatorEngine) -> int:
        """Find least-loaded non-isolated core."""
        best = 0
        best_load = float("inf")
        for core in engine.cores:
            if core.isolated:
                continue
            load = len(core.run_queue) + (1 if core.current_task is not None else 0)
            if load < best_load:
                best_load = load
                best = core.core_id
        return best

    def _should_preempt(self, engine: SimulatorEngine, new_task, core_id: int) -> bool:
        """Preempt if new task has a nearer deadline than current task."""
        if new_task.deadline_ns == 0:
            return False

        core = engine.cores[core_id]
        if core.current_task is None:
            return False

        current = engine.tasks.get(core.current_task)
        if current is None or current.state != TaskState.RUNNING:
            return False

        # No deadline on current task -> new deadline task should preempt
        if current.deadline_ns == 0:
            return True

        # Both have deadlines: preempt if new task's deadline is nearer
        return new_task.deadline_ns < current.deadline_ns
