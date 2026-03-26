"""Weighted Multi-Objective expert policy.

Scores (task, core) pairs using a weighted combination of deadline urgency,
priority, cache affinity, core utilization, and power efficiency.
See plan Section 5.3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from slm_sim.actions import SchedulingAction
from slm_sim.experts import ExpertPolicy
from slm_sim.models import PRIORITY_MAX, CoreType, TaskState

if TYPE_CHECKING:
    from slm_sim.engine import SimulatorEngine


class WeightedExpertPolicy(ExpertPolicy):
    """Multi-objective scoring policy."""

    def __init__(
        self,
        w_deadline: float = 0.35,
        w_priority: float = 0.25,
        w_cache: float = 0.20,
        w_utilization: float = 0.15,
        w_power: float = 0.05,
        preempt_threshold: float = 0.3,
    ):
        self.w_deadline = w_deadline
        self.w_priority = w_priority
        self.w_cache = w_cache
        self.w_utilization = w_utilization
        self.w_power = w_power
        self.preempt_threshold = preempt_threshold

    @property
    def name(self) -> str:
        return "weighted_multi_objective"

    def decide(self, engine: SimulatorEngine, task_id: int) -> SchedulingAction:
        """Score all (task, core) pairs and pick the best assignment.

        score(task, core) = w1 * deadline_urgency(task)
                          + w2 * priority_norm(task)
                          + w3 * cache_affinity(task, core)
                          + w4 * (1 - core_utilization(core))
                          + w5 * power_efficiency(core)

        Preempt if score exceeds current task's score by > threshold.
        """
        task = engine.tasks[task_id]

        best_core_id = 0
        best_score = -float("inf")

        for core in engine.cores:
            if core.isolated:
                continue
            score = self._score(engine, task, core)
            if score > best_score:
                best_score = score
                best_core_id = core.core_id

        # Priority adjustment based on deadline urgency
        priority_adj = 1  # keep
        if task.deadline_ns > 0:
            remaining = task.deadline_ns - engine.clock_ns
            if remaining < 50_000_000:  # < 50 ms
                priority_adj = 2  # raise

        # Preemption check
        preempt = False
        core = engine.cores[best_core_id]
        if core.current_task is not None:
            current = engine.tasks.get(core.current_task)
            if current is not None and current.state == TaskState.RUNNING:
                current_score = self._score(engine, current, core)
                if best_score - current_score > self.preempt_threshold:
                    preempt = True

        return SchedulingAction(
            core_assignment=best_core_id,
            priority_adj=priority_adj,
            preempt=preempt,
        )

    def _score(self, engine: SimulatorEngine, task, core) -> float:
        """Compute the multi-objective score for a (task, core) pair."""
        # Deadline urgency: 0 if no deadline, else scales with proximity
        urgency = 0.0
        if task.deadline_ns > 0:
            remaining = task.deadline_ns - engine.clock_ns
            if remaining <= 0:
                urgency = 1.0
            else:
                # Map 100ms -> 0, 0ms -> 1
                urgency = max(0.0, min(1.0, 1.0 - remaining / 100_000_000))

        # Priority normalized
        priority_norm = task.effective_priority / PRIORITY_MAX if PRIORITY_MAX > 0 else 0.0

        # Cache affinity: 1.0 if working set fits in remaining cache
        cache_mb = core.cache_size_kb / 1024.0
        # Estimate remaining cache as total minus current pressure
        cache_remaining = max(0.0, cache_mb - core.cache_pressure * cache_mb)
        cache_affinity = 1.0 if task.working_set_mb <= cache_remaining else 0.0

        # Core availability: prefer less utilized cores
        availability = 1.0 - core.utilization_pct

        # Power efficiency: prefer efficiency cores (lower power weight = more efficient)
        max_power = max(c.power_weight for c in engine.cores)
        power_eff = 1.0 - (core.power_weight / max_power) if max_power > 0 else 0.5

        return (self.w_deadline * urgency
                + self.w_priority * priority_norm
                + self.w_cache * cache_affinity
                + self.w_utilization * availability
                + self.w_power * power_eff)
