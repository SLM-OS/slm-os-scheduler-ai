"""SLM-OS Hybrid expert policy (Phase 3 scheduler replica).

Replicates the existing SLM-OS Phase 3 scheduling heuristics:
priority queue, deadline boost, working set affinity, find_target_cpu().
See plan Section 5.1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from slm_sim.actions import SchedulingAction
from slm_sim.experts import ExpertPolicy
from slm_sim.models import (
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    CoreType,
    TaskState,
)

if TYPE_CHECKING:
    from slm_sim.engine import SimulatorEngine

# Deadline boost thresholds (matching kernel sched.c)
DEADLINE_CRITICAL_NS = 10_000_000   # 10 ms
DEADLINE_HIGH_NS = 50_000_000       # 50 ms
DEADLINE_BOOST_NS = 100_000_000     # 100 ms

# Working set threshold for core affinity
LARGE_WORKING_SET_MB = 8.0


class HybridExpertPolicy(ExpertPolicy):
    """Replicates the Phase 3 SLM-OS scheduler heuristics."""

    @property
    def name(self) -> str:
        return "slm_os_hybrid"

    def decide(self, engine: SimulatorEngine, task_id: int) -> SchedulingAction:
        """Schedule using SLM-OS Phase 3 logic.

        - Deadline boost: <10ms -> CRITICAL; <50ms -> HIGH; <100ms -> +1
        - Working set < 8 MB -> any core; >= 8 MB -> performance core
        - Deadline tasks pinned to non-boot CPUs
        - find_target_cpu() picks least-loaded non-isolated core
        - No preemption (always enqueue)
        """
        task = engine.tasks[task_id]

        # --- Deadline boost ---
        priority_adj = 1  # keep by default
        if task.deadline_ns > 0:
            remaining_ns = task.deadline_ns - engine.clock_ns
            if remaining_ns < DEADLINE_CRITICAL_NS:
                # Boost to CRITICAL
                if task.effective_priority < PRIORITY_CRITICAL:
                    priority_adj = 2  # raise
            elif remaining_ns < DEADLINE_HIGH_NS:
                # Boost to HIGH
                if task.effective_priority < PRIORITY_HIGH:
                    priority_adj = 2  # raise
            elif remaining_ns < DEADLINE_BOOST_NS:
                # Minor boost (+1 level)
                priority_adj = 2  # raise

        # --- Core assignment ---
        target_core = self._find_target_cpu(engine, task)

        # No preemption in the hybrid policy
        return SchedulingAction(
            core_assignment=target_core,
            priority_adj=priority_adj,
            preempt=False,
        )

    def _find_target_cpu(self, engine: SimulatorEngine, task) -> int:
        """Find the best core for this task.

        Logic from kernel sched.c:
        - Large working set or deadline tasks -> performance cores, non-boot (core > 0)
        - Regular tasks -> least-loaded non-isolated core
        """
        has_deadline = task.deadline_ns > 0
        large_working_set = task.working_set_mb >= LARGE_WORKING_SET_MB

        if has_deadline or large_working_set:
            return self._find_performance_cpu(engine)
        else:
            return self._find_least_loaded_cpu(engine)

    def _find_performance_cpu(self, engine: SimulatorEngine) -> int:
        """Find least-loaded performance core, preferring non-boot (core > 0)."""
        best_core = None
        best_load = float("inf")

        for core in engine.cores:
            if core.isolated:
                continue
            if core.core_type != CoreType.PERFORMANCE:
                continue

            load = len(core.run_queue) + (1 if core.current_task is not None else 0)
            # Penalize boot CPU (core 0)
            if core.core_id == 0:
                load += 100

            if load < best_load:
                best_load = load
                best_core = core.core_id

        if best_core is not None:
            return best_core

        # Fallback: any non-isolated core
        return self._find_least_loaded_cpu(engine)

    def _find_least_loaded_cpu(self, engine: SimulatorEngine) -> int:
        """Find the non-isolated core with the lowest load."""
        best_core = 0
        best_load = float("inf")

        for core in engine.cores:
            if core.isolated:
                continue
            load = len(core.run_queue) + (1 if core.current_task is not None else 0)
            if load < best_load:
                best_load = load
                best_core = core.core_id

        return best_core
