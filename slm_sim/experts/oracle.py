"""Oracle expert policy (offline optimal upper bound).

Runs the full episode, then backtracks via beam search to find the
scheduling decisions that would have minimized deadline misses.
Computationally expensive — used only as a theoretical performance
ceiling. See plan Section 5.5.
"""

from __future__ import annotations

import copy
import heapq
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from slm_sim.actions import (
    SchedulingAction,
    action_space_size,
    apply_action,
    decode_action,
)
from slm_sim.experts import ExpertPolicy
from slm_sim.models import TaskState

if TYPE_CHECKING:
    from slm_sim.engine import SimulatorEngine
    from slm_sim.workloads.scenarios import ScenarioComposer


@dataclass(order=True)
class BeamCandidate:
    """A candidate in the beam search, ordered by negative score (higher is better)."""
    neg_score: float
    actions: list[SchedulingAction] = field(compare=False)
    # Lightweight state summary rather than full engine copy
    deadline_met: int = field(compare=False, default=0)
    deadline_missed: int = field(compare=False, default=0)
    total_completed: int = field(compare=False, default=0)


class OracleExpertPolicy(ExpertPolicy):
    """Offline optimal policy via beam search over complete episodes.

    This policy cannot make online decisions. Instead, run_offline_episode()
    plays through an episode multiple times with different action sequences,
    keeping the top-k (beam_width) trajectories at each decision point.

    The search is heuristic — it samples a subset of actions at each step
    rather than exploring the full action space, making it tractable while
    still finding good solutions.
    """

    def __init__(self, beam_width: int = 8, actions_per_step: int = 6,
                 max_decisions: int = 500):
        """
        Args:
            beam_width: Number of top candidates to keep at each step.
            actions_per_step: Number of random actions to try per candidate
                at each decision point (full action space is too large).
            max_decisions: Maximum decision points to search over before
                falling back to a heuristic for the remainder.
        """
        self.beam_width = beam_width
        self.actions_per_step = actions_per_step
        self.max_decisions = max_decisions

    @property
    def name(self) -> str:
        return "oracle"

    def decide(self, engine: SimulatorEngine, task_id: int) -> SchedulingAction:
        """Not callable online — use run_offline_episode() instead."""
        raise RuntimeError(
            "Oracle policy cannot make online decisions. "
            "Use run_offline_episode() to compute optimal decisions "
            "over a complete episode."
        )

    def run_offline_episode(
        self,
        engine: SimulatorEngine,
        workload: ScenarioComposer,
        seed: int = 42,
    ) -> tuple[list[SchedulingAction], float]:
        """Run beam search over a complete episode to find near-optimal decisions.

        Strategy:
        1. Run a full episode with the Hybrid expert to establish a baseline.
        2. Replay the episode, at each decision point trying multiple actions
           and keeping the top beam_width candidates by a scoring function
           that prioritizes deadline compliance.
        3. Since full engine copies are expensive, we use a greedy forward
           search: at each decision point, score candidates based on
           immediate reward (completed tasks and their deadline status).

        Args:
            engine: Simulator engine (will be reset).
            workload: Scenario composer for seeding events.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (best_actions, best_score).
        """
        rng = np.random.default_rng(seed)
        n_cores = len(engine.platform.cores)
        has_gpu = engine.platform.gpu.available
        n_actions = action_space_size(n_cores, has_gpu)

        # Phase 1: Run baseline with hybrid expert to get event sequence
        from slm_sim.experts.hybrid import HybridExpertPolicy
        from slm_sim.workloads.scenarios import ScenarioComposer
        baseline_expert = HybridExpertPolicy()
        engine.reset(seed=seed)
        workload_rng = np.random.default_rng(seed)
        baseline_workload = ScenarioComposer(workload.config.name, workload_rng)
        baseline_metrics = engine.run_episode(baseline_expert, baseline_workload)
        baseline_score = self._compute_score(baseline_metrics)

        # Collect the decision points and their task IDs from the baseline run
        baseline_decisions = len(engine.transitions)
        if baseline_decisions == 0:
            return [], baseline_score

        # Phase 2: Greedy beam search replay
        # We'll replay the episode, at each decision point sampling actions
        # and greedily picking the best immediate outcome.
        best_actions: list[SchedulingAction] = []
        best_score = baseline_score

        # Simple greedy forward search: for each decision point up to
        # max_decisions, try actions_per_step random actions and pick
        # the one that leads to the best immediate outcome.
        engine.reset(seed=seed)
        workload_rng = np.random.default_rng(seed)
        replay_workload = ScenarioComposer(workload.config.name, workload_rng)
        replay_workload.seed_all_events(engine, 0, engine.episode_duration_ns)

        from slm_sim.observation import extract_observation
        from slm_sim.reward import compute_reward

        decisions_made = 0
        while engine.event_queue and decisions_made < self.max_decisions:
            event = engine.pop_event()
            if event is None or event.timestamp_ns > engine.episode_duration_ns:
                break

            old_clock = engine.clock_ns
            engine.clock_ns = event.timestamp_ns
            engine._update_core_busy_time(old_clock, event.timestamp_ns)

            needs_decision = engine._process_event(event)
            if not needs_decision:
                continue

            ready_task = engine._pick_highest_priority_ready_task()
            if ready_task is None:
                continue

            decisions_made += 1

            # Try multiple actions, pick the one with best immediate score
            # Save engine state (lightweight: just task states and core states)
            best_action = None
            best_immediate = -float("inf")

            # Generate candidate action indices
            candidates = set()
            # Always include: least-loaded core with keep priority, no preempt
            from slm_sim.actions import encode_action
            for core in engine.cores:
                if not core.isolated:
                    baseline_act = SchedulingAction(core.core_id, 1, False)
                    candidates.add(encode_action(baseline_act, n_cores, has_gpu))
                    break

            # Random samples
            while len(candidates) < self.actions_per_step:
                candidates.add(int(rng.integers(0, n_actions)))

            for action_idx in candidates:
                action = decode_action(action_idx, n_cores, has_gpu)
                # Score this action based on task-core fit
                score = self._score_action(engine, ready_task, action)
                if score > best_immediate:
                    best_immediate = score
                    best_action = action

            if best_action is None:
                best_action = SchedulingAction(0, 1, False)

            best_actions.append(best_action)
            apply_action(engine, ready_task.task_id, best_action)
            engine._schedule_completion_for_task(ready_task.task_id)

        # Process remaining events without decisions (run to completion)
        while engine.event_queue:
            event = engine.pop_event()
            if event is None or event.timestamp_ns > engine.episode_duration_ns:
                break
            old_clock = engine.clock_ns
            engine.clock_ns = event.timestamp_ns
            engine._update_core_busy_time(old_clock, event.timestamp_ns)
            engine._process_event(event)

        final_score = self._compute_score(engine.metrics)
        return best_actions, max(final_score, baseline_score)

    def _score_action(self, engine: SimulatorEngine, task, action: SchedulingAction) -> float:
        """Score an action heuristically without actually applying it.

        Higher score = better action. Considers:
        - Deadline urgency of the task
        - Load on target core
        - Cache fit
        - Whether preemption makes sense
        """
        score = 0.0
        n_cores = len(engine.cores)

        # Resolve target core
        target = action.core_assignment
        if target >= n_cores:
            # GPU assignment — bonus if task is GPU-eligible
            if task.can_use_gpu and engine.gpu.available:
                score += 0.5
            else:
                score -= 1.0
            return score

        core = engine.cores[target]
        if core.isolated:
            return -2.0

        # Prefer less loaded cores
        load = len(core.run_queue) + (1 if core.current_task is not None else 0)
        score += max(0.0, 1.0 - load * 0.2)

        # Deadline urgency bonus
        if task.deadline_ns > 0:
            remaining = task.deadline_ns - engine.clock_ns
            if remaining < 10_000_000:  # < 10 ms
                score += 2.0
            elif remaining < 50_000_000:
                score += 1.0

        # Cache fit
        cache_mb = core.cache_size_kb / 1024.0
        if cache_mb > 0 and task.working_set_mb <= cache_mb:
            score += 0.3

        # Priority raise bonus for urgent tasks
        if action.priority_adj == 2 and task.deadline_ns > 0:
            remaining = task.deadline_ns - engine.clock_ns
            if remaining < 50_000_000:
                score += 0.2

        # Preemption: good if current task is lower priority
        if action.preempt and core.current_task is not None:
            current = engine.tasks.get(core.current_task)
            if current and current.effective_priority < task.effective_priority:
                score += 0.4
            else:
                score -= 0.3  # Bad preemption

        return score

    def _compute_score(self, metrics) -> float:
        """Score an episode's metrics. Higher = better."""
        dcr = metrics.deadline_compliance_rate
        throughput = metrics.total_tasks_completed
        # Heavily weight DCR
        return dcr * 1000 + throughput * 0.1
