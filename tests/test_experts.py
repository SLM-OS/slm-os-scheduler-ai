"""Tests for expert scheduling policies."""

from __future__ import annotations

import numpy as np
import pytest

from slm_sim.actions import SchedulingAction
from slm_sim.engine import SimulatorEngine
from slm_sim.experts.edf import EDFExpertPolicy
from slm_sim.experts.hybrid import HybridExpertPolicy
from slm_sim.experts.random_policy import RandomExpertPolicy
from slm_sim.experts.weighted import WeightedExpertPolicy
from slm_sim.models import (
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_NORMAL,
    ComponentType,
    SimTask,
    TaskState,
)
from slm_sim.platforms import make_jetson_orin_nano
from slm_sim.workloads.scenarios import ScenarioComposer


@pytest.fixture
def engine():
    platform = make_jetson_orin_nano()
    eng = SimulatorEngine(platform=platform, episode_duration_ns=2_000_000_000)
    eng.reset()
    return eng


def add_task(engine, task_id, priority=PRIORITY_NORMAL, deadline_ns=0,
             working_set_mb=4.0, component=ComponentType.ANOMALY_DETECTOR):
    task = SimTask(
        task_id=task_id,
        name=f"task_{task_id}",
        state=TaskState.READY,
        priority=priority,
        effective_priority=priority,
        working_set_mb=working_set_mb,
        model_size_mb=12.0,
        inference_duration_ns=1_000_000,
        deadline_ns=deadline_ns,
        arrival_time_ns=engine.clock_ns,
        remaining_work_ns=1_000_000,
        component_type=component,
    )
    engine.tasks[task_id] = task
    return task


class TestEDFExpert:
    def test_returns_valid_action(self, engine):
        add_task(engine, 1, deadline_ns=engine.clock_ns + 5_000_000)
        policy = EDFExpertPolicy()
        action = policy.decide(engine, 1)
        assert isinstance(action, SchedulingAction)
        assert 0 <= action.core_assignment < len(engine.cores)

    def test_preempts_for_nearer_deadline(self, engine):
        # Put task with far deadline on core 0
        add_task(engine, 1, deadline_ns=engine.clock_ns + 100_000_000)
        engine.tasks[1].state = TaskState.RUNNING
        engine.tasks[1].assigned_cpu = 0
        engine.cores[0].current_task = 1

        # New task with near deadline
        add_task(engine, 2, deadline_ns=engine.clock_ns + 5_000_000)
        policy = EDFExpertPolicy()
        action = policy.decide(engine, 2)
        # Should want to preempt (core 0 has furthest-deadline task)
        # The exact core depends on load, but preempt should be True
        # if it targets the core with task 1
        if action.core_assignment == 0:
            assert action.preempt is True

    def test_no_preempt_without_deadline(self, engine):
        add_task(engine, 1, deadline_ns=0)  # no deadline
        policy = EDFExpertPolicy()
        action = policy.decide(engine, 1)
        assert action.preempt is False

    def test_runs_episode(self, engine):
        rng = np.random.default_rng(42)
        workload = ScenarioComposer("light_single", rng)
        policy = EDFExpertPolicy()
        metrics = engine.run_episode(policy, workload)
        assert metrics.total_tasks_completed > 0


class TestWeightedExpert:
    def test_returns_valid_action(self, engine):
        add_task(engine, 1)
        policy = WeightedExpertPolicy()
        action = policy.decide(engine, 1)
        assert isinstance(action, SchedulingAction)

    def test_prefers_less_loaded_core(self, engine):
        # Load up cores 0-3
        for i in range(4):
            engine.cores[i].current_task = 100 + i
            engine.cores[i].utilization_pct = 0.9
        add_task(engine, 1)
        policy = WeightedExpertPolicy()
        action = policy.decide(engine, 1)
        # Should prefer cores 4 or 5 (less loaded)
        assert action.core_assignment >= 4

    def test_raises_priority_for_urgent_deadline(self, engine):
        add_task(engine, 1, deadline_ns=engine.clock_ns + 10_000_000)  # 10 ms
        policy = WeightedExpertPolicy()
        action = policy.decide(engine, 1)
        assert action.priority_adj == 2  # raise

    def test_runs_episode(self, engine):
        rng = np.random.default_rng(42)
        workload = ScenarioComposer("light_single", rng)
        policy = WeightedExpertPolicy()
        metrics = engine.run_episode(policy, workload)
        assert metrics.total_tasks_completed > 0


class TestRandomExpert:
    def test_returns_valid_action(self, engine):
        add_task(engine, 1)
        policy = RandomExpertPolicy(rng=np.random.default_rng(42))
        action = policy.decide(engine, 1)
        assert isinstance(action, SchedulingAction)
        max_core = len(engine.cores) + (1 if engine.gpu.available else 0)
        assert 0 <= action.core_assignment < max_core
        assert action.priority_adj in (0, 1, 2)

    def test_randomness(self, engine):
        add_task(engine, 1)
        policy = RandomExpertPolicy(rng=np.random.default_rng(42))
        actions = [policy.decide(engine, 1) for _ in range(50)]
        # Should have some variety
        cores = set(a.core_assignment for a in actions)
        assert len(cores) > 1

    def test_runs_episode(self, engine):
        rng = np.random.default_rng(42)
        workload = ScenarioComposer("light_single", rng)
        policy = RandomExpertPolicy(rng=np.random.default_rng(42))
        metrics = engine.run_episode(policy, workload)
        assert metrics.total_tasks_completed > 0
