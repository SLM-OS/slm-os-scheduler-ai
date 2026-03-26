"""Tests for slm_sim.actions — action encoding/decoding and application."""

from __future__ import annotations

import pytest

from slm_sim.actions import (
    SchedulingAction,
    _adjust_priority,
    _find_least_loaded_core,
    action_space_size,
    apply_action,
    decode_action,
    encode_action,
)
from slm_sim.engine import SimulatorEngine
from slm_sim.models import (
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_IDLE,
    PRIORITY_LOW,
    PRIORITY_NORMAL,
    ComponentType,
    SimTask,
    TaskState,
)
from slm_sim.platforms import make_jetson_orin_nano


@pytest.fixture
def engine():
    platform = make_jetson_orin_nano()
    eng = SimulatorEngine(platform=platform)
    eng.reset()
    return eng


class TestActionSpaceSize:
    def test_jetson_action_space(self):
        assert action_space_size(6, gpu_available=True) == 42

    def test_pi5_action_space(self):
        assert action_space_size(4, gpu_available=False) == 24

    def test_biglittle_action_space(self):
        assert action_space_size(6, gpu_available=False) == 36


class TestActionEncoding:
    def test_roundtrip_all_jetson_actions(self):
        n_cores = 6
        gpu = True
        total = action_space_size(n_cores, gpu)
        for idx in range(total):
            action = decode_action(idx, n_cores, gpu)
            re_encoded = encode_action(action, n_cores, gpu)
            assert re_encoded == idx, f"Roundtrip failed for idx={idx}"

    def test_roundtrip_all_pi5_actions(self):
        n_cores = 4
        gpu = False
        total = action_space_size(n_cores, gpu)
        for idx in range(total):
            action = decode_action(idx, n_cores, gpu)
            re_encoded = encode_action(action, n_cores, gpu)
            assert re_encoded == idx

    def test_decode_known_action(self):
        action = decode_action(0, 6, True)
        assert action.core_assignment == 0
        assert action.priority_adj == 0
        assert action.preempt is False

    def test_decode_last_action(self):
        action = decode_action(41, 6, True)
        assert action.core_assignment == 6
        assert action.priority_adj == 2
        assert action.preempt is True

    def test_encode_known_action(self):
        action = SchedulingAction(core_assignment=0, priority_adj=1, preempt=False)
        idx = encode_action(action, 6, True)
        assert idx == 0 * 3 * 2 + 1 * 2 + 0  # = 2


class TestAdjustPriority:
    def test_keep(self):
        assert _adjust_priority(PRIORITY_NORMAL, 1) == PRIORITY_NORMAL

    def test_raise_from_normal(self):
        assert _adjust_priority(PRIORITY_NORMAL, 2) == PRIORITY_HIGH

    def test_lower_from_normal(self):
        assert _adjust_priority(PRIORITY_NORMAL, 0) == PRIORITY_LOW

    def test_raise_from_critical_stays(self):
        assert _adjust_priority(PRIORITY_CRITICAL, 2) == PRIORITY_CRITICAL

    def test_lower_from_idle_stays(self):
        assert _adjust_priority(PRIORITY_IDLE, 0) == PRIORITY_IDLE

    def test_raise_from_high_to_critical(self):
        assert _adjust_priority(PRIORITY_HIGH, 2) == PRIORITY_CRITICAL


class TestApplyAction:
    def _make_task(self, engine, task_id=1, priority=PRIORITY_NORMAL):
        task = SimTask(
            task_id=task_id,
            name=f"task_{task_id}",
            state=TaskState.READY,
            priority=priority,
            effective_priority=priority,
            component_type=ComponentType.ANOMALY_DETECTOR,
        )
        engine.tasks[task_id] = task
        return task

    def test_assign_to_idle_core(self, engine):
        self._make_task(engine, task_id=1)
        action = SchedulingAction(core_assignment=2, priority_adj=1, preempt=False)
        apply_action(engine, 1, action)

        assert engine.tasks[1].state == TaskState.RUNNING
        assert engine.tasks[1].assigned_cpu == 2
        assert engine.cores[2].current_task == 1

    def test_enqueue_on_busy_core(self, engine):
        self._make_task(engine, task_id=1)
        self._make_task(engine, task_id=2)

        # Put task 1 on core 0
        apply_action(engine, 1, SchedulingAction(0, 1, False))
        # Enqueue task 2 on same core
        apply_action(engine, 2, SchedulingAction(0, 1, False))

        assert engine.cores[0].current_task == 1
        assert 2 in engine.cores[0].run_queue

    def test_preempt_lower_priority(self, engine):
        self._make_task(engine, task_id=1, priority=PRIORITY_LOW)
        self._make_task(engine, task_id=2, priority=PRIORITY_HIGH)

        apply_action(engine, 1, SchedulingAction(0, 1, False))
        assert engine.cores[0].current_task == 1

        apply_action(engine, 2, SchedulingAction(0, 1, True))
        assert engine.cores[0].current_task == 2
        assert 1 in engine.cores[0].run_queue
        assert engine.tasks[1].state == TaskState.READY

    def test_preempt_fails_if_higher_priority(self, engine):
        self._make_task(engine, task_id=1, priority=PRIORITY_HIGH)
        self._make_task(engine, task_id=2, priority=PRIORITY_LOW)

        apply_action(engine, 1, SchedulingAction(0, 1, False))
        apply_action(engine, 2, SchedulingAction(0, 1, True))

        # Should not preempt — task 1 has higher priority
        assert engine.cores[0].current_task == 1
        assert 2 in engine.cores[0].run_queue

    def test_priority_adjustment_applied(self, engine):
        self._make_task(engine, task_id=1, priority=PRIORITY_NORMAL)
        apply_action(engine, 1, SchedulingAction(0, 2, False))  # raise
        assert engine.tasks[1].effective_priority == PRIORITY_HIGH

    def test_invalid_core_falls_back(self, engine):
        self._make_task(engine, task_id=1)
        # Core 6 = GPU slot, but task not GPU-eligible
        apply_action(engine, 1, SchedulingAction(6, 1, False))
        # Should fall back to least-loaded core
        assert engine.tasks[1].state == TaskState.RUNNING
        assert engine.tasks[1].assigned_cpu is not None

    def test_isolated_core_falls_back(self, engine):
        engine.cores[0].isolated = True
        self._make_task(engine, task_id=1)
        apply_action(engine, 1, SchedulingAction(0, 1, False))
        # Should skip isolated core
        assert engine.tasks[1].assigned_cpu != 0


class TestFindLeastLoadedCore:
    def test_all_idle(self, engine):
        core_id = _find_least_loaded_core(engine)
        assert 0 <= core_id < len(engine.cores)

    def test_skips_busy_cores(self, engine):
        engine.cores[0].current_task = 99
        engine.cores[1].current_task = 98
        core_id = _find_least_loaded_core(engine)
        assert core_id >= 2

    def test_skips_isolated(self, engine):
        for c in engine.cores:
            c.isolated = True
        # Only unisolate core 3
        engine.cores[3].isolated = False
        assert _find_least_loaded_core(engine) == 3
