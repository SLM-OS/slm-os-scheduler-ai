"""Tests for slm_sim.observation — state vector extraction."""

from __future__ import annotations

import numpy as np
import pytest

from slm_sim.engine import SimulatorEngine
from slm_sim.models import (
    PRIORITY_HIGH,
    PRIORITY_NORMAL,
    ComponentType,
    SimTask,
    TaskState,
)
from slm_sim.observation import (
    FEATURES_PER_CORE,
    FEATURES_PER_TASK,
    GLOBAL_FEATURES,
    MAX_CORES,
    MAX_PENDING_TASKS,
    TOTAL_FEATURES,
    extract_observation,
)
from slm_sim.platforms import make_jetson_orin_nano, make_raspberry_pi5


@pytest.fixture
def engine():
    platform = make_jetson_orin_nano()
    eng = SimulatorEngine(platform=platform)
    eng.reset()
    return eng


class TestObservationDimensions:
    def test_total_features(self):
        expected = (FEATURES_PER_CORE * MAX_CORES
                    + FEATURES_PER_TASK * MAX_PENDING_TASKS
                    + GLOBAL_FEATURES)
        assert TOTAL_FEATURES == expected
        assert TOTAL_FEATURES == 108


class TestExtractObservation:
    def test_empty_state_shape(self, engine):
        obs = extract_observation(engine)
        assert obs.shape == (108,)
        assert obs.dtype == np.float32

    def test_empty_state_all_in_range(self, engine):
        obs = extract_observation(engine)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_with_ready_tasks(self, engine):
        # Add some tasks
        for i in range(3):
            task = SimTask(
                task_id=i + 1,
                name=f"task_{i}",
                state=TaskState.READY,
                priority=PRIORITY_NORMAL,
                effective_priority=PRIORITY_NORMAL,
                working_set_mb=4.0,
                model_size_mb=12.0,
                inference_duration_ns=1_000_000,
                component_type=ComponentType.ANOMALY_DETECTOR,
                arrival_time_ns=0,
            )
            engine.tasks[task.task_id] = task

        obs = extract_observation(engine)
        assert obs.shape == (108,)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

        # Task features should be non-zero for first 3 task slots
        task_start = FEATURES_PER_CORE * MAX_CORES
        assert obs[task_start] > 0  # first task priority_norm > 0

    def test_core_utilization_reflected(self, engine):
        engine.cores[0].utilization_pct = 0.75
        obs = extract_observation(engine)
        assert obs[0] == pytest.approx(0.75)  # core 0 utilization

    def test_pi5_fewer_cores_padded(self):
        platform = make_raspberry_pi5()
        eng = SimulatorEngine(platform=platform)
        eng.reset()
        obs = extract_observation(eng)
        assert obs.shape == (108,)
        # Cores 4-5 should be zero-padded
        core4_start = 4 * FEATURES_PER_CORE
        core5_start = 5 * FEATURES_PER_CORE
        assert np.all(obs[core4_start:core4_start + FEATURES_PER_CORE] == 0.0)
        assert np.all(obs[core5_start:core5_start + FEATURES_PER_CORE] == 0.0)

    def test_more_than_8_tasks_only_top_k(self, engine):
        for i in range(12):
            task = SimTask(
                task_id=i + 1,
                name=f"task_{i}",
                state=TaskState.READY,
                priority=PRIORITY_NORMAL,
                effective_priority=PRIORITY_NORMAL + (i % 3),
                component_type=ComponentType.ANOMALY_DETECTOR,
                arrival_time_ns=0,
            )
            engine.tasks[task.task_id] = task

        obs = extract_observation(engine)
        assert obs.shape == (108,)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)
