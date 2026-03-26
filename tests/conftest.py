"""Shared test fixtures for the SLM-OS scheduler simulator."""

from __future__ import annotations

import numpy as np
import pytest

from slm_sim.models import (
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_NORMAL,
    ComponentType,
    CoreState,
    CoreType,
    SimTask,
    TaskState,
)
from slm_sim.platforms import (
    PlatformProfile,
    make_big_little,
    make_jetson_orin_nano,
    make_raspberry_pi5,
)


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random generator for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def jetson_platform() -> PlatformProfile:
    """Jetson Orin Nano platform profile."""
    return make_jetson_orin_nano()


@pytest.fixture
def pi5_platform() -> PlatformProfile:
    """Raspberry Pi 5 platform profile."""
    return make_raspberry_pi5()


@pytest.fixture
def biglittle_platform() -> PlatformProfile:
    """big.LITTLE platform profile."""
    return make_big_little()


@pytest.fixture
def sample_task() -> SimTask:
    """A sample anomaly detection task."""
    return SimTask(
        task_id=1,
        name="anomaly_det_0",
        state=TaskState.READY,
        priority=PRIORITY_NORMAL,
        effective_priority=PRIORITY_NORMAL,
        model_handle=1,
        model_size_mb=12.0,
        working_set_mb=4.0,
        ops_per_inference=1.2,
        inference_duration_ns=1_000_000,  # 1 ms
        can_use_gpu=False,
        deadline_ns=5_000_000,  # 5 ms
        arrival_time_ns=0,
        remaining_work_ns=1_000_000,
        component_type=ComponentType.ANOMALY_DETECTOR,
    )


@pytest.fixture
def sample_deadline_task() -> SimTask:
    """A sample predictive maintenance task with tight deadline."""
    return SimTask(
        task_id=2,
        name="pred_maint_0",
        state=TaskState.READY,
        priority=PRIORITY_HIGH,
        effective_priority=PRIORITY_HIGH,
        model_handle=2,
        model_size_mb=45.0,
        working_set_mb=16.0,
        ops_per_inference=4.8,
        inference_duration_ns=25_000_000,  # 25 ms
        can_use_gpu=True,
        deadline_ns=50_000_000,  # 50 ms
        arrival_time_ns=0,
        remaining_work_ns=25_000_000,
        component_type=ComponentType.PRED_MAINT,
    )
