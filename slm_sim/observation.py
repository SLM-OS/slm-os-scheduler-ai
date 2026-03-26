"""State vector extraction for scheduling agents.

Extracts a fixed-size 108-dimensional observation vector from the
simulator state, as specified in plan Section 2.1. All features
normalized to [0, 1].
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from slm_sim.models import PRIORITY_MAX, ComponentType, TaskState

if TYPE_CHECKING:
    from slm_sim.engine import SimulatorEngine

# Feature counts (Section 2.1)
FEATURES_PER_CORE = 6
MAX_CORES = 6  # padded to max across platforms
FEATURES_PER_TASK = 8
MAX_PENDING_TASKS = 8  # top-K by priority
GLOBAL_FEATURES = 8

TOTAL_FEATURES = (FEATURES_PER_CORE * MAX_CORES
                  + FEATURES_PER_TASK * MAX_PENDING_TASKS
                  + GLOBAL_FEATURES)  # 36 + 64 + 8 = 108

# Normalization constants
MAX_RUN_QUEUE_DEPTH = 32
MAX_CACHE_PRESSURE = 5.0
MAX_WORKING_SET_MB = 64.0
MAX_MODEL_SIZE_MB = 128.0
MAX_INFERENCE_DURATION_NS = 100_000_000  # 100 ms
MAX_WAIT_TIME_NS = 1_000_000_000  # 1 second
MAX_READY_COUNT = 64
MAX_GPU_QUEUE = 8
MAX_DEADLINE_NS = 1_000_000_000  # 1 second for urgency calc

# Number of component types for normalization
_NUM_COMPONENT_TYPES = 4  # anomaly, predmaint, security, other

# Precomputed reciprocals for fast normalization
_INV_PRIORITY_MAX = 1.0 / PRIORITY_MAX
_INV_MAX_RUN_QUEUE = 1.0 / MAX_RUN_QUEUE_DEPTH
_INV_MAX_CACHE = 1.0 / MAX_CACHE_PRESSURE
_INV_MAX_WS = 1.0 / MAX_WORKING_SET_MB
_INV_MAX_MODEL = 1.0 / MAX_MODEL_SIZE_MB
_INV_MAX_INFER = 1.0 / MAX_INFERENCE_DURATION_NS
_INV_MAX_WAIT = 1.0 / MAX_WAIT_TIME_NS
_INV_MAX_READY = 1.0 / MAX_READY_COUNT
_INV_MAX_GPU_Q = 1.0 / MAX_GPU_QUEUE
_INV_MAX_DEADLINE = 1.0 / MAX_DEADLINE_NS
_INV_NUM_COMP = 1.0 / _NUM_COMPONENT_TYPES


def _clamp01(x: float) -> float:
    """Clamp a float to [0, 1] without numpy overhead."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def extract_observation(engine: SimulatorEngine) -> np.ndarray:
    """Extract the 108-dimensional state vector from current simulator state.

    Returns:
        Float32 numpy array of shape (TOTAL_FEATURES,), all values in [0, 1].
    """
    buf = np.zeros(TOTAL_FEATURES, dtype=np.float32)
    _fill_core_features(engine, buf, 0)
    _fill_task_features(engine, buf, FEATURES_PER_CORE * MAX_CORES)
    _fill_global_features(engine, buf, FEATURES_PER_CORE * MAX_CORES + FEATURES_PER_TASK * MAX_PENDING_TASKS)
    return buf


def _fill_core_features(engine: SimulatorEngine, buf: np.ndarray, start: int) -> None:
    """Fill per-core features into buf starting at offset `start`."""
    tasks = engine.tasks
    for i, core in enumerate(engine.cores):
        if i >= MAX_CORES:
            break
        o = start + i * FEATURES_PER_CORE
        buf[o] = _clamp01(core.utilization_pct)
        buf[o + 1] = _clamp01(len(core.run_queue) * _INV_MAX_RUN_QUEUE)
        buf[o + 2] = _clamp01(core.cache_pressure * _INV_MAX_CACHE)
        buf[o + 3] = float(core.core_type)
        buf[o + 4] = 1.0 if core.isolated else 0.0
        ct = core.current_task
        if ct is not None and ct in tasks:
            buf[o + 5] = tasks[ct].effective_priority * _INV_PRIORITY_MAX
        # else: already 0.0


def _fill_task_features(engine: SimulatorEngine, buf: np.ndarray, start: int) -> None:
    """Fill features for top-K pending tasks by effective_priority."""
    clock = engine.clock_ns
    ready = [t for t in engine.tasks.values() if t.state == TaskState.READY]
    ready.sort(key=lambda t: (-t.effective_priority, t.arrival_time_ns))

    for i, task in enumerate(ready[:MAX_PENDING_TASKS]):
        o = start + i * FEATURES_PER_TASK
        buf[o] = task.effective_priority * _INV_PRIORITY_MAX

        if task.deadline_ns > 0:
            ttd = task.deadline_ns - clock
            if ttd <= 0:
                buf[o + 1] = 1.0
            else:
                buf[o + 1] = _clamp01(1.0 - ttd * _INV_MAX_DEADLINE)

        buf[o + 2] = _clamp01(task.working_set_mb * _INV_MAX_WS)
        buf[o + 3] = _clamp01(task.model_size_mb * _INV_MAX_MODEL)
        buf[o + 4] = _clamp01(task.inference_duration_ns * _INV_MAX_INFER)
        buf[o + 5] = 1.0 if task.can_use_gpu else 0.0
        buf[o + 6] = _clamp01((clock - task.arrival_time_ns) * _INV_MAX_WAIT)
        buf[o + 7] = _clamp01(task.component_type * _INV_NUM_COMP)


def _fill_global_features(engine: SimulatorEngine, buf: np.ndarray, start: int) -> None:
    """Fill global system features."""
    ready_count = sum(1 for t in engine.tasks.values() if t.state == TaskState.READY)
    buf[start] = _clamp01(ready_count * _INV_MAX_READY)

    m = engine.metrics
    if m.total_deadline_tasks > 0:
        buf[start + 1] = m.deadline_missed / m.total_deadline_tasks

    if m.total_tasks_completed > 0:
        avg_lat = m.total_latency_ns / m.total_tasks_completed
        buf[start + 2] = _clamp01(avg_lat / 10_000_000)

    buf[start + 3] = _clamp01(engine.memory.weight_pool_pressure)
    buf[start + 4] = _clamp01(engine.memory.workspace_pool_pressure)

    gpu_depth = len(engine.gpu.queue) + (1 if engine.gpu.current_job else 0)
    buf[start + 5] = _clamp01(gpu_depth * _INV_MAX_GPU_Q)

    # Load imbalance (CV) — computed with plain Python to avoid numpy overhead
    cores = engine.cores
    n = len(cores)
    if n > 0:
        total = sum(c.utilization_pct for c in cores)
        mean_u = total / n
        if mean_u > 1e-9:
            var_sum = sum((c.utilization_pct - mean_u) ** 2 for c in cores)
            std_u = math.sqrt(var_sum / n)
            buf[start + 6] = _clamp01(std_u / mean_u)

    if engine.episode_duration_ns > 0:
        buf[start + 7] = _clamp01(engine.clock_ns / engine.episode_duration_ns)
