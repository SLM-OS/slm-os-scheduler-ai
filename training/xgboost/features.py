"""Feature engineering for XGBoost classifiers.

Adds derived features beyond the 108-dim state vector to help XGBoost
learn feature interactions. See plan Section 7.2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from slm_sim.observation import (
    FEATURES_PER_CORE,
    FEATURES_PER_TASK,
    GLOBAL_FEATURES,
    MAX_CORES,
    MAX_PENDING_TASKS,
    TOTAL_FEATURES,
)

# Offsets into the 108-dim state vector
_CORE_START = 0
_TASK_START = FEATURES_PER_CORE * MAX_CORES  # 36
_GLOBAL_START = _TASK_START + FEATURES_PER_TASK * MAX_PENDING_TASKS  # 100

# Per-core feature offsets (within each core's 6 features)
_CORE_UTIL = 0
_CORE_QUEUE = 1
_CORE_CACHE = 2
_CORE_TYPE = 3
_CORE_ISOLATED = 4
_CORE_TASK_PRIO = 5

# Per-task feature offsets (within each task's 8 features)
_TASK_PRIO = 0
_TASK_DEADLINE_URG = 1
_TASK_WS = 2
_TASK_MODEL = 3
_TASK_INFER = 4
_TASK_GPU = 5
_TASK_WAIT = 6
_TASK_COMP = 7

# Global feature offsets
_GLOB_READY = 0
_GLOB_MISS_RATE = 1
_GLOB_AVG_LAT = 2
_GLOB_WEIGHT_PRESS = 3
_GLOB_WS_PRESS = 4
_GLOB_GPU_DEPTH = 5
_GLOB_IMBALANCE = 6
_GLOB_TIME = 7

# Number of derived features
N_DERIVED_FEATURES = 5
FEATURE_NAMES_DERIVED = [
    "max_deadline_urgency",
    "core_util_std",
    "deadline_task_count",
    "gpu_should_use",
    "best_cache_fit_core",
]


def add_derived_features(states: np.ndarray) -> np.ndarray:
    """Add derived features to state vectors.

    Args:
        states: Array of shape (N, 108) with raw state features.

    Returns:
        Array of shape (N, 108 + 5) with derived features appended.
    """
    n = states.shape[0]
    derived = np.zeros((n, N_DERIVED_FEATURES), dtype=np.float32)

    for i in range(n):
        s = states[i]

        # 1. max_deadline_urgency: max urgency across top-K pending tasks
        max_urg = 0.0
        for t in range(MAX_PENDING_TASKS):
            off = _TASK_START + t * FEATURES_PER_TASK
            urg = s[off + _TASK_DEADLINE_URG]
            if urg > max_urg:
                max_urg = urg
        derived[i, 0] = max_urg

        # 2. core_util_std: std dev of core utilizations (normalized)
        utils = []
        for c in range(MAX_CORES):
            off = _CORE_START + c * FEATURES_PER_CORE
            u = s[off + _CORE_UTIL]
            # Only include non-padding cores (type=1 for perf, or util > 0)
            if s[off + _CORE_TYPE] > 0 or u > 0 or c < 4:
                utils.append(u)
        if len(utils) > 1:
            derived[i, 1] = float(np.std(utils))
        else:
            derived[i, 1] = 0.0

        # 3. deadline_task_count: number of pending tasks with deadline urgency > 0
        count = 0
        for t in range(MAX_PENDING_TASKS):
            off = _TASK_START + t * FEATURES_PER_TASK
            if s[off + _TASK_DEADLINE_URG] > 0:
                count += 1
        derived[i, 2] = count / MAX_PENDING_TASKS  # normalize to [0, 1]

        # 4. gpu_should_use: 1 if GPU available AND most urgent task is GPU-eligible
        #    AND GPU queue depth < 0.25 (normalized, means < 2 items)
        gpu_depth = s[_GLOBAL_START + _GLOB_GPU_DEPTH]
        top_task_gpu = s[_TASK_START + _TASK_GPU]  # top-priority task
        if top_task_gpu > 0.5 and gpu_depth < 0.25:
            derived[i, 3] = 1.0
        else:
            derived[i, 3] = 0.0

        # 5. best_cache_fit_core: normalized ID of the core where the top task's
        #    working set fits best (lowest cache pressure among non-isolated cores)
        top_ws = s[_TASK_START + _TASK_WS]
        best_core = 0
        best_pressure = float("inf")
        for c in range(MAX_CORES):
            off = _CORE_START + c * FEATURES_PER_CORE
            if s[off + _CORE_ISOLATED] > 0.5:
                continue
            pressure = s[off + _CORE_CACHE]
            if pressure < best_pressure:
                best_pressure = pressure
                best_core = c
        derived[i, 4] = best_core / max(MAX_CORES - 1, 1)

    return np.hstack([states, derived])


def get_feature_names() -> list[str]:
    """Get all feature names (108 base + 5 derived)."""
    names = []
    for c in range(MAX_CORES):
        for feat in ["util", "queue", "cache", "type", "isolated", "task_prio"]:
            names.append(f"core{c}_{feat}")
    for t in range(MAX_PENDING_TASKS):
        for feat in ["prio", "deadline_urg", "ws", "model", "infer", "gpu", "wait", "comp"]:
            names.append(f"task{t}_{feat}")
    names.extend(["glob_ready", "glob_miss_rate", "glob_avg_lat",
                   "glob_weight_press", "glob_ws_press", "glob_gpu_depth",
                   "glob_imbalance", "glob_time"])
    names.extend(FEATURE_NAMES_DERIVED)
    return names


def load_features_and_labels(
    parquet_path: Path | str,
    experts: set[str] | None = None,
    max_rows: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load features and decomposed action labels from a Parquet file.

    Streams row groups one at a time to avoid holding the full PyArrow
    table alongside the numpy arrays.

    Args:
        parquet_path: Path to Parquet file.
        experts: Expert names to include. None = all non-random.
        max_rows: If set, randomly subsample to at most this many rows.
        seed: Random seed for subsampling.

    Returns:
        Tuple of (X, y_core, y_priority, y_preempt) where:
            X: (N, 113) feature array with derived features
            y_core: (N,) core assignment labels
            y_priority: (N,) priority adjustment labels
            y_preempt: (N,) preempt decision labels
    """
    parquet_path = Path(parquet_path)
    if experts is None:
        experts = {"slm_os_hybrid", "edf", "weighted_multi_objective"}

    needed_cols = (
        [f"state_{j:03d}" for j in range(TOTAL_FEATURES)]
        + ["action_core", "action_priority", "action_preempt", "expert_policy"]
    )
    pf = pq.ParquetFile(parquet_path)

    # Pass 1: count expert rows
    total_expert_rows = 0
    for i in range(pf.metadata.num_row_groups):
        rg = pf.read_row_group(i, columns=["expert_policy"])
        total_expert_rows += sum(1 for e in rg.column("expert_policy").to_pylist()
                                 if e in experts)
        del rg

    if total_expert_rows == 0:
        empty = np.zeros((0, TOTAL_FEATURES + N_DERIVED_FEATURES), dtype=np.float32)
        return empty, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    if max_rows and total_expert_rows > max_rows:
        n = max_rows
        sample_ratio = max_rows / total_expert_rows
    else:
        n = total_expert_rows
        sample_ratio = None

    rng = np.random.default_rng(seed)

    # Allocate output arrays
    states = np.zeros((n, TOTAL_FEATURES), dtype=np.float32)
    y_core = np.zeros(n, dtype=np.int32)
    y_priority = np.zeros(n, dtype=np.int32)
    y_preempt = np.zeros(n, dtype=np.int32)
    write_pos = 0

    # Pass 2: stream row groups
    for i in range(pf.metadata.num_row_groups):
        rg = pf.read_row_group(i, columns=needed_cols)
        expert_col = rg.column("expert_policy").to_pylist()
        mask = np.array([e in experts for e in expert_col])
        if mask.sum() == 0:
            del rg
            continue

        rg = rg.filter(mask)
        rg_n = len(rg)

        if sample_ratio is not None:
            keep = int(round(rg_n * sample_ratio))
            if keep == 0:
                del rg
                continue
            keep = min(keep, n - write_pos)
            idx = rng.choice(rg_n, size=keep, replace=False)
            idx.sort()
            for j in range(TOTAL_FEATURES):
                states[write_pos:write_pos + keep, j] = (
                    rg.column(f"state_{j:03d}").to_numpy()[idx]
                )
            y_core[write_pos:write_pos + keep] = rg.column("action_core").to_numpy().astype(np.int32)[idx]
            y_priority[write_pos:write_pos + keep] = rg.column("action_priority").to_numpy().astype(np.int32)[idx]
            y_preempt[write_pos:write_pos + keep] = rg.column("action_preempt").to_numpy().astype(np.int32)[idx]
            write_pos += keep
        else:
            chunk = min(rg_n, n - write_pos)
            for j in range(TOTAL_FEATURES):
                states[write_pos:write_pos + chunk, j] = (
                    rg.column(f"state_{j:03d}").to_numpy()[:chunk]
                )
            y_core[write_pos:write_pos + chunk] = rg.column("action_core").to_numpy().astype(np.int32)[:chunk]
            y_priority[write_pos:write_pos + chunk] = rg.column("action_priority").to_numpy().astype(np.int32)[:chunk]
            y_preempt[write_pos:write_pos + chunk] = rg.column("action_preempt").to_numpy().astype(np.int32)[:chunk]
            write_pos += chunk

        del rg
        if write_pos >= n:
            break

    # Trim if needed
    if write_pos < n:
        states = states[:write_pos]
        y_core = y_core[:write_pos]
        y_priority = y_priority[:write_pos]
        y_preempt = y_preempt[:write_pos]

    # Add derived features
    X = add_derived_features(states)
    del states

    return X, y_core, y_priority, y_preempt
