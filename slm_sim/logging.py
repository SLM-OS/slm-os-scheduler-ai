"""Parquet logging for transition tuples.

Converts engine.transitions (list of dicts) into columnar Parquet format
matching the schema from plan Section 6.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from slm_sim.actions import decode_action
from slm_sim.observation import TOTAL_FEATURES


def transitions_to_table(
    transitions: list[dict],
    expert_policy: str,
    scenario: str,
    platform: str,
    episode_id: int,
    num_cores: int,
    gpu_available: bool,
) -> pa.Table:
    """Convert a list of transition dicts to a PyArrow Table.

    Args:
        transitions: List of dicts from engine.transitions with keys:
            state, action, reward, reward_deadline, reward_latency,
            reward_balance, reward_power, next_state, done, sim_time_ns.
        expert_policy: Name of the expert that generated this data.
        scenario: Scenario name.
        platform: Platform name.
        episode_id: Unique episode identifier.
        num_cores: Number of CPU cores (for action decoding).
        gpu_available: Whether GPU is available (for action decoding).

    Returns:
        PyArrow Table with the dataset schema.
    """
    n = len(transitions)
    if n == 0:
        return _empty_table()

    # Pre-allocate arrays
    states = np.zeros((n, TOTAL_FEATURES), dtype=np.float32)
    next_states = np.zeros((n, TOTAL_FEATURES), dtype=np.float32)
    actions = np.zeros(n, dtype=np.int32)
    action_cores = np.zeros(n, dtype=np.int32)
    action_priorities = np.zeros(n, dtype=np.int32)
    action_preempts = np.zeros(n, dtype=np.int32)
    rewards = np.zeros(n, dtype=np.float32)
    reward_deadlines = np.zeros(n, dtype=np.float32)
    reward_latencies = np.zeros(n, dtype=np.float32)
    reward_balances = np.zeros(n, dtype=np.float32)
    reward_powers = np.zeros(n, dtype=np.float32)
    dones = np.zeros(n, dtype=bool)
    sim_times = np.zeros(n, dtype=np.int64)
    steps = np.arange(n, dtype=np.int32)

    for i, t in enumerate(transitions):
        states[i] = t["state"]
        next_states[i] = t["next_state"]
        actions[i] = t["action"]
        rewards[i] = t["reward"]
        reward_deadlines[i] = t.get("reward_deadline", 0.0)
        reward_latencies[i] = t.get("reward_latency", 0.0)
        reward_balances[i] = t.get("reward_balance", 0.0)
        reward_powers[i] = t.get("reward_power", 0.0)
        dones[i] = t["done"]
        sim_times[i] = t.get("sim_time_ns", 0)

        # Decompose action
        decoded = decode_action(t["action"], num_cores, gpu_available)
        action_cores[i] = decoded.core_assignment
        action_priorities[i] = decoded.priority_adj
        action_preempts[i] = 1 if decoded.preempt else 0

    # Build table
    # State and next_state stored as fixed-size lists
    state_arrays = [pa.array(states[:, j]) for j in range(TOTAL_FEATURES)]
    next_state_arrays = [pa.array(next_states[:, j]) for j in range(TOTAL_FEATURES)]

    columns = {
        "action": pa.array(actions),
        "action_core": pa.array(action_cores),
        "action_priority": pa.array(action_priorities),
        "action_preempt": pa.array(action_preempts),
        "reward": pa.array(rewards),
        "reward_deadline": pa.array(reward_deadlines),
        "reward_latency": pa.array(reward_latencies),
        "reward_balance": pa.array(reward_balances),
        "reward_power": pa.array(reward_powers),
        "done": pa.array(dones),
        "expert_policy": pa.array([expert_policy] * n),
        "scenario": pa.array([scenario] * n),
        "platform": pa.array([platform] * n),
        "episode_id": pa.array(np.full(n, episode_id, dtype=np.int64)),
        "step_in_episode": pa.array(steps),
        "sim_time_ns": pa.array(sim_times),
    }

    # Add state features as individual columns (easier to work with than nested)
    for j in range(TOTAL_FEATURES):
        columns[f"state_{j:03d}"] = state_arrays[j]
        columns[f"next_state_{j:03d}"] = next_state_arrays[j]

    return pa.table(columns)


def write_parquet(
    table: pa.Table,
    path: Path | str,
    compression: str = "snappy",
) -> None:
    """Write a PyArrow Table to a Parquet file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression=compression)


def read_parquet(path: Path | str) -> pa.Table:
    """Read a Parquet file into a PyArrow Table."""
    return pq.read_table(Path(path))


def concat_tables(tables: list[pa.Table]) -> pa.Table:
    """Concatenate multiple tables with the same schema."""
    if not tables:
        return _empty_table()
    return pa.concat_tables(tables, promote_options="default")


def _empty_table() -> pa.Table:
    """Return an empty table with the correct schema."""
    columns = {
        "action": pa.array([], type=pa.int32()),
        "action_core": pa.array([], type=pa.int32()),
        "action_priority": pa.array([], type=pa.int32()),
        "action_preempt": pa.array([], type=pa.int32()),
        "reward": pa.array([], type=pa.float32()),
        "reward_deadline": pa.array([], type=pa.float32()),
        "reward_latency": pa.array([], type=pa.float32()),
        "reward_balance": pa.array([], type=pa.float32()),
        "reward_power": pa.array([], type=pa.float32()),
        "done": pa.array([], type=pa.bool_()),
        "expert_policy": pa.array([], type=pa.string()),
        "scenario": pa.array([], type=pa.string()),
        "platform": pa.array([], type=pa.string()),
        "episode_id": pa.array([], type=pa.int64()),
        "step_in_episode": pa.array([], type=pa.int32()),
        "sim_time_ns": pa.array([], type=pa.int64()),
    }
    for j in range(TOTAL_FEATURES):
        columns[f"state_{j:03d}"] = pa.array([], type=pa.float32())
        columns[f"next_state_{j:03d}"] = pa.array([], type=pa.float32())
    return pa.table(columns)
