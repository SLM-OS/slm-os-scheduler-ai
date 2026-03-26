"""Reward computation for scheduling decisions.

Implements the four-component reward function from plan Section 2.3:
R = w_d * R_deadline + w_l * R_latency + w_b * R_balance + w_p * R_power
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slm_sim.engine import SimulatorEngine

# Default reward weights (Section 2.3)
DEFAULT_WEIGHT_DEADLINE = 0.50
DEFAULT_WEIGHT_LATENCY = 0.25
DEFAULT_WEIGHT_BALANCE = 0.15
DEFAULT_WEIGHT_POWER = 0.10

# Per-component target latencies in nanoseconds (Section 2.3)
TARGET_LATENCY_NS: dict[int, int] = {
    0: 5_000_000,     # ANOMALY_DETECTOR: 5 ms
    1: 50_000_000,    # PRED_MAINT: 50 ms
    2: 20_000_000,    # SECURITY_MON: 20 ms
    3: 10_000_000,    # SYSTEM: 10 ms
    4: 10_000_000,    # OTHER: 10 ms default
}


@dataclass
class RewardComponents:
    """Individual reward components for logging and analysis."""
    deadline: float = 0.0
    latency: float = 0.0
    balance: float = 0.0
    power: float = 0.0
    total: float = 0.0


@dataclass
class RewardConfig:
    """Configurable reward weights."""
    w_deadline: float = DEFAULT_WEIGHT_DEADLINE
    w_latency: float = DEFAULT_WEIGHT_LATENCY
    w_balance: float = DEFAULT_WEIGHT_BALANCE
    w_power: float = DEFAULT_WEIGHT_POWER


def compute_reward(engine: SimulatorEngine,
                   completed_tasks: list[dict],
                   config: RewardConfig | None = None) -> RewardComponents:
    """Compute the four-component reward after a scheduling decision.

    Args:
        engine: Current simulator state.
        completed_tasks: List of dicts with keys:
            - task_id: int
            - arrival_time_ns: int
            - completion_time_ns: int
            - deadline_ns: int (0 = no deadline)
            - component_type: int (ComponentType value)
        config: Reward weight configuration.

    Returns:
        RewardComponents with individual and total reward.
    """
    if config is None:
        config = RewardConfig()

    r_deadline = _compute_deadline_reward(completed_tasks)
    r_latency = _compute_latency_reward(completed_tasks)
    r_balance = _compute_balance_reward(engine)
    r_power = _compute_power_reward(engine)

    total = (config.w_deadline * r_deadline
             + config.w_latency * r_latency
             + config.w_balance * r_balance
             + config.w_power * r_power)

    return RewardComponents(
        deadline=r_deadline,
        latency=r_latency,
        balance=r_balance,
        power=r_power,
        total=total,
    )


def _compute_deadline_reward(completed_tasks: list[dict]) -> float:
    """R_deadline: reward for deadline compliance.

    For each completed task:
      - completed before deadline: +1.0
      - completed after deadline: -2.0 * (overshoot_ms / deadline_ms), clipped to -2.0
      - no deadline: +0.1
    Averaged over completed tasks.
    """
    if not completed_tasks:
        return 0.0

    total = 0.0
    for t in completed_tasks:
        deadline_ns = t["deadline_ns"]
        if deadline_ns == 0:
            total += 0.1
            continue

        completion_ns = t["completion_time_ns"]
        arrival_ns = t["arrival_time_ns"]
        # Deadline is absolute; overshoot = completion - deadline
        overshoot_ns = completion_ns - deadline_ns
        if overshoot_ns <= 0:
            total += 1.0
        else:
            # Penalty proportional to overshoot relative to deadline window
            deadline_window_ns = deadline_ns - arrival_ns
            if deadline_window_ns <= 0:
                total += -2.0
            else:
                penalty = -2.0 * (overshoot_ns / deadline_window_ns)
                total += max(penalty, -2.0)

    return total / len(completed_tasks)


def _compute_latency_reward(completed_tasks: list[dict]) -> float:
    """R_latency: reward for inference latency.

    R_latency = 1.0 - (actual_latency / target_latency), clipped to [-1, 1].
    Averaged over completed tasks.
    """
    if not completed_tasks:
        return 0.0

    total = 0.0
    for t in completed_tasks:
        actual_ns = t["completion_time_ns"] - t["arrival_time_ns"]
        component = t["component_type"]
        target_ns = TARGET_LATENCY_NS.get(component, 10_000_000)
        if target_ns <= 0:
            total += 0.0
            continue
        reward = 1.0 - (actual_ns / target_ns)
        total += max(-1.0, min(1.0, reward))

    return total / len(completed_tasks)


def _compute_balance_reward(engine: SimulatorEngine) -> float:
    """R_balance: reward for even core utilization.

    R_balance = 1.0 - coefficient_of_variation(core_utilizations)
    CV = std_dev / mean, clipped result to [0, 1].
    """
    cores = engine.cores
    n = len(cores)
    if n == 0:
        return 1.0
    total_u = sum(c.utilization_pct for c in cores)
    mean_util = total_u / n
    if mean_util < 1e-9:
        return 1.0
    var_sum = sum((c.utilization_pct - mean_util) ** 2 for c in cores)
    std_util = math.sqrt(var_sum / n)
    cv = std_util / mean_util
    return max(0.0, min(1.0, 1.0 - cv))


def _compute_power_reward(engine: SimulatorEngine) -> float:
    """R_power: reward for power efficiency.

    R_power = 1.0 - (weighted_active_power / max_possible_power)
    where weighted_active = sum(utilization[i] * power_weight[i])
    """
    weighted_active = sum(
        c.utilization_pct * c.power_weight for c in engine.cores
    )
    max_possible = sum(c.power_weight for c in engine.cores)
    if max_possible < 1e-9:
        return 1.0
    r = 1.0 - (weighted_active / max_possible)
    return max(0.0, min(1.0, r))
