"""Tests for slm_sim.reward — reward computation."""

from __future__ import annotations

import numpy as np
import pytest

from slm_sim.models import ComponentType, CoreState, CoreType
from slm_sim.platforms import make_jetson_orin_nano
from slm_sim.engine import SimulatorEngine
from slm_sim.reward import (
    DEFAULT_WEIGHT_BALANCE,
    DEFAULT_WEIGHT_DEADLINE,
    DEFAULT_WEIGHT_LATENCY,
    DEFAULT_WEIGHT_POWER,
    TARGET_LATENCY_NS,
    RewardConfig,
    _compute_balance_reward,
    _compute_deadline_reward,
    _compute_latency_reward,
    _compute_power_reward,
    compute_reward,
)


@pytest.fixture
def engine():
    platform = make_jetson_orin_nano()
    eng = SimulatorEngine(platform=platform)
    eng.reset()
    return eng


class TestRewardConfig:
    def test_default_weights(self):
        config = RewardConfig()
        assert config.w_deadline == DEFAULT_WEIGHT_DEADLINE
        assert config.w_latency == DEFAULT_WEIGHT_LATENCY
        assert config.w_balance == DEFAULT_WEIGHT_BALANCE
        assert config.w_power == DEFAULT_WEIGHT_POWER

    def test_weights_sum_to_one(self):
        config = RewardConfig()
        total = config.w_deadline + config.w_latency + config.w_balance + config.w_power
        assert abs(total - 1.0) < 1e-9

    def test_target_latencies(self):
        assert TARGET_LATENCY_NS[0] == 5_000_000   # anomaly: 5 ms
        assert TARGET_LATENCY_NS[1] == 50_000_000  # pred_maint: 50 ms
        assert TARGET_LATENCY_NS[2] == 20_000_000  # security: 20 ms
        assert TARGET_LATENCY_NS[3] == 10_000_000  # system: 10 ms


class TestDeadlineReward:
    def test_no_completed_tasks(self):
        assert _compute_deadline_reward([]) == 0.0

    def test_met_deadline(self):
        tasks = [{"deadline_ns": 100_000, "arrival_time_ns": 0,
                  "completion_time_ns": 50_000, "component_type": 0}]
        assert _compute_deadline_reward(tasks) == 1.0

    def test_missed_deadline(self):
        # Deadline at 100k, completed at 150k, arrival at 0
        # overshoot = 50k, window = 100k, penalty = -2 * 50k/100k = -1.0
        tasks = [{"deadline_ns": 100_000, "arrival_time_ns": 0,
                  "completion_time_ns": 150_000, "component_type": 0}]
        assert _compute_deadline_reward(tasks) == pytest.approx(-1.0)

    def test_badly_missed_deadline_clipped(self):
        # Overshoot >> deadline window -> clipped to -2.0
        tasks = [{"deadline_ns": 100_000, "arrival_time_ns": 0,
                  "completion_time_ns": 500_000, "component_type": 0}]
        assert _compute_deadline_reward(tasks) == pytest.approx(-2.0)

    def test_no_deadline_gives_small_bonus(self):
        tasks = [{"deadline_ns": 0, "arrival_time_ns": 0,
                  "completion_time_ns": 50_000, "component_type": 3}]
        assert _compute_deadline_reward(tasks) == pytest.approx(0.1)

    def test_mixed_tasks_averaged(self):
        tasks = [
            {"deadline_ns": 100_000, "arrival_time_ns": 0,
             "completion_time_ns": 50_000, "component_type": 0},  # +1.0
            {"deadline_ns": 0, "arrival_time_ns": 0,
             "completion_time_ns": 50_000, "component_type": 3},  # +0.1
        ]
        assert _compute_deadline_reward(tasks) == pytest.approx(0.55)


class TestLatencyReward:
    def test_no_completed_tasks(self):
        assert _compute_latency_reward([]) == 0.0

    def test_perfect_latency(self):
        # Actual = 0 -> reward = 1.0
        tasks = [{"arrival_time_ns": 0, "completion_time_ns": 0,
                  "component_type": 0}]
        assert _compute_latency_reward(tasks) == pytest.approx(1.0)

    def test_at_target_latency(self):
        # Actual = target -> reward = 0.0
        tasks = [{"arrival_time_ns": 0,
                  "completion_time_ns": 5_000_000,  # 5ms = anomaly target
                  "component_type": 0}]
        assert _compute_latency_reward(tasks) == pytest.approx(0.0)

    def test_double_target_latency(self):
        # Actual = 2x target -> reward = -1.0 (clipped)
        tasks = [{"arrival_time_ns": 0,
                  "completion_time_ns": 10_000_000,
                  "component_type": 0}]
        assert _compute_latency_reward(tasks) == pytest.approx(-1.0)

    def test_way_over_target_clipped(self):
        # 10x target -> 1 - 10 = -9, clipped to -1.0
        tasks = [{"arrival_time_ns": 0,
                  "completion_time_ns": 50_000_000,
                  "component_type": 0}]
        assert _compute_latency_reward(tasks) == pytest.approx(-1.0)


class TestBalanceReward:
    def test_all_idle(self, engine):
        # All cores at 0% -> perfectly balanced
        assert _compute_balance_reward(engine) == pytest.approx(1.0)

    def test_all_equal(self, engine):
        for c in engine.cores:
            c.utilization_pct = 0.5
        assert _compute_balance_reward(engine) == pytest.approx(1.0)

    def test_imbalanced(self, engine):
        # One core at 100%, rest at 0%
        engine.cores[0].utilization_pct = 1.0
        reward = _compute_balance_reward(engine)
        assert reward < 1.0
        assert reward >= 0.0


class TestPowerReward:
    def test_all_idle(self, engine):
        # No utilization -> reward = 1.0
        assert _compute_power_reward(engine) == pytest.approx(1.0)

    def test_all_maxed(self, engine):
        for c in engine.cores:
            c.utilization_pct = 1.0
        assert _compute_power_reward(engine) == pytest.approx(0.0)

    def test_half_utilized(self, engine):
        for c in engine.cores:
            c.utilization_pct = 0.5
        assert _compute_power_reward(engine) == pytest.approx(0.5)


class TestComputeReward:
    def test_full_reward_computation(self, engine):
        tasks = [{"deadline_ns": 100_000, "arrival_time_ns": 0,
                  "completion_time_ns": 50_000, "component_type": 0}]
        result = compute_reward(engine, tasks)
        assert result.deadline == pytest.approx(1.0)
        assert result.total > 0

    def test_empty_tasks_all_idle(self, engine):
        result = compute_reward(engine, [])
        # deadline=0, latency=0, balance=1.0, power=1.0
        assert result.deadline == 0.0
        assert result.latency == 0.0
        assert result.balance == pytest.approx(1.0)
        assert result.power == pytest.approx(1.0)
