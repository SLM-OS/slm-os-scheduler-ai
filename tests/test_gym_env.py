"""Tests for slm_sim.gym_env — Gymnasium environment wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from slm_sim.gym_env import SlmSchedulerEnv
from slm_sim.observation import TOTAL_FEATURES


class TestSlmSchedulerEnv:
    def test_construction(self):
        env = SlmSchedulerEnv(
            platform_name="jetson_orin_nano",
            scenario_name="light_single",
            episode_duration_ns=1_000_000_000,
            seed=42,
        )
        assert env.observation_space.shape == (TOTAL_FEATURES,)
        assert env.action_space.n == 42  # Jetson: 7 * 3 * 2

    def test_pi5_action_space(self):
        env = SlmSchedulerEnv(
            platform_name="raspberry_pi5",
            scenario_name="light_single",
            episode_duration_ns=1_000_000_000,
        )
        assert env.action_space.n == 24  # Pi5: 4 * 3 * 2 (no GPU)

    def test_reset_returns_valid_obs(self):
        env = SlmSchedulerEnv(
            platform_name="jetson_orin_nano",
            scenario_name="light_single",
            episode_duration_ns=1_000_000_000,
            seed=42,
        )
        obs, info = env.reset(seed=42)
        assert obs.shape == (TOTAL_FEATURES,)
        assert obs.dtype == np.float32
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)
        assert "dcr" in info

    def test_step_returns_valid(self):
        env = SlmSchedulerEnv(
            platform_name="jetson_orin_nano",
            scenario_name="light_single",
            episode_duration_ns=1_000_000_000,
            seed=42,
        )
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (TOTAL_FEATURES,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_episode_runs_to_completion(self):
        env = SlmSchedulerEnv(
            platform_name="jetson_orin_nano",
            scenario_name="light_single",
            episode_duration_ns=500_000_000,  # 0.5 seconds
            seed=42,
        )
        obs, info = env.reset(seed=42)
        steps = 0
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
            if steps > 10000:
                pytest.fail("Episode did not terminate within 10000 steps")

        assert steps > 0
        assert info["tasks_completed"] > 0

    def test_multiple_resets(self):
        env = SlmSchedulerEnv(
            platform_name="jetson_orin_nano",
            scenario_name="light_single",
            episode_duration_ns=500_000_000,
            seed=42,
        )
        for episode in range(3):
            obs, info = env.reset(seed=episode)
            assert obs.shape == (TOTAL_FEATURES,)
            for _ in range(10):
                action = env.action_space.sample()
                obs, _, terminated, _, _ = env.step(action)
                if terminated:
                    break
