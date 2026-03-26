"""Vectorized environment for parallel PPO training.

Wraps multiple SlmSchedulerEnv instances for SB3's vectorized
environment interface. See plan Section 7.3.
"""

from __future__ import annotations

from typing import Optional

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from slm_sim.gym_env import SlmSchedulerEnv
from slm_sim.reward import RewardConfig


def make_vec_env(
    n_envs: int = 16,
    platform_name: str = "jetson_orin_nano",
    scenario_name: str = "medium_mixed",
    episode_duration_ns: int = 10_000_000_000,
    warmup_ns: int = 1_000_000_000,
    reward_config: Optional[RewardConfig] = None,
    base_seed: int = 0,
    use_subproc: bool = False,
) -> DummyVecEnv | SubprocVecEnv:
    """Create a vectorized environment for PPO training.

    Args:
        n_envs: Number of parallel environments.
        platform_name: Hardware platform profile.
        scenario_name: Workload scenario.
        episode_duration_ns: Episode duration.
        warmup_ns: Warmup period.
        reward_config: Reward weight configuration.
        base_seed: Starting seed (each env gets base_seed + i).
        use_subproc: Use SubprocVecEnv (separate processes) vs DummyVecEnv (same process).

    Returns:
        Vectorized environment compatible with SB3.
    """
    def make_env(seed: int):
        def _init():
            env = SlmSchedulerEnv(
                platform_name=platform_name,
                scenario_name=scenario_name,
                episode_duration_ns=episode_duration_ns,
                warmup_ns=warmup_ns,
                reward_config=reward_config,
                seed=seed,
            )
            return env
        return _init

    env_fns = [make_env(base_seed + i) for i in range(n_envs)]

    if use_subproc:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)
