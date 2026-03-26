"""Phase S6 tests: RL/PPO training pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


class TestVecEnv:
    def test_dummy_vec_env_creation(self):
        from training.rl.vec_env import make_vec_env
        env = make_vec_env(
            n_envs=2,
            platform_name="jetson_orin_nano",
            scenario_name="light_single",
            episode_duration_ns=500_000_000,
        )
        assert env.num_envs == 2
        obs = env.reset()
        assert obs.shape == (2, 108)
        env.close()

    def test_vec_env_step(self):
        from training.rl.vec_env import make_vec_env
        env = make_vec_env(
            n_envs=2,
            scenario_name="light_single",
            episode_duration_ns=500_000_000,
        )
        obs = env.reset()
        actions = np.array([env.action_space.sample() for _ in range(2)])
        obs, rewards, dones, infos = env.step(actions)
        assert obs.shape == (2, 108)
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        env.close()


class TestActorCriticNetwork:
    def test_policy_construction(self):
        from training.rl.vec_env import make_vec_env
        from training.rl.network import SchedulerActorCriticPolicy
        from stable_baselines3 import PPO

        env = make_vec_env(
            n_envs=1,
            scenario_name="light_single",
            episode_duration_ns=500_000_000,
        )
        model = PPO(
            policy=SchedulerActorCriticPolicy,
            env=env,
            n_steps=64,
            batch_size=32,
            verbose=0,
            policy_kwargs={"backbone_dims": (256, 256), "head_dim": 128},
        )
        # Check parameter count is in expected range (~164K)
        total_params = sum(p.numel() for p in model.policy.parameters())
        assert total_params > 100_000
        assert total_params < 250_000
        env.close()

    def test_forward_pass(self):
        from training.rl.network import SchedulerActorCriticNet
        net = SchedulerActorCriticNet(feature_dim=108)
        x = torch.randn(4, 108)
        pi, vf = net(x)
        assert pi.shape == (4, 128)
        assert vf.shape == (4, 128)


class TestCurriculum:
    def test_phase_progression(self):
        from training.rl.curriculum import CurriculumSchedule
        schedule = CurriculumSchedule()
        assert schedule.current_phase.name == "A"

        schedule.update(500_000)
        assert schedule.current_phase.name == "A"

        schedule.update(1_500_000)
        assert schedule.current_phase.name == "B"

        schedule.update(4_000_000)
        assert schedule.current_phase.name == "C"

        schedule.update(6_000_000)
        assert schedule.current_phase.name == "D"

    def test_scenario_sampling(self):
        from training.rl.curriculum import CurriculumSchedule
        schedule = CurriculumSchedule(seed=42)

        # Phase A should only return light scenarios
        scenarios_a = set(schedule.sample_scenario() for _ in range(50))
        assert scenarios_a.issubset({"light_single", "light_mixed"})

        # Phase D should include hard scenarios
        schedule.update(6_000_000)
        scenarios_d = set(schedule.sample_scenario() for _ in range(100))
        assert "heavy_inference" in scenarios_d or "burst_storm" in scenarios_d


class TestPPOTraining:
    def test_short_training_run(self):
        """Train PPO for a very small number of steps to verify pipeline."""
        from training.rl.train import PPOConfig, train_ppo

        config = PPOConfig(
            total_timesteps=256,
            n_envs=2,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            eval_freq=1_000_000,  # disable eval during short test
            initial_scenario="light_single",
            episode_duration_ns=500_000_000,
            use_curriculum=False,
        )
        model = train_ppo(config)
        assert model is not None

        # Verify model can predict
        obs = np.random.rand(1, 108).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (1,)

    def test_onnx_export(self):
        from training.rl.train import PPOConfig, train_ppo, export_actor_onnx

        config = PPOConfig(
            total_timesteps=128,
            n_envs=1,
            n_steps=64,
            batch_size=32,
            n_epochs=1,
            eval_freq=1_000_000,
            initial_scenario="light_single",
            episode_duration_ns=500_000_000,
            use_curriculum=False,
        )
        model = train_ppo(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy.onnx"
            export_actor_onnx(model, path)
            assert path.exists()
            assert path.stat().st_size > 0


class TestBehavioralCloning:
    def test_pretrain_reduces_loss(self):
        """Pre-train actor via behavioral cloning on small expert data."""
        from training.rl.train import PPOConfig, pretrain_behavioral_cloning
        from training.rl.network import SchedulerActorCriticPolicy
        from training.rl.vec_env import make_vec_env
        from stable_baselines3 import PPO

        # Generate small dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            splits_dir = Path(tmpdir) / "splits"

            from scripts.generate_dataset import generate
            generate(raw_dir, small=True)
            from scripts.split_dataset import split_dataset
            split_dataset(raw_dir, splits_dir)

            train_path = splits_dir / "train.parquet"

            env = make_vec_env(
                n_envs=1,
                scenario_name="light_single",
                episode_duration_ns=500_000_000,
            )
            model = PPO(
                policy=SchedulerActorCriticPolicy,
                env=env,
                n_steps=64,
                batch_size=32,
                verbose=0,
                policy_kwargs={"backbone_dims": (256, 256), "head_dim": 128},
            )

            loss = pretrain_behavioral_cloning(
                model, train_path, n_epochs=3, batch_size=64,
            )
            assert loss < 10.0  # Should converge somewhat
            env.close()
