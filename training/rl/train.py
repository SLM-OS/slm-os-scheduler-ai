"""PPO training loop with curriculum learning.

Uses Stable Baselines 3 PPO with custom actor-critic network,
behavioral cloning pre-training, and curriculum schedule.
See plan Sections 4.3 and 7.3.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

from training.rl.curriculum import CurriculumCallback, CurriculumSchedule
from training.rl.network import SchedulerActorCriticPolicy
from training.rl.vec_env import make_vec_env


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ProgressCallback(BaseCallback):
    """Prints timestamped progress at regular intervals."""

    def __init__(self, total_timesteps: int, log_interval: int = 100_000, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self.next_log = log_interval
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print(f"[{_ts()}] PPO training started: {self.total_timesteps:,} total timesteps",
              flush=True)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_log:
            elapsed = time.time() - self.start_time
            pct = 100 * self.num_timesteps / self.total_timesteps
            rate = self.num_timesteps / elapsed if elapsed > 0 else 0
            eta_s = (self.total_timesteps - self.num_timesteps) / rate if rate > 0 else 0
            eta_m = eta_s / 60

            print(f"[{_ts()}] PPO: {self.num_timesteps:>10,}/{self.total_timesteps:,} "
                  f"({pct:5.1f}%)  {rate:,.0f} steps/s  "
                  f"ETA {eta_m:.0f}m", flush=True)
            self.next_log += self.log_interval
        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self.start_time
        print(f"[{_ts()}] PPO training complete in {elapsed / 60:.1f} minutes",
              flush=True)


@dataclass
class PPOConfig:
    """PPO training configuration matching plan Section 4.3/7.3."""
    # PPO hyperparameters
    learning_rate: float = 3e-4
    clip_range: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    n_steps: int = 2048         # steps per rollout per env
    batch_size: int = 256       # mini-batch size
    n_epochs: int = 4           # PPO epochs per update
    max_grad_norm: float = 0.5

    # Training schedule
    total_timesteps: int = 5_000_000
    n_envs: int = 16

    # Environment
    platform_name: str = "jetson_orin_nano"
    initial_scenario: str = "light_single"
    episode_duration_ns: int = 10_000_000_000

    # Evaluation
    eval_freq: int = 10_000     # evaluate every N steps
    n_eval_episodes: int = 20

    # Architecture
    backbone_dims: tuple = (256, 256)
    head_dim: int = 128

    # Curriculum
    use_curriculum: bool = True

    # Paths
    tensorboard_log: Optional[str] = None
    save_dir: Optional[str] = None


def train_ppo(config: PPOConfig = PPOConfig()) -> PPO:
    """Train a PPO agent for scheduling.

    Args:
        config: Training configuration.

    Returns:
        Trained PPO model.
    """
    # Create training environments
    train_env = make_vec_env(
        n_envs=config.n_envs,
        platform_name=config.platform_name,
        scenario_name=config.initial_scenario,
        episode_duration_ns=config.episode_duration_ns,
        use_subproc=False,
    )

    # Create eval environment (single env, different seed range)
    eval_env = make_vec_env(
        n_envs=1,
        platform_name=config.platform_name,
        scenario_name="medium_mixed",
        episode_duration_ns=config.episode_duration_ns,
        base_seed=100_000,
        use_subproc=False,
    )

    # Create PPO model with custom policy
    model = PPO(
        policy=SchedulerActorCriticPolicy,
        env=train_env,
        learning_rate=config.learning_rate,
        clip_range=config.clip_range,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        max_grad_norm=config.max_grad_norm,
        tensorboard_log=config.tensorboard_log,
        verbose=0,
        policy_kwargs={
            "backbone_dims": config.backbone_dims,
            "head_dim": config.head_dim,
        },
    )

    # Callbacks
    callbacks = []

    # Progress logging every 100K steps
    callbacks.append(ProgressCallback(
        total_timesteps=config.total_timesteps,
        log_interval=100_000,
    ))

    if config.use_curriculum:
        schedule = CurriculumSchedule()
        callbacks.append(CurriculumCallback(schedule, verbose=1))

    if config.save_dir:
        save_path = Path(config.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save checkpoint every 500K steps
        callbacks.append(CheckpointCallback(
            save_freq=max(500_000 // config.n_envs, 1),
            save_path=str(save_path / "checkpoints"),
            name_prefix="ppo",
            verbose=0,
        ))

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(save_path),
            log_path=str(save_path / "logs"),
            eval_freq=max(config.eval_freq // config.n_envs, 1),
            n_eval_episodes=config.n_eval_episodes,
            deterministic=True,
            verbose=0,
        )
        callbacks.append(eval_callback)

    callback = CallbackList(callbacks)

    # Train
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callback,
        progress_bar=False,
    )

    train_env.close()
    eval_env.close()

    return model


def export_actor_onnx(model: PPO, path: Path | str) -> None:
    """Export the actor (policy) network to ONNX.

    Only exports the actor, not the critic (not needed at inference time).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    policy = model.policy
    policy.eval()

    obs_dim = model.observation_space.shape[0]
    dummy = torch.randn(1, obs_dim).to(policy.device)

    # Extract just the actor forward path
    class ActorWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.features_extractor = policy.features_extractor
            self.mlp_extractor = policy.mlp_extractor
            self.action_net = policy.action_net

        def forward(self, obs):
            features = self.features_extractor(obs)
            latent_pi, _ = self.mlp_extractor(features)
            return self.action_net(latent_pi)

    actor = ActorWrapper(policy)
    actor.eval()

    torch.onnx.export(
        actor,
        dummy,
        str(path),
        input_names=["state"],
        output_names=["action_logits"],
        dynamic_axes={"state": {0: "batch"}, "action_logits": {0: "batch"}},
        opset_version=17,
    )


def pretrain_behavioral_cloning(
    model: PPO,
    train_parquet: Path | str,
    n_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> float:
    """Pre-train the actor network via behavioral cloning from expert data.

    Initializes the policy with expert behavior before PPO exploration.
    Returns final training loss.
    """
    from training.mlp.dataset import SchedulerDataset

    dataset = SchedulerDataset(train_parquet)
    if len(dataset) == 0:
        return float("inf")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
    )

    policy = model.policy
    policy.train()
    device = policy.device

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    final_loss = float("inf")
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        count = 0
        for states, actions, _weights in loader:
            states = states.to(device)
            actions = actions.to(device)

            features = policy.features_extractor(states)
            latent_pi, _ = policy.mlp_extractor(features)
            logits = policy.action_net(latent_pi)

            loss = loss_fn(logits, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(states)
            count += len(states)

        final_loss = epoch_loss / max(count, 1)

    return final_loss
