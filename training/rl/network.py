"""Custom actor-critic network for SB3 PPO.

Shared backbone with separate actor (policy) and critic (value) heads.
See plan Section 4.3.

Architecture:
  Shared:  Input(108) -> Dense(256, ReLU) -> Dense(256, ReLU)
  Actor:   -> Dense(128, ReLU) -> Dense(N_ACTIONS)
  Critic:  -> Dense(128, ReLU) -> Dense(1)
  ~164K parameters total.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Type

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SchedulerFeaturesExtractor(BaseFeaturesExtractor):
    """Identity feature extractor — observations are already feature vectors."""

    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=observation_space.shape[0])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations


class SchedulerActorCriticNet(nn.Module):
    """Shared backbone + separate actor/critic heads.

    This is used as the mlp_extractor in SB3's ActorCriticPolicy.
    """

    def __init__(
        self,
        feature_dim: int,
        backbone_dims: Tuple[int, int] = (256, 256),
        head_dim: int = 128,
    ):
        super().__init__()

        self.latent_dim_pi = head_dim
        self.latent_dim_vf = head_dim

        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, backbone_dims[0]),
            nn.ReLU(),
            nn.Linear(backbone_dims[0], backbone_dims[1]),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.policy_net = nn.Sequential(
            nn.Linear(backbone_dims[1], head_dim),
            nn.ReLU(),
        )

        # Critic head (value function)
        self.value_net = nn.Sequential(
            nn.Linear(backbone_dims[1], head_dim),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_out = self.shared(features)
        return self.policy_net(shared_out), self.value_net(shared_out)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(self.shared(features))

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(self.shared(features))


class SchedulerActorCriticPolicy(ActorCriticPolicy):
    """Custom ActorCriticPolicy using the SchedulerActorCriticNet."""

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        backbone_dims: Tuple[int, int] = (256, 256),
        head_dim: int = 128,
        *args,
        **kwargs,
    ):
        self.backbone_dims = backbone_dims
        self.head_dim = head_dim
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=SchedulerFeaturesExtractor,
            features_extractor_kwargs={},
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SchedulerActorCriticNet(
            feature_dim=self.features_dim,
            backbone_dims=self.backbone_dims,
            head_dim=self.head_dim,
        )
