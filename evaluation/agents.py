"""Agent wrappers for trained models.

Wraps MLP, XGBoost, and PPO models to implement the ExpertPolicy
interface so they can be used by the evaluation framework.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from slm_sim.actions import SchedulingAction, decode_action
from slm_sim.experts import ExpertPolicy
from slm_sim.observation import TOTAL_FEATURES, extract_observation

if TYPE_CHECKING:
    from slm_sim.engine import SimulatorEngine


class MLPAgent(ExpertPolicy):
    """Wraps a trained MLP as a scheduling agent."""

    def __init__(self, model_path: Path | str, num_cores: int, gpu_available: bool):
        from training.mlp.model import SchedulerMLP

        state_dict = torch.load(
            str(model_path), weights_only=True, map_location="cpu"
        )
        n_actions = state_dict["network.11.bias"].shape[0]

        self.model = SchedulerMLP(n_actions=n_actions)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.num_cores = num_cores
        self.gpu_available = gpu_available

    @property
    def name(self) -> str:
        return "mlp"

    def decide(self, engine: SimulatorEngine, task_id: int) -> SchedulingAction:
        obs = extract_observation(engine)
        state = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(state)
            action_idx = logits.argmax(dim=-1).item()
        return decode_action(action_idx, self.num_cores, self.gpu_available)


class PPOAgent(ExpertPolicy):
    """Wraps a trained PPO actor as a scheduling agent."""

    def __init__(self, model_path: Path | str, num_cores: int, gpu_available: bool):
        from stable_baselines3 import PPO

        self.ppo = PPO.load(str(model_path), device="cpu")
        self.ppo.policy.eval()
        self.num_cores = num_cores
        self.gpu_available = gpu_available

    @property
    def name(self) -> str:
        return "ppo"

    def decide(self, engine: SimulatorEngine, task_id: int) -> SchedulingAction:
        obs = extract_observation(engine)
        state = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            features = self.ppo.policy.features_extractor(state)
            latent_pi, _ = self.ppo.policy.mlp_extractor(features)
            logits = self.ppo.policy.action_net(latent_pi)
            action_idx = logits.argmax(dim=-1).item()
        return decode_action(action_idx, self.num_cores, self.gpu_available)


class XGBoostAgent(ExpertPolicy):
    """Wraps a trained XGBoost triple classifier as a scheduling agent."""

    def __init__(self, model_dir: Path | str, num_cores: int, gpu_available: bool):
        from training.xgboost.features import add_derived_features
        from training.xgboost.train import TripleClassifier

        self.triple = TripleClassifier.load(Path(model_dir))
        self.add_derived_features = add_derived_features
        self.num_cores = num_cores
        self.gpu_available = gpu_available

    @property
    def name(self) -> str:
        return "xgboost"

    def decide(self, engine: SimulatorEngine, task_id: int) -> SchedulingAction:
        obs = extract_observation(engine)
        # XGBoost expects (1, 113) with derived features
        X = self.add_derived_features(obs.reshape(1, -1))
        core, priority, preempt = self.triple.predict(X)

        return SchedulingAction(
            core_assignment=int(core[0]),
            priority_adj=int(priority[0]),
            preempt=bool(preempt[0]),
        )
