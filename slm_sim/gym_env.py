"""Gymnasium environment wrapper for the SLM-OS simulator.

Wraps SimulatorEngine as a standard Gymnasium Env so it can be used
with Stable Baselines 3 and other RL frameworks.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from slm_sim.actions import (
    SchedulingAction,
    action_space_size,
    apply_action,
    decode_action,
    encode_action,
)
from slm_sim.engine import EventType, SimulatorEngine
from slm_sim.observation import TOTAL_FEATURES, extract_observation
from slm_sim.platforms import PlatformProfile, get_platform
from slm_sim.reward import RewardConfig, compute_reward
from slm_sim.workloads.scenarios import ScenarioComposer


class SlmSchedulerEnv(gym.Env):
    """Gymnasium environment for SLM-OS scheduling.

    Observation: 108-dimensional float32 vector (all features in [0, 1]).
    Action: Single discrete integer encoding (core, priority_adj, preempt).

    Each step:
      1. Decode action and apply to the highest-priority ready task
      2. Advance simulation to the next scheduling decision point
      3. Compute reward from tasks completed since last step
      4. Return new observation
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        platform_name: str = "jetson_orin_nano",
        scenario_name: str = "medium_mixed",
        episode_duration_ns: int = 10_000_000_000,
        warmup_ns: int = 1_000_000_000,
        reward_config: Optional[RewardConfig] = None,
        seed: int = 42,
    ):
        super().__init__()

        self.platform = get_platform(platform_name)
        self.scenario_name = scenario_name
        self.reward_config = reward_config or RewardConfig()
        self._episode_duration_ns = episode_duration_ns
        self._warmup_ns = warmup_ns
        self._base_seed = seed

        self.engine = SimulatorEngine(
            platform=self.platform,
            episode_duration_ns=episode_duration_ns,
            warmup_ns=warmup_ns,
            seed=seed,
        )

        n_actions = action_space_size(
            self.platform.num_cores,
            self.platform.gpu.available,
        )

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(TOTAL_FEATURES,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(n_actions)

        self._current_task_id: Optional[int] = None
        self._episode_seed: int = seed

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        if seed is not None:
            self._episode_seed = seed
        else:
            self._episode_seed += 1

        self.engine.reset(seed=self._episode_seed)
        rng = np.random.default_rng(self._episode_seed)
        self._workload = ScenarioComposer(self.scenario_name, rng)
        self._workload.seed_all_events(
            self.engine, 0, self._episode_duration_ns
        )

        # Advance to first scheduling decision
        self._advance_to_next_decision()

        obs = extract_observation(self.engine)
        info = self._make_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take a scheduling action and advance to next decision point.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Clear recently completed tasks for reward tracking
        self.engine._recently_completed = []

        # Apply action to the current pending task
        if self._current_task_id is not None:
            decoded = decode_action(
                action,
                len(self.engine.cores),
                self.engine.gpu.available,
            )
            apply_action(self.engine, self._current_task_id, decoded)
            self.engine._schedule_completion_for_task(self._current_task_id)

        # Advance to next decision point
        terminated = not self._advance_to_next_decision()
        truncated = False

        # Compute reward
        reward_result = compute_reward(
            self.engine,
            self.engine._recently_completed,
            self.reward_config,
        )

        obs = extract_observation(self.engine)
        info = self._make_info()
        info["reward_components"] = {
            "deadline": reward_result.deadline,
            "latency": reward_result.latency,
            "balance": reward_result.balance,
            "power": reward_result.power,
        }

        return obs, reward_result.total, terminated, truncated, info

    def _advance_to_next_decision(self) -> bool:
        """Process events until a scheduling decision is needed.

        Returns True if a decision point was reached, False if episode ended.
        """
        while self.engine.event_queue:
            event = self.engine.pop_event()
            if event.timestamp_ns > self.engine.episode_duration_ns:
                self._current_task_id = None
                return False

            old_clock = self.engine.clock_ns
            self.engine.clock_ns = event.timestamp_ns
            self.engine._update_core_busy_time(old_clock, event.timestamp_ns)

            needs_decision = self.engine._process_event(event)

            if needs_decision:
                ready_task = self.engine._pick_highest_priority_ready_task()
                if ready_task is not None:
                    self._current_task_id = ready_task.task_id
                    self.engine.metrics.scheduling_decisions += 1
                    return True

        self._current_task_id = None
        return False

    def _make_info(self) -> dict[str, Any]:
        """Build the info dict for Gymnasium."""
        m = self.engine.metrics
        return {
            "tasks_arrived": m.total_tasks_arrived,
            "tasks_completed": m.total_tasks_completed,
            "dcr": m.deadline_compliance_rate,
            "mean_latency_ns": m.mean_latency_ns,
            "scheduling_decisions": m.scheduling_decisions,
            "pending_task_id": self._current_task_id,
        }

    def render(self) -> None:
        """Render current state (optional, for debugging)."""
        m = self.engine.metrics
        print(
            f"t={self.engine.clock_ns/1e6:.1f}ms | "
            f"arrived={m.total_tasks_arrived} "
            f"completed={m.total_tasks_completed} "
            f"DCR={m.deadline_compliance_rate:.2%}"
        )
