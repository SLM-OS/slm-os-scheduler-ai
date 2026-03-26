"""DAgger (Dataset Aggregation) loop for MLP refinement.

Runs the trained MLP in the simulator, has the expert correct its
decisions, and adds the corrected data back to the training set.
3 iterations of 1000 episodes each. See plan Section 7.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from slm_sim.actions import SchedulingAction, decode_action, encode_action
from slm_sim.engine import SimulatorEngine
from slm_sim.experts import ExpertPolicy
from slm_sim.experts.hybrid import HybridExpertPolicy
from slm_sim.logging import transitions_to_table
from slm_sim.observation import TOTAL_FEATURES, extract_observation
from slm_sim.platforms import PlatformProfile, get_platform
from slm_sim.workloads.scenarios import ScenarioComposer
from training.mlp.model import SchedulerMLP


class MLPAgent:
    """Wraps a trained MLP as a scheduling agent for the simulator."""

    def __init__(self, model: SchedulerMLP, num_cores: int, gpu_available: bool):
        self.model = model
        self.num_cores = num_cores
        self.gpu_available = gpu_available
        self.model.eval()

    def decide(self, engine: SimulatorEngine, task_id: int) -> SchedulingAction:
        obs = extract_observation(engine)
        state = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(state)
            action_idx = logits.argmax(dim=-1).item()
        return decode_action(action_idx, self.num_cores, self.gpu_available)

    @property
    def name(self) -> str:
        return "mlp_agent"


def run_dagger(
    model: SchedulerMLP,
    expert: ExpertPolicy,
    platform_name: str = "jetson_orin_nano",
    scenario_name: str = "medium_mixed",
    n_iterations: int = 3,
    episodes_per_iteration: int = 1000,
    episode_duration_ns: int = 2_000_000_000,
    base_seed: int = 10000,
    output_dir: Optional[Path] = None,
) -> list[pa.Table]:
    """Run the DAgger loop.

    At each iteration:
    1. Run the MLP in the simulator, collecting (state, mlp_action) pairs.
    2. For each state, also ask the expert what it would do.
    3. The expert's action becomes the label (corrected data).
    4. New corrected data is mixed with original training data.

    Args:
        model: The trained MLP to refine.
        expert: Expert policy for correction.
        platform_name: Platform to run on.
        scenario_name: Scenario to use.
        n_iterations: Number of DAgger rounds (default 3).
        episodes_per_iteration: Episodes per round (default 1000).
        episode_duration_ns: Duration of each episode.
        base_seed: Starting seed.
        output_dir: Where to save DAgger data.

    Returns:
        List of PyArrow Tables containing the corrected data from each round.
    """
    platform = get_platform(platform_name)
    num_cores = len(platform.cores)
    gpu_available = platform.gpu.available
    mlp_agent = MLPAgent(model, num_cores, gpu_available)

    all_tables = []

    for iteration in range(n_iterations):
        corrected_transitions = []

        for ep in range(episodes_per_iteration):
            seed = base_seed + iteration * episodes_per_iteration + ep
            engine = SimulatorEngine(
                platform=platform,
                episode_duration_ns=episode_duration_ns,
                seed=seed,
            )
            engine.reset()
            rng = np.random.default_rng(seed)
            workload = ScenarioComposer(scenario_name, rng)
            workload.seed_all_events(engine, 0, episode_duration_ns)

            # Run episode: MLP makes decisions, expert provides corrections
            while engine.event_queue:
                event = engine.pop_event()
                if event.timestamp_ns > engine.episode_duration_ns:
                    break

                old_clock = engine.clock_ns
                engine.clock_ns = event.timestamp_ns
                engine._update_core_busy_time(old_clock, event.timestamp_ns)

                needs_decision = engine._process_event(event)
                if not needs_decision:
                    continue

                ready_task = engine._pick_highest_priority_ready_task()
                if ready_task is None:
                    continue

                # MLP's observation
                obs = extract_observation(engine)

                # Expert's correction
                expert_action = expert.decide(engine, ready_task.task_id)
                expert_idx = encode_action(expert_action, num_cores, gpu_available)

                corrected_transitions.append({
                    "state": obs,
                    "action": expert_idx,
                    "next_state": obs,  # placeholder
                    "reward": 0.0,
                    "done": False,
                    "sim_time_ns": engine.clock_ns,
                })

                # Apply the MLP's action (not the expert's — DAgger uses
                # the learner's policy for exploration)
                from slm_sim.actions import apply_action
                mlp_action = mlp_agent.decide(engine, ready_task.task_id)
                apply_action(engine, ready_task.task_id, mlp_action)
                engine._schedule_completion_for_task(ready_task.task_id)

        # Convert to table
        if corrected_transitions:
            table = transitions_to_table(
                corrected_transitions,
                expert_policy=f"dagger_round_{iteration}",
                scenario=scenario_name,
                platform=platform_name,
                episode_id=iteration,
                num_cores=num_cores,
                gpu_available=gpu_available,
            )
            all_tables.append(table)

            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                path = output_dir / f"dagger_round_{iteration}.parquet"
                pq.write_table(table, path, compression="snappy")

    return all_tables
