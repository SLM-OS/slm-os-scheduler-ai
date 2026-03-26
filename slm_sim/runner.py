"""Parallel episode runner for dataset generation.

Runs episodes across combinations of (expert, scenario, platform, seed)
using multiprocessing, collects transitions, and writes to Parquet.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from slm_sim.engine import SimulatorEngine
from slm_sim.experts import ExpertPolicy
from slm_sim.experts.edf import EDFExpertPolicy
from slm_sim.experts.hybrid import HybridExpertPolicy
from slm_sim.experts.random_policy import RandomExpertPolicy
from slm_sim.experts.weighted import WeightedExpertPolicy
from slm_sim.logging import concat_tables, transitions_to_table, write_parquet
from slm_sim.platforms import get_platform
from slm_sim.workloads.scenarios import ScenarioComposer


@dataclass
class RunConfig:
    """Configuration for a batch of episodes."""
    expert_name: str
    scenario_name: str
    platform_name: str
    n_episodes: int
    episode_duration_ns: int = 10_000_000_000  # 10 seconds
    base_seed: int = 0
    output_dir: Optional[str] = None


def _make_expert(name: str, seed: int = 0) -> ExpertPolicy:
    """Instantiate an expert policy by name."""
    if name == "slm_os_hybrid":
        return HybridExpertPolicy()
    elif name == "edf":
        return EDFExpertPolicy()
    elif name == "weighted_multi_objective":
        return WeightedExpertPolicy()
    elif name == "random":
        return RandomExpertPolicy(rng=np.random.default_rng(seed))
    else:
        raise ValueError(f"Unknown expert: {name}")


def run_single_episode(args: tuple) -> dict:
    """Run a single episode and return transition data.

    Args is a tuple for multiprocessing compatibility:
        (expert_name, scenario_name, platform_name, episode_duration_ns,
         seed, episode_id)

    Returns dict with transitions and metadata.
    """
    (expert_name, scenario_name, platform_name,
     episode_duration_ns, seed, episode_id) = args

    platform = get_platform(platform_name)
    engine = SimulatorEngine(
        platform=platform,
        episode_duration_ns=episode_duration_ns,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    workload = ScenarioComposer(scenario_name, rng)
    expert = _make_expert(expert_name, seed=seed)

    metrics = engine.run_episode(expert, workload)

    return {
        "transitions": engine.transitions,
        "expert_name": expert_name,
        "scenario_name": scenario_name,
        "platform_name": platform_name,
        "episode_id": episode_id,
        "num_cores": len(platform.cores),
        "gpu_available": platform.gpu.available,
        "metrics": {
            "dcr": metrics.deadline_compliance_rate,
            "tasks_completed": metrics.total_tasks_completed,
            "tasks_arrived": metrics.total_tasks_arrived,
            "mean_latency_ns": metrics.mean_latency_ns,
            "decisions": metrics.scheduling_decisions,
        },
    }


def run_batch(
    config: RunConfig,
    n_workers: Optional[int] = None,
    progress_callback=None,
) -> list[dict]:
    """Run a batch of episodes, optionally in parallel.

    Args:
        config: Batch configuration.
        n_workers: Number of parallel workers. None = sequential.
            Set to 0 for cpu_count().
        progress_callback: Called with (completed, total) after each episode.

    Returns:
        List of result dicts from run_single_episode.
    """
    args_list = [
        (config.expert_name, config.scenario_name, config.platform_name,
         config.episode_duration_ns, config.base_seed + i, i)
        for i in range(config.n_episodes)
    ]

    results = []
    if n_workers is None or n_workers == 1:
        # Sequential
        for i, args in enumerate(args_list):
            result = run_single_episode(args)
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, config.n_episodes)
    else:
        workers = n_workers if n_workers > 0 else mp.cpu_count()
        with mp.Pool(workers) as pool:
            for i, result in enumerate(pool.imap_unordered(run_single_episode, args_list)):
                results.append(result)
                if progress_callback:
                    progress_callback(i + 1, config.n_episodes)

    return results


def batch_to_parquet(
    results: list[dict],
    output_path: Path | str,
) -> int:
    """Convert batch results to a single Parquet file.

    Args:
        results: List of dicts from run_batch.
        output_path: Where to write the Parquet file.

    Returns:
        Total number of transition rows written.
    """
    tables = []
    for r in results:
        if not r["transitions"]:
            continue
        table = transitions_to_table(
            transitions=r["transitions"],
            expert_policy=r["expert_name"],
            scenario=r["scenario_name"],
            platform=r["platform_name"],
            episode_id=r["episode_id"],
            num_cores=r["num_cores"],
            gpu_available=r["gpu_available"],
        )
        tables.append(table)

    if not tables:
        return 0

    combined = concat_tables(tables)
    write_parquet(combined, output_path)
    return len(combined)
