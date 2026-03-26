"""Run all models on identical test episodes for head-to-head comparison.

Evaluates expert policies (and optionally trained models) on held-out
test episodes with identical random seeds. See plan Section 8.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from evaluation.metrics import EvalMetrics, compute_eval_metrics, metrics_to_dict
from slm_sim.engine import SimulatorEngine
from slm_sim.experts import ExpertPolicy
from slm_sim.experts.edf import EDFExpertPolicy
from slm_sim.experts.hybrid import HybridExpertPolicy
from slm_sim.experts.random_policy import RandomExpertPolicy
from slm_sim.experts.weighted import WeightedExpertPolicy
from slm_sim.platforms import get_platform
from slm_sim.workloads.scenarios import SCENARIOS, ScenarioComposer


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    n_episodes: int = 200
    episode_duration_ns: int = 10_000_000_000
    platform_name: str = "jetson_orin_nano"
    scenarios: list[str] | None = None  # None = all scenarios
    base_seed: int = 500_000


def get_default_experts() -> dict[str, ExpertPolicy]:
    """Get all expert policies for evaluation."""
    return {
        "hybrid": HybridExpertPolicy(),
        "edf": EDFExpertPolicy(),
        "weighted": WeightedExpertPolicy(),
        "random": RandomExpertPolicy(rng=np.random.default_rng(42)),
    }


def evaluate_agent(
    agent: ExpertPolicy,
    agent_name: str,
    config: EvalConfig,
) -> pd.DataFrame:
    """Evaluate a single agent across all configured scenarios.

    Args:
        agent: The scheduling agent to evaluate.
        agent_name: Name for the results.
        config: Evaluation configuration.

    Returns:
        DataFrame with one row per (scenario, episode).
    """
    platform = get_platform(config.platform_name)
    scenarios = config.scenarios or list(SCENARIOS.keys())
    rows = []

    for scenario in scenarios:
        for ep in range(config.n_episodes):
            seed = config.base_seed + ep
            engine = SimulatorEngine(
                platform=platform,
                episode_duration_ns=config.episode_duration_ns,
                seed=seed,
            )
            rng = np.random.default_rng(seed)
            workload = ScenarioComposer(scenario, rng)
            engine.run_episode(agent, workload)
            engine._update_utilization_stats()

            em = compute_eval_metrics(engine)
            row = metrics_to_dict(em)
            row["agent"] = agent_name
            row["scenario"] = scenario
            row["platform"] = config.platform_name
            row["episode"] = ep
            row["seed"] = seed
            rows.append(row)

    return pd.DataFrame(rows)


def run_full_evaluation(
    agents: dict[str, ExpertPolicy] | None = None,
    config: EvalConfig | None = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run all agents on all scenarios and compile results.

    Args:
        agents: Dict of {name: agent}. None = default experts.
        config: Evaluation config. None = defaults.
        output_path: Where to save the CSV. None = don't save.

    Returns:
        Combined DataFrame with all results.
    """
    if agents is None:
        agents = get_default_experts()
    if config is None:
        config = EvalConfig()

    all_dfs = []
    total = len(agents)
    for i, (name, agent) in enumerate(agents.items()):
        print(f"[{i+1}/{total}] Evaluating {name}...", flush=True)
        df = evaluate_agent(agent, name, config)
        all_dfs.append(df)
        # Print summary
        for scenario in df["scenario"].unique():
            mask = df["scenario"] == scenario
            dcr = df.loc[mask, "dcr"].mean()
            lat = df.loc[mask, "mean_latency_ns"].mean() / 1e6
            print(f"  {scenario}: DCR={dcr:.3f}, latency={lat:.2f}ms")

    combined = pd.concat(all_dfs, ignore_index=True)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")

    return combined


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics grouped by (agent, scenario).

    Returns DataFrame with mean, std, and 95% CI for key metrics.
    """
    key_metrics = ["dcr", "mean_latency_ns", "throughput",
                   "utilization_balance", "power_efficiency"]

    summaries = []
    for (agent, scenario), group in df.groupby(["agent", "scenario"]):
        row = {"agent": agent, "scenario": scenario, "n_episodes": len(group)}
        for metric in key_metrics:
            vals = group[metric].values
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals))
            n = len(vals)
            if n > 1:
                se = float(np.std(vals, ddof=1) / np.sqrt(n))
                row[f"{metric}_ci95"] = 1.96 * se
            else:
                row[f"{metric}_ci95"] = 0.0
        summaries.append(row)

    return pd.DataFrame(summaries)
