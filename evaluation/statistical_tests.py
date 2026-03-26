"""Statistical significance testing for model comparison.

Paired t-tests, 95% confidence intervals over test episodes.
See plan Section 8.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ComparisonResult:
    """Result of a paired comparison between two agents."""
    agent_a: str
    agent_b: str
    scenario: str
    metric: str
    mean_a: float
    mean_b: float
    mean_diff: float       # a - b
    std_diff: float
    ci95_low: float        # 95% CI lower bound for diff
    ci95_high: float       # 95% CI upper bound for diff
    t_statistic: float
    p_value: float
    significant: bool      # p < 0.05
    n_episodes: int


def paired_comparison(
    df: pd.DataFrame,
    agent_a: str,
    agent_b: str,
    metric: str = "dcr",
    scenario: str | None = None,
) -> ComparisonResult:
    """Run a paired t-test comparing two agents on the same episodes.

    Episodes are matched by seed, so this is a true paired comparison.

    Args:
        df: Full evaluation results DataFrame.
        agent_a: First agent name.
        agent_b: Second agent name.
        metric: Metric column to compare.
        scenario: Specific scenario (None = all scenarios pooled).

    Returns:
        ComparisonResult with test statistics.
    """
    df_a = df[df["agent"] == agent_a]
    df_b = df[df["agent"] == agent_b]

    if scenario:
        df_a = df_a[df_a["scenario"] == scenario]
        df_b = df_b[df_b["scenario"] == scenario]

    # Match by seed
    merged = pd.merge(
        df_a[["seed", "scenario", metric]],
        df_b[["seed", "scenario", metric]],
        on=["seed", "scenario"],
        suffixes=("_a", "_b"),
    )

    if len(merged) < 2:
        return ComparisonResult(
            agent_a=agent_a, agent_b=agent_b,
            scenario=scenario or "all", metric=metric,
            mean_a=0, mean_b=0, mean_diff=0, std_diff=0,
            ci95_low=0, ci95_high=0,
            t_statistic=0, p_value=1.0,
            significant=False, n_episodes=len(merged),
        )

    vals_a = merged[f"{metric}_a"].values
    vals_b = merged[f"{metric}_b"].values
    diffs = vals_a - vals_b

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    n = len(diffs)
    se = std_diff / np.sqrt(n)

    t_stat, p_val = stats.ttest_rel(vals_a, vals_b)

    ci_margin = stats.t.ppf(0.975, df=n - 1) * se

    return ComparisonResult(
        agent_a=agent_a,
        agent_b=agent_b,
        scenario=scenario or "all",
        metric=metric,
        mean_a=float(np.mean(vals_a)),
        mean_b=float(np.mean(vals_b)),
        mean_diff=mean_diff,
        std_diff=std_diff,
        ci95_low=mean_diff - ci_margin,
        ci95_high=mean_diff + ci_margin,
        t_statistic=float(t_stat),
        p_value=float(p_val),
        significant=p_val < 0.05,
        n_episodes=n,
    )


def run_all_comparisons(
    df: pd.DataFrame,
    baseline: str = "random",
    metric: str = "dcr",
) -> pd.DataFrame:
    """Compare all agents against a baseline across all scenarios.

    Returns DataFrame with comparison results.
    """
    agents = [a for a in df["agent"].unique() if a != baseline]
    scenarios = df["scenario"].unique()

    results = []
    for agent in agents:
        for scenario in scenarios:
            cr = paired_comparison(df, agent, baseline, metric, scenario)
            results.append({
                "agent": cr.agent_a,
                "baseline": cr.agent_b,
                "scenario": cr.scenario,
                "metric": cr.metric,
                "agent_mean": cr.mean_a,
                "baseline_mean": cr.mean_b,
                "diff": cr.mean_diff,
                "ci95_low": cr.ci95_low,
                "ci95_high": cr.ci95_high,
                "p_value": cr.p_value,
                "significant": cr.significant,
                "n_episodes": cr.n_episodes,
            })

    return pd.DataFrame(results)
