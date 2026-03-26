"""Charts and tables for evaluation results.

Generates comparison plots, bar charts, and summary tables.
See plan Sections 7.3 and 8.4.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_dcr_comparison(
    df: pd.DataFrame,
    output_path: Path | str,
    scenarios: list[str] | None = None,
) -> None:
    """Bar chart comparing DCR across agents and scenarios.

    Args:
        df: Evaluation results DataFrame with columns: agent, scenario, dcr.
        output_path: Where to save the chart image.
        scenarios: Scenarios to include. None = all.
    """
    if scenarios:
        df = df[df["scenario"].isin(scenarios)]

    summary = df.groupby(["agent", "scenario"])["dcr"].agg(["mean", "std"]).reset_index()
    agents = sorted(summary["agent"].unique())
    scenario_list = sorted(summary["scenario"].unique())

    fig, ax = plt.subplots(figsize=(max(10, len(scenario_list) * 2), 6))
    x = np.arange(len(scenario_list))
    width = 0.8 / len(agents)

    for i, agent in enumerate(agents):
        agent_data = summary[summary["agent"] == agent]
        means = []
        stds = []
        for s in scenario_list:
            row = agent_data[agent_data["scenario"] == s]
            means.append(row["mean"].values[0] if len(row) > 0 else 0)
            stds.append(row["std"].values[0] if len(row) > 0 else 0)

        offset = (i - len(agents) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=agent,
               capsize=3, alpha=0.85)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Deadline Compliance Rate")
    ax.set_title("DCR Comparison Across Agents and Scenarios")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_list, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_latency_comparison(
    df: pd.DataFrame,
    output_path: Path | str,
    scenarios: list[str] | None = None,
) -> None:
    """Bar chart comparing mean latency across agents and scenarios."""
    if scenarios:
        df = df[df["scenario"].isin(scenarios)]

    summary = df.groupby(["agent", "scenario"])["mean_latency_ns"].agg(
        ["mean", "std"]
    ).reset_index()
    # Convert to ms
    summary["mean"] /= 1e6
    summary["std"] /= 1e6

    agents = sorted(summary["agent"].unique())
    scenario_list = sorted(summary["scenario"].unique())

    fig, ax = plt.subplots(figsize=(max(10, len(scenario_list) * 2), 6))
    x = np.arange(len(scenario_list))
    width = 0.8 / len(agents)

    for i, agent in enumerate(agents):
        agent_data = summary[summary["agent"] == agent]
        means = []
        stds = []
        for s in scenario_list:
            row = agent_data[agent_data["scenario"] == s]
            means.append(row["mean"].values[0] if len(row) > 0 else 0)
            stds.append(row["std"].values[0] if len(row) > 0 else 0)

        offset = (i - len(agents) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=agent,
               capsize=3, alpha=0.85)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_title("Mean Inference Latency Across Agents and Scenarios")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_list, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_throughput_comparison(
    df: pd.DataFrame,
    output_path: Path | str,
    scenarios: list[str] | None = None,
) -> None:
    """Bar chart comparing throughput (tasks/sec) across agents."""
    if scenarios:
        df = df[df["scenario"].isin(scenarios)]

    summary = df.groupby(["agent", "scenario"])["throughput"].agg(
        ["mean", "std"]
    ).reset_index()

    agents = sorted(summary["agent"].unique())
    scenario_list = sorted(summary["scenario"].unique())

    fig, ax = plt.subplots(figsize=(max(10, len(scenario_list) * 2), 6))
    x = np.arange(len(scenario_list))
    width = 0.8 / len(agents)

    for i, agent in enumerate(agents):
        agent_data = summary[summary["agent"] == agent]
        means = []
        stds = []
        for s in scenario_list:
            row = agent_data[agent_data["scenario"] == s]
            means.append(row["mean"].values[0] if len(row) > 0 else 0)
            stds.append(row["std"].values[0] if len(row) > 0 else 0)

        offset = (i - len(agents) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=agent,
               capsize=3, alpha=0.85)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Throughput (tasks/sec)")
    ax.set_title("Task Throughput Across Agents and Scenarios")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_list, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_summary_table(
    summary_df: pd.DataFrame,
    output_path: Path | str,
) -> None:
    """Save summary statistics as a formatted CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False, float_format="%.4f")


def generate_all_charts(
    df: pd.DataFrame,
    output_dir: Path | str,
    scenarios: list[str] | None = None,
) -> list[Path]:
    """Generate all standard evaluation charts.

    Returns list of generated file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = []

    dcr_path = output_dir / "dcr_comparison.png"
    plot_dcr_comparison(df, dcr_path, scenarios)
    files.append(dcr_path)

    lat_path = output_dir / "latency_comparison.png"
    plot_latency_comparison(df, lat_path, scenarios)
    files.append(lat_path)

    tp_path = output_dir / "throughput_comparison.png"
    plot_throughput_comparison(df, tp_path, scenarios)
    files.append(tp_path)

    return files
