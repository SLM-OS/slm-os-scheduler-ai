#!/usr/bin/env python3
"""Run full evaluation: trained models vs expert baselines.

Evaluates all agents on identical episodes with paired seeds,
computes metrics, runs statistical significance tests, and
generates comparison charts.

Usage:
    python scripts/run_evaluation.py [--episodes N] [--platform NAME]
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.agents import MLPAgent, PPOAgent, XGBoostAgent
from evaluation.run_eval import EvalConfig, run_full_evaluation, summarize_results
from evaluation.statistical_tests import run_all_comparisons
from slm_sim.platforms import get_platform


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description="Run full model evaluation")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Episodes per scenario per agent (default: 200)")
    parser.add_argument("--platform", default="jetson_orin_nano",
                        help="Platform to evaluate on (default: jetson_orin_nano)")
    args = parser.parse_args()

    platform = get_platform(args.platform)
    num_cores = platform.num_cores
    gpu = platform.gpu.available

    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Build agent dict: experts + trained models ---
    from evaluation.run_eval import get_default_experts
    agents = get_default_experts()

    # MLP
    mlp_path = project_root / "models" / "mlp" / "best.pt"
    if mlp_path.exists():
        agents["mlp"] = MLPAgent(mlp_path, num_cores, gpu)
        print(f"[{_ts()}] Loaded MLP from {mlp_path}")
    else:
        print(f"[{_ts()}] SKIP MLP: {mlp_path} not found")

    # PPO
    ppo_path = project_root / "models" / "ppo" / "best_model.zip"
    if ppo_path.exists():
        agents["ppo"] = PPOAgent(ppo_path, num_cores, gpu)
        print(f"[{_ts()}] Loaded PPO from {ppo_path}")
    else:
        print(f"[{_ts()}] SKIP PPO: {ppo_path} not found")

    # XGBoost
    xgb_dir = project_root / "models" / "xgboost"
    if (xgb_dir / "meta.json").exists():
        agents["xgboost"] = XGBoostAgent(xgb_dir, num_cores, gpu)
        print(f"[{_ts()}] Loaded XGBoost from {xgb_dir}")
    else:
        print(f"[{_ts()}] SKIP XGBoost: {xgb_dir}/meta.json not found")

    print(f"\n[{_ts()}] Evaluating {len(agents)} agents: {list(agents.keys())}")
    print(f"  Platform: {args.platform} ({num_cores} cores, GPU={'yes' if gpu else 'no'})")
    print(f"  Episodes per scenario: {args.episodes}")
    print(f"  Scenarios: 8")
    print(f"  Total episodes: {len(agents) * 8 * args.episodes:,}")
    print(flush=True)

    config = EvalConfig(
        n_episodes=args.episodes,
        platform_name=args.platform,
    )

    # --- Run evaluation ---
    t0 = time.time()
    results = run_full_evaluation(
        agents=agents,
        config=config,
        output_path=results_dir / "eval_results.csv",
    )
    elapsed = time.time() - t0
    print(f"\n[{_ts()}] Evaluation complete in {elapsed / 60:.1f} minutes")

    # --- Summary statistics ---
    print(f"\n{'=' * 70}")
    print("SUMMARY (mean +/- 95% CI)")
    print(f"{'=' * 70}")
    summary = summarize_results(results)
    summary.to_csv(results_dir / "eval_summary.csv", index=False)

    # Print a readable table per scenario
    for scenario in sorted(results["scenario"].unique()):
        print(f"\n--- {scenario} ---")
        s = summary[summary["scenario"] == scenario].sort_values("dcr_mean", ascending=False)
        print(f"  {'Agent':<12} {'DCR':>8} {'Latency(ms)':>12} {'Throughput':>11} {'Balance':>8}")
        for _, row in s.iterrows():
            dcr = f"{row['dcr_mean']:.3f}"
            lat = f"{row['mean_latency_ns_mean'] / 1e6:.2f}"
            thr = f"{row['throughput_mean']:.0f}"
            bal = f"{row['utilization_balance_mean']:.3f}"
            print(f"  {row['agent']:<12} {dcr:>8} {lat:>12} {thr:>11} {bal:>8}")

    # --- Statistical comparisons vs hybrid baseline ---
    print(f"\n{'=' * 70}")
    print("STATISTICAL COMPARISONS vs hybrid (paired t-test, p < 0.05)")
    print(f"{'=' * 70}")

    for metric in ["dcr", "mean_latency_ns", "throughput"]:
        print(f"\n--- {metric} ---")
        comparisons = run_all_comparisons(results, baseline="hybrid", metric=metric)
        comparisons.to_csv(results_dir / f"comparisons_{metric}.csv", index=False)

        for _, row in comparisons.iterrows():
            sig = "*" if row["significant"] else " "
            direction = "+" if row["diff"] > 0 else ""
            print(f"  {row['agent']:<12} vs hybrid on {row['scenario']:<20} "
                  f"diff={direction}{row['diff']:.4f} "
                  f"p={row['p_value']:.4f} {sig}")

    print(f"\n[{_ts()}] Results saved to {results_dir}/")

    # --- Charts ---
    try:
        from evaluation.visualization import generate_all_charts
        generate_all_charts(results, output_dir=str(results_dir / "charts"))
        print(f"[{_ts()}] Charts saved to {results_dir}/charts/")
    except Exception as e:
        print(f"[{_ts()}] Chart generation failed: {e}")


if __name__ == "__main__":
    main()
