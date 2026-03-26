"""Phase S7 tests: evaluation pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from slm_sim.engine import SimulatorEngine
from slm_sim.experts.hybrid import HybridExpertPolicy
from slm_sim.experts.random_policy import RandomExpertPolicy
from slm_sim.platforms import make_jetson_orin_nano
from slm_sim.workloads.scenarios import ScenarioComposer


class TestEvalMetrics:
    def test_compute_metrics_after_episode(self):
        from evaluation.metrics import compute_eval_metrics

        platform = make_jetson_orin_nano()
        engine = SimulatorEngine(
            platform=platform, episode_duration_ns=1_000_000_000, seed=42,
        )
        rng = np.random.default_rng(42)
        workload = ScenarioComposer("light_single", rng)
        engine.run_episode(HybridExpertPolicy(), workload)
        engine._update_utilization_stats()

        em = compute_eval_metrics(engine)
        assert 0.0 <= em.dcr <= 1.0
        assert em.mean_latency_ns >= 0
        assert em.throughput > 0
        assert em.tasks_completed > 0
        assert 0.0 <= em.utilization_balance <= 1.0
        assert 0.0 <= em.power_efficiency <= 1.0

    def test_metrics_to_dict(self):
        from evaluation.metrics import EvalMetrics, metrics_to_dict

        em = EvalMetrics(dcr=0.95, mean_latency_ns=1_000_000, throughput=500)
        d = metrics_to_dict(em)
        assert d["dcr"] == 0.95
        assert d["throughput"] == 500
        assert "mean_latency_ns" in d


class TestEvalRunner:
    def test_evaluate_single_agent(self):
        from evaluation.run_eval import EvalConfig, evaluate_agent

        config = EvalConfig(
            n_episodes=3,
            episode_duration_ns=500_000_000,
            scenarios=["light_single"],
        )
        df = evaluate_agent(HybridExpertPolicy(), "hybrid", config)
        assert len(df) == 3
        assert "dcr" in df.columns
        assert "agent" in df.columns
        assert df["agent"].iloc[0] == "hybrid"
        assert df["scenario"].iloc[0] == "light_single"

    def test_full_evaluation(self):
        from evaluation.run_eval import EvalConfig, run_full_evaluation, summarize_results

        agents = {
            "hybrid": HybridExpertPolicy(),
            "random": RandomExpertPolicy(rng=np.random.default_rng(42)),
        }
        config = EvalConfig(
            n_episodes=3,
            episode_duration_ns=500_000_000,
            scenarios=["light_single", "medium_mixed"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            df = run_full_evaluation(agents, config, csv_path)

            assert len(df) == 2 * 2 * 3  # 2 agents x 2 scenarios x 3 episodes
            assert csv_path.exists()

            summary = summarize_results(df)
            assert len(summary) == 4  # 2 agents x 2 scenarios
            assert "dcr_mean" in summary.columns
            assert "dcr_ci95" in summary.columns


class TestStatisticalTests:
    def test_paired_comparison(self):
        from evaluation.run_eval import EvalConfig, run_full_evaluation
        from evaluation.statistical_tests import paired_comparison

        agents = {
            "hybrid": HybridExpertPolicy(),
            "random": RandomExpertPolicy(rng=np.random.default_rng(42)),
        }
        config = EvalConfig(
            n_episodes=5,
            episode_duration_ns=1_000_000_000,
            scenarios=["medium_mixed"],
        )
        df = run_full_evaluation(agents, config)

        result = paired_comparison(df, "hybrid", "random", "dcr", "medium_mixed")
        assert result.n_episodes == 5
        assert result.mean_a >= result.mean_b  # hybrid should be >= random
        assert 0 <= result.p_value <= 1

    def test_all_comparisons(self):
        from evaluation.run_eval import EvalConfig, run_full_evaluation
        from evaluation.statistical_tests import run_all_comparisons

        agents = {
            "hybrid": HybridExpertPolicy(),
            "edf": EDFExpertPolicy(),
            "random": RandomExpertPolicy(rng=np.random.default_rng(42)),
        }
        config = EvalConfig(
            n_episodes=3,
            episode_duration_ns=500_000_000,
            scenarios=["light_single"],
        )
        df = run_full_evaluation(agents, config)
        comparisons = run_all_comparisons(df, baseline="random", metric="dcr")

        assert len(comparisons) == 2  # hybrid vs random, edf vs random
        assert "diff" in comparisons.columns
        assert "p_value" in comparisons.columns


class TestVisualization:
    def test_generate_all_charts(self):
        from evaluation.run_eval import EvalConfig, run_full_evaluation
        from evaluation.visualization import generate_all_charts

        agents = {
            "hybrid": HybridExpertPolicy(),
            "random": RandomExpertPolicy(rng=np.random.default_rng(42)),
        }
        config = EvalConfig(
            n_episodes=3,
            episode_duration_ns=500_000_000,
            scenarios=["light_single", "medium_mixed"],
        )
        df = run_full_evaluation(agents, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            charts_dir = Path(tmpdir) / "charts"
            files = generate_all_charts(df, charts_dir)

            assert len(files) == 3
            for f in files:
                assert f.exists()
                assert f.stat().st_size > 0

    def test_summary_table(self):
        from evaluation.run_eval import EvalConfig, run_full_evaluation, summarize_results
        from evaluation.visualization import generate_summary_table

        agents = {"hybrid": HybridExpertPolicy()}
        config = EvalConfig(
            n_episodes=3, episode_duration_ns=500_000_000,
            scenarios=["light_single"],
        )
        df = run_full_evaluation(agents, config)
        summary = summarize_results(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.csv"
            generate_summary_table(summary, path)
            assert path.exists()

            loaded = pd.read_csv(path)
            assert len(loaded) == 1
            assert "dcr_mean" in loaded.columns


# Import needed for test_all_comparisons
from slm_sim.experts.edf import EDFExpertPolicy
