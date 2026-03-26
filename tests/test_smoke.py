"""Phase S1 smoke test: run episodes with hybrid expert policy.

Verifies:
- Simulator runs to completion without errors
- Hybrid expert achieves >90% DCR on light_single
- Throughput > 1000 episodes/minute (= >16 episodes/second)
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from slm_sim.engine import SimulatorEngine
from slm_sim.experts.hybrid import HybridExpertPolicy
from slm_sim.platforms import make_jetson_orin_nano, make_raspberry_pi5
from slm_sim.workloads.scenarios import ScenarioComposer


class TestSmokeHybridPolicy:
    """Integration test: full episode with hybrid expert."""

    def test_single_episode_completes(self):
        platform = make_jetson_orin_nano()
        engine = SimulatorEngine(
            platform=platform,
            episode_duration_ns=1_000_000_000,  # 1 second (short)
            seed=42,
        )
        rng = np.random.default_rng(42)
        workload = ScenarioComposer("light_single", rng)
        agent = HybridExpertPolicy()

        metrics = engine.run_episode(agent, workload)

        assert metrics.total_tasks_arrived > 0
        assert metrics.total_tasks_completed > 0
        assert metrics.scheduling_decisions > 0

    def test_dcr_above_90_percent_light_single(self):
        """Run 20 episodes and check average DCR > 90%."""
        platform = make_jetson_orin_nano()
        agent = HybridExpertPolicy()
        dcrs = []

        for seed in range(20):
            engine = SimulatorEngine(
                platform=platform,
                episode_duration_ns=2_000_000_000,  # 2 seconds
                seed=seed,
            )
            rng = np.random.default_rng(seed)
            workload = ScenarioComposer("light_single", rng)
            metrics = engine.run_episode(agent, workload)
            dcrs.append(metrics.deadline_compliance_rate)

        avg_dcr = np.mean(dcrs)
        assert avg_dcr > 0.90, (
            f"Average DCR {avg_dcr:.3f} is below 90% threshold. "
            f"Individual DCRs: {[f'{d:.3f}' for d in dcrs]}"
        )

    def test_throughput(self):
        """Verify we can run >16 episodes/second (= 1000/min)."""
        platform = make_jetson_orin_nano()
        agent = HybridExpertPolicy()
        n_episodes = 50

        start = time.perf_counter()
        for seed in range(n_episodes):
            engine = SimulatorEngine(
                platform=platform,
                episode_duration_ns=1_000_000_000,  # 1 second
                seed=seed,
            )
            rng = np.random.default_rng(seed)
            workload = ScenarioComposer("light_single", rng)
            engine.run_episode(agent, workload)
        elapsed = time.perf_counter() - start

        eps_per_sec = n_episodes / elapsed
        eps_per_min = eps_per_sec * 60
        assert eps_per_min > 1000, (
            f"Throughput {eps_per_min:.0f} episodes/min is below 1000. "
            f"({eps_per_sec:.1f} eps/sec, {elapsed:.2f}s for {n_episodes} episodes)"
        )

    def test_transitions_logged(self):
        """Verify transition tuples are generated."""
        platform = make_jetson_orin_nano()
        engine = SimulatorEngine(
            platform=platform,
            episode_duration_ns=1_000_000_000,
            seed=42,
        )
        rng = np.random.default_rng(42)
        workload = ScenarioComposer("light_single", rng)
        agent = HybridExpertPolicy()

        engine.run_episode(agent, workload)

        assert len(engine.transitions) > 0
        t = engine.transitions[0]
        assert "state" in t
        assert "action" in t
        assert "reward" in t
        assert "next_state" in t
        assert "done" in t
        assert t["state"].shape == (108,)

    def test_pi5_platform_runs(self):
        """Verify simulation works on Pi 5 profile too."""
        platform = make_raspberry_pi5()
        engine = SimulatorEngine(
            platform=platform,
            episode_duration_ns=1_000_000_000,
            seed=42,
        )
        rng = np.random.default_rng(42)
        workload = ScenarioComposer("light_single", rng)
        agent = HybridExpertPolicy()

        metrics = engine.run_episode(agent, workload)
        assert metrics.total_tasks_completed > 0
