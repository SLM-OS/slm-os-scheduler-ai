"""Phase S3 integration test: generate, validate, and split a small dataset."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from slm_sim.engine import SimulatorEngine
from slm_sim.experts.hybrid import HybridExpertPolicy
from slm_sim.logging import transitions_to_table, write_parquet, read_parquet
from slm_sim.observation import TOTAL_FEATURES
from slm_sim.platforms import make_jetson_orin_nano
from slm_sim.runner import RunConfig, run_batch, batch_to_parquet, run_single_episode
from slm_sim.workloads.scenarios import ScenarioComposer


class TestTransitionLogger:
    def test_single_episode_to_table(self):
        platform = make_jetson_orin_nano()
        engine = SimulatorEngine(platform=platform, episode_duration_ns=1_000_000_000, seed=42)
        rng = np.random.default_rng(42)
        workload = ScenarioComposer("light_single", rng)
        expert = HybridExpertPolicy()
        engine.run_episode(expert, workload)

        table = transitions_to_table(
            engine.transitions, "slm_os_hybrid", "light_single",
            "jetson_orin_nano", 0, 6, True,
        )
        assert len(table) > 0
        assert "action" in table.column_names
        assert "reward" in table.column_names
        assert "state_000" in table.column_names
        assert f"state_{TOTAL_FEATURES-1:03d}" in table.column_names
        assert "next_state_000" in table.column_names
        assert "expert_policy" in table.column_names
        assert "episode_id" in table.column_names

    def test_write_and_read_parquet(self):
        platform = make_jetson_orin_nano()
        engine = SimulatorEngine(platform=platform, episode_duration_ns=500_000_000, seed=42)
        rng = np.random.default_rng(42)
        workload = ScenarioComposer("light_single", rng)
        expert = HybridExpertPolicy()
        engine.run_episode(expert, workload)

        table = transitions_to_table(
            engine.transitions, "slm_os_hybrid", "light_single",
            "jetson_orin_nano", 0, 6, True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            write_parquet(table, path)
            assert path.exists()

            loaded = read_parquet(path)
            assert len(loaded) == len(table)
            assert loaded.column_names == table.column_names


class TestRunner:
    def test_run_single_episode(self):
        args = ("slm_os_hybrid", "light_single", "jetson_orin_nano",
                1_000_000_000, 42, 0)
        result = run_single_episode(args)
        assert len(result["transitions"]) > 0
        assert result["metrics"]["tasks_completed"] > 0
        assert result["metrics"]["dcr"] >= 0.0

    def test_run_batch_sequential(self):
        config = RunConfig(
            expert_name="slm_os_hybrid",
            scenario_name="light_single",
            platform_name="jetson_orin_nano",
            n_episodes=3,
            episode_duration_ns=500_000_000,
        )
        results = run_batch(config, n_workers=None)
        assert len(results) == 3
        for r in results:
            assert len(r["transitions"]) > 0

    def test_batch_to_parquet(self):
        config = RunConfig(
            expert_name="edf",
            scenario_name="light_single",
            platform_name="jetson_orin_nano",
            n_episodes=3,
            episode_duration_ns=500_000_000,
        )
        results = run_batch(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "batch.parquet"
            n_rows = batch_to_parquet(results, path)
            assert n_rows > 0
            assert path.exists()

            table = pq.read_table(path)
            assert len(table) == n_rows
            # Should have 3 unique episode IDs
            episode_ids = table.column("episode_id").to_pylist()
            assert len(set(episode_ids)) == 3


class TestFullPipeline:
    """End-to-end: generate small dataset, validate, split."""

    def test_generate_validate_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            splits_dir = Path(tmpdir) / "splits"

            # Generate small dataset: 2 experts x 2 scenarios x 1 platform
            from scripts.generate_dataset import generate
            generate(raw_dir, small=True)

            # Verify files were created
            parquet_files = list(raw_dir.glob("*.parquet"))
            assert len(parquet_files) > 0

            # Validate
            from scripts.validate_dataset import validate_dataset
            passed = validate_dataset(raw_dir)
            assert passed, "Dataset validation failed"

            # Split
            from scripts.split_dataset import split_dataset
            stats = split_dataset(raw_dir, splits_dir)
            assert stats["total_rows"] > 0
            assert stats["train_rows"] > 0
            assert stats["val_rows"] >= 0
            assert stats["test_rows"] >= 0
            assert (stats["train_rows"] + stats["val_rows"] + stats["test_rows"]
                    == stats["total_rows"])

            # Verify split files exist
            assert (splits_dir / "train.parquet").exists()
            assert (splits_dir / "val.parquet").exists()
            assert (splits_dir / "test.parquet").exists()

            # Verify normalization stats
            norm_path = Path(tmpdir) / "normalization.json"
            assert norm_path.exists()
            with open(norm_path) as f:
                norm = json.load(f)
            assert norm["n_features"] == TOTAL_FEATURES
            assert len(norm["mean"]) == TOTAL_FEATURES
            assert len(norm["std"]) == TOTAL_FEATURES

    def test_state_features_in_range(self):
        """Verify all state features are in [0, 1] in generated data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            from scripts.generate_dataset import generate
            generate(raw_dir, small=True)

            for path in raw_dir.glob("*.parquet"):
                table = pq.read_table(path)
                for j in range(TOTAL_FEATURES):
                    col = f"state_{j:03d}"
                    arr = table.column(col).to_numpy()
                    assert np.all(arr >= -0.01), f"{path.name} {col}: min={arr.min()}"
                    assert np.all(arr <= 1.01), f"{path.name} {col}: max={arr.max()}"
