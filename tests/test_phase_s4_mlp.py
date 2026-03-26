"""Phase S4 tests: MLP training pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

from slm_sim.observation import TOTAL_FEATURES


@pytest.fixture
def small_dataset_dir():
    """Generate a small dataset and return its directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"
        splits_dir = Path(tmpdir) / "splits"

        from scripts.generate_dataset import generate
        generate(raw_dir, small=True)

        from scripts.split_dataset import split_dataset
        split_dataset(raw_dir, splits_dir)

        yield tmpdir


class TestSchedulerMLP:
    def test_model_construction(self):
        from training.mlp.model import SchedulerMLP
        model = SchedulerMLP(n_features=108, n_actions=42)
        assert model.param_count() > 100_000  # ~131K expected
        assert model.param_count() < 200_000

    def test_forward_pass(self):
        from training.mlp.model import SchedulerMLP
        model = SchedulerMLP(n_features=108, n_actions=42)
        x = torch.randn(4, 108)
        logits = model(x)
        assert logits.shape == (4, 42)

    def test_predict(self):
        from training.mlp.model import SchedulerMLP
        model = SchedulerMLP(n_features=108, n_actions=42)
        x = torch.randn(4, 108)
        actions = model.predict(x)
        assert actions.shape == (4,)
        assert actions.min() >= 0
        assert actions.max() < 42


class TestSchedulerDataset:
    def test_loads_from_parquet(self, small_dataset_dir):
        from training.mlp.dataset import SchedulerDataset
        train_path = Path(small_dataset_dir) / "splits" / "train.parquet"
        ds = SchedulerDataset(train_path)
        assert len(ds) > 0

        state, action, weight = ds[0]
        assert state.shape == (108,)
        assert action.dtype == torch.long
        assert weight.dtype == torch.float32
        assert weight.item() > 0

    def test_normalization(self, small_dataset_dir):
        from training.mlp.dataset import SchedulerDataset
        train_path = Path(small_dataset_dir) / "splits" / "train.parquet"
        norm_path = Path(small_dataset_dir) / "normalization.json"
        ds = SchedulerDataset(train_path, normalization_path=norm_path)
        assert len(ds) > 0

    def test_filter_experts(self, small_dataset_dir):
        from training.mlp.dataset import SchedulerDataset
        train_path = Path(small_dataset_dir) / "splits" / "train.parquet"
        ds_all = SchedulerDataset(train_path, experts=None)
        ds_one = SchedulerDataset(train_path, experts={"slm_os_hybrid"})
        assert len(ds_one) <= len(ds_all)


class TestMLPTraining:
    def test_training_loss_decreases(self, small_dataset_dir):
        from training.mlp.dataset import SchedulerDataset
        from training.mlp.model import SchedulerMLP
        from training.mlp.train import TrainConfig, train_mlp

        train_path = Path(small_dataset_dir) / "splits" / "train.parquet"
        val_path = Path(small_dataset_dir) / "splits" / "val.parquet"

        train_ds = SchedulerDataset(train_path)
        val_ds = SchedulerDataset(val_path)

        if len(train_ds) == 0 or len(val_ds) == 0:
            pytest.skip("Not enough data in splits")

        config = TrainConfig(
            n_epochs=10,
            batch_size=64,
            lr=1e-3,
            patience=20,  # don't early stop in this test
        )

        save_path = Path(small_dataset_dir) / "model.pt"
        model, result = train_mlp(train_ds, val_ds, config, save_path)

        assert len(result.train_losses) > 1
        # Loss should generally decrease (first few epochs)
        assert result.train_losses[-1] < result.train_losses[0], (
            f"Loss did not decrease: {result.train_losses[0]:.4f} -> {result.train_losses[-1]:.4f}"
        )
        assert save_path.exists()

    def test_onnx_export(self, small_dataset_dir):
        from training.mlp.model import SchedulerMLP
        from training.mlp.train import export_onnx

        model = SchedulerMLP(n_features=108, n_actions=42)
        onnx_path = Path(small_dataset_dir) / "model.onnx"
        export_onnx(model, onnx_path)
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0


class TestDAgger:
    def test_dagger_single_round(self):
        """Run 1 DAgger iteration with very few episodes."""
        from training.mlp.model import SchedulerMLP
        from training.mlp.dagger import run_dagger
        from slm_sim.experts.hybrid import HybridExpertPolicy

        model = SchedulerMLP(n_features=108, n_actions=42)
        expert = HybridExpertPolicy()

        tables = run_dagger(
            model=model,
            expert=expert,
            platform_name="jetson_orin_nano",
            scenario_name="light_single",
            n_iterations=1,
            episodes_per_iteration=3,
            episode_duration_ns=500_000_000,
        )

        assert len(tables) == 1
        assert len(tables[0]) > 0
