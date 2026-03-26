"""Phase S5 tests: XGBoost training pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def dataset_dir():
    """Generate a small dataset once for all tests in this module."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"
        splits_dir = Path(tmpdir) / "splits"

        from scripts.generate_dataset import generate
        generate(raw_dir, small=True)

        from scripts.split_dataset import split_dataset
        split_dataset(raw_dir, splits_dir)

        yield tmpdir


class TestFeatureEngineering:
    def test_derived_features_shape(self):
        from training.xgboost.features import add_derived_features, N_DERIVED_FEATURES
        from slm_sim.observation import TOTAL_FEATURES
        states = np.random.rand(10, TOTAL_FEATURES).astype(np.float32)
        X = add_derived_features(states)
        assert X.shape == (10, TOTAL_FEATURES + N_DERIVED_FEATURES)

    def test_feature_names_count(self):
        from training.xgboost.features import get_feature_names, N_DERIVED_FEATURES
        from slm_sim.observation import TOTAL_FEATURES
        names = get_feature_names()
        assert len(names) == TOTAL_FEATURES + N_DERIVED_FEATURES

    def test_load_features_and_labels(self, dataset_dir):
        from training.xgboost.features import load_features_and_labels
        train_path = Path(dataset_dir) / "splits" / "train.parquet"
        X, y_core, y_prio, y_pre = load_features_and_labels(train_path)
        assert X.shape[0] > 0
        assert X.shape[1] == 113  # 108 + 5 derived
        assert len(y_core) == X.shape[0]
        assert len(y_prio) == X.shape[0]
        assert len(y_pre) == X.shape[0]
        # Labels should be valid
        assert y_core.min() >= 0
        assert y_prio.min() >= 0
        assert set(np.unique(y_pre)).issubset({0, 1})


class TestTripleClassifier:
    def test_train_and_predict(self, dataset_dir):
        from training.xgboost.train import train_triple_classifier, XGBConfig

        train_path = Path(dataset_dir) / "splits" / "train.parquet"
        val_path = Path(dataset_dir) / "splits" / "val.parquet"

        # Use small configs for speed
        cfg = XGBConfig(n_estimators=10, max_depth=4)
        triple, metrics = train_triple_classifier(
            train_path, val_path,
            core_config=cfg, priority_config=cfg, preempt_config=cfg,
        )

        assert metrics["train_samples"] > 0
        assert metrics["train_core_acc"] > 0
        assert metrics["train_priority_acc"] > 0
        assert metrics["train_preempt_acc"] > 0

        if "val_core_acc" in metrics:
            assert metrics["val_core_acc"] > 0

    def test_save_and_load(self, dataset_dir):
        from training.xgboost.train import train_triple_classifier, XGBConfig

        train_path = Path(dataset_dir) / "splits" / "train.parquet"
        cfg = XGBConfig(n_estimators=5, max_depth=3)
        triple, _ = train_triple_classifier(
            train_path, core_config=cfg, priority_config=cfg, preempt_config=cfg,
        )

        with tempfile.TemporaryDirectory() as model_dir:
            model_path = Path(model_dir)
            triple.save(model_path)

            from training.xgboost.train import TripleClassifier
            loaded = TripleClassifier.load(model_path)

            # Predict with both and compare
            from training.xgboost.features import load_features_and_labels
            X, _, _, _ = load_features_and_labels(train_path)
            c1, p1, pr1 = triple.predict(X[:10])
            c2, p2, pr2 = loaded.predict(X[:10])
            np.testing.assert_array_equal(c1, c2)
            np.testing.assert_array_equal(p1, p2)
            np.testing.assert_array_equal(pr1, pr2)

    def test_feature_importance(self, dataset_dir):
        from training.xgboost.train import (
            train_triple_classifier,
            get_feature_importance,
            XGBConfig,
        )

        train_path = Path(dataset_dir) / "splits" / "train.parquet"
        cfg = XGBConfig(n_estimators=10, max_depth=4)
        triple, _ = train_triple_classifier(
            train_path, core_config=cfg, priority_config=cfg, preempt_config=cfg,
        )

        importances = get_feature_importance(triple)
        assert "core" in importances
        assert "priority" in importances
        assert "preempt" in importances
        # Core classifier has 113 features
        assert len(importances["core"]) == 113


class TestOptunaSearch:
    def test_small_search(self, dataset_dir):
        from training.xgboost.train import optuna_search

        train_path = Path(dataset_dir) / "splits" / "train.parquet"
        val_path = Path(dataset_dir) / "splits" / "val.parquet"

        best_params = optuna_search(
            train_path, val_path,
            classifier_name="core",
            n_trials=3,  # very small for speed
        )
        assert "n_estimators" in best_params
        assert "max_depth" in best_params
        assert "learning_rate" in best_params
