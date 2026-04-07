"""Three-classifier XGBoost training pipeline.

Trains separate classifiers for core assignment, priority adjustment,
and preempt decision. Includes Optuna hyperparameter search.
See plan Sections 4.2 and 7.2.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from training.xgboost.features import (
    N_DERIVED_FEATURES,
    add_derived_features,
    get_feature_names,
    load_features_and_labels,
)
from slm_sim.observation import TOTAL_FEATURES


@dataclass
class XGBConfig:
    """Configuration for a single XGBoost classifier."""
    n_estimators: int = 200
    max_depth: int = 8
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    n_jobs: int = -1
    random_state: int = 42


@dataclass
class TripleClassifier:
    """Three XGBoost classifiers for decomposed action prediction."""
    core_clf: xgb.XGBClassifier
    priority_clf: xgb.XGBClassifier
    preempt_clf: xgb.XGBClassifier
    feature_names: list[str] = field(default_factory=list)
    # Label encoders to map original labels to 0-based contiguous labels
    core_le: LabelEncoder = field(default_factory=LabelEncoder)
    priority_le: LabelEncoder = field(default_factory=LabelEncoder)
    preempt_le: LabelEncoder = field(default_factory=LabelEncoder)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict decomposed actions (returns original label values).

        Returns:
            (core_preds, priority_preds, preempt_preds)
        """
        core_enc = self.core_clf.predict(X)
        core = self.core_le.inverse_transform(core_enc)
        X_with_core = np.hstack([X, core.reshape(-1, 1)])
        prio_enc = self.priority_clf.predict(X_with_core)
        priority = self.priority_le.inverse_transform(prio_enc)
        X_with_both = np.hstack([X_with_core, priority.reshape(-1, 1)])
        pre_enc = self.preempt_clf.predict(X_with_both)
        preempt = self.preempt_le.inverse_transform(pre_enc)
        return core, priority, preempt

    def save(self, directory: Path) -> None:
        """Save all three classifiers and label encoders to JSON."""
        directory.mkdir(parents=True, exist_ok=True)
        self.core_clf.save_model(directory / "core_clf.json")
        self.priority_clf.save_model(directory / "priority_clf.json")
        self.preempt_clf.save_model(directory / "preempt_clf.json")
        meta = {
            "feature_names": self.feature_names,
            "core_classes": self.core_le.classes_.tolist(),
            "priority_classes": self.priority_le.classes_.tolist(),
            "preempt_classes": self.preempt_le.classes_.tolist(),
        }
        with open(directory / "meta.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, directory: Path) -> TripleClassifier:
        """Load all three classifiers and label encoders from JSON."""
        core_clf = xgb.XGBClassifier()
        core_clf.load_model(directory / "core_clf.json")
        priority_clf = xgb.XGBClassifier()
        priority_clf.load_model(directory / "priority_clf.json")
        preempt_clf = xgb.XGBClassifier()
        preempt_clf.load_model(directory / "preempt_clf.json")
        with open(directory / "meta.json") as f:
            meta = json.load(f)
        core_le = LabelEncoder()
        core_le.classes_ = np.array(meta["core_classes"])
        priority_le = LabelEncoder()
        priority_le.classes_ = np.array(meta["priority_classes"])
        preempt_le = LabelEncoder()
        preempt_le.classes_ = np.array(meta["preempt_classes"])
        return cls(core_clf, priority_clf, preempt_clf,
                   meta["feature_names"], core_le, priority_le, preempt_le)


def train_triple_classifier(
    train_path: Path | str,
    val_path: Optional[Path | str] = None,
    core_config: Optional[XGBConfig] = None,
    priority_config: Optional[XGBConfig] = None,
    preempt_config: Optional[XGBConfig] = None,
    max_rows: Optional[int] = None,
) -> tuple[TripleClassifier, dict]:
    """Train the three-classifier XGBoost ensemble.

    Args:
        train_path: Path to training Parquet file.
        val_path: Path to validation Parquet file (for eval metrics).
        core_config: Config for core assignment classifier.
        priority_config: Config for priority adjustment classifier.
        preempt_config: Config for preempt decision classifier.
        max_rows: If set, subsample training/val data to this many rows.

    Returns:
        Tuple of (TripleClassifier, metrics_dict).
    """
    if core_config is None:
        core_config = XGBConfig(n_estimators=200, max_depth=8)
    if priority_config is None:
        priority_config = XGBConfig(n_estimators=100, max_depth=6)
    if preempt_config is None:
        preempt_config = XGBConfig(n_estimators=100, max_depth=6)

    # Load training data
    X_train, y_core, y_priority, y_preempt = load_features_and_labels(
        train_path, max_rows=max_rows,
    )
    feature_names = get_feature_names()

    if len(X_train) == 0:
        raise ValueError("No training data after filtering")

    # Encode labels to 0-based contiguous integers (XGBoost sklearn requirement)
    core_le = LabelEncoder()
    priority_le = LabelEncoder()
    preempt_le = LabelEncoder()

    y_core_enc = core_le.fit_transform(y_core)
    y_priority_enc = priority_le.fit_transform(y_priority)
    y_preempt_enc = preempt_le.fit_transform(y_preempt)

    def _make_clf(cfg: XGBConfig) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            min_child_weight=cfg.min_child_weight,
            n_jobs=cfg.n_jobs,
            random_state=cfg.random_state,
            eval_metric="mlogloss",
        )

    print(f"[{_ts()}] Training on {len(X_train):,} samples, "
          f"{len(np.unique(y_core))} core classes, "
          f"{len(np.unique(y_priority))} priority classes, "
          f"{len(np.unique(y_preempt))} preempt classes", flush=True)
    train_start = time.time()

    # --- Classifier 1: Core Assignment ---
    print(f"[{_ts()}] Training core assignment classifier "
          f"({core_config.n_estimators} trees, depth {core_config.max_depth})...", flush=True)
    t0 = time.time()
    core_clf = _make_clf(core_config)
    core_clf.fit(X_train, y_core_enc)
    print(f"[{_ts()}] Core classifier done ({time.time() - t0:.1f}s)", flush=True)

    # --- Classifier 2: Priority Adjustment (conditioned on predicted core) ---
    print(f"[{_ts()}] Training priority classifier "
          f"({priority_config.n_estimators} trees, depth {priority_config.max_depth})...", flush=True)
    t0 = time.time()
    core_pred_enc = core_clf.predict(X_train)
    core_pred_orig = core_le.inverse_transform(core_pred_enc)
    X_train_c = np.hstack([X_train, core_pred_orig.reshape(-1, 1)])

    priority_clf = _make_clf(priority_config)
    priority_clf.fit(X_train_c, y_priority_enc)
    print(f"[{_ts()}] Priority classifier done ({time.time() - t0:.1f}s)", flush=True)

    # --- Classifier 3: Preempt Decision (conditioned on predicted core + priority) ---
    print(f"[{_ts()}] Training preempt classifier "
          f"({preempt_config.n_estimators} trees, depth {preempt_config.max_depth})...", flush=True)
    t0 = time.time()
    prio_pred_enc = priority_clf.predict(X_train_c)
    prio_pred_orig = priority_le.inverse_transform(prio_pred_enc)
    X_train_cp = np.hstack([X_train_c, prio_pred_orig.reshape(-1, 1)])

    preempt_clf = _make_clf(preempt_config)
    preempt_clf.fit(X_train_cp, y_preempt_enc)
    print(f"[{_ts()}] Preempt classifier done ({time.time() - t0:.1f}s)", flush=True)

    triple = TripleClassifier(
        core_clf, priority_clf, preempt_clf, feature_names,
        core_le, priority_le, preempt_le,
    )

    # Evaluate
    print(f"[{_ts()}] Evaluating on training set...", flush=True)
    metrics = {"train_samples": len(X_train)}
    train_c, train_p, train_pr = triple.predict(X_train)
    metrics["train_core_acc"] = float((train_c == y_core).mean())
    metrics["train_priority_acc"] = float((train_p == y_priority).mean())
    metrics["train_preempt_acc"] = float((train_pr == y_preempt).mean())
    print(f"[{_ts()}] Train — core={metrics['train_core_acc']:.4f}  "
          f"priority={metrics['train_priority_acc']:.4f}  "
          f"preempt={metrics['train_preempt_acc']:.4f}", flush=True)

    if val_path:
        print(f"[{_ts()}] Loading validation data...", flush=True)
        val_max = max_rows // 4 if max_rows else None
        X_val, vy_core, vy_priority, vy_preempt = load_features_and_labels(
            val_path, max_rows=val_max,
        )
        if len(X_val) > 0:
            print(f"[{_ts()}] Evaluating on {len(X_val):,} val samples...", flush=True)
            val_c, val_p, val_pr = triple.predict(X_val)
            metrics["val_samples"] = len(X_val)
            metrics["val_core_acc"] = float((val_c == vy_core).mean())
            metrics["val_priority_acc"] = float((val_p == vy_priority).mean())
            metrics["val_preempt_acc"] = float((val_pr == vy_preempt).mean())
            print(f"[{_ts()}] Val   — core={metrics['val_core_acc']:.4f}  "
                  f"priority={metrics['val_priority_acc']:.4f}  "
                  f"preempt={metrics['val_preempt_acc']:.4f}", flush=True)

    total_elapsed = time.time() - train_start
    print(f"[{_ts()}] XGBoost training complete in {total_elapsed:.1f}s", flush=True)

    return triple, metrics


def get_feature_importance(triple: TripleClassifier) -> dict[str, np.ndarray]:
    """Extract feature importances from all three classifiers.

    Returns dict mapping classifier name to importance array.
    """
    return {
        "core": triple.core_clf.feature_importances_,
        "priority": triple.priority_clf.feature_importances_,
        "preempt": triple.preempt_clf.feature_importances_,
    }


def optuna_search(
    train_path: Path | str,
    val_path: Path | str,
    classifier_name: str = "core",
    n_trials: int = 100,
) -> dict:
    """Run Optuna hyperparameter search for one classifier.

    Args:
        train_path: Training data path.
        val_path: Validation data path.
        classifier_name: Which classifier to tune ("core", "priority", "preempt").
        n_trials: Number of Optuna trials.

    Returns:
        Best hyperparameters as a dict.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train, y_core, y_priority, y_preempt = load_features_and_labels(train_path)
    X_val, vy_core, vy_priority, vy_preempt = load_features_and_labels(val_path)

    if classifier_name == "core":
        y_train, y_val = y_core, vy_core
    elif classifier_name == "priority":
        y_train, y_val = y_priority, vy_priority
    elif classifier_name == "preempt":
        y_train, y_val = y_preempt, vy_preempt
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        clf = xgb.XGBClassifier(
            **params,
            n_jobs=-1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        return float((preds == y_val).mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params
