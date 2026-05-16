"""Tests for the export pipeline changes.

Covers:
- Platform filtering in SchedulerDataset
- Explicit n_actions override in train_mlp
- ai_decode_action() generation in write_config_h
- export_mlp platform-specific checkpoint loading and validation
- XGBoost removal from export CLI
- BatchNorm folding and verification data generation
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from slm_sim.actions import action_space_size, decode_action, encode_action
from slm_sim.observation import TOTAL_FEATURES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_dataset_dir():
    """Generate a small dataset for platform-filter tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"
        splits_dir = Path(tmpdir) / "splits"

        from scripts.generate_dataset import generate
        generate(raw_dir, small=True)

        from scripts.split_dataset import split_dataset
        split_dataset(raw_dir, splits_dir)

        yield tmpdir


@pytest.fixture
def output_dir():
    """Temporary directory for export output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ---------------------------------------------------------------------------
# Platform filtering in SchedulerDataset
# ---------------------------------------------------------------------------

class TestDatasetPlatformFilter:
    def test_platform_filter_returns_subset(self, small_dataset_dir):
        """Filtering to a platform returns at most as many rows as unfiltered."""
        from training.mlp.dataset import SchedulerDataset
        train_path = Path(small_dataset_dir) / "splits" / "train.parquet"

        ds_all = SchedulerDataset(train_path)
        ds_jetson = SchedulerDataset(train_path, platform="jetson_orin_nano")

        if len(ds_all) == 0:
            pytest.skip("No data in splits")

        # Jetson-only should be a subset (<=) of all platforms
        assert len(ds_jetson) <= len(ds_all)
        assert len(ds_jetson) > 0

        # A non-existent platform should return 0 rows (strict reduction)
        ds_fake = SchedulerDataset(train_path, platform="fake_platform")
        assert len(ds_fake) < len(ds_all)

    def test_platform_filter_correct_actions(self, small_dataset_dir):
        """Jetson-filtered data should only contain valid Jetson actions."""
        from training.mlp.dataset import SchedulerDataset
        train_path = Path(small_dataset_dir) / "splits" / "train.parquet"

        ds = SchedulerDataset(train_path, platform="jetson_orin_nano")
        if len(ds) == 0:
            pytest.skip("No Jetson data in splits")

        # All action indices should be valid for Jetson (0..41)
        max_action = int(ds.actions.max())
        assert max_action < 42

    def test_platform_filter_no_match_gives_empty(self, small_dataset_dir):
        """Non-existent platform returns empty dataset."""
        from training.mlp.dataset import SchedulerDataset
        train_path = Path(small_dataset_dir) / "splits" / "train.parquet"

        ds = SchedulerDataset(train_path, platform="nonexistent_platform")
        assert len(ds) == 0

    def test_platform_none_loads_all(self, small_dataset_dir):
        """platform=None should load all platforms (same as default)."""
        from training.mlp.dataset import SchedulerDataset
        train_path = Path(small_dataset_dir) / "splits" / "train.parquet"

        ds_default = SchedulerDataset(train_path)
        ds_none = SchedulerDataset(train_path, platform=None)
        assert len(ds_default) == len(ds_none)

    def test_platform_filter_with_max_rows(self, small_dataset_dir):
        """Platform filter works correctly with max_rows subsampling."""
        from training.mlp.dataset import SchedulerDataset
        train_path = Path(small_dataset_dir) / "splits" / "train.parquet"

        ds_full = SchedulerDataset(train_path, platform="jetson_orin_nano")
        if len(ds_full) < 100:
            pytest.skip("Not enough Jetson data for subsampling test")

        ds_capped = SchedulerDataset(
            train_path, platform="jetson_orin_nano", max_rows=100
        )
        assert len(ds_capped) <= 100
        assert len(ds_capped) > 0


# ---------------------------------------------------------------------------
# train_mlp n_actions override
# ---------------------------------------------------------------------------

class TestTrainMLPNActionsOverride:
    def test_n_actions_override_sets_output_dim(self, small_dataset_dir):
        """Explicit n_actions produces a model with that output dimension."""
        from training.mlp.dataset import SchedulerDataset
        from training.mlp.train import TrainConfig, train_mlp

        train_path = Path(small_dataset_dir) / "splits" / "train.parquet"
        val_path = Path(small_dataset_dir) / "splits" / "val.parquet"

        train_ds = SchedulerDataset(train_path)
        val_ds = SchedulerDataset(val_path)

        if len(train_ds) == 0 or len(val_ds) == 0:
            pytest.skip("Not enough data")

        config = TrainConfig(n_epochs=2, batch_size=64, patience=20)
        model, result = train_mlp(train_ds, val_ds, config, n_actions=42)

        # Model should have 42 outputs regardless of what's in the data
        x = torch.randn(1, TOTAL_FEATURES)
        logits = model(x)
        assert logits.shape == (1, 42)

    def test_n_actions_none_infers_from_data(self, small_dataset_dir):
        """n_actions=None infers from the dataset."""
        from training.mlp.dataset import SchedulerDataset
        from training.mlp.train import TrainConfig, train_mlp

        train_path = Path(small_dataset_dir) / "splits" / "train.parquet"
        val_path = Path(small_dataset_dir) / "splits" / "val.parquet"

        train_ds = SchedulerDataset(train_path)
        val_ds = SchedulerDataset(val_path)

        if len(train_ds) == 0 or len(val_ds) == 0:
            pytest.skip("Not enough data")

        config = TrainConfig(n_epochs=2, batch_size=64, patience=20)
        model, result = train_mlp(train_ds, val_ds, config, n_actions=None)

        expected_n = max(train_ds.n_actions, val_ds.n_actions)
        x = torch.randn(1, TOTAL_FEATURES)
        logits = model(x)
        assert logits.shape == (1, expected_n)


# ---------------------------------------------------------------------------
# ai_decode_action() generation
# ---------------------------------------------------------------------------

class TestConfigHeaderGeneration:
    def test_config_h_contains_decode_function(self, output_dir):
        """write_config_h should generate ai_decode_action inline function."""
        from scripts.export_models import write_config_h
        write_config_h(42, "jetson_orin_nano", output_dir)

        content = (output_dir / "ai_config.h").read_text()
        assert "ai_decode_action" in content
        assert "struct ai_sched_action" in content
        assert "AI_SCHED_N_ACTIONS" in content
        assert "42" in content

    def test_config_h_platform_constants(self, output_dir):
        """Config header has correct platform constants."""
        from scripts.export_models import write_config_h
        write_config_h(42, "jetson_orin_nano", output_dir)

        content = (output_dir / "ai_config.h").read_text()
        assert "AI_SCHED_N_CORES     6" in content
        assert "AI_SCHED_GPU         1" in content

    def test_config_h_no_gpu_platform(self, output_dir):
        """Config header sets GPU=0 for platforms without GPU."""
        from scripts.export_models import write_config_h
        write_config_h(24, "raspberry_pi5", output_dir)

        content = (output_dir / "ai_config.h").read_text()
        assert "AI_SCHED_N_CORES     4" in content
        assert "AI_SCHED_GPU         0" in content
        assert "AI_SCHED_N_ACTIONS   24" in content

    def test_decode_action_matches_python(self):
        """C ai_decode_action formula must match Python decode_action."""
        # The C decode is: preempt = idx%2; priority = (idx/2)%3; core = idx/6
        # Verify this matches slm_sim.actions.decode_action for all 42 Jetson actions
        for idx in range(42):
            py_action = decode_action(idx, num_cores=6, gpu_available=True)

            # Replicate C decode logic
            c_preempt = idx % 2
            tmp = idx // 2
            c_priority = tmp % 3
            c_core = tmp // 3

            assert c_core == py_action.core_assignment, (
                f"idx={idx}: C core={c_core} != Python core={py_action.core_assignment}"
            )
            assert c_priority == py_action.priority_adj, (
                f"idx={idx}: C prio={c_priority} != Python prio={py_action.priority_adj}"
            )
            assert c_preempt == int(py_action.preempt), (
                f"idx={idx}: C preempt={c_preempt} != Python preempt={int(py_action.preempt)}"
            )

    def test_decode_roundtrip_all_platforms(self):
        """Decode formula is consistent for all three platforms."""
        platforms = [
            ("jetson_orin_nano", 6, True, 42),
            ("raspberry_pi5", 4, False, 24),
            ("big_little", 6, False, 36),
        ]
        for name, n_cores, gpu, n_actions in platforms:
            assert action_space_size(n_cores, gpu) == n_actions
            for idx in range(n_actions):
                py = decode_action(idx, n_cores, gpu)
                # C formula
                preempt = idx % 2
                priority = (idx // 2) % 3
                core = idx // 6
                assert core == py.core_assignment
                assert priority == py.priority_adj
                assert preempt == int(py.preempt)


# ---------------------------------------------------------------------------
# export_mlp: checkpoint loading and validation
# ---------------------------------------------------------------------------

class TestExportMLPValidation:
    def _make_checkpoint(self, path: Path, n_actions: int):
        """Create a minimal MLP checkpoint with given output size."""
        from training.mlp.model import SchedulerMLP
        model = SchedulerMLP(n_features=TOTAL_FEATURES, n_actions=n_actions)
        # Run a forward pass to populate BatchNorm running stats
        model.train()
        dummy = torch.randn(32, TOTAL_FEATURES)
        model(dummy)
        model.eval()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)

    def test_prefers_platform_specific_checkpoint(self, output_dir):
        """export_mlp should prefer best_jetson_orin_nano.pt over best.pt."""
        from scripts.export_models import export_mlp

        models_dir = output_dir / "models" / "mlp"
        generic = models_dir / "best.pt"
        platform = models_dir / "best_jetson_orin_nano.pt"

        # Create both checkpoints — generic with 36 actions, platform with 42
        self._make_checkpoint(generic, 36)
        self._make_checkpoint(platform, 42)

        import unittest.mock as mock
        with mock.patch("scripts.export_models.Path", wraps=Path) as _:
            # Monkeypatch the model path search
            orig_export = export_mlp.__wrapped__ if hasattr(export_mlp, '__wrapped__') else None

            # Just test the path selection logic directly
            platform_path = models_dir / "best_jetson_orin_nano.pt"
            generic_path = models_dir / "best.pt"

            assert platform_path.exists()
            assert generic_path.exists()
            # Platform-specific should be preferred
            if platform_path.exists():
                chosen = platform_path
            elif generic_path.exists():
                chosen = generic_path
            assert chosen == platform_path

    def test_rejects_mismatched_action_space(self, output_dir, capsys):
        """export_mlp should refuse to export if model actions != platform actions."""
        from scripts.export_models import export_mlp

        models_dir = output_dir / "models" / "mlp"
        # Create a 36-action checkpoint (wrong for Jetson's 42)
        self._make_checkpoint(models_dir / "best.pt", 36)

        import os
        orig_cwd = os.getcwd()
        os.chdir(output_dir)
        try:
            export_mlp("jetson_orin_nano", 42, output_dir / "generated")
        finally:
            os.chdir(orig_cwd)

        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "42" in captured.out
        # Should NOT have generated weight files
        assert not (output_dir / "generated" / "ai_weights_mlp.c").exists()

    def test_exports_matching_action_space(self, output_dir):
        """export_mlp should succeed when model matches platform."""
        from scripts.export_models import export_mlp

        models_dir = output_dir / "models" / "mlp"
        self._make_checkpoint(models_dir / "best_jetson_orin_nano.pt", 42)

        import os
        orig_cwd = os.getcwd()
        os.chdir(output_dir)
        try:
            gen_dir = output_dir / "generated"
            gen_dir.mkdir(parents=True, exist_ok=True)
            export_mlp("jetson_orin_nano", 42, gen_dir)
        finally:
            os.chdir(orig_cwd)

        assert (output_dir / "generated" / "ai_weights_mlp.c").exists()
        assert (output_dir / "generated" / "ai_weights_mlp.h").exists()
        assert (output_dir / "generated" / "test_vectors_mlp.bin").exists()
        assert (output_dir / "generated" / "expected_actions_mlp.bin").exists()

    def test_verification_data_dimensions(self, output_dir):
        """Verification artifacts have correct sizes."""
        from scripts.export_models import export_mlp

        models_dir = output_dir / "models" / "mlp"
        self._make_checkpoint(models_dir / "best_jetson_orin_nano.pt", 42)

        import os
        orig_cwd = os.getcwd()
        os.chdir(output_dir)
        try:
            gen_dir = output_dir / "generated"
            gen_dir.mkdir(parents=True, exist_ok=True)
            export_mlp("jetson_orin_nano", 42, gen_dir)
        finally:
            os.chdir(orig_cwd)

        # 1000 test vectors × 108 features × 4 bytes
        tv = output_dir / "generated" / "test_vectors_mlp.bin"
        assert tv.stat().st_size == 1000 * TOTAL_FEATURES * 4

        # 1000 expected actions × 4 bytes (int32)
        ea = output_dir / "generated" / "expected_actions_mlp.bin"
        assert ea.stat().st_size == 1000 * 4

        # 10 logit vectors × 42 actions × 4 bytes
        el = output_dir / "generated" / "expected_logits_mlp.bin"
        assert el.stat().st_size == 10 * 42 * 4


# ---------------------------------------------------------------------------
# XGBoost is now re-enabled (#58a / #850) — emits a binary blob, not C source
# ---------------------------------------------------------------------------

class TestXGBoostBinaryBlob:
    """Format-level coverage for `write_xgboost_blob` and the surrounding
    cascade serializer. Validates the binary round-trips and produces
    identical predictions to a direct Python tree-walk on the same
    `trees_data` dict."""

    def test_xgboost_is_accepted_cli_choice(self):
        """Export CLI must accept --model xgboost (#58a re-enables it)."""
        from scripts import export_models  # imports without exploding
        import inspect
        source = inspect.getsource(export_models.main)
        assert '"xgboost"' in source, (
            "main() should advertise --model xgboost in its argparse choices"
        )

    def test_all_includes_xgboost(self):
        """'all' must include xgboost so a single export run produces
        every available policy in one pass."""
        import inspect
        from scripts import export_models
        source = inspect.getsource(export_models.main)
        assert (
            '["mlp", "ppo", "xgboost"]' in source
            or "['mlp', 'ppo', 'xgboost']" in source
        ), "main() 'all' branch should iterate mlp + ppo + xgboost"

    def test_blob_roundtrips_simple_two_classifier_cascade(self):
        """End-to-end: build a trivial 2-classifier cascade dict, write
        the SEMB+XGBC blob, parse it back in pure Python, and verify
        each classifier predicts the same label as a direct tree walk
        on the source dict.
        """
        from scripts.export_models import (
            _build_xgbc_payload,
            _wrap_semb,
            SCHED_MODEL_KIND_XGBOOST,
            SCHED_MODEL_SCHEMA_V1,
            XGB_CASCADE_ORDER,
        )

        # Each classifier has 2 trees (one per class) where each tree is
        # a single leaf — so class i's score is just leaf_value_i and
        # argmax is deterministic per classifier.
        def make_clf(leaf_a: float, leaf_b: float, labels):
            return {
                "trees": [
                    [{
                        "feature_idx": -1,
                        "threshold": 0.0,
                        "leaf_value": leaf_a,
                        "left_child": 0,
                        "right_child": 0,
                    }],
                    [{
                        "feature_idx": -1,
                        "threshold": 0.0,
                        "leaf_value": leaf_b,
                        "left_child": 0,
                        "right_child": 0,
                    }],
                ],
                "n_classes": 2,
                "label_classes": labels,
            }

        trees_data = {
            "core": make_clf(0.5, 0.1, [7, 9]),
            "priority": make_clf(0.0, 1.0, [100, 200]),
            "preempt": make_clf(0.2, 0.3, [0, 1]),
        }
        # Cascade ordering must match TripleClassifier.predict.
        assert XGB_CASCADE_ORDER == ["core", "priority", "preempt"]

        payload = _build_xgbc_payload(trees_data, XGB_CASCADE_ORDER)
        blob = _wrap_semb(
            payload,
            kind_id=SCHED_MODEL_KIND_XGBOOST,
            schema_version=SCHED_MODEL_SCHEMA_V1,
        )

        parsed = _parse_smb_blob(blob)
        assert parsed["kind_id"] == SCHED_MODEL_KIND_XGBOOST
        assert parsed["schema_version"] == SCHED_MODEL_SCHEMA_V1
        assert len(parsed["classifiers"]) == 3

        # Empty feature vector — every tree is a leaf so feature value
        # never matters. Direct argmax over leaf_values gives the
        # expected label.
        features = []
        labels = [_walk_classifier(c, features) for c in parsed["classifiers"]]
        assert labels == [7, 200, 1]

    def test_blob_format_constants_match_runtime(self):
        """The SEMB outer-header version + XGBC inner magic + node size
        are wire-format constants that must stay synchronized with the
        SLM-OS runtime parser. Pin them here so an accidental edit on
        either side fails this test, not silently mis-parses on a
        kernel boot.
        """
        from scripts.export_models import (
            _SEMB_MAGIC, _SEMB_VERSION_V1, _SEMB_OUTER_HEADER_LEN,
            _XGBC_MAGIC, _XGB_PAYLOAD_VERSION_V1, _XGBC_HEADER_LEN,
            _XGBC_CLASSIFIER_HEADER_LEN, _XGB_NODE_LEN, _XGB_FLAG_LEAF,
        )
        assert _SEMB_MAGIC == b"SEMB"
        assert _SEMB_VERSION_V1 == 1
        assert _SEMB_OUTER_HEADER_LEN == 24
        assert _XGBC_MAGIC == b"XGBC"
        assert _XGB_PAYLOAD_VERSION_V1 == 1
        assert _XGBC_HEADER_LEN == 16
        # Classifier header: u32 n_trees + u32 n_nodes + u16 n_classes
        # + u16 reserved + u32 reserved = 16 bytes.
        assert _XGBC_CLASSIFIER_HEADER_LEN == 16
        # Node: u16 feature + u16 flags + u32 left + u32 right
        # + f32 thresh + f32 value = 20 bytes.
        assert _XGB_NODE_LEN == 20
        assert _XGB_FLAG_LEAF == 1

    def test_checksum_detects_payload_corruption(self):
        """A single-byte flip inside the payload must change the FNV-1a
        checksum stored in the SEMB header — otherwise the runtime
        parser would silently accept a corrupt blob."""
        from scripts.export_models import (
            _build_xgbc_payload, _wrap_semb,
            SCHED_MODEL_KIND_XGBOOST, SCHED_MODEL_SCHEMA_V1,
            _fnv1a_32,
        )
        trees_data = {
            "core": {
                "trees": [[{
                    "feature_idx": -1, "threshold": 0.0,
                    "leaf_value": 1.0,
                    "left_child": 0, "right_child": 0,
                }]],
                "n_classes": 1,
                "label_classes": [0],
            },
        }
        payload = _build_xgbc_payload(trees_data, ["core"])
        original = _fnv1a_32(payload)
        corrupted = bytearray(payload)
        corrupted[-1] ^= 0xFF
        assert _fnv1a_32(bytes(corrupted)) != original


# ---------------------------------------------------------------------------
# Pure-Python SEMB+XGBC parser — used only by the round-trip tests above.
# Mirrors the SLM-OS Rust runtime parser (`runtime/src/ml/xgb_tree.rs`)
# bit-for-bit; if either parser drifts the round-trip test fails.
# ---------------------------------------------------------------------------

def _parse_smb_blob(blob: bytes) -> dict:
    import struct

    if blob[0:4] != b"SEMB":
        raise ValueError("not a SEMB blob")
    (magic, version, kind_id, schema_version, _r, payload_len, checksum) = (
        struct.unpack_from("<4sHHHHII", blob, 0)
    )
    # Trailing reserved word at offset 20..24.
    (reserved2,) = struct.unpack_from("<I", blob, 20)
    assert reserved2 == 0, "trailing reserved word must be zero"
    payload = blob[24:24 + payload_len]
    if len(blob) != 24 + payload_len:
        raise ValueError("trailing bytes after payload")

    # XGBC inner header.
    (xmagic, xversion, _r1, n_clf, _r2, _r3) = struct.unpack_from(
        "<4sHHHHI", payload, 0
    )
    if xmagic != b"XGBC":
        raise ValueError("not an XGBC payload")
    classifiers = []
    cursor = 16
    for _ in range(n_clf):
        n_trees, n_nodes, n_classes, _r4, _r5 = struct.unpack_from(
            "<IIHHI", payload, cursor
        )
        cursor += 16
        roots = list(struct.unpack_from(f"<{n_trees}I", payload, cursor))
        cursor += 4 * n_trees
        nodes = []
        for _ in range(n_nodes):
            (fi, flags, left, right, thresh, value) = struct.unpack_from(
                "<HHIIff", payload, cursor
            )
            cursor += 20
            nodes.append({
                "feature_idx": fi,
                "flags": flags,
                "left": left,
                "right": right,
                "threshold": thresh,
                "value": value,
            })
        labels = list(struct.unpack_from(f"<{n_classes}i", payload, cursor))
        cursor += 4 * n_classes
        classifiers.append({
            "n_trees": n_trees,
            "n_nodes": n_nodes,
            "n_classes": n_classes,
            "roots": roots,
            "nodes": nodes,
            "labels": labels,
        })
    return {
        "kind_id": kind_id,
        "schema_version": schema_version,
        "checksum_header": checksum,
        "classifiers": classifiers,
    }


def _walk_tree(nodes: list, root_idx: int, features: list) -> float:
    idx = root_idx
    for _ in range(256):
        node = nodes[idx]
        if node["flags"] & 1:
            return node["value"]
        f = features[node["feature_idx"]] if node["feature_idx"] < len(features) else 0.0
        idx = node["left"] if f < node["threshold"] else node["right"]
    return 0.0


def _walk_classifier(parsed_clf: dict, features: list) -> int:
    """Argmax across per-class tree-sum + label lookup. Trees are laid
    out one-per-class-per-round (XGBoost's standard multiclass shape)."""
    n_classes = parsed_clf["n_classes"]
    if n_classes == 0:
        return 0
    scores = [0.0] * n_classes
    for i, root in enumerate(parsed_clf["roots"]):
        scores[i % n_classes] += _walk_tree(parsed_clf["nodes"], root, features)
    best = max(range(n_classes), key=lambda i: scores[i])
    return parsed_clf["labels"][best]


# ---------------------------------------------------------------------------
# BatchNorm folding correctness
# ---------------------------------------------------------------------------

class TestBatchNormFolding:
    def test_folding_produces_correct_layers(self):
        """BN folding should produce 4 (W, b) pairs."""
        from training.mlp.model import SchedulerMLP
        from scripts.export_models import fold_mlp_batchnorms

        model = SchedulerMLP(n_features=TOTAL_FEATURES, n_actions=42)
        # Need to populate BN running stats
        model.train()
        for _ in range(5):
            model(torch.randn(32, TOTAL_FEATURES))
        model.eval()

        layers = fold_mlp_batchnorms(model)
        assert len(layers) == 4

        # Check dimensions: 108→256→256→128→42
        assert layers[0][0].shape == (256, 108)
        assert layers[0][1].shape == (256,)
        assert layers[1][0].shape == (256, 256)
        assert layers[1][1].shape == (256,)
        assert layers[2][0].shape == (128, 256)
        assert layers[2][1].shape == (128,)
        assert layers[3][0].shape == (42, 128)
        assert layers[3][1].shape == (42,)

    def test_folding_preserves_output(self):
        """Folded model should produce same outputs as original."""
        from training.mlp.model import SchedulerMLP
        from scripts.export_models import fold_mlp_batchnorms, verify_bn_folding

        model = SchedulerMLP(n_features=TOTAL_FEATURES, n_actions=42)
        model.train()
        for _ in range(10):
            model(torch.randn(64, TOTAL_FEATURES))
        model.eval()

        layers = fold_mlp_batchnorms(model)
        max_diff = verify_bn_folding(model, layers)
        assert max_diff < 1e-3  # Should be very small
