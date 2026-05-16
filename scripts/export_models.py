#!/usr/bin/env python3
"""Export trained models to C deployment format.

MLP:     BatchNorm folding → 4-layer weights → C arrays
PPO:     Actor extraction → 4-layer weights → C arrays
XGBoost: Tree node arrays + label encoders + derived features

Usage:
    python scripts/export_models.py --model mlp --platform jetson_orin_nano
    python scripts/export_models.py --model all --platform jetson_orin_nano
    python scripts/export_models.py --model mlp --platform jetson_orin_nano \
        --output-dir ~/projects/CS-496-Capstone-SLM-Operating-System/kernel/sched/ai/
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slm_sim.actions import action_space_size
from slm_sim.observation import TOTAL_FEATURES
from slm_sim.platforms import get_platform


# ---------------------------------------------------------------------------
# BatchNorm folding
# ---------------------------------------------------------------------------

def fold_bn_into_next_linear(
    bn: nn.BatchNorm1d,
    linear: nn.Linear,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold a BatchNorm layer into the following Linear layer.

    Given y_bn = gamma * (x - mean) / sqrt(var + eps) + beta,
    and z = W_next @ y_bn + b_next,
    returns (W_new, b_new) such that z = W_new @ x + b_new.
    """
    gamma = bn.weight.data          # (C,)
    beta = bn.bias.data             # (C,)
    mean = bn.running_mean.data     # (C,)
    var = bn.running_var.data       # (C,)
    eps = bn.eps

    # BN scale and shift: y_bn = scale * x + shift
    scale = gamma / torch.sqrt(var + eps)   # (C,)
    shift = beta - scale * mean             # (C,)

    # Fold into next linear: z = W @ (scale * x + shift) + b
    #                          = (W * scale) @ x + (W @ shift + b)
    W = linear.weight.data          # (out, in)
    b = linear.bias.data            # (out,)

    W_new = W * scale.unsqueeze(0)  # broadcast: (out, in) * (1, in)
    b_new = W @ shift + b           # (out,)

    return W_new, b_new


def fold_mlp_batchnorms(model: nn.Module) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Fold all BatchNorm layers in the MLP into adjacent Linear layers.

    Returns list of (W, b) tuples for the 4 inference layers:
      Layer 0: original Linear[0] (before first BN)
      Layer 1: BN[2] folded into Linear[4]
      Layer 2: BN[6] folded into Linear[8]
      Layer 3: BN[10] folded into Linear[11]
    """
    model.eval()
    seq = model.network

    # Layer 0: Linear(108, 256) — no BN before it
    layer0_W = seq[0].weight.data.clone()
    layer0_b = seq[0].bias.data.clone()

    # BN at index 2, fold into Linear at index 4
    layer1_W, layer1_b = fold_bn_into_next_linear(seq[2], seq[4])

    # BN at index 6, fold into Linear at index 8
    layer2_W, layer2_b = fold_bn_into_next_linear(seq[6], seq[8])

    # BN at index 10, fold into Linear at index 11
    layer3_W, layer3_b = fold_bn_into_next_linear(seq[10], seq[11])

    return [
        (layer0_W, layer0_b),
        (layer1_W, layer1_b),
        (layer2_W, layer2_b),
        (layer3_W, layer3_b),
    ]


def verify_bn_folding(
    model: nn.Module,
    folded_layers: list[tuple[torch.Tensor, torch.Tensor]],
    n_tests: int = 1000,
) -> float:
    """Verify that the folded model produces identical outputs.

    Returns max absolute difference across all test cases.
    """
    model.eval()
    rng = torch.manual_seed(12345)
    test_inputs = torch.rand(n_tests, model.n_features)

    with torch.no_grad():
        expected = model(test_inputs)

        # Run folded inference
        x = test_inputs
        for i, (W, b) in enumerate(folded_layers):
            x = x @ W.T + b
            if i < len(folded_layers) - 1:
                x = torch.relu(x)

    max_diff = (expected - x).abs().max().item()
    return max_diff


# ---------------------------------------------------------------------------
# PPO actor extraction
# ---------------------------------------------------------------------------

def extract_ppo_actor_layers(
    model_path: Path,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Extract the 4-layer actor network from an SB3 PPO checkpoint."""
    from stable_baselines3 import PPO

    ppo = PPO.load(str(model_path), device="cpu")
    policy = ppo.policy
    policy.eval()

    extractor = policy.mlp_extractor

    # Shared backbone
    shared = extractor.shared
    # shared is Sequential: Linear(108,256), ReLU, Linear(256,256), ReLU
    layer0_W = shared[0].weight.data.clone()
    layer0_b = shared[0].bias.data.clone()
    layer1_W = shared[2].weight.data.clone()
    layer1_b = shared[2].bias.data.clone()

    # Actor head
    actor_head = extractor.policy_net
    # policy_net is Sequential: Linear(256,128), ReLU
    layer2_W = actor_head[0].weight.data.clone()
    layer2_b = actor_head[0].bias.data.clone()

    # Final action layer
    action_net = policy.action_net
    layer3_W = action_net.weight.data.clone()
    layer3_b = action_net.bias.data.clone()

    return [
        (layer0_W, layer0_b),
        (layer1_W, layer1_b),
        (layer2_W, layer2_b),
        (layer3_W, layer3_b),
    ]


def verify_ppo_actor(
    model_path: Path,
    layers: list[tuple[torch.Tensor, torch.Tensor]],
    n_tests: int = 1000,
) -> float:
    """Verify extracted PPO actor matches SB3 inference."""
    from stable_baselines3 import PPO

    ppo = PPO.load(str(model_path), device="cpu")
    policy = ppo.policy
    policy.eval()

    test_inputs = torch.rand(n_tests, TOTAL_FEATURES)

    with torch.no_grad():
        # SB3 actor forward path
        features = policy.features_extractor(test_inputs)
        latent_pi, _ = policy.mlp_extractor(features)
        expected = policy.action_net(latent_pi)

        # Folded forward
        x = test_inputs
        for i, (W, b) in enumerate(layers):
            x = x @ W.T + b
            if i < len(layers) - 1:
                x = torch.relu(x)

    return (expected - x).abs().max().item()


# ---------------------------------------------------------------------------
# XGBoost export
# ---------------------------------------------------------------------------

def export_xgboost_trees(model_dir: Path) -> dict:
    """Export XGBoost classifiers to compact node arrays.

    Returns dict with keys: core, priority, preempt.
    Each value is a dict with 'nodes' (list of list of node dicts) and
    'label_classes' (list of original label values).
    """
    import xgboost as xgb

    with open(model_dir / "meta.json") as f:
        meta = json.load(f)

    result = {}
    for name, clf_file, classes_key in [
        ("core", "core_clf.json", "core_classes"),
        ("priority", "priority_clf.json", "priority_classes"),
        ("preempt", "preempt_clf.json", "preempt_classes"),
    ]:
        clf = xgb.XGBClassifier()
        clf.load_model(model_dir / clf_file)
        booster = clf.get_booster()

        # Get tree dump as JSON
        tree_dump = booster.get_dump(dump_format="json")
        all_trees = []
        for tree_json in tree_dump:
            tree = json.loads(tree_json)
            nodes = []
            _flatten_tree(tree, nodes)
            all_trees.append(nodes)

        result[name] = {
            "trees": all_trees,
            "n_classes": len(meta[classes_key]),
            "label_classes": meta[classes_key],
        }

    return result


def _flatten_tree(node: dict, nodes: list, idx: int = 0) -> int:
    """Recursively flatten an XGBoost tree node into a list.

    Returns the next available index.
    """
    current_idx = len(nodes)
    if "leaf" in node:
        nodes.append({
            "feature_idx": -1,
            "threshold": 0.0,
            "leaf_value": node["leaf"],
            "left_child": 0,
            "right_child": 0,
        })
        return current_idx + 1

    # Interior node — reserve this slot
    nodes.append(None)

    # Recurse left ("yes" branch) and right ("no" branch)
    left_start = len(nodes)
    _flatten_tree(node["children"][0], nodes)
    right_start = len(nodes)
    _flatten_tree(node["children"][1], nodes)

    # Fill in this node
    nodes[current_idx] = {
        "feature_idx": int(node["split"].replace("f", "")),
        "threshold": float(node["split_condition"]),
        "leaf_value": 0.0,
        "left_child": left_start,
        "right_child": right_start,
    }
    return len(nodes)


# ---------------------------------------------------------------------------
# C code generation
# ---------------------------------------------------------------------------

def float_to_hex(f: float) -> str:
    """Convert float to C99 hexadecimal float literal for bit-exact repr."""
    return float.hex(f)

def write_nn_weights_c(
    layers: list[tuple[torch.Tensor, torch.Tensor]],
    model_name: str,
    output_dir: Path,
) -> None:
    """Write neural network weights as C source and header files."""
    prefix = model_name  # e.g., "mlp" or "ppo"

    # Header file
    h_lines = [
        f"/* ai_weights_{prefix}.h — Auto-generated by export_models.py */",
        f"#ifndef AI_WEIGHTS_{prefix.upper()}_H",
        f"#define AI_WEIGHTS_{prefix.upper()}_H",
        "",
        f"#define {prefix.upper()}_N_LAYERS 4",
        "",
    ]
    for i, (W, b) in enumerate(layers):
        out_sz, in_sz = W.shape
        h_lines.append(f"#define {prefix.upper()}_L{i}_IN  {in_sz}")
        h_lines.append(f"#define {prefix.upper()}_L{i}_OUT {out_sz}")
        h_lines.append(f"extern const float {prefix}_w{i}[{out_sz} * {in_sz}];")
        h_lines.append(f"extern const float {prefix}_b{i}[{out_sz}];")
        h_lines.append("")

    h_lines.append(f"#endif /* AI_WEIGHTS_{prefix.upper()}_H */")
    h_lines.append("")

    (output_dir / f"ai_weights_{prefix}.h").write_text("\n".join(h_lines))

    # Source file
    c_lines = [
        f"/* ai_weights_{prefix}.c — Auto-generated by export_models.py */",
        f'#include "ai_weights_{prefix}.h"',
        "",
    ]
    for i, (W, b) in enumerate(layers):
        out_sz, in_sz = W.shape
        W_flat = W.detach().cpu().numpy().flatten()
        b_flat = b.detach().cpu().numpy().flatten()

        c_lines.append(f"__attribute__((aligned(64)))")
        c_lines.append(f"const float {prefix}_w{i}[{out_sz} * {in_sz}] = {{")
        _write_float_array(c_lines, W_flat)
        c_lines.append("};")
        c_lines.append("")

        c_lines.append(f"__attribute__((aligned(64)))")
        c_lines.append(f"const float {prefix}_b{i}[{out_sz}] = {{")
        _write_float_array(c_lines, b_flat)
        c_lines.append("};")
        c_lines.append("")

    (output_dir / f"ai_weights_{prefix}.c").write_text("\n".join(c_lines))


def _write_float_array(lines: list[str], arr: np.ndarray, cols: int = 4) -> None:
    """Write float array values as hex literals, `cols` per line."""
    values = arr.tolist()
    for i in range(0, len(values), cols):
        chunk = values[i:i + cols]
        formatted = ", ".join(float_to_hex(v) for v in chunk)
        suffix = "," if i + cols < len(values) else ""
        lines.append(f"    {formatted}{suffix}")


# ---------------------------------------------------------------------------
# XGBoost binary blob exporter (SLM-OS scheduler runtime-load format)
# ---------------------------------------------------------------------------

# Outer header constants — must match runtime/src/mm/eviction/blob.rs in the
# SLM-OS repo. Layout is shared between the eviction store (kind 1..3) and
# the scheduler model store (kind 0x1001..0x1006). The scheduler XGBoost
# kind id is 0x1006 (SCHED_MODEL_KIND_XGBOOST), introduced in #58c.
_SEMB_MAGIC = b"SEMB"
_SEMB_VERSION_V1 = 1
_SEMB_OUTER_HEADER_LEN = 24

# Sched-side blob-kind ids. Coordinated with kernel/sched/ai/runtime_model.h.
SCHED_MODEL_KIND_XGBOOST = 0x1006
SCHED_MODEL_SCHEMA_V1 = 1

# Inner cascade payload — must match runtime/src/ml/xgb_tree.rs (XgbCascade).
_XGBC_MAGIC = b"XGBC"
_XGB_PAYLOAD_VERSION_V1 = 1
_XGBC_HEADER_LEN = 16
# Per-classifier header: u32 n_trees, u32 n_nodes, u16 n_classes, u16 r,
# u32 reserved.
_XGBC_CLASSIFIER_HEADER_LEN = 16
# Per-node record layout (cascade form, u32 children — must match
# `runtime/src/ml/xgb_tree.rs::CASCADE_NODE_LEN_V1`). All fields LE.
#   u16 feature_idx, u16 flags, u32 left, u32 right, f32 threshold, f32 value
_XGB_NODE_LEN = 20
_XGB_FLAG_LEAF = 1


def _fnv1a_32(data: bytes) -> int:
    """FNV-1a 32-bit checksum. Mirrors blob.rs::checksum32."""
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _xgb_pack_node(node: dict) -> bytes:
    """Pack a flattened XGBoost node into the 20-byte cascade record.

    `_flatten_tree` emits dicts with `feature_idx, threshold, leaf_value,
    left_child, right_child`. Leaves are tagged with `feature_idx == -1`
    by the flattener; here we set the FLAG_LEAF bit and store the leaf
    value in the `value` field. Interior nodes carry the split feature
    index + threshold and the left/right child offsets within the
    classifier's flat node array.

    Children are u32 — required to address the trained `core_clf`
    (~450 K nodes) which overflows the eviction-side u16 wire format.
    """
    is_leaf = node["feature_idx"] < 0
    if is_leaf:
        return struct.pack(
            "<HHIIff",
            0,                          # feature_idx unused
            _XGB_FLAG_LEAF,
            0,                          # left unused
            0,                          # right unused
            0.0,                        # threshold unused
            float(node["leaf_value"]),
        )
    feature_idx = int(node["feature_idx"])
    if feature_idx < 0 or feature_idx > 0xFFFF:
        raise ValueError(f"feature_idx {feature_idx} out of u16 range")
    left = int(node["left_child"])
    right = int(node["right_child"])
    if max(left, right) > 0xFFFFFFFF:
        raise ValueError(
            f"node child index out of u32 range (left={left}, right={right})"
        )
    return struct.pack(
        "<HHIIff",
        feature_idx,
        0,
        left,
        right,
        float(node["threshold"]),
        0.0,
    )


# Cascade limits — must mirror runtime/src/ml/xgb_tree.rs.
_XGB_MAX_CLASSIFIERS = 8
_XGB_MAX_TREES_CASCADE = 16_384
_XGB_MAX_NODES_CASCADE = 2_000_000
_XGB_MAX_LABEL_CLASSES = 64


def _build_classifier_section(clf_data: dict) -> bytes:
    """Serialize one classifier (header + roots + nodes + label map)."""
    trees = clf_data["trees"]
    label_classes = list(clf_data["label_classes"])
    n_classes = len(label_classes)

    # Flatten trees into one per-classifier node array. `tree_offsets`
    # gives the index in the flat array where each tree's root sits.
    tree_offsets: list[int] = []
    flat_nodes: list[dict] = []
    for nodes in trees:
        offset = len(flat_nodes)
        tree_offsets.append(offset)
        # Re-base every interior node's child indices into the flat array.
        for node in nodes:
            if node["feature_idx"] < 0:
                flat_nodes.append(node)
            else:
                flat_nodes.append({
                    "feature_idx": node["feature_idx"],
                    "threshold": node["threshold"],
                    "leaf_value": 0.0,
                    "left_child": node["left_child"] + offset,
                    "right_child": node["right_child"] + offset,
                })

    n_trees = len(trees)
    n_nodes = len(flat_nodes)
    if n_trees == 0 or n_nodes == 0:
        raise ValueError("classifier has no trees or nodes")
    if n_trees > _XGB_MAX_TREES_CASCADE:
        raise ValueError(
            f"classifier has {n_trees} trees — exceeds runtime cap "
            f"{_XGB_MAX_TREES_CASCADE} (raise MAX_TREES_CASCADE in xgb_tree.rs)"
        )
    if n_nodes > _XGB_MAX_NODES_CASCADE:
        raise ValueError(
            f"classifier has {n_nodes} nodes — exceeds runtime cap "
            f"{_XGB_MAX_NODES_CASCADE} (raise MAX_NODES_CASCADE in xgb_tree.rs)"
        )
    if n_classes > _XGB_MAX_LABEL_CLASSES:
        raise ValueError(
            f"too many label classes: {n_classes} > {_XGB_MAX_LABEL_CLASSES}"
        )

    out = bytearray()
    out += struct.pack(
        "<IIHHI",
        n_trees,
        n_nodes,
        n_classes,
        0,                  # reserved (u16)
        0,                  # reserved (u32)
    )
    for off in tree_offsets:
        out += struct.pack("<I", off)
    for node in flat_nodes:
        out += _xgb_pack_node(node)
    for cls in label_classes:
        out += struct.pack("<i", int(cls))
    return bytes(out)


def _build_xgbc_payload(trees_data: dict, classifier_order: list[str]) -> bytes:
    """Build the XGBC cascade payload (header + N classifier sections)."""
    n = len(classifier_order)
    if n == 0:
        raise ValueError("cascade has zero classifiers")
    if n > _XGB_MAX_CLASSIFIERS:
        raise ValueError(
            f"cascade has {n} classifiers — exceeds runtime cap "
            f"{_XGB_MAX_CLASSIFIERS} (raise MAX_CLASSIFIERS in xgb_tree.rs)"
        )
    header = struct.pack(
        "<4sHHHHI",
        _XGBC_MAGIC,
        _XGB_PAYLOAD_VERSION_V1,
        0,                  # reserved
        n,
        0,                  # reserved
        0,                  # reserved
    )
    sections = b"".join(
        _build_classifier_section(trees_data[name])
        for name in classifier_order
    )
    return header + sections


def _wrap_semb(payload: bytes, kind_id: int, schema_version: int) -> bytes:
    """Wrap an inner payload in the outer SEMB header expected by the
    scheduler model store (kernel/sched/ai/runtime_model.c).
    """
    if not (0 <= kind_id <= 0xFFFF):
        raise ValueError(f"kind_id {kind_id} out of u16 range")
    if not (0 <= schema_version <= 0xFFFF):
        raise ValueError(f"schema_version {schema_version} out of u16 range")
    payload_len = len(payload)
    checksum = _fnv1a_32(payload)
    # SEMB header layout (24 bytes total — must match
    # `runtime/src/mm/eviction/blob.rs::HEADER_LEN`):
    #   off  0..4   "SEMB" magic
    #   off  4..6   u16  version
    #   off  6..8   u16  kind_id
    #   off  8..10  u16  schema_version
    #   off 10..12  u16  reserved (must be 0)
    #   off 12..16  u32  payload_len
    #   off 16..20  u32  checksum (FNV-1a 32 over payload)
    #   off 20..24  u32  reserved (must be 0)
    # `<4sHHHHII` packs the first 20 bytes; the trailing reserved
    # u32 is appended separately so the layout is greppable against
    # the field comments above.
    header = struct.pack(
        "<4sHHHHII",
        _SEMB_MAGIC,
        _SEMB_VERSION_V1,
        kind_id,
        schema_version,
        0,                  # reserved (offset 10..12)
        payload_len,
        checksum,
    )
    header += struct.pack("<I", 0)  # reserved (offset 20..24)
    return header + payload


# Cascade ordering must match TripleClassifier.predict in
# training/xgboost/train.py: core → priority(+core) → preempt(+core+priority).
# The on-target Rust predictor walks classifiers in this order and appends
# each prediction to the feature vector before invoking the next.
XGB_CASCADE_ORDER = ["core", "priority", "preempt"]


def write_xgboost_blob(
    trees_data: dict,
    output_dir: Path,
    blob_name: str = "xgb_sched.smb",
) -> Path:
    """Write the trained XGBoost cascade as a single binary blob suitable
    for runtime loading via `slm.sched_model_stage("xgboost", path)`.

    Replaces the rejected Plan A C-source form. Returns the path written.
    """
    payload = _build_xgbc_payload(trees_data, XGB_CASCADE_ORDER)
    blob = _wrap_semb(
        payload,
        kind_id=SCHED_MODEL_KIND_XGBOOST,
        schema_version=SCHED_MODEL_SCHEMA_V1,
    )
    out_path = output_dir / blob_name
    out_path.write_bytes(blob)

    total_trees = sum(len(trees_data[n]["trees"]) for n in XGB_CASCADE_ORDER)
    total_nodes = sum(
        sum(len(t) for t in trees_data[n]["trees"])
        for n in XGB_CASCADE_ORDER
    )
    print(
        f"  XGBoost blob: {len(blob):,} bytes "
        f"({total_trees} trees, {total_nodes:,} nodes across "
        f"{len(XGB_CASCADE_ORDER)} classifiers)"
    )
    return out_path


# ---------------------------------------------------------------------------
# XGBoost verification artifacts
# ---------------------------------------------------------------------------

def _load_triple_classifier(model_dir: Path):
    """Load the trained TripleClassifier the cascade was emitted from.
    Used only to compute expected-action ground truth for verification.
    """
    from training.xgboost.train import TripleClassifier
    return TripleClassifier.load(model_dir)


def _add_derived_features_numpy(states: np.ndarray) -> np.ndarray:
    """Wrapper around training.xgboost.features.add_derived_features that
    keeps this module's import surface flat. Returns (N, 113)."""
    from training.xgboost.features import add_derived_features
    return add_derived_features(states)


def generate_xgb_verification_data(
    model_dir: Path,
    output_dir: Path,
    n_tests: int = 1000,
) -> None:
    """Generate `test_vectors_xgb.bin` (raw 108-dim states), the matching
    `expected_actions_xgb.bin` (Python ground-truth `(core, priority,
    preempt)` triples), and `expected_logits_xgb.bin` (per-classifier
    raw scores for the first 10 vectors — debugging aid).

    The 108-dim states are written raw so SLM-OS can compute the 5
    derived features in-kernel and self-check the derivation logic.
    """
    triple = _load_triple_classifier(model_dir)
    rng = np.random.default_rng(54321)
    states = rng.random((n_tests, TOTAL_FEATURES), dtype=np.float32)

    X = _add_derived_features_numpy(states)
    core, priority, preempt = triple.predict(X)
    actions = np.stack(
        [
            core.astype(np.int32),
            priority.astype(np.int32),
            preempt.astype(np.int32),
        ],
        axis=1,
    )

    states.tofile(output_dir / "test_vectors_xgb.bin")
    actions.tofile(output_dir / "expected_actions_xgb.bin")

    # Per-classifier raw scores for the first 10 vectors. Each row is
    # `core_logits || priority_logits || preempt_logits` flattened, so
    # row stride depends on how many classes each classifier has.
    debug_logits: list[np.ndarray] = []
    X10 = X[:10]
    core_enc = triple.core_clf.predict(X10)
    core_orig = triple.core_le.inverse_transform(core_enc)
    X10_c = np.hstack([X10, core_orig.reshape(-1, 1)])
    prio_enc = triple.priority_clf.predict(X10_c)
    prio_orig = triple.priority_le.inverse_transform(prio_enc)
    X10_cp = np.hstack([X10_c, prio_orig.reshape(-1, 1)])
    debug_logits.append(triple.core_clf.predict_proba(X10).astype(np.float32))
    debug_logits.append(
        triple.priority_clf.predict_proba(X10_c).astype(np.float32)
    )
    debug_logits.append(
        triple.preempt_clf.predict_proba(X10_cp).astype(np.float32)
    )
    # Concatenate by row: each test vector's row is the per-classifier
    # probability vectors back-to-back. Use a fixed-width record so
    # the consuming side can mmap predictably.
    flat = np.concatenate(
        [arr.reshape(arr.shape[0], -1) for arr in debug_logits],
        axis=1,
    )
    flat.astype(np.float32).tofile(output_dir / "expected_logits_xgb.bin")

    print(
        f"  Verification: {n_tests} state vectors + ground-truth actions "
        f"+ debug logits (first 10)"
    )


def write_xgboost_c(
    trees_data: dict,
    output_dir: Path,
) -> None:
    """Write XGBoost trees as C source and header files."""
    h_lines = [
        "/* ai_xgboost_trees.h — Auto-generated by export_models.py */",
        "#ifndef AI_XGBOOST_TREES_H",
        "#define AI_XGBOOST_TREES_H",
        "",
        "#include <stdint.h>",
        "",
        "struct xgb_node {",
        "    int16_t  feature_idx;   /* -1 = leaf */",
        "    float    threshold;",
        "    float    leaf_value;",
        "    uint16_t left_child;",
        "    uint16_t right_child;",
        "};",
        "",
    ]

    c_lines = [
        "/* ai_xgboost_trees.c — Auto-generated by export_models.py */",
        '#include "ai_xgboost_trees.h"',
        "",
    ]

    total_nodes = 0
    total_trees = 0

    for clf_name in ["core", "priority", "preempt"]:
        clf_data = trees_data[clf_name]
        trees = clf_data["trees"]
        n_classes = clf_data["n_classes"]
        label_classes = clf_data["label_classes"]

        h_lines.append(f"#define XGB_{clf_name.upper()}_N_TREES {len(trees)}")
        h_lines.append(f"#define XGB_{clf_name.upper()}_N_CLASSES {n_classes}")
        h_lines.append(f"extern const int xgb_{clf_name}_label_map[{n_classes}];")
        h_lines.append("")

        # Label map
        c_lines.append(f"const int xgb_{clf_name}_label_map[{n_classes}] = {{")
        c_lines.append("    " + ", ".join(str(int(c)) for c in label_classes))
        c_lines.append("};")
        c_lines.append("")

        # Tree node arrays and offset table
        tree_offsets = []
        tree_sizes = []
        all_nodes_for_clf = []

        for t_idx, tree_nodes in enumerate(trees):
            offset = len(all_nodes_for_clf)
            tree_offsets.append(offset)
            tree_sizes.append(len(tree_nodes))
            all_nodes_for_clf.extend(tree_nodes)

        total_n = len(all_nodes_for_clf)
        total_nodes += total_n
        total_trees += len(trees)

        h_lines.append(f"#define XGB_{clf_name.upper()}_TOTAL_NODES {total_n}")
        h_lines.append(f"extern const struct xgb_node xgb_{clf_name}_nodes[{total_n}];")
        h_lines.append(f"extern const int xgb_{clf_name}_tree_offsets[{len(trees)}];")
        h_lines.append(f"extern const int xgb_{clf_name}_tree_sizes[{len(trees)}];")
        h_lines.append("")

        # Nodes array
        c_lines.append(f"const struct xgb_node xgb_{clf_name}_nodes[{total_n}] = {{")
        for node in all_nodes_for_clf:
            c_lines.append(
                f"    {{ {node['feature_idx']}, "
                f"{float_to_hex(node['threshold'])}, "
                f"{float_to_hex(node['leaf_value'])}, "
                f"{node['left_child']}, {node['right_child']} }},"
            )
        c_lines.append("};")
        c_lines.append("")

        # Offsets and sizes
        c_lines.append(f"const int xgb_{clf_name}_tree_offsets[{len(trees)}] = {{")
        c_lines.append("    " + ", ".join(str(o) for o in tree_offsets))
        c_lines.append("};")
        c_lines.append("")

        c_lines.append(f"const int xgb_{clf_name}_tree_sizes[{len(trees)}] = {{")
        c_lines.append("    " + ", ".join(str(s) for s in tree_sizes))
        c_lines.append("};")
        c_lines.append("")

    h_lines.append("#endif /* AI_XGBOOST_TREES_H */")
    h_lines.append("")

    (output_dir / "ai_xgboost_trees.h").write_text("\n".join(h_lines))
    (output_dir / "ai_xgboost_trees.c").write_text("\n".join(c_lines))

    print(f"  XGBoost: {total_trees} trees, {total_nodes} total nodes, "
          f"~{total_nodes * 14 / 1024:.0f} KB")


def write_config_h(n_actions: int, platform_name: str, output_dir: Path) -> None:
    """Write ai_config.h with platform-specific constants and action decoder."""
    platform = get_platform(platform_name)
    n_cores = platform.num_cores
    gpu = 1 if platform.gpu.available else 0

    lines = [
        "/* ai_config.h — Auto-generated by export_models.py */",
        "#ifndef AI_CONFIG_H",
        "#define AI_CONFIG_H",
        "",
        f"#define AI_SCHED_STATE_SIZE  {TOTAL_FEATURES}",
        f"#define AI_SCHED_N_ACTIONS   {n_actions}",
        f'#define AI_SCHED_PLATFORM    "{platform_name}"',
        f"#define AI_SCHED_N_CORES     {n_cores}",
        f"#define AI_SCHED_GPU         {gpu}",
        "",
        "struct ai_sched_action {",
        "    unsigned char core_assignment;  /* 0..N_CORES-1, or N_CORES for GPU */",
        "    unsigned char priority_adj;     /* 0=lower, 1=keep, 2=raise */",
        "    unsigned char preempt;          /* 1=preempt current task on core */",
        "};",
        "",
        "/* Decode flat action index → structured action.",
        f" * Encoding: idx = core * 6 + priority_adj * 2 + preempt",
        f" * Action space: ({n_cores} cores + {gpu} GPU) × 3 priority × 2 preempt = {n_actions}",
        " */",
        "static inline void ai_decode_action(int idx, struct ai_sched_action *out) {",
        "    out->preempt        = idx % 2;  idx /= 2;",
        "    out->priority_adj   = idx % 3;  idx /= 3;",
        "    out->core_assignment = idx;",
        "}",
        "",
        "#endif /* AI_CONFIG_H */",
        "",
    ]
    (output_dir / "ai_config.h").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Verification artifacts
# ---------------------------------------------------------------------------

def generate_verification_data(
    layers: list[tuple[torch.Tensor, torch.Tensor]],
    model_name: str,
    output_dir: Path,
    n_tests: int = 1000,
) -> None:
    """Generate test vectors and expected outputs for host-side C verification."""
    torch.manual_seed(54321)
    test_inputs = torch.rand(n_tests, TOTAL_FEATURES)

    with torch.no_grad():
        x = test_inputs
        for i, (W, b) in enumerate(layers):
            x = x @ W.T + b
            if i < len(layers) - 1:
                x = torch.relu(x)
        logits = x
        actions = logits.argmax(dim=-1)

    # Write binary files
    test_inputs.numpy().astype(np.float32).tofile(
        output_dir / f"test_vectors_{model_name}.bin"
    )
    actions.numpy().astype(np.int32).tofile(
        output_dir / f"expected_actions_{model_name}.bin"
    )
    # First 10 logit vectors for layer-by-layer debugging
    logits[:10].numpy().astype(np.float32).tofile(
        output_dir / f"expected_logits_{model_name}.bin"
    )

    print(f"  Verification: {n_tests} test vectors written")


def write_verify_c(output_dir: Path, model_name: str, n_actions: int) -> None:
    """Write a standalone C test for host-side verification."""
    prefix = model_name
    c_code = f"""\
/* verify_inference_{prefix}.c — Host-side bit-exact verification.
 *
 * Compile: gcc -O2 -o verify_{prefix} verify_inference_{prefix}.c \\
 *          ai_weights_{prefix}.c -lm
 * Run:     ./verify_{prefix}
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ai_weights_{prefix}.h"
#include "ai_config.h"

#define N_TESTS 1000

static void matvec(const float *W, const float *bias,
                   const float *in, float *out, int M, int N) {{
    for (int i = 0; i < M; i++) {{
        float sum = bias[i];
        const float *row = &W[i * N];
        for (int j = 0; j < N; j++)
            sum += row[j] * in[j];
        out[i] = sum;
    }}
}}

static void relu(float *x, int n) {{
    for (int i = 0; i < n; i++)
        if (x[i] < 0.0f) x[i] = 0.0f;
}}

static int argmax(const float *x, int n) {{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}}

static int run_inference(const float *state) {{
    float a[256], b[256];

    matvec({prefix}_w0, {prefix}_b0, state, a,
           {prefix.upper()}_L0_OUT, {prefix.upper()}_L0_IN);
    relu(a, {prefix.upper()}_L0_OUT);

    matvec({prefix}_w1, {prefix}_b1, a, b,
           {prefix.upper()}_L1_OUT, {prefix.upper()}_L1_IN);
    relu(b, {prefix.upper()}_L1_OUT);

    matvec({prefix}_w2, {prefix}_b2, b, a,
           {prefix.upper()}_L2_OUT, {prefix.upper()}_L2_IN);
    relu(a, {prefix.upper()}_L2_OUT);

    matvec({prefix}_w3, {prefix}_b3, a, b,
           {prefix.upper()}_L3_OUT, {prefix.upper()}_L3_IN);

    return argmax(b, AI_SCHED_N_ACTIONS);
}}

int main(void) {{
    FILE *fv = fopen("test_vectors_{prefix}.bin", "rb");
    FILE *fa = fopen("expected_actions_{prefix}.bin", "rb");
    if (!fv || !fa) {{
        fprintf(stderr, "Cannot open test files\\n");
        return 1;
    }}

    float state[AI_SCHED_STATE_SIZE];
    int expected, predicted;
    int pass = 0, fail = 0;

    for (int i = 0; i < N_TESTS; i++) {{
        if (fread(state, sizeof(float), AI_SCHED_STATE_SIZE, fv) != AI_SCHED_STATE_SIZE) break;
        if (fread(&expected, sizeof(int), 1, fa) != 1) break;

        predicted = run_inference(state);
        if (predicted == expected)
            pass++;
        else {{
            fail++;
            if (fail <= 5)
                printf("MISMATCH test %d: expected %d, got %d\\n", i, expected, predicted);
        }}
    }}

    printf("Results: %d/%d passed", pass, pass + fail);
    if (fail > 0)
        printf(" (%d FAILED)", fail);
    printf("\\n");

    fclose(fv);
    fclose(fa);
    return fail > 0 ? 1 : 0;
}}
"""
    (output_dir / f"verify_inference_{prefix}.c").write_text(c_code)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def export_mlp(platform_name: str, n_actions: int, output_dir: Path) -> None:
    """Export MLP model with BatchNorm folding."""
    from training.mlp.model import SchedulerMLP

    # Prefer platform-specific model, fall back to generic
    platform_path = Path(f"models/mlp/best_{platform_name}.pt")
    generic_path = Path("models/mlp/best.pt")
    if platform_path.exists():
        model_path = platform_path
    elif generic_path.exists():
        model_path = generic_path
    else:
        print(f"  SKIP: no MLP model found ({platform_path} or {generic_path})")
        return

    # Detect output dimension from checkpoint
    state_dict = torch.load(model_path, weights_only=True, map_location="cpu")
    model_n_actions = state_dict["network.11.bias"].shape[0]
    if model_n_actions != n_actions:
        print(f"  ERROR: model {model_path.name} has {model_n_actions} actions "
              f"but platform {platform_name} requires {n_actions}")
        print(f"  Retrain with: python scripts/_train_mlp.py --platform {platform_name}")
        return

    # Load model
    model = SchedulerMLP(n_actions=n_actions)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"  Loaded MLP from {model_path}: {model.param_count()} params, {n_actions} actions")

    # Fold BatchNorm
    layers = fold_mlp_batchnorms(model)
    max_diff = verify_bn_folding(model, layers)
    print(f"  BatchNorm folding verified: max diff = {max_diff:.2e}")
    if max_diff > 1e-4:
        print(f"  WARNING: large folding difference ({max_diff:.2e}), proceeding anyway")

    # Write C files
    write_nn_weights_c(layers, "mlp", output_dir)
    generate_verification_data(layers, "mlp", output_dir)
    write_verify_c(output_dir, "mlp", n_actions)

    total_params = sum(W.numel() + b.numel() for W, b in layers)
    print(f"  MLP exported: {total_params} params, ~{total_params * 4 / 1024:.0f} KB")


def export_ppo(platform_name: str, n_actions: int, output_dir: Path) -> None:
    """Export PPO actor network."""
    model_path = Path("models/ppo/best_model.zip")
    if not model_path.exists():
        print(f"  SKIP: {model_path} not found (training may still be running)")
        return

    layers = extract_ppo_actor_layers(model_path)
    max_diff = verify_ppo_actor(model_path, layers)
    print(f"  PPO actor extraction verified: max diff = {max_diff:.2e}")

    # Check output dimension matches platform
    out_dim = layers[-1][0].shape[0]
    if out_dim != n_actions:
        print(f"  ERROR: PPO output dim ({out_dim}) != platform actions ({n_actions})")
        return

    write_nn_weights_c(layers, "ppo", output_dir)
    generate_verification_data(layers, "ppo", output_dir)
    write_verify_c(output_dir, "ppo", n_actions)

    total_params = sum(W.numel() + b.numel() for W, b in layers)
    print(f"  PPO exported: {total_params} params, ~{total_params * 4 / 1024:.0f} KB")


def export_xgboost(
    platform_name: str,
    n_actions: int,
    output_dir: Path,
    emit_c_source: bool = False,
) -> None:
    """Export XGBoost cascade as a runtime-load binary blob (default) and
    optionally also as the legacy C-source form (debug aid).

    The binary blob (`xgb_sched.smb`) is the form SLM-OS consumes via
    `slm.sched_model_stage("xgboost", path)`. The C-source form was
    Plan A's original output but was rejected for ballooning the
    kernel image; it is kept behind `--xgb-emit-c-source` for local
    inspection only and never shipped.
    """
    model_dir = Path("models/xgboost")
    if not (model_dir / "meta.json").exists():
        print(f"  SKIP: {model_dir}/meta.json not found")
        return

    trees_data = export_xgboost_trees(model_dir)
    write_xgboost_blob(trees_data, output_dir)
    generate_xgb_verification_data(model_dir, output_dir)
    if emit_c_source:
        # Retained debug path; produces ~24 MB of C source. Useful only
        # for visually inspecting tree structure in a text editor.
        write_xgboost_c(trees_data, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Export trained models to C deployment format"
    )
    parser.add_argument(
        "--model", choices=["mlp", "ppo", "xgboost", "all"], default="all",
        help="Which model to export (default: all)",
    )
    parser.add_argument(
        "--platform", default="jetson_orin_nano",
        choices=["jetson_orin_nano", "raspberry_pi5", "big_little"],
        help="Target platform (determines action space size)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: deploy/generated/)",
    )
    parser.add_argument(
        "--xgb-emit-c-source", action="store_true",
        help="(debug) Also emit the XGBoost cascade as legacy C source. "
             "Default off; ~24 MB of C is large and not shipped.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parent.parent / "deploy" / "generated"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get platform action space
    platform = get_platform(args.platform)
    n_actions = action_space_size(platform.num_cores, platform.gpu.available)

    print(f"Platform: {args.platform} ({platform.num_cores} cores, "
          f"GPU={'yes' if platform.gpu.available else 'no'}, "
          f"{n_actions} actions)")
    print(f"Output:   {output_dir}")
    print()

    models = [args.model] if args.model != "all" else ["mlp", "ppo", "xgboost"]

    for model_name in models:
        print(f"--- Exporting {model_name.upper()} ---")
        if model_name == "mlp":
            export_mlp(args.platform, n_actions, output_dir)
        elif model_name == "ppo":
            export_ppo(args.platform, n_actions, output_dir)
        elif model_name == "xgboost":
            export_xgboost(
                args.platform, n_actions, output_dir,
                emit_c_source=args.xgb_emit_c_source,
            )
        print()

    # Write config header with the platform's action space
    write_config_h(n_actions, args.platform, output_dir)
    print(f"Config: AI_SCHED_N_ACTIONS={n_actions}")
    print("Done.")


if __name__ == "__main__":
    main()
