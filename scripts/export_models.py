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


def export_xgboost(platform_name: str, n_actions: int, output_dir: Path) -> None:
    """Export XGBoost classifiers."""
    model_dir = Path("models/xgboost")
    if not (model_dir / "meta.json").exists():
        print(f"  SKIP: {model_dir}/meta.json not found")
        return

    trees_data = export_xgboost_trees(model_dir)
    write_xgboost_c(trees_data, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Export trained models to C deployment format"
    )
    parser.add_argument(
        "--model", choices=["mlp", "ppo", "all"], default="all",
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

    models = [args.model] if args.model != "all" else ["mlp", "ppo"]

    for model_name in models:
        print(f"--- Exporting {model_name.upper()} ---")
        if model_name == "mlp":
            export_mlp(args.platform, n_actions, output_dir)
        elif model_name == "ppo":
            export_ppo(args.platform, n_actions, output_dir)
        print()

    # Write config header with the platform's action space
    write_config_h(n_actions, args.platform, output_dir)
    print(f"Config: AI_SCHED_N_ACTIONS={n_actions}")
    print("Done.")


if __name__ == "__main__":
    main()
