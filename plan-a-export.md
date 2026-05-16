# Plan A: Export Pipeline (slm-os-scheduler-ai)

## Context

The trained models (MLP, PPO, XGBoost) need to be exported as C weight arrays and verification data for integration into the SLM-OS kernel. This work is self-contained within the `slm-os-scheduler-ai` repo.

**Repo:** `~/projects/slm-os-scheduler-ai`

---

## Step 1: BatchNorm Folding for MLP

The MLP architecture is `Linear → ReLU → BatchNorm → [Dropout] → Linear → ...`. At inference, Dropout is identity and each BatchNorm (affine transform on post-ReLU output) folds into the *next* Linear layer:
- `W_new = W_next * diag(gamma / sqrt(var + eps))`
- `b_new = W_next * (beta - gamma * mean / sqrt(var + eps)) + b_next`

Fold pairs (from `training/mlp/model.py` Sequential indices):
- BN[2] → folds into Linear[4]
- BN[6] → folds into Linear[8]
- BN[10] → folds into Linear[11]

Result: 4-layer network with no BN/Dropout:
```
Layer 0: W[256×108] + b[256] → ReLU
Layer 1: W[256×256] + b[256] → ReLU  (BN folded in)
Layer 2: W[128×256] + b[128] → ReLU  (BN folded in)
Layer 3: W[42×128]  + b[42]          (BN folded in)
```
~132K params × 4 bytes = **~515 KB**.

**Verification:** Run 1000 random inputs through both the original model (eval mode) and the folded model. Outputs must be identical (or within 1e-6 tolerance due to FP ordering).

**Platform-specific training:** The MLP must be trained on Jetson-only data so the output dimension matches the platform's 42-action space. Use `python scripts/_train_mlp.py --platform jetson_orin_nano` to produce `models/mlp/best_jetson_orin_nano.pt`. The export pipeline looks for the platform-specific checkpoint first.

## Step 2: PPO Actor Extraction

No BatchNorm — extract weights directly from SB3's policy:
```
shared.0:    Linear(108, 256) → ReLU
shared.2:    Linear(256, 256) → ReLU
policy_net:  Linear(256, 128) → ReLU
action_net:  Linear(128, 42)
```
Access via `model.policy.{features_extractor, mlp_extractor, action_net}` (see `training/rl/train.py` ActorWrapper class for the path). ~515 KB.

## Step 3: Implement `scripts/export_models.py`

Rewrite the stub. CLI:
```
python scripts/export_models.py --model mlp|ppo|xgboost|all \
    --platform jetson_orin_nano|raspberry_pi5|big_little \
    --output-dir deploy/generated/
```

**Generates per model:**
- `ai_weights_{model}.h` — extern declarations for weight arrays
- `ai_weights_{model}.c` — weight arrays as `static const float[]`, cache-line aligned (`__attribute__((aligned(64)))`), using `%a` hex float format for bit-exact reproduction

**Generates once:**
- `ai_config.h` — `AI_SCHED_N_ACTIONS` (42 Jetson, 24 Pi5, 36 big.LITTLE), computed from platform profile in `slm_sim/platforms.py`

**Action space per platform** (from `slm_sim/actions.py`: `(num_cores + gpu_available) × 3 × 2`):
- Jetson Orin Nano: 6 cores + GPU → 42
- Raspberry Pi 5: 4 cores, no GPU → 24
- big.LITTLE: 6 cores, no GPU → 36

**Validation:** Script checks that model output dimension matches target platform's action space. Refuses to export if mismatched and prints retrain instructions.

## Step 4: XGBoost Export — runtime-binary form (re-enabled, #58a / #850)

The original drop rationale was twofold: (1) the C-source form ballooned the
kernel image by ~24 MB, and (2) XGBoost's 97.0% DCR underperforms MLP's 99.3%.
SLM-OS reframed both: per #848 (pluggable-policy framing), policies are
options that expand the comparison space, not winners chasing a leaderboard;
and per #58 the cascade is *runtime-loaded* via the existing scheduler blob
subsystem, so it never enters the kernel image.

**Output:** `deploy/generated/xgb_sched.smb` — single SEMB-wrapped binary
holding all three classifiers (`core / priority / preempt`) in cascade form.
Walk it via `slm.sched_model_stage("xgboost", path)` →
`slm.sched_model_activate("xgboost")` → `slm.sched_set_policy("ai_xgb")`.

**Wire format** (kept in sync with `runtime/src/ml/xgb_tree.rs` in the
SLM-OS repo):

- 24-byte SEMB outer header (magic `SEMB`, version 1, kind `SCHED_MODEL_KIND_XGBOOST` = 0x1006, schema 1, payload length, FNV-1a checksum).
- 16-byte XGBC payload header (magic `XGBC`, version 1, classifier count).
- Per classifier: 16-byte section header (`u32 n_trees`, `u32 n_nodes`, `u16 n_classes`, padding) + `u32` root-offset table + 20-byte node records (`u16 feature_idx, u16 flags, u32 left, u32 right, f32 threshold, f32 value`) + `i32` label-class map.

The 20-byte cascade node format and `u32` indices are required: the trained
`core_clf` flattens to ~450 K nodes — well past the eviction-side u16
ceiling.

**Verification artifacts** (mirror MLP/PPO):
- `test_vectors_xgb.bin` — 1000 raw 108-float state vectors. SLM-OS computes
  the 5 derived features in-kernel and self-checks the derivation logic.
- `expected_actions_xgb.bin` — 1000 ground-truth `(core, priority, preempt)`
  triples from `TripleClassifier.predict` in Python.
- `expected_logits_xgb.bin` — per-classifier `predict_proba` for the first
  10 vectors. Debugging aid; not required for the bit-equality test.

**CLI:** `python scripts/export_models.py --model xgboost --platform <p>`.
The legacy C-source form is retained behind `--xgb-emit-c-source` for local
inspection only and is never shipped.

## Step 5: Verification Artifacts

Generate for each exported model:
- `test_vectors.bin` — 1000 state vectors (1000 × 108 × 4 bytes = 432 KB)
- `expected_actions.bin` — 1000 expected action indices (1000 × 4 bytes)
- `expected_logits_sample.bin` — full logit output for first 10 vectors (for layer-by-layer debugging)

Source vectors from held-out validation data or random inputs.

Also generate a standalone C test:
- `verify_inference.c` — loads test vectors, runs C inference, compares to expected actions
- Can be compiled with host GCC: `gcc -o verify verify_inference.c ai_weights_mlp.c -lm`

## Step 6: Action Decoding (reference implementation)

Include in generated output:
```c
static inline void ai_decode_action(int idx, struct ai_sched_action *out) {
    out->preempt = idx % 2; idx /= 2;
    out->priority_adj = idx % 3; idx /= 3;
    out->core_assignment = idx;
}
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `scripts/export_models.py` | Full export pipeline (MLP + PPO only) |
| `scripts/_train_mlp.py` | Added `--platform` argument for platform-specific training |
| `training/mlp/dataset.py` | Added `platform` parameter for platform-filtered data loading |
| `deploy/generated/` | Output directory (add to `.gitignore`) |

## Verification

1. BN folding: 1000 test inputs, original vs folded model outputs match
2. Weight export: hex float roundtrip preserves all values
3. Host C test: `verify_inference` reports 100% action match
4. Each model type exports and verifies independently
