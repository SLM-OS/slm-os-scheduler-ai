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

## ~~Step 4: XGBoost Export~~ — DROPPED

XGBoost export produced 24 MB of tree node data (462K nodes across 1,400 trees) — far too large for bare-metal kernel deployment. Evaluation also showed XGBoost (97.0% DCR) underperforming MLP (99.3%). Dropping XGBoost from kernel integration; focusing on MLP + PPO.

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
