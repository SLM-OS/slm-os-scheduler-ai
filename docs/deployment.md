# Deployment and Export

## Overview

Trained models are exported as C source files containing weight arrays and a standalone inference engine. These compile directly into the SLM-OS bare-metal kernel — no Python, no ONNX runtime, no dynamic allocation at inference time.

## Export Pipeline (`scripts/export_models.py`)

```bash
python scripts/export_models.py --model mlp --platform jetson_orin_nano
python scripts/export_models.py --model all --platform jetson_orin_nano \
    --output-dir ~/projects/CS-496-Capstone-SLM-Operating-System/kernel/sched/ai/
```

### What It Generates

| File | Contents |
|------|----------|
| `ai_config.h` | `AI_SCHED_STATE_SIZE` (108), `AI_SCHED_N_ACTIONS` (platform-dependent) |
| `ai_weights_mlp.h` | Extern declarations for MLP weight arrays |
| `ai_weights_mlp.c` | Weight arrays as `static const float[]` with hex float literals |
| `ai_weights_ppo.h/c` | Same for PPO actor weights |
| `ai_xgboost_trees.h/c` | Tree node arrays for 3 XGBoost classifiers |
| `verify_inference_*.c` | Host-side C verification test |
| `test_vectors_*.bin` | 1000 test state vectors |
| `expected_actions_*.bin` | Expected action indices |

### Weight Format

- Arrays use `%a` hex float format for bit-exact reproduction
- Cache-line aligned: `__attribute__((aligned(64)))`
- Row-major layout: `W[out_features][in_features]`
- Placed in `.rodata` via `const` qualifier

Example:
```c
__attribute__((aligned(64)))
const float mlp_w0[256 * 108] = {
    0x1.eaa65c0000000p-1, -0x1.4fb09a0000000p-2, ...
};
```

## BatchNorm Folding (MLP Only)

The MLP has `Linear → ReLU → BatchNorm` ordering. At inference, Dropout is removed and each BatchNorm folds into the *next* Linear layer:

```
W_new = W_next * diag(gamma / sqrt(var + eps))
b_new = W_next * (beta - gamma * mean / sqrt(var + eps)) + b_next
```

This eliminates BatchNorm from inference entirely, leaving 4 simple layers:

```
Layer 0: W[256x108], b[256] → ReLU
Layer 1: W[256x256], b[256] → ReLU   (BN folded in)
Layer 2: W[128x256], b[128] → ReLU   (BN folded in)
Layer 3: W[36x128],  b[36]           (BN folded in, no activation)
```

Verification: 1000/1000 test inputs produce identical argmax actions before and after folding.

## PPO Actor Extraction

The PPO model is an actor-critic with shared backbone. Only the actor path is exported:

```
features_extractor (identity) → shared[0] (Linear 108→256) → ReLU
    → shared[2] (Linear 256→256) → ReLU
    → policy_net[0] (Linear 256→128) → ReLU
    → action_net (Linear 128→N_ACTIONS)
```

No BatchNorm, so extraction is exact (zero diff).

## XGBoost Export

Three classifiers exported as compact tree node arrays:

```c
struct xgb_node {
    int16_t  feature_idx;   // -1 = leaf
    float    threshold;      // split threshold
    float    leaf_value;     // leaf output (if leaf)
    uint16_t left_child;     // index of left child
    uint16_t right_child;    // index of right child
};
```

Also exports:
- Tree offset/size tables for locating individual trees in the flat array
- Label encoder mappings (`xgb_*_label_map[]`) for decoding predictions back to kernel priority values

**Size warning:** The current XGBoost model produces ~462K nodes (~6.3 MB), which may be too large for a bare-metal kernel. MLP/PPO at ~515 KB each are better candidates for initial deployment.

## C Inference Engine

The inference code is straightforward — three functions:

```c
// Matrix-vector multiply: out = W * in + bias
void matvec(const float *W, const float *bias,
            const float *in, float *out, int M, int N);

// In-place ReLU
void relu(float *x, int n);

// Return index of maximum value
int argmax(const float *x, int n);
```

Full inference chains 4 layers:
```c
int ai_schedule_mlp(const float state[108], struct ai_sched_action *action) {
    float a[256], b[256];
    matvec(mlp_w0, mlp_b0, state, a, 256, 108); relu(a, 256);
    matvec(mlp_w1, mlp_b1, a, b, 256, 256);     relu(b, 256);
    matvec(mlp_w2, mlp_b2, b, a, 128, 256);     relu(a, 128);
    matvec(mlp_w3, mlp_b3, a, b, N_ACTIONS, 128);
    int idx = argmax(b, N_ACTIONS);
    decode_action(idx, action);
    return 0;
}
```

**Performance estimate:** ~131K MACs. With NEON auto-vectorization at 1.5 GHz: ~22 us. Scalar: ~87 us.

## Host-Side Verification

After exporting, verify bit-exact match between Python and C:

```bash
cd deploy/generated
gcc -O2 -o verify_mlp verify_inference_mlp.c ai_weights_mlp.c -lm
./verify_mlp
# Results: 1000/1000 passed
```

## C API (`deploy/ai_scheduler.h`)

The kernel-facing interface:

```c
#define AI_SCHED_STATE_SIZE 108

struct ai_sched_action {
    uint8_t core_assignment;   // 0..N_CORES-1, or N_CORES for GPU
    uint8_t priority_adj;      // 0=lower, 1=keep, 2=raise
    uint8_t preempt;           // 1=preempt current task
};

int ai_scheduler_init(void);
int ai_schedule(const float state[108], struct ai_sched_action *action);
void ai_scheduler_shutdown(void);
```

The kernel calls `ai_schedule()` at each scheduling decision point, passing the current state vector. If it returns non-zero, the kernel falls back to the heuristic scheduler.

## Export Sizes

| Model | Parameters | .rodata Size |
|-------|-----------|-------------|
| MLP (BN-folded) | 131,236 | ~513 KB |
| PPO (actor only) | 132,010 | ~516 KB |
| XGBoost (3 classifiers) | 462,412 nodes | ~6.3 MB |

## Action Space Mismatch

The training dataset includes data from all 3 platforms. The MLP was trained with `n_actions = max(action_index) + 1` from the subsampled data, which may differ from the target platform's theoretical action space. The export script detects this and reports it:

```
NOTE: model was trained with 36 actions (platform wants 42)
```

For deployment on a specific platform, retrain with data from that platform only, or ensure the training data includes the full action range.
