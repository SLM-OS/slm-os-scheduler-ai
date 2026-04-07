# Training Pipelines

Three model families are trained on expert demonstrations, each with different strengths.

## MLP (Behavioral Cloning)

**Architecture** (~131K parameters):
```
Input(108) → Linear(256) → ReLU → BatchNorm → Dropout(0.1)
           → Linear(256) → ReLU → BatchNorm → Dropout(0.1)
           → Linear(128) → ReLU → BatchNorm
           → Linear(N_ACTIONS)
```

**How it works:** Supervised learning that imitates expert policies. Each training sample is a (state, expert_action) pair. The loss is cross-entropy weighted by `|reward|` so that high-impact decisions count more.

**Training configuration:**
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-3 → 1e-5 (cosine annealing) |
| Weight decay | 1e-4 |
| Batch size | 1024 |
| Max epochs | 50 |
| Early stopping | Patience 5 epochs on val loss |
| Device | CPU |

**Training flow:**
1. Load Parquet (train/val splits), filter to expert policies only (exclude random)
2. Compute per-sample weights: `|reward| / mean(|reward|)` (normalized to mean 1.0)
3. Train with reward-weighted cross-entropy loss
4. Save best model checkpoint by validation loss
5. Optional: DAgger refinement

**DAgger refinement** (`training/mlp/dagger.py`):

Addresses distribution shift — the MLP was trained on states visited by experts, but at inference it visits different states. DAgger fixes this:

1. Run the MLP's own policy in the simulator (3 iterations x 1000 episodes)
2. At each decision, ask the expert what it would do
3. Expert's answer becomes the label
4. Retrain on combined original + corrected data

**Results (from first training run):**
- Best epoch: 6, val loss: 0.2608, val accuracy: 90.4%

**Files:**
- `training/mlp/model.py` — `SchedulerMLP` architecture
- `training/mlp/dataset.py` — `SchedulerDataset` Parquet loader with streaming reads
- `training/mlp/train.py` — Training loop with timestamped progress logging
- `training/mlp/dagger.py` — DAgger refinement loop

---

## XGBoost (3-Classifier Cascade)

**Architecture:** Three separate gradient-boosted tree classifiers, predicting action components sequentially:

```
State (113 features = 108 base + 5 derived)
  │
  ├─→ Core Assignment Classifier (200 trees, max_depth=8)
  │         │
  │         ▼ predicted core appended to features (114 features)
  ├─→ Priority Adjustment Classifier (100 trees, max_depth=6)
  │         │
  │         ▼ predicted priority appended (115 features)
  └─→ Preempt Decision Classifier (100 trees, max_depth=6)
```

Each classifier conditions on previous predictions, capturing dependencies like "if assigned to a busy core, more likely to preempt."

**5 Derived Features** (`training/xgboost/features.py`):

| Feature | Description |
|---------|-------------|
| `max_deadline_urgency` | Max urgency across top-8 pending tasks |
| `core_util_std` | Std dev of core utilizations (imbalance indicator) |
| `deadline_task_count` | Fraction of pending tasks with active deadlines |
| `gpu_should_use` | Binary: top task is GPU-eligible AND GPU queue < 25% full |
| `best_cache_fit_core` | Normalized ID of core with lowest cache pressure |

Tree models can't learn cross-feature interactions as easily as neural nets, so these features encode domain knowledge explicitly.

**Training configuration:**
| Parameter | Core clf | Priority clf | Preempt clf |
|-----------|----------|-------------|-------------|
| Trees | 200 | 100 | 100 |
| Max depth | 8 | 6 | 6 |
| Learning rate | 0.1 | 0.1 | 0.1 |
| Subsample | 0.8 | 0.8 | 0.8 |
| Col sample | 0.8 | 0.8 | 0.8 |
| Min child weight | 3 | 3 | 3 |

**Label encoding:** Kernel priority values are non-contiguous (0, 2, 4, 6, 7), so `LabelEncoder` maps them to 0-based contiguous integers for XGBoost, then maps back on prediction.

**Optuna hyperparameter search** (`optuna_search()`): 100 trials per classifier, tuning n_estimators (50-500), max_depth (4-12), learning_rate (0.01-0.3), subsample (0.6-1.0), colsample_bytree (0.6-1.0), min_child_weight (1-10).

**Results:**
- Core assignment accuracy: 93.4% (val)
- Priority accuracy: 99.97% (val)
- Preempt accuracy: 99.95% (val)

Priority and preempt decisions are nearly trivial; the learning challenge is core assignment.

**Files:**
- `training/xgboost/features.py` — Derived features and Parquet loading
- `training/xgboost/train.py` — `TripleClassifier`, training loop, Optuna search

---

## PPO (Reinforcement Learning)

**Architecture** (~164K parameters):
```
                Input(108)
                    │
              ┌─────┴─────┐
              │  Shared    │
              │  Backbone  │
              │ 256 → 256  │
              └─────┬─────┘
                    │
            ┌───────┴───────┐
            │               │
       Actor Head      Critic Head
       128 → 42        128 → 1
     (action probs)  (state value)
```

Unlike MLP/XGBoost which imitate experts, PPO learns by trial and error in the simulator. It can potentially discover strategies the experts never used.

**How PPO works:**
1. **Rollout:** Run the current policy in 16 parallel environments for 2048 steps each
2. **Advantage estimation:** Using the critic, compute "was this action better or worse than expected?" (GAE with gamma=0.99, lambda=0.95)
3. **Policy update:** Adjust actor to make good actions more likely, clipped so no update is too large (clip_range=0.2)
4. **Repeat** for 5 million total timesteps

**Training configuration:**
| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Stable Baselines 3) |
| Learning rate | 3e-4 |
| Clip range | 0.2 |
| Discount (gamma) | 0.99 |
| GAE lambda | 0.95 |
| Entropy coefficient | 0.01 |
| Value function coefficient | 0.5 |
| Steps per rollout | 2048 per env |
| Mini-batch size | 256 |
| PPO epochs per update | 4 |
| Max gradient norm | 0.5 |
| Parallel environments | 16 |
| Total timesteps | 5,000,000 |

**Curriculum learning** (4 phases):

| Phase | Steps | Scenarios | Purpose |
|-------|-------|-----------|---------|
| A | 0-1M | light_single, light_mixed | Learn basics |
| B | 1M-3M | + medium_mixed, deadline_pressure | Add load balancing, time pressure |
| C | 3M-5M | All 8 equally | Full distribution |
| D | 5M+ | All 8, weighted (heavy: 30%, burst: 25%) | Stress hardest cases |

**Behavioral cloning pre-training:** Before PPO starts, the actor is initialized by imitating expert data for 10 epochs. This avoids the slow random exploration at the beginning of RL training.

**Checkpoints:** Saved every 500K steps to `models/ppo/checkpoints/`. Best model by eval score saved to `models/ppo/best_model.zip`.

**At deployment:** Only the actor network is used (critic is training-only). Exported via `export_actor_onnx()`.

**Files:**
- `training/rl/network.py` — `SchedulerActorCriticPolicy` and `SchedulerActorCriticNet`
- `training/rl/vec_env.py` — Vectorized environment factory
- `training/rl/curriculum.py` — `CurriculumSchedule` with 4 phases, `CurriculumCallback`
- `training/rl/train.py` — `train_ppo()`, `ProgressCallback`, `pretrain_behavioral_cloning()`

---

## Comparison

| | MLP | XGBoost | PPO |
|---|---|---|---|
| **Learns from** | Expert demonstrations | Expert demonstrations | Its own experience |
| **Can exceed experts?** | No | No | Yes |
| **Training time** | Minutes | Minutes | Hours |
| **Inference** | 1 forward pass | 3 tree traversals | 1 forward pass (actor) |
| **Parameters** | ~131K | N/A (tree nodes) | ~164K (actor+critic) |
| **Interpretability** | Low | High (feature importance) | Low |
| **Export size** | ~515 KB | ~6.3 MB | ~515 KB |
