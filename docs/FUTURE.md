# Future Refinements

Potential improvements that are implemented but not yet exercised, or known issues worth addressing.

---

## 1. Run Full Evaluation Suite

The evaluation framework (`evaluation/`) is complete but hasn't been run against the trained models.

**What it does:** Runs all agents (MLP, XGBoost, PPO, plus expert baselines) on identical test episodes with paired random seeds, computes DCR/latency/balance/power metrics, and runs paired t-tests for statistical significance.

**How to run:**
```python
from evaluation.run_eval import run_full_evaluation, summarize_results
from evaluation.statistical_tests import run_all_comparisons
from evaluation.visualization import generate_all_charts

results = run_full_evaluation()
summary = summarize_results(results)
comparisons = run_all_comparisons(results, baseline="slm_os_hybrid")
generate_all_charts(results, output_dir="results/charts/")
```

**Why it matters:** Without this, we don't have quantitative evidence of how the trained models compare to the expert baselines. The MLP's 90.4% val accuracy and XGBoost's 93.4% core accuracy are training metrics — evaluation on held-out test scenarios with proper metrics (DCR, latency, etc.) is the real measure.

---

## 2. DAgger Refinement for MLP

**Status:** Implemented in `training/mlp/dagger.py`, never run.

**What it does:** Addresses distribution shift. The MLP was trained on states visited by experts, but at inference visits different states due to its own mistakes. DAgger runs the MLP in the simulator while the expert corrects its decisions, then retrains on the combined data.

**Configuration:** 3 iterations x 1000 episodes, using `medium_mixed` scenario on `jetson_orin_nano`.

**How to run:**
```python
from training.mlp.dagger import run_dagger
from training.mlp.model import SchedulerMLP
from slm_sim.experts.hybrid import HybridExpertPolicy

model = SchedulerMLP(n_actions=36)
model.load_state_dict(torch.load("models/mlp/best.pt", weights_only=True))

expert = HybridExpertPolicy()
tables = run_dagger(model, expert, output_dir=Path("data/dagger/"))
# Then retrain MLP on original + DAgger data
```

**Expected impact:** Could improve accuracy on states the MLP visits during its own execution, particularly in scenarios where the MLP's errors compound.

---

## 3. Optuna Hyperparameter Search for XGBoost

**Status:** Implemented in `training/xgboost/train.py:optuna_search()`, never run. Current model uses defaults.

**What it does:** Runs 100 Optuna trials per classifier, searching over: `n_estimators` (50-500), `max_depth` (4-12), `learning_rate` (0.01-0.3), `subsample` (0.6-1.0), `colsample_bytree` (0.6-1.0), `min_child_weight` (1-10).

**How to run:**
```python
from training.xgboost.train import optuna_search

best_core_params = optuna_search("data/splits/train.parquet",
                                  "data/splits/val.parquet",
                                  classifier_name="core", n_trials=100)
best_prio_params = optuna_search(..., classifier_name="priority")
best_pre_params = optuna_search(..., classifier_name="preempt")
# Then retrain with optimized params
```

**Expected impact:** The core assignment classifier (93.4% accuracy) has the most room for improvement. Optimized hyperparameters could also reduce tree count/depth, shrinking the 6.3 MB export size.

---

## 4. Re-export PPO After Training Completes

**Status:** PPO training is at ~70% (3.5M/5M steps). Current export uses a mid-training checkpoint.

**Action needed:** Once `train_all.py` finishes, re-run:
```bash
python scripts/export_models.py --model ppo --platform jetson_orin_nano
cd deploy/generated && gcc -O2 -o verify_ppo verify_inference_ppo.c ai_weights_ppo.c -lm && ./verify_ppo
```

The final model will likely have better evaluation performance than the mid-training snapshot, especially on harder scenarios (curriculum phases C and D).

---

## 5. Action Space Mismatch

**Issue:** The MLP was trained on data from all 3 platforms (Jetson/Pi5/big.LITTLE) subsampled together. The resulting model has 36 output actions — matching big.LITTLE's action space (6 cores, no GPU, 6x3x2=36) — rather than Jetson's 42 (7 targets including GPU).

**Impact:** The model cannot select the GPU target on Jetson (actions 36-41 don't exist in its output). Core assignments 0-5 and all priority/preempt combinations work fine.

**Fix options:**
1. **Retrain with Jetson-only data:** Filter the training Parquet to `platform == "jetson_orin_nano"` before loading. This gives the full 42-action output but loses cross-platform generalization.
2. **Retrain with all data but pad actions:** Ensure the training set includes at least one sample of each action index up to the platform's maximum.
3. **Accept 36 actions:** If GPU offload isn't critical, the 36-action model still covers all CPU core assignments and priority/preempt decisions.

---

## 6. XGBoost Export Size

**Issue:** The exported XGBoost model is ~6.3 MB (462K tree nodes across 1400 trees). This is likely too large for bare-metal kernel `.rodata`.

**MLP and PPO are ~515 KB each** — much more practical for deployment.

**Options if XGBoost deployment is desired:**
1. **Reduce tree count/depth** via Optuna search (see item 3). Fewer, shallower trees = smaller export.
2. **Prune trees** post-training by removing low-importance nodes.
3. **Use treelite** for C codegen instead of the interpretive traversal — produces if/else chains that may be more compact after compiler optimization.
4. **Skip XGBoost for deployment** and use MLP or PPO instead. XGBoost's main advantage (interpretability via feature importance) is a training-time benefit, not a deployment requirement.

---

## 7. Missing Kernel Features in State Vector

**Context:** 5 of the 8 per-task features in the 108-dim state vector (working_set, model_size, inference_duration, can_use_gpu, component_type) have no equivalent in the current SLM-OS kernel task struct. They will be zero when running on real hardware.

**Impact:** The model was trained with these features varying, so zeroing them degrades prediction quality. The remaining features (priority, deadline urgency, queue depth, utilization) carry most of the scheduling signal, but accuracy will be lower than simulation results suggest.

**Fix options:**
1. **Retrain with features masked:** Zero out these 5 features during training so the model learns to ignore them. This gives a model robust to their absence.
2. **Add fields to kernel task struct:** Extend `struct task` with an optional `struct slm_task_info` pointer, populated by the Rust runtime when creating ML inference tasks. This is the long-term solution described in Plan B.
3. **Hybrid approach:** Retrain with masked features for initial deployment, then switch to the full-feature model once kernel support is added.
