# Future Refinements

Potential improvements that are implemented but not yet exercised, or known issues worth addressing.

---

## ~~1. Run Full Evaluation Suite~~ (DONE)

Completed 2026-04-07. Ran 7 agents (MLP, PPO, XGBoost, hybrid, EDF, weighted, random) × 8 scenarios × 200 episodes = 11,200 episodes. Results in `results/`.

**Key findings:** MLP dominated among trained models (99.3% DCR on medium_mixed vs 92.7% PPO, 97.0% XGBoost). Statistically significant but small gap from hybrid baseline on hardest 3 scenarios (0.7-0.9% lower DCR). See `results/eval_summary.csv` and `results/charts/`.

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

## 4. ~~Re-export PPO After Training Completes~~ (DONE)

Completed 2026-04-07. Final PPO model exported with 42 actions (full Jetson action space including GPU target), 132,010 params, ~516 KB. Host-side C verification: 1000/1000 bit-exact.

---

## ~~5. Action Space Mismatch~~ (RESOLVED)

Resolved 2026-04-11 via platform-specific training. `SchedulerDataset` now accepts a `platform` parameter to filter training data. `_train_mlp.py --platform jetson_orin_nano` computes the platform's full action space (42) and passes it to `train_mlp(n_actions=42)`, ensuring the model output dimension matches even if experts never use all actions (e.g., GPU target).

Jetson-specific MLP trained and exported: 42 actions, 88.8% val accuracy, 1000/1000 bit-exact C verification. Checkpoint at `models/mlp/best_jetson_orin_nano.pt`.

---

## ~~6. XGBoost Export Size~~ (DROPPED)

XGBoost export produced 24 MB of tree node data (462K nodes across 1,400 trees) — far too large for bare-metal kernel `.rodata`. Evaluation also showed XGBoost (97.0% DCR) underperforming MLP (99.3%). XGBoost dropped from kernel integration; export pipeline now only supports MLP and PPO. XGBoost remains available for training and analysis but is not deployed.

---

## 7. Missing Kernel Features in State Vector

**Context:** 5 of the 8 per-task features in the 108-dim state vector (working_set, model_size, inference_duration, can_use_gpu, component_type) have no equivalent in the current SLM-OS kernel task struct. They will be zero when running on real hardware.

**Impact:** The model was trained with these features varying, so zeroing them degrades prediction quality. The remaining features (priority, deadline urgency, queue depth, utilization) carry most of the scheduling signal, but accuracy will be lower than simulation results suggest.

**Fix options:**
1. **Retrain with features masked:** Zero out these 5 features during training so the model learns to ignore them. This gives a model robust to their absence.
2. **Add fields to kernel task struct:** Extend `struct task` with an optional `struct slm_task_info` pointer, populated by the Rust runtime when creating ML inference tasks. This is the long-term solution described in Plan B.
3. **Hybrid approach:** Retrain with masked features for initial deployment, then switch to the full-feature model once kernel support is added.

---

## 8. Ablation Studies

**Plan reference:** Section 8.4

The plan specifies four ablation studies that have not been implemented:

1. **State features:** Remove feature groups (per-core, per-task, global) and measure impact on DCR/latency
2. **Reward weights:** Vary w_deadline, w_latency, w_balance, w_power and plot Pareto frontier
3. **Model size:** Test MLP variants (64, 128, 256, 512 hidden neurons) for latency-accuracy tradeoff
4. **Action space:** Compare flat N-way classification vs decomposed 3-classifier cascade

No ablation infrastructure exists yet. Would need evaluation framework (item 1) running first, then scripts to retrain with modified configurations and compare.

---

## 9. Out-of-Distribution Test Scenarios

**Plan reference:** Section 8.2

The plan describes two novel OOD scenarios not present in the codebase:

- **`cascading_failure`:** A component crash triggers load spikes on remaining cores, testing recovery behavior
- **`model_swap`:** Hot-swap of ML model mid-episode (model size changes), testing adaptation

The existing 8 scenarios in `slm_sim/workloads/scenarios.py` cover normal operating conditions. These OOD scenarios would test robustness to failure modes not seen during training.

Additionally, `burst_storm` was intended to be held out from MLP/XGBoost training as an OOD generalization test, but was included in the training data.

---

## 10. PPO Reward Shaping

**Plan reference:** Section 7.3

The plan describes two auxiliary reward signals for PPO that are not implemented:

- **Step penalty:** -0.001 per scheduling step (encourages faster decisions)
- **Exploration bonus:** +0.05 for choosing a core not used in the last 5 decisions (encourages action diversity)

Currently, PPO uses only the base 4-component reward from `slm_sim/reward.py`. Adding these could improve RL convergence speed and action space exploration.

---

## 11. INT8 Quantization

**Plan reference:** Sections 4.1, 9.2

The plan describes an export path: PyTorch → ONNX → INT8 quantization → C inference. Currently, all exports are FP32.

INT8 quantization would:
- Reduce weight size from ~515 KB to ~131 KB per model
- Reduce inference latency (important if scalar FP at ~87us exceeds the 50us target)
- Require validation that quantization doesn't degrade action accuracy

ONNX Runtime provides post-training quantization tools. Would need a calibration dataset (subset of validation data) and accuracy verification after quantization.

---

## 12. Behavioral Cloning Pre-training for PPO

**Plan reference:** Section 7.3

`pretrain_behavioral_cloning()` exists in `training/rl/train.py` but is not called by the training pipeline (`train_ppo()` or `_train_ppo.py`). The plan specifies initializing the actor network with 10 epochs of behavioral cloning before PPO exploration begins.

Integrating this into the pipeline could significantly reduce PPO's initial exploration phase, where it makes near-random decisions. The function exists and works (tested), but needs to be wired into `_train_ppo.py` before `train_ppo()` is called.

---

## 13. SHAP Interpretability for XGBoost

**Plan reference:** Section 7.2

The plan mentions SHAP (SHapley Additive exPlanations) analysis for per-sample decision explanations on the XGBoost model. The `shap` package is in `requirements.txt` but no SHAP analysis code exists.

Basic feature importance is implemented (`get_feature_importance()` in `training/xgboost/train.py`), but SHAP provides richer per-decision explanations showing which features pushed toward which action.

---

## 14. Fallback Rate Tracking

**Plan reference:** Section 9.3

When deployed in the kernel, invalid AI actions (e.g., assigning to an offline core, GPU target when unavailable) should fall back to the heuristic scheduler. The plan specifies:

- Tracking the fallback rate as a runtime metric
- If rate exceeds 5%, flagging the model for retraining

No fallback detection or rate tracking exists in the current C API or evaluation framework. This should be implemented in the kernel-side integration (Plan B) and validated during QEMU testing.

---

## 15. Analysis Notebooks

**Plan reference:** Section 12

Three Jupyter notebooks were planned but not created:

- `notebooks/dataset_analysis.ipynb` — Distribution of rewards, actions, features across experts/scenarios
- `notebooks/xgboost_shap.ipynb` — SHAP analysis of XGBoost decisions
- `notebooks/rl_policy_analysis.ipynb` — PPO learning curves, curriculum phase transitions, policy behavior visualization

These are useful for understanding model behavior and presenting results but are not required for deployment.
