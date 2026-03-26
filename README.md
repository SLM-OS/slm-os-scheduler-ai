# SLM-OS AI Scheduler

An AI-driven process scheduler for [SLM-OS](https://github.com/your-org/CS-496-Capstone-SLM-Operating-System), replacing hand-tuned heuristics with learned policies that optimize deadline compliance, inference latency, power efficiency, and core utilization simultaneously.

The system comprises a discrete-event scheduling simulator, synthetic training dataset generation from expert policies, three trained scheduling models (MLP, XGBoost, PPO), and an integration path back into SLM-OS's Rust/C runtime.

## Architecture

```
slm-os-scheduler-ai/
├── slm_sim/                  # Discrete-event simulator
│   ├── engine.py             # Core simulation loop (event queue, clock, handlers)
│   ├── models.py             # SimTask, CoreState, MemorySubsystem, GPUModel
│   ├── platforms.py          # Hardware profiles (Jetson, Pi 5, big.LITTLE)
│   ├── observation.py        # 108-dim state vector extraction
│   ├── actions.py            # Action encoding/decoding, apply logic
│   ├── reward.py             # 4-component reward (deadline, latency, balance, power)
│   ├── gym_env.py            # Gymnasium environment wrapper (SB3-compatible)
│   ├── logging.py            # Parquet transition logger
│   ├── runner.py             # Parallel episode runner
│   ├── workloads/            # Workload profiles
│   │   ├── anomaly.py        #   100 Hz, 12 MB model, 5 ms deadline
│   │   ├── predmaint.py      #   1 Hz + burst, 45 MB model, 50 ms deadline, GPU
│   │   ├── security.py       #   10 Hz, 8 MB model, 20 ms deadline
│   │   ├── system.py         #   100 Hz OS overhead, no deadline
│   │   └── scenarios.py      #   8 scenario compositions
│   └── experts/              # Expert scheduling policies
│       ├── hybrid.py         #   SLM-OS Phase 3 heuristic replica
│       ├── edf.py            #   Earliest Deadline First
│       ├── weighted.py       #   Multi-objective scoring
│       ├── random_policy.py  #   Random baseline
│       └── oracle.py         #   Offline beam search upper bound
├── training/                 # Model training pipelines
│   ├── mlp/                  #   Feedforward NN (131K params)
│   │   ├── model.py          #     108->256->256->128->N arch
│   │   ├── dataset.py        #     Parquet loader, reward weighting
│   │   ├── train.py          #     AdamW, cosine LR, early stopping
│   │   └── dagger.py         #     DAgger refinement loop
│   ├── xgboost/              #   Gradient-boosted trees (interpretable)
│   │   ├── features.py       #     5 derived features, feature names
│   │   └── train.py          #     3-classifier cascade, Optuna search
│   └── rl/                   #   PPO reinforcement learning
│       ├── network.py        #     Custom actor-critic for SB3 (164K params)
│       ├── vec_env.py        #     Vectorized env factory
│       ├── curriculum.py     #     4-phase difficulty progression
│       └── train.py          #     PPO + behavioral cloning pre-training
├── evaluation/               # Comparative evaluation framework
│   ├── metrics.py            #   Primary + secondary metrics
│   ├── run_eval.py           #   Head-to-head evaluation runner
│   ├── statistical_tests.py  #   Paired t-tests, 95% CIs
│   └── visualization.py      #   Bar charts, summary tables
├── scripts/                  # CLI scripts
│   ├── generate_dataset.py   #   Full dataset generation orchestrator
│   ├── validate_dataset.py   #   NaN/range/completeness checks
│   ├── split_dataset.py      #   70/15/15 episode-level split + normalization
│   └── export_models.py      #   ONNX + C weight export (stub)
├── deploy/
│   └── ai_scheduler.h        # C API header for SLM-OS kernel integration
├── tests/                    # 204 tests across 17 files
├── pyproject.toml
└── requirements.txt
```

## Key Design Decisions

- **Priority values match the real SLM-OS kernel**: IDLE=0, LOW=2, NORMAL=4, HIGH=6, CRITICAL=7 (with gaps, not contiguous).
- **Platform profiles are configurable**: Jetson Orin Nano (6-core, GPU), Raspberry Pi 5 (4-core), and big.LITTLE (2 perf + 4 efficiency) via a registry.
- **108-dimensional state vector**: 6 features per core (x6 max), 8 features per pending task (top-8), 8 global features. All normalized to [0, 1].
- **Decomposed action space**: (core_assignment) x (priority_adj: lower/keep/raise) x (preempt: yes/no). Jetson = 42 discrete actions.
- **4-component reward**: `R = 0.50*deadline + 0.25*latency + 0.15*balance + 0.10*power`.

## Quick Start

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Generate Dataset

```bash
# Small test dataset (~30 seconds)
python scripts/generate_dataset.py --small

# Full dataset (3 experts x 8 scenarios x 500 episodes x 3 platforms)
python scripts/generate_dataset.py --workers 4

# Validate and split
python scripts/validate_dataset.py
python scripts/split_dataset.py
```

### Train Models

```python
# MLP
from training.mlp.dataset import SchedulerDataset
from training.mlp.train import train_mlp, TrainConfig

train_ds = SchedulerDataset("data/splits/train.parquet")
val_ds = SchedulerDataset("data/splits/val.parquet")
model, result = train_mlp(train_ds, val_ds, TrainConfig(n_epochs=50))

# XGBoost
from training.xgboost.train import train_triple_classifier

triple, metrics = train_triple_classifier("data/splits/train.parquet",
                                           "data/splits/val.parquet")

# PPO
from training.rl.train import train_ppo, PPOConfig

model = train_ppo(PPOConfig(total_timesteps=5_000_000))
```

### Evaluate

```python
from evaluation.run_eval import run_full_evaluation, EvalConfig

df = run_full_evaluation(config=EvalConfig(n_episodes=200))
```

## Simulator Details

### Event-Driven Architecture

The simulator uses a min-heap event queue with 7 event types: `TASK_ARRIVAL`, `TASK_COMPLETION`, `GPU_COMPLETION`, `QUANTUM_EXPIRED`, `DEADLINE_CHECK` (1ms periodic), `SCHEDULING_DECISION`, and `MIGRATION_CHECK` (10ms periodic).

At each scheduling decision point:
1. Extract 108-dim state vector
2. Agent selects action (core, priority_adj, preempt)
3. Apply action to simulator state
4. Log (state, action, reward, next_state) transition

### Workload Scenarios

| Scenario | Description | Load |
|----------|-------------|------|
| `light_single` | Anomaly detector only | ~10% |
| `light_mixed` | Anomaly + Security | ~25% |
| `medium_mixed` | All components + System tasks | ~50% |
| `heavy_inference` | All at 2x rate | ~80% |
| `burst_storm` | Normal + periodic 3x bursts | spikes ~95% |
| `deadline_pressure` | PredMaint at 3x rate | ~60% |
| `memory_pressure` | 4x PredMaint instances | pool exhaustion |
| `asymmetric` | Components pinned to specific cores | ~40% unbalanced |

### Expert Policies

| Expert | Strategy | DCR on medium_mixed |
|--------|----------|-------------------|
| **Hybrid** | SLM-OS Phase 3 replica (deadline boost + load balance) | ~100% |
| **EDF** | Earliest Deadline First with preemption | ~91% |
| **Weighted** | Multi-objective (task,core) scoring | ~73% |
| **Random** | Uniform random (negative baseline) | ~71% |
| **Oracle** | Offline beam search upper bound | best achievable |

## Model Architectures

### MLP (~131K parameters)
```
Input(108) -> Dense(256, ReLU) -> BN -> Dropout(0.1)
           -> Dense(256, ReLU) -> BN -> Dropout(0.1)
           -> Dense(128, ReLU) -> BN
           -> Dense(N_ACTIONS)
```
Trained with reward-weighted cross-entropy, AdamW + cosine LR, DAgger refinement.

### XGBoost (3 cascaded classifiers)
- Classifier 1: Core assignment (200 trees, depth 8)
- Classifier 2: Priority adjustment conditioned on predicted core (100 trees, depth 6)
- Classifier 3: Preempt decision conditioned on core + priority (100 trees, depth 6)

Uses 113 features (108 base + 5 derived). Optuna hyperparameter search.

### PPO (~164K parameters)
```
Shared:  Input(108) -> Dense(256, ReLU) -> Dense(256, ReLU)
Actor:   -> Dense(128, ReLU) -> Dense(N_ACTIONS)
Critic:  -> Dense(128, ReLU) -> Dense(1)
```
4-phase curriculum (light -> mixed -> full -> hard-weighted). Optional behavioral cloning pre-training.

## SLM-OS Integration

The `deploy/ai_scheduler.h` header defines the C API for kernel integration:

```c
int ai_scheduler_init(void);
int ai_schedule(const float state[108], struct ai_sched_action *action);
void ai_scheduler_shutdown(void);
```

Models are exported via ONNX and converted to C weight arrays for bare-metal inference within the SLM-OS kernel scheduler.

## Performance

- **Simulator throughput**: >1,600 episodes/minute on a single core
- **Test suite**: 204 tests, all passing in ~2 minutes
- **Expert-random gap**: statistically significant on heavy workloads (paired t-test, p < 0.05)

## Requirements

- Python >= 3.12
- PyTorch >= 2.1 (CPU sufficient for training)
- XGBoost >= 2.0
- Stable Baselines 3 >= 2.2
- See `requirements.txt` for full list

## License

Private repository. Part of CS-496 Capstone: SLM Operating System.
