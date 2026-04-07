# Architecture Overview

## Purpose

This project trains AI models to replace hand-tuned heuristics in the SLM-OS kernel's process scheduler. The trained models optimize for deadline compliance, latency, core load balance, and power efficiency across heterogeneous multi-core ARM platforms.

## System Flow

```
                        Simulator (slm_sim/)
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         Expert Policies  Workloads      Platforms
         (hybrid, EDF,   (anomaly,    (Jetson, Pi5,
          weighted,       predmaint,   big.LITTLE)
          random,         security,
          oracle)         system)
              │               │               │
              └───────┬───────┘               │
                      │                       │
                      ▼                       │
              Dataset Generation ─────────────┘
              (scripts/generate_dataset.py)
                      │
                      ▼
              Parquet Files (data/raw/)
                      │
              ┌───────┼───────┐
              │       │       │
              ▼       ▼       ▼
             MLP   XGBoost   PPO
           (train)  (train)  (train)
              │       │       │
              └───────┼───────┘
                      │
                      ▼
              Evaluation (evaluation/)
              Head-to-head comparison
              with statistical tests
                      │
                      ▼
              Export (scripts/export_models.py)
              C weight arrays for kernel
```

## Project Structure

```
slm-os-scheduler-ai/
├── slm_sim/                 # Discrete-event simulator
│   ├── engine.py            #   Core simulation loop
│   ├── models.py            #   Task, core, memory, GPU dataclasses
│   ├── platforms.py         #   Hardware profiles (3 platforms)
│   ├── observation.py       #   108-dim state vector extraction
│   ├── actions.py           #   Decomposed action space
│   ├── reward.py            #   4-component reward function
│   ├── gym_env.py           #   Gymnasium wrapper for RL
│   ├── logging.py           #   Parquet transition logging
│   ├── runner.py            #   Parallel episode runner
│   ├── workloads/           #   4 workload profiles + 8 scenarios
│   └── experts/             #   5 expert scheduling policies
│
├── training/                # Model training pipelines
│   ├── mlp/                 #   Behavioral cloning MLP
│   ├── xgboost/             #   3-classifier cascade
│   └── rl/                  #   PPO with curriculum learning
│
├── evaluation/              # Comparative evaluation
│   ├── metrics.py           #   DCR, latency, balance, power
│   ├── run_eval.py          #   Head-to-head runner
│   ├── statistical_tests.py #   Paired t-tests
│   └── visualization.py    #   Charts
│
├── scripts/                 # CLI entry points
├── deploy/                  # C API header + generated code
├── tests/                   # 204 tests (pytest)
└── docs/                    # This documentation
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Simulation granularity | Scheduling-decision level (not cycle-accurate) | Fast enough for millions of training episodes |
| Priority values | IDLE=0, LOW=2, NORMAL=4, HIGH=6, CRITICAL=7 | Match real SLM-OS kernel values exactly |
| Observation size | 108 dimensions, all [0,1] | Fixed-size for neural networks, normalized for training stability |
| Action decomposition | core x priority_adj x preempt | Smaller per-dimension space vs flat 42-way classification |
| Reward weighting | 50% deadline, 25% latency, 15% balance, 10% power | Deadline compliance is the primary objective |
| Training data | Expert imitation + RL exploration | Imitation provides fast baseline; RL can exceed experts |
| Memory efficiency | Streaming Parquet reads, subprocess isolation | Dataset (11GB train) doesn't fit in RAM as a single array |

## Dependencies

Core: `numpy`, `gymnasium`, `pyarrow`, `torch`, `xgboost`, `stable-baselines3`, `optuna`, `scipy`

Visualization: `matplotlib`, `seaborn`

Export: `onnx`, `onnxruntime`

Full list in `requirements.txt`. Python 3.12+.
