# Evaluation Framework

The evaluation framework (`evaluation/`) compares trained models against expert baselines using statistical rigor.

## Metrics (`metrics.py`)

### Primary Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Deadline Compliance Rate (DCR) | >= 95% | Fraction of deadline-bearing tasks completed on time |
| Mean latency | < component target | Average scheduling + execution latency |
| P99 latency | < 2x component target | 99th percentile latency |
| Scheduling overhead | < 5% of quantum | Time spent in scheduling decisions |

### Secondary Metrics

| Metric | Description |
|--------|-------------|
| Core utilization balance | 1 - CV(utilizations); higher = more balanced |
| Throughput | Tasks completed per second |
| Power efficiency | 1 - weighted active power |
| Deadline miss severity | Mean overshoot for missed deadlines |
| Starvation count | Tasks waiting > 10x expected duration |
| Priority inversion count | Lower-priority task running while higher waits |

All metrics are computed per-episode by `compute_eval_metrics()`, which returns an `EvalMetrics` dataclass.

## Evaluation Runner (`run_eval.py`)

`run_full_evaluation()` runs all agents on identical test episodes:

1. Load test scenarios (from held-out test split or fresh generation)
2. For each (agent, scenario, platform) combination:
   - Run 200 episodes with identical random seeds (same seeds across agents for paired comparison)
   - Collect per-episode metrics
3. `summarize_results()` computes mean, std, and 95% confidence intervals

`evaluate_agent()` runs a single agent on a batch of episodes and returns metrics.

## Statistical Tests (`statistical_tests.py`)

### Paired Comparison

`paired_comparison(metrics_a, metrics_b, metric_name)` runs a paired t-test:

- Uses `scipy.stats.ttest_rel` (paired, two-sided)
- Same episodes with same seeds, so differences are paired
- Returns: t-statistic, p-value, mean difference, 95% CI of difference
- Significance threshold: p < 0.05

### Full Comparison Suite

`run_all_comparisons(all_results, baseline="slm_os_hybrid")` compares every agent against the baseline on all primary metrics, returning a structured results dict.

## Visualization (`visualization.py`)

Four chart generators:

| Function | Chart Type | Shows |
|----------|-----------|-------|
| `plot_dcr_comparison()` | Grouped bar | DCR by agent across scenarios |
| `plot_latency_comparison()` | Grouped bar | Mean latency by agent |
| `plot_throughput_comparison()` | Grouped bar | Tasks/sec by agent |
| `generate_all_charts()` | All above | Saves to `results/charts/` |

## Evaluation Design Principles

- **Paired testing:** Same random seeds across agents eliminates noise from different task sequences
- **Episode-level aggregation:** Each episode is one data point; avoids autocorrelation within episodes
- **Out-of-distribution testing:** Models can be tested on scenarios not seen during training (e.g., `burst_storm` held out for MLP/XGBoost)
- **Multiple baselines:** Compare against heuristic (production), EDF (simple optimal), random (floor), and oracle (ceiling)
