#!/usr/bin/env python3
"""Generate the complete training dataset.

Runs expert policies across all scenarios and platforms, logging
transition tuples to Parquet files. See plan Section 6/Phase S3.

Usage:
    python scripts/generate_dataset.py [--small] [--workers N]

    --small     Generate a small test dataset (5 episodes per combo)
    --workers N Number of parallel workers (default: sequential)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slm_sim.runner import RunConfig, batch_to_parquet, run_batch
from slm_sim.workloads.scenarios import SCENARIOS

# Expert policies to generate data from (Section 6.2)
EXPERTS = ["slm_os_hybrid", "edf", "weighted_multi_objective"]
RANDOM_EXPERT = "random"

# Platforms
PLATFORMS = ["jetson_orin_nano", "raspberry_pi5", "big_little"]

# Dataset size (Section 6.2)
EPISODES_PER_EXPERT = 500
EPISODES_RANDOM = 200
EPISODE_DURATION_NS = 10_000_000_000  # 10 seconds

# Small dataset for testing
SMALL_EPISODES_PER_EXPERT = 5
SMALL_EPISODES_RANDOM = 3
SMALL_EPISODE_DURATION_NS = 2_000_000_000  # 2 seconds
SMALL_SCENARIOS = ["light_single", "medium_mixed"]
SMALL_PLATFORMS = ["jetson_orin_nano"]


def generate(
    output_dir: Path,
    small: bool = False,
    n_workers: int | None = None,
) -> None:
    """Generate the full dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = SMALL_SCENARIOS if small else list(SCENARIOS.keys())
    platforms = SMALL_PLATFORMS if small else PLATFORMS
    expert_episodes = SMALL_EPISODES_PER_EXPERT if small else EPISODES_PER_EXPERT
    random_episodes = SMALL_EPISODES_RANDOM if small else EPISODES_RANDOM
    duration = SMALL_EPISODE_DURATION_NS if small else EPISODE_DURATION_NS

    total_combos = (len(EXPERTS) * len(scenarios) * len(platforms)
                    + len(scenarios) * len(platforms))  # random
    completed = 0
    total_rows = 0
    start_time = time.time()

    # Expert policies
    for expert in EXPERTS:
        for scenario in scenarios:
            for platform in platforms:
                completed += 1
                label = f"[{completed}/{total_combos}] {expert}/{scenario}/{platform}"
                print(f"{label}: generating {expert_episodes} episodes...", flush=True)

                config = RunConfig(
                    expert_name=expert,
                    scenario_name=scenario,
                    platform_name=platform,
                    n_episodes=expert_episodes,
                    episode_duration_ns=duration,
                    base_seed=hash((expert, scenario, platform)) & 0x7FFFFFFF,
                )

                results = run_batch(config, n_workers=n_workers)
                filename = f"{expert}__{scenario}__{platform}.parquet"
                rows = batch_to_parquet(results, output_dir / filename)
                total_rows += rows
                print(f"  -> {rows} transitions written", flush=True)

    # Random baseline
    for scenario in scenarios:
        for platform in platforms:
            completed += 1
            label = f"[{completed}/{total_combos}] random/{scenario}/{platform}"
            print(f"{label}: generating {random_episodes} episodes...", flush=True)

            config = RunConfig(
                expert_name=RANDOM_EXPERT,
                scenario_name=scenario,
                platform_name=platform,
                n_episodes=random_episodes,
                episode_duration_ns=duration,
                base_seed=hash(("random", scenario, platform)) & 0x7FFFFFFF,
            )

            results = run_batch(config, n_workers=n_workers)
            filename = f"random__{scenario}__{platform}.parquet"
            rows = batch_to_parquet(results, output_dir / filename)
            total_rows += rows
            print(f"  -> {rows} transitions written", flush=True)

    elapsed = time.time() - start_time
    print(f"\nDone: {total_rows:,} total transitions in {elapsed:.1f}s")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate SLM-OS scheduler training dataset")
    parser.add_argument("--small", action="store_true",
                        help="Generate small test dataset")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: sequential)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: data/raw/)")
    args = parser.parse_args()

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).resolve().parent.parent / "data" / "raw"

    generate(output_dir, small=args.small, n_workers=args.workers)


if __name__ == "__main__":
    main()
