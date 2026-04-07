#!/usr/bin/env python3
"""Train PPO model. Called by train_all.py in a subprocess."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.rl.train import train_ppo, PPOConfig

model = train_ppo(PPOConfig(
    total_timesteps=5_000_000,
    save_dir="models/ppo",
))

print("PPO training complete")
