"""MLP training loop with reward-weighted cross-entropy.

AdamW optimizer, cosine annealing lr schedule, early stopping.
See plan Section 7.1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from training.mlp.dataset import SchedulerDataset
from training.mlp.model import SchedulerMLP


def _ts() -> str:
    """Current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class TrainConfig:
    """Training configuration matching plan Section 7.1."""
    n_epochs: int = 50
    batch_size: int = 1024
    lr: float = 1e-3
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    patience: int = 5           # early stopping patience
    hidden1: int = 256
    hidden2: int = 256
    hidden3: int = 128
    dropout: float = 0.1
    device: str = "cpu"


@dataclass
class TrainResult:
    """Training result summary."""
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_acc: float = 0.0


def train_mlp(
    train_dataset: SchedulerDataset,
    val_dataset: SchedulerDataset,
    config: TrainConfig = TrainConfig(),
    save_path: Optional[Path] = None,
) -> tuple[SchedulerMLP, TrainResult]:
    """Train the MLP scheduler model.

    Args:
        train_dataset: Training data.
        val_dataset: Validation data.
        config: Training configuration.
        save_path: Where to save the best model checkpoint.

    Returns:
        Tuple of (trained model, training results).
    """
    device = torch.device(config.device)
    n_actions = max(train_dataset.n_actions, val_dataset.n_actions)
    if n_actions == 0:
        n_actions = 42  # default Jetson

    model = SchedulerMLP(
        n_features=train_dataset.n_features,
        n_actions=n_actions,
        hidden1=config.hidden1,
        hidden2=config.hidden2,
        hidden3=config.hidden3,
        dropout=config.dropout,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.n_epochs,
        eta_min=config.lr_min,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    result = TrainResult()
    patience_counter = 0
    train_start = time.time()
    n_batches = len(train_loader)

    print(f"[{_ts()}] Starting MLP training: {config.n_epochs} epochs, "
          f"{len(train_dataset)} train / {len(val_dataset)} val samples, "
          f"{n_batches} batches/epoch", flush=True)

    for epoch in range(config.n_epochs):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for states, actions, weights in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            weights = weights.to(device)

            logits = model(states)
            # Per-sample cross-entropy weighted by |reward|
            loss_unreduced = nn.functional.cross_entropy(
                logits, actions, reduction="none"
            )
            loss = (loss_unreduced * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(states)
            train_count += len(states)

        scheduler.step()
        train_loss = train_loss_sum / max(train_count, 1)
        result.train_losses.append(train_loss)

        # --- Validate ---
        val_loss, val_acc = _evaluate(model, val_loader, device)
        result.val_losses.append(val_loss)
        result.val_accuracies.append(val_acc)

        epoch_elapsed = time.time() - epoch_start
        improved = ""

        # --- Early stopping ---
        if val_loss < result.best_val_loss:
            result.best_val_loss = val_loss
            result.best_val_acc = val_acc
            result.best_epoch = epoch
            patience_counter = 0
            improved = " *best*"
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"[{_ts()}] Early stopping at epoch {epoch} "
                      f"(no improvement for {config.patience} epochs)", flush=True)
                break

        lr = optimizer.param_groups[0]["lr"]
        print(f"[{_ts()}] Epoch {epoch:3d}/{config.n_epochs} — "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  lr={lr:.2e}  "
              f"({epoch_elapsed:.1f}s){improved}", flush=True)

    total_elapsed = time.time() - train_start
    print(f"[{_ts()}] MLP training complete in {total_elapsed:.1f}s — "
          f"best epoch {result.best_epoch}, "
          f"val_loss={result.best_val_loss:.4f}, "
          f"val_acc={result.best_val_acc:.4f}", flush=True)

    # Load best model if we saved one
    if save_path and save_path.exists():
        model.load_state_dict(torch.load(save_path, weights_only=True))

    return model, result


def _evaluate(
    model: SchedulerMLP,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a dataset. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for states, actions, weights in loader:
            states = states.to(device)
            actions = actions.to(device)
            weights = weights.to(device)

            logits = model(states)
            loss_unreduced = nn.functional.cross_entropy(
                logits, actions, reduction="none"
            )
            total_loss += (loss_unreduced * weights).sum().item()

            preds = logits.argmax(dim=-1)
            correct += (preds == actions).sum().item()
            total += len(states)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def export_onnx(model: SchedulerMLP, path: Path | str, device: str = "cpu") -> None:
    """Export trained model to ONNX format."""
    model.eval()
    model.to(device)
    dummy_input = torch.randn(1, model.n_features, device=device)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(path),
        input_names=["state"],
        output_names=["logits"],
        dynamic_axes={"state": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
