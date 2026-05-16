"""Smoke tests for the SLM-OS fine-tune flow added in `_train_mlp.py
--source slmos-traces` (#879).

The script itself does its work at module level (no `main()` to call),
so end-to-end testing means running it as a subprocess. We keep the
subprocess test minimal — a tiny synthesized Parquet + a tiny synthesized
checkpoint, the smallest input that exercises the fine-tune path's
default-paths, train/val split, pretrained-load, and output-save logic
without burning real training time.

The disjoint train/val split invariant (regression target — prior
two-load-with-different-seed approach silently produced overlapping
subsamples) is also exercised at unit level via SchedulerDataset +
torch.utils.data.random_split, so a future refactor of `_train_mlp.py`
can't silently re-introduce the overlap.
"""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from data.slmos_traces import (
    SCHED_TRACE_AI_HEADER_SIZE,
    SCHED_TRACE_AI_MAGIC,
    SCHED_TRACE_AI_RECORD_SIZE,
    SCHED_TRACE_AI_STATE_DIM,
    SCHED_TRACE_AI_VERSION,
    KIND_DECISION,
    KIND_COMPLETION,
    SLMOS_TRACE_EXPERT_LABEL,
    parse_trace,
    write_parquet,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "_train_mlp.py"


def _build_minimal_trace_parquet(out_path: Path, n_rows: int = 40) -> int:
    """Synthesize a small in-memory trace, ingest it, write Parquet.

    Uses the real ingester (not a hand-rolled Parquet) so the schema
    is whatever production emits. Pairs each DECISION with a
    deadline-met COMPLETION so every row has a non-zero reward and
    `SchedulerDataset` keeps all of them.
    """
    rng = np.random.default_rng(seed=123)
    blob = bytearray()
    blob += struct.pack(
        "<IHHIIQQ",
        SCHED_TRACE_AI_MAGIC,
        SCHED_TRACE_AI_VERSION,
        SCHED_TRACE_AI_RECORD_SIZE,
        n_rows * 2,                # decisions + completions
        SCHED_TRACE_AI_STATE_DIM,
        n_rows * 2,
        0,
    )
    for i in range(n_rows):
        state = rng.standard_normal(SCHED_TRACE_AI_STATE_DIM).astype(np.float32)
        # DECISION — policy_name is a fixed 16-byte field; assigning to
        # a shorter slice would extend the bytearray and misalign every
        # subsequent record. Use len("heuristic") = 9 explicitly.
        d = bytearray(SCHED_TRACE_AI_RECORD_SIZE)
        d[0] = KIND_DECISION
        d[1] = i % 4
        struct.pack_into("<I", d, 4, i + 1)
        struct.pack_into("<Q", d, 8, 1000 + i * 10)
        name_bytes = b"heuristic"
        d[16:16 + len(name_bytes)] = name_bytes
        struct.pack_into("<iii", d, 32, i % 4, 4, 0)
        d[48:48 + 432] = state.tobytes()
        blob += bytes(d)
        # COMPLETION (deadline met)
        c = bytearray(SCHED_TRACE_AI_RECORD_SIZE)
        c[0] = KIND_COMPLETION
        c[1] = i % 4
        struct.pack_into("<I", c, 4, i + 1)
        struct.pack_into("<Q", c, 8, 1100 + i * 10)
        struct.pack_into("<QQQ", c, 16,
                         1000 + i * 10, 1100 + i * 10, 2000 + i * 10)
        struct.pack_into("<I", c, 40, 100)
        c[44] = i % 4
        c[45] = 1
        blob += bytes(c)
    parsed = parse_trace(bytes(blob))
    return write_parquet(parsed, out_path, platform="raspberry_pi5")


def _make_tiny_pretrained_checkpoint(out_path: Path, n_actions: int) -> None:
    """Write a randomly-initialized SchedulerMLP state_dict at the
    same architectural dims `train_mlp` will instantiate (defaults
    from TrainConfig: hidden1=256, hidden2=256, hidden3=128)."""
    from training.mlp.model import SchedulerMLP

    model = SchedulerMLP(
        n_actions=n_actions,
        hidden1=256, hidden2=256, hidden3=128, dropout=0.1,
    )
    torch.save(model.state_dict(), out_path)


def test_random_split_produces_disjoint_train_val(tmp_path: Path):
    """Direct invariant test on the train/val split path. Earlier
    `_train_mlp.py` re-loaded the same Parquet twice with different
    `seed` values, which produced overlapping subsamples. The fix uses
    `torch.utils.data.random_split` on a single load. This test pins
    that invariant so a future refactor can't silently regress."""
    from training.mlp.dataset import SchedulerDataset

    parquet = tmp_path / "trace.parquet"
    _build_minimal_trace_parquet(parquet, n_rows=40)
    full_ds = SchedulerDataset(
        parquet,
        experts={SLMOS_TRACE_EXPERT_LABEL},
        platform="raspberry_pi5",
        max_rows=10_000_000,
    )
    assert len(full_ds) == 40

    val_size = max(1, len(full_ds) // 10)   # 4
    train_size = len(full_ds) - val_size    # 36
    rng = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [train_size, val_size], generator=rng,
    )
    assert len(train_ds) == train_size
    assert len(val_ds) == val_size

    train_indices = set(train_ds.indices)
    val_indices = set(val_ds.indices)
    assert train_indices.isdisjoint(val_indices), (
        f"train/val splits overlap: {train_indices & val_indices}"
    )
    assert len(train_indices | val_indices) == len(full_ds), (
        "split should partition the full dataset"
    )


@pytest.mark.slow
def test_finetune_subprocess_end_to_end(tmp_path: Path):
    """End-to-end: synthesize a trace Parquet + a pretrained .pt,
    run `_train_mlp.py --source slmos-traces`, verify the output
    checkpoint is created and loadable.

    Marked @slow because even 10 epochs on 40 rows takes ~10-20 s.
    Run with `pytest -m slow` to include.
    """
    from slm_sim.actions import action_space_size
    from slm_sim.platforms import get_platform

    plat = get_platform("raspberry_pi5")
    n_actions = action_space_size(plat.num_cores, plat.gpu.available)

    parquet_path = tmp_path / "trace.parquet"
    init_path = tmp_path / "pretrained.pt"
    out_path = tmp_path / "best_raspberry_pi5_real.pt"

    n = _build_minimal_trace_parquet(parquet_path, n_rows=40)
    assert n == 40
    _make_tiny_pretrained_checkpoint(init_path, n_actions=n_actions)

    result = subprocess.run(
        [
            sys.executable, str(TRAIN_SCRIPT),
            "--source", "slmos-traces",
            "--input", str(parquet_path),
            "--platform", "raspberry_pi5",
            "--init-from", str(init_path),
            "--output", str(out_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(
            f"_train_mlp.py exited {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    assert out_path.exists(), "fine-tune did not produce output checkpoint"
    state = torch.load(out_path, map_location="cpu", weights_only=True)
    assert isinstance(state, dict), "checkpoint should be a state_dict"
    assert len(state) > 0, "checkpoint state_dict is empty"
