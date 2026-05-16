"""Tests for data.slmos_traces — the SLM-OS sched-trace ingester (#879).

Builds a binary blob that matches the documented v1 format exactly
(see SLM-OS `docs/sched-trace-format.md`) and round-trips it through
the parser. Pins the wire-format constants the ingester depends on
so a future change to SLM-OS's record layout that doesn't also bump
the trace `version` is caught here.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from data.slmos_traces import (
    KIND_COMPLETION,
    KIND_DECISION,
    SCHED_TRACE_AI_HEADER_SIZE,
    SCHED_TRACE_AI_MAGIC,
    SCHED_TRACE_AI_RECORD_SIZE,
    SCHED_TRACE_AI_STATE_DIM,
    SCHED_TRACE_AI_VERSION,
    SLMOS_TRACE_EXPERT_LABEL,
    TraceFormatError,
    encode_action_index,
    parse_trace,
    write_parquet,
)


def _build_header(record_count: int, total: int = 0, dropped: int = 0) -> bytes:
    return struct.pack(
        "<IHHIIQQ",
        SCHED_TRACE_AI_MAGIC,
        SCHED_TRACE_AI_VERSION,
        SCHED_TRACE_AI_RECORD_SIZE,
        record_count,
        SCHED_TRACE_AI_STATE_DIM,
        total or record_count,
        dropped,
    )


def _build_decision(
    *,
    cpu: int,
    task_id: int,
    timestamp_ns: int,
    policy: str,
    core: int,
    priority: int,
    preempt: int,
    state: np.ndarray,
) -> bytes:
    """Build one 480-byte DECISION record."""
    assert state.shape == (SCHED_TRACE_AI_STATE_DIM,)
    assert state.dtype == np.float32
    body = bytearray(SCHED_TRACE_AI_RECORD_SIZE)
    body[0] = KIND_DECISION
    body[1] = cpu
    struct.pack_into("<I", body, 4, task_id)
    struct.pack_into("<Q", body, 8, timestamp_ns)
    name_bytes = policy.encode("ascii")[:15]
    body[16:16 + len(name_bytes)] = name_bytes
    # body[16+len(name_bytes):32] already zero (null-padded)
    struct.pack_into("<iii", body, 32, core, priority, preempt)
    # body[44:48] padding stays zero
    state_bytes = state.tobytes()
    assert len(state_bytes) == 432
    body[48:48 + 432] = state_bytes
    return bytes(body)


def _build_completion(
    *,
    cpu: int,
    task_id: int,
    timestamp_ns: int,
    dispatch_ns: int,
    completion_ns: int,
    deadline_ns: int,
    latency_us: int,
    ran_on_cpu: int,
    deadline_met: int,
) -> bytes:
    body = bytearray(SCHED_TRACE_AI_RECORD_SIZE)
    body[0] = KIND_COMPLETION
    body[1] = cpu
    struct.pack_into("<I", body, 4, task_id)
    struct.pack_into("<Q", body, 8, timestamp_ns)
    struct.pack_into("<QQQ", body, 16, dispatch_ns, completion_ns, deadline_ns)
    struct.pack_into("<I", body, 40, latency_us)
    body[44] = ran_on_cpu
    body[45] = deadline_met
    return bytes(body)


def test_action_encoding_matches_simulator():
    # encode_action_index uses (core * 3 + 1) * 2 + 0 = core * 6 + 2.
    assert encode_action_index(0) == 2
    assert encode_action_index(1) == 8
    assert encode_action_index(6) == 38  # GPU slot on 6-core platforms


def test_parse_rejects_bad_magic():
    blob = bytearray(_build_header(record_count=0))
    blob[0:4] = b"NOPE"
    with pytest.raises(TraceFormatError, match="magic"):
        parse_trace(bytes(blob))


def test_parse_rejects_wrong_version():
    blob = bytearray(_build_header(record_count=0))
    # Stomp the version field (offset 4, u16)
    struct.pack_into("<H", blob, 4, 99)
    with pytest.raises(TraceFormatError, match="version"):
        parse_trace(bytes(blob))


def test_parse_rejects_truncated_body():
    header = _build_header(record_count=3)
    # Provide only one record where header advertises three.
    one_record = bytes(SCHED_TRACE_AI_RECORD_SIZE)
    with pytest.raises(TraceFormatError, match="truncated"):
        parse_trace(header + one_record)


def test_roundtrip_single_decision():
    state = np.arange(SCHED_TRACE_AI_STATE_DIM, dtype=np.float32) / 100.0
    decision = _build_decision(
        cpu=2, task_id=42, timestamp_ns=1_000_000,
        policy="ai_mlp", core=3, priority=5, preempt=0,
        state=state,
    )
    blob = _build_header(record_count=1) + decision
    parsed = parse_trace(blob)

    assert parsed.header.record_count == 1
    assert parsed.header.state_dim == SCHED_TRACE_AI_STATE_DIM
    assert len(parsed.decisions) == 1
    assert len(parsed.completions) == 0

    d = parsed.decisions[0]
    assert d.cpu_recorded == 2
    assert d.task_id == 42
    assert d.timestamp_ns == 1_000_000
    assert d.policy_name == "ai_mlp"
    assert d.action_core == 3
    assert d.action_priority == 5
    assert d.action_preempt == 0
    np.testing.assert_array_equal(d.state, state)


def test_roundtrip_completion_correlation():
    """A DECISION + matching COMPLETION (deadline_met=1) should
    produce reward=+1.0 in the emitted Parquet."""
    state = np.zeros(SCHED_TRACE_AI_STATE_DIM, dtype=np.float32)
    blob = (
        _build_header(record_count=2)
        + _build_decision(
            cpu=0, task_id=7, timestamp_ns=100, policy="heuristic",
            core=1, priority=4, preempt=0, state=state,
        )
        + _build_completion(
            cpu=0, task_id=7, timestamp_ns=200,
            dispatch_ns=100, completion_ns=200, deadline_ns=500,
            latency_us=100, ran_on_cpu=1, deadline_met=1,
        )
    )
    parsed = parse_trace(blob)
    assert len(parsed.decisions) == 1
    assert len(parsed.completions) == 1
    assert parsed.completions[0].deadline_met is True


def test_write_parquet_schema_matches_dataset(tmp_path: Path):
    """The ingester's Parquet schema must match the columns
    `SchedulerDataset` reads: state_000..state_107, action, reward,
    expert_policy, platform. Extra columns (timestamp_ns) are fine."""
    state = np.linspace(0.0, 1.0, SCHED_TRACE_AI_STATE_DIM, dtype=np.float32)
    blob = (
        _build_header(record_count=2)
        + _build_decision(
            cpu=0, task_id=1, timestamp_ns=10, policy="heuristic",
            core=2, priority=4, preempt=0, state=state,
        )
        + _build_completion(
            cpu=0, task_id=1, timestamp_ns=20,
            dispatch_ns=10, completion_ns=20, deadline_ns=15,
            latency_us=10, ran_on_cpu=2, deadline_met=0,  # missed
        )
    )
    parsed = parse_trace(blob)

    out = tmp_path / "trace.parquet"
    n = write_parquet(parsed, out, platform="raspi5")
    assert n == 1

    table = pq.read_table(out)
    cols = table.column_names
    # All 108 state columns present.
    for j in range(SCHED_TRACE_AI_STATE_DIM):
        assert f"state_{j:03d}" in cols
    assert "action" in cols
    assert "reward" in cols
    assert "expert_policy" in cols
    assert "platform" in cols
    assert "timestamp_ns" in cols

    # Single row, missed-deadline → reward -1.0
    row = table.to_pylist()[0]
    assert row["action"] == encode_action_index(2)
    assert row["reward"] == pytest.approx(-1.0)
    assert row["expert_policy"] == SLMOS_TRACE_EXPERT_LABEL
    assert row["platform"] == "raspi5"


def test_write_parquet_default_reward_for_unmatched_decision(tmp_path: Path):
    """A DECISION without a matching COMPLETION gets reward=+0.1
    (we observed the decision; no outcome info)."""
    state = np.ones(SCHED_TRACE_AI_STATE_DIM, dtype=np.float32)
    blob = _build_header(record_count=1) + _build_decision(
        cpu=0, task_id=99, timestamp_ns=10, policy="ai_xgb",
        core=0, priority=6, preempt=0, state=state,
    )
    parsed = parse_trace(blob)
    out = tmp_path / "trace.parquet"
    write_parquet(parsed, out)
    row = pq.read_table(out).to_pylist()[0]
    assert row["reward"] == pytest.approx(0.1)


def test_unknown_kind_skipped(tmp_path: Path):
    """Records with kind > 2 are silently skipped so a future SLM-OS
    format extension (adding kind=3 = MIGRATION, say) doesn't break
    older ingesters."""
    state = np.zeros(SCHED_TRACE_AI_STATE_DIM, dtype=np.float32)
    unknown = bytearray(SCHED_TRACE_AI_RECORD_SIZE)
    unknown[0] = 99  # not 1 or 2
    decision = _build_decision(
        cpu=0, task_id=1, timestamp_ns=10, policy="heuristic",
        core=0, priority=4, preempt=0, state=state,
    )
    blob = _build_header(record_count=2) + bytes(unknown) + decision
    parsed = parse_trace(blob)
    assert len(parsed.decisions) == 1
    assert len(parsed.completions) == 0


def test_format_constants_pin():
    """Pin the wire-format constants — these must move in lockstep
    with SLM-OS's `kernel/include/sched_trace_ai.h`. If SLM-OS bumps
    record_size or state_dim without also bumping version, this test
    fires and the maintainer chases the symmetric change."""
    assert SCHED_TRACE_AI_MAGIC == 0x53544C53
    assert SCHED_TRACE_AI_VERSION == 1
    assert SCHED_TRACE_AI_RECORD_SIZE == 480
    assert SCHED_TRACE_AI_STATE_DIM == 108
    assert SCHED_TRACE_AI_HEADER_SIZE == 32
