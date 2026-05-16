"""SLM-OS scheduler-trace ingester (#879, sub-ticket of #61).

Parses the binary trace format emitted by SLM-OS's `sched aitrace
dump <path>` shell verb (defined in the SLM-OS repo at
`kernel/sched/ai/sched_trace_ai.c`, format spec in
`docs/sched-trace-format.md`) and converts it into the same Parquet
shape the simulator's `runner.py` produces. That Parquet then feeds
`scripts/_train_mlp.py --source slmos-traces` for fine-tuning the
synthetic-trained MLP weights on real SLM-OS scheduling behaviour.

Wire format (matches SLM-OS docs/sched-trace-format.md v1):

    File header (32 bytes, little-endian):
      u32  magic          = 0x53544C53 (ASCII 'SLTS')
      u16  version        = 1
      u16  record_size    = 480
      u32  record_count   = N
      u32  state_dim      = 108
      u64  total_events_since_start
      u64  dropped_events

    Record (480 bytes, two kinds):
      u8   kind            (1=DECISION, 2=COMPLETION)
      u8   cpu_recorded
      u16  _hdr_pad
      u32  task_id
      u64  timestamp_ns
      Then (kind-dependent), padded to 480.

    DECISION body:
      char policy_name[16]
      i32  action_core
      i32  action_priority
      i32  action_preempt
      u32  _decision_pad
      f32  state[108]

    COMPLETION body:
      u64  dispatch_ns           (always 0 in v1; correlate by task_id)
      u64  completion_ns
      u64  deadline_ns
      u32  latency_to_complete_us
      u8   ran_on_cpu
      u8   deadline_met
      u8   _completion_pad[2]
      u8   _tail_pad[432]
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCHED_TRACE_AI_MAGIC = 0x53544C53  # 'SLTS' little-endian
SCHED_TRACE_AI_VERSION = 1
SCHED_TRACE_AI_RECORD_SIZE = 480
SCHED_TRACE_AI_STATE_DIM = 108
SCHED_TRACE_AI_HEADER_SIZE = 32

KIND_DECISION = 1
KIND_COMPLETION = 2

# Identifier used in the `expert_policy` Parquet column so that the
# downstream `SchedulerDataset` knows which rows came from real SLM-OS
# traces vs. simulated expert demonstrations. The trainer's
# `--source slmos-traces` flag overrides the dataset's default
# expert filter to admit this label.
SLMOS_TRACE_EXPERT_LABEL = "slmos_trace"


class TraceFormatError(ValueError):
    """Raised when the binary doesn't match the documented format."""


@dataclass
class TraceFileHeader:
    magic: int
    version: int
    record_size: int
    record_count: int
    state_dim: int
    total_events_since_start: int
    dropped_events: int


@dataclass
class DecisionRecord:
    cpu_recorded: int
    task_id: int
    timestamp_ns: int
    policy_name: str
    action_core: int
    action_priority: int
    action_preempt: int
    state: np.ndarray   # f32[108]


@dataclass
class CompletionRecord:
    cpu_recorded: int
    task_id: int
    timestamp_ns: int
    dispatch_ns: int
    completion_ns: int
    deadline_ns: int
    latency_to_complete_us: int
    ran_on_cpu: int
    deadline_met: bool


@dataclass
class ParsedTrace:
    header: TraceFileHeader
    decisions: list[DecisionRecord] = field(default_factory=list)
    completions: list[CompletionRecord] = field(default_factory=list)
    # Records whose `kind` byte didn't match KIND_DECISION or
    # KIND_COMPLETION. Forward-compat for a future SLM-OS format that
    # adds a new variant (e.g., MIGRATION). Surfacing the count lets
    # `main()` warn that the ingester is silently dropping records,
    # which is a hint the format/version pinning needs to move.
    unknown_records: int = 0


def _read_header(blob: bytes) -> TraceFileHeader:
    if len(blob) < SCHED_TRACE_AI_HEADER_SIZE:
        raise TraceFormatError(
            f"trace truncated: got {len(blob)} bytes, "
            f"need at least {SCHED_TRACE_AI_HEADER_SIZE}"
        )
    magic, version, record_size, record_count, state_dim, \
        total, dropped = struct.unpack_from("<IHHIIQQ", blob, 0)
    if magic != SCHED_TRACE_AI_MAGIC:
        raise TraceFormatError(
            f"bad magic 0x{magic:08x}, expected 0x{SCHED_TRACE_AI_MAGIC:08x}"
        )
    if version != SCHED_TRACE_AI_VERSION:
        raise TraceFormatError(
            f"unsupported version {version}, this ingester handles "
            f"v{SCHED_TRACE_AI_VERSION}"
        )
    if record_size != SCHED_TRACE_AI_RECORD_SIZE:
        raise TraceFormatError(
            f"record_size mismatch: file says {record_size}, "
            f"expected {SCHED_TRACE_AI_RECORD_SIZE}"
        )
    if state_dim != SCHED_TRACE_AI_STATE_DIM:
        raise TraceFormatError(
            f"state_dim mismatch: file says {state_dim}, "
            f"expected {SCHED_TRACE_AI_STATE_DIM}"
        )
    expected_size = SCHED_TRACE_AI_HEADER_SIZE + record_count * record_size
    if len(blob) < expected_size:
        raise TraceFormatError(
            f"trace truncated: header advertises {record_count} records "
            f"({expected_size} bytes total), got {len(blob)}"
        )
    return TraceFileHeader(
        magic=magic,
        version=version,
        record_size=record_size,
        record_count=record_count,
        state_dim=state_dim,
        total_events_since_start=total,
        dropped_events=dropped,
    )


def _decode_policy_name(raw: bytes) -> str:
    # Null-terminated, ASCII, 16-byte fixed field.
    null_pos = raw.find(b"\x00")
    if null_pos >= 0:
        raw = raw[:null_pos]
    return raw.decode("ascii", errors="replace")


def _parse_decision(blob: bytes, offset: int) -> DecisionRecord:
    # 16-byte common header already consumed by caller; we get the
    # full 480-byte slot here for simplicity.
    cpu_recorded = blob[offset + 1]
    task_id = struct.unpack_from("<I", blob, offset + 4)[0]
    timestamp_ns = struct.unpack_from("<Q", blob, offset + 8)[0]
    policy_name = _decode_policy_name(blob[offset + 16:offset + 32])
    action_core, action_priority, action_preempt = struct.unpack_from(
        "<iii", blob, offset + 32
    )
    # state[] starts at offset + 48, 432 bytes (108 × f32 LE).
    state = np.frombuffer(
        blob, dtype="<f4", count=SCHED_TRACE_AI_STATE_DIM,
        offset=offset + 48,
    ).copy()  # detach from the input buffer
    return DecisionRecord(
        cpu_recorded=cpu_recorded,
        task_id=task_id,
        timestamp_ns=timestamp_ns,
        policy_name=policy_name,
        action_core=action_core,
        action_priority=action_priority,
        action_preempt=action_preempt,
        state=state,
    )


def _parse_completion(blob: bytes, offset: int) -> CompletionRecord:
    cpu_recorded = blob[offset + 1]
    task_id = struct.unpack_from("<I", blob, offset + 4)[0]
    timestamp_ns = struct.unpack_from("<Q", blob, offset + 8)[0]
    dispatch_ns, completion_ns, deadline_ns = struct.unpack_from(
        "<QQQ", blob, offset + 16
    )
    latency_to_complete_us = struct.unpack_from("<I", blob, offset + 40)[0]
    ran_on_cpu = blob[offset + 44]
    deadline_met = bool(blob[offset + 45])
    return CompletionRecord(
        cpu_recorded=cpu_recorded,
        task_id=task_id,
        timestamp_ns=timestamp_ns,
        dispatch_ns=dispatch_ns,
        completion_ns=completion_ns,
        deadline_ns=deadline_ns,
        latency_to_complete_us=latency_to_complete_us,
        ran_on_cpu=ran_on_cpu,
        deadline_met=deadline_met,
    )


def parse_trace(blob: bytes) -> ParsedTrace:
    """Parse a binary trace blob into typed Python records.

    Raises TraceFormatError if the bytes don't match the documented
    v1 format. Returns a ParsedTrace with separated DECISION and
    COMPLETION lists.
    """
    header = _read_header(blob)
    parsed = ParsedTrace(header=header)
    for i in range(header.record_count):
        offset = SCHED_TRACE_AI_HEADER_SIZE + i * header.record_size
        kind = blob[offset]
        if kind == KIND_DECISION:
            parsed.decisions.append(_parse_decision(blob, offset))
        elif kind == KIND_COMPLETION:
            parsed.completions.append(_parse_completion(blob, offset))
        else:
            # Unknown kind: skip but count. Future versions may add
            # new variants; SLM-OS docs/sched-trace-format.md will
            # bump the `version` field if the existing kinds change
            # shape, so an unknown kind here just means "additional
            # record type we haven't been taught." main() surfaces
            # the count so an operator can tell at a glance.
            parsed.unknown_records += 1
            continue
    return parsed


# Defensive upper bound on action_core. SLM-OS targets max out around
# 6 cores + 1 GPU slot (Jetson Orin Nano); anything above 16 is almost
# certainly corrupted trace data and should fail loudly rather than
# silently propagate into an out-of-range Parquet action column.
MAX_ACTION_CORE = 16


def encode_action_index(action_core: int) -> int:
    """Map a DECISION record's `action_core` to the simulator's
    integer action index.

    The simulator's 42-action space encodes `(core * 3 + priority_adj)
    * 2 + preempt` (see slm_sim/actions.py::encode_action). The trace
    captures `action_core` directly and `action_priority` as the
    task's effective priority *after* the decision — which doesn't
    map cleanly to the simulator's `priority_adj ∈ {lower, keep,
    raise}` triplet (we'd need to know the prior priority).

    For fine-tuning, v1 maps `priority_adj = 1` (keep) and `preempt
    = 0` uniformly. Concretely this means the cross-entropy loss will
    push the model's priority_adj output toward 'keep' on every fine-
    tune sample, gradually eroding the synthetic baseline's prio /
    preempt behaviour over many epochs. The v1 config (10 epochs,
    lr=1e-4) is deliberately conservative to limit this erosion;
    masking the prio/preempt outputs from the loss is the proper
    long-term fix and is tracked for v2.

    Raises TraceFormatError if `action_core` is outside [0,
    MAX_ACTION_CORE) — defends against corrupted trace data leaking
    an out-of-range action index into the downstream Parquet.
    """
    if action_core < 0 or action_core >= MAX_ACTION_CORE:
        raise TraceFormatError(
            f"action_core={action_core} out of range "
            f"[0, {MAX_ACTION_CORE}); trace data is likely corrupted"
        )
    priority_adj = 1  # 'keep'
    preempt = 0
    return (action_core * 3 + priority_adj) * 2 + preempt


def _build_reward_for_decisions(
    parsed: ParsedTrace,
) -> dict[int, float]:
    """Compute a reward per DECISION record, by `task_id`.

    Strategy:
    - Matching COMPLETION + deadline + met:    reward = +1.0
    - Matching COMPLETION + deadline + missed: reward = -1.0
    - Matching COMPLETION + no deadline:       reward = +0.1
        (legitimately positive — task finished successfully)
    - No matching COMPLETION (still in-flight at dump time, or
      task slot reused): reward = 0.0
        (unknown outcome — sample weight is |reward| so these
        rows contribute nothing to gradient, equivalent to dropping
        without changing row counts. Prior +0.1 default conflated
        "no deadline" and "no completion" and added survivor-bias
        positive signal for tasks that may have been about to miss
        their deadlines.)
    - Aggregates COMPLETIONs by `task_id`. If multiple COMPLETIONs
      collide on the same id (task slot reuse) we keep the last one;
      the trace ring is short-lived enough that this is rare.
    """
    by_task: dict[int, CompletionRecord] = {}
    for c in parsed.completions:
        by_task[c.task_id] = c

    rewards: dict[int, float] = {}
    for d in parsed.decisions:
        c = by_task.get(d.task_id)
        if c is None:
            # No completion observed — outcome unknown, neutral reward.
            rewards[d.task_id] = 0.0
        elif c.deadline_ns == 0:
            # Completion observed, no deadline — task finished.
            rewards[d.task_id] = 0.1
        else:
            rewards[d.task_id] = 1.0 if c.deadline_met else -1.0
    return rewards


def write_parquet(
    parsed: ParsedTrace,
    out_path: Path | str,
    *,
    platform: str = "raspberry_pi5",
    expert_label: str = SLMOS_TRACE_EXPERT_LABEL,
) -> int:
    """Serialize the trace's DECISION records into a Parquet file with
    the same schema `SchedulerDataset` expects:

      state_000 .. state_107  (float32)
      action                  (int32)
      reward                  (float32)
      expert_policy           (string)
      platform                (string)
      timestamp_ns            (uint64)   — extra column, ignored by the
                                            dataset but useful for
                                            downstream timeline tooling

    Returns the number of rows written.
    """
    out_path = Path(out_path)
    rewards = _build_reward_for_decisions(parsed)

    n = len(parsed.decisions)
    if n == 0:
        # Still emit an empty Parquet so the trainer can `--source
        # slmos-traces` against an empty file and exit cleanly with
        # a helpful message rather than crashing on a missing file.
        empty_schema = pa.schema(
            [(f"state_{j:03d}", pa.float32()) for j in range(SCHED_TRACE_AI_STATE_DIM)]
            + [("action", pa.int32()),
               ("reward", pa.float32()),
               ("expert_policy", pa.string()),
               ("platform", pa.string()),
               ("timestamp_ns", pa.uint64())]
        )
        pq.write_table(empty_schema.empty_table(), out_path)
        return 0

    states = np.stack([d.state for d in parsed.decisions], axis=0)
    actions = np.array(
        [encode_action_index(d.action_core) for d in parsed.decisions],
        dtype=np.int32,
    )
    rewards_arr = np.array(
        [rewards.get(d.task_id, 0.1) for d in parsed.decisions],
        dtype=np.float32,
    )
    timestamps = np.array(
        [d.timestamp_ns for d in parsed.decisions], dtype=np.uint64,
    )

    columns = {
        f"state_{j:03d}": states[:, j].astype(np.float32)
        for j in range(SCHED_TRACE_AI_STATE_DIM)
    }
    columns["action"] = actions
    columns["reward"] = rewards_arr
    columns["expert_policy"] = np.array([expert_label] * n, dtype=object)
    columns["platform"] = np.array([platform] * n, dtype=object)
    columns["timestamp_ns"] = timestamps

    table = pa.table(columns)
    pq.write_table(table, out_path)
    return n


def ingest_path(
    trace_path: Path | str,
    parquet_path: Path | str,
    *,
    platform: str = "raspberry_pi5",
) -> int:
    """End-to-end convenience: read a .bin trace, parse it, and emit
    a Parquet file. Returns row count written."""
    trace_path = Path(trace_path)
    blob = trace_path.read_bytes()
    parsed = parse_trace(blob)
    return write_parquet(parsed, parquet_path, platform=platform)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert an SLM-OS sched-trace .bin into Parquet"
    )
    parser.add_argument("--input", required=True, help="path to the .bin trace")
    parser.add_argument("--output", required=True, help="path to write the .parquet")
    parser.add_argument(
        "--platform",
        default="raspberry_pi5",
        help="platform label written to the `platform` column. Must "
             "match a key in `slm_sim.platforms.PLATFORMS` (currently "
             "'raspberry_pi5', 'jetson_orin_nano', or 'big_little') so "
             "the downstream trainer's `--platform` filter sees the "
             "rows.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress summary output"
    )
    args = parser.parse_args(argv)

    trace_path = Path(args.input)
    parquet_path = Path(args.output)
    blob = trace_path.read_bytes()
    parsed = parse_trace(blob)
    n = write_parquet(parsed, parquet_path, platform=args.platform)
    if not args.quiet:
        print(
            f"slmos-trace ingester: {trace_path} → {parquet_path}\n"
            f"  decisions:   {len(parsed.decisions)}\n"
            f"  completions: {len(parsed.completions)}\n"
            f"  unknown:     {parsed.unknown_records}\n"
            f"  dropped:     {parsed.header.dropped_events}\n"
            f"  rows out:    {n}\n"
            f"  platform:    {args.platform}\n"
        )
        if parsed.unknown_records > 0:
            print(
                f"warning: {parsed.unknown_records} record(s) had an "
                f"unknown `kind` byte and were skipped. The SLM-OS "
                f"format may have added a variant; check whether "
                f"SCHED_TRACE_AI_VERSION needs to move.",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
