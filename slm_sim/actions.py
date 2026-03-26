"""Action space definition and application logic.

The scheduling decision is decomposed into three sub-actions
(Section 2.2): core assignment, priority adjustment, and preempt flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from slm_sim.models import (
    PRIORITY_LEVELS,
    PRIORITY_MAX,
    PRIORITY_MIN,
    GPURequest,
    TaskState,
)

if TYPE_CHECKING:
    from slm_sim.engine import SimulatorEngine


@dataclass
class SchedulingAction:
    """Decomposed scheduling action."""
    core_assignment: int  # 0..N_CORES-1 = specific core, N_CORES = GPU
    priority_adj: int     # 0 = lower, 1 = keep, 2 = raise
    preempt: bool         # True = preempt current task on target core


def action_space_size(num_cores: int, gpu_available: bool) -> int:
    """Total number of discrete actions: (N_CORES + gpu) * 3 * 2."""
    core_choices = num_cores + (1 if gpu_available else 0)
    return core_choices * 3 * 2


def encode_action(action: SchedulingAction, num_cores: int,
                  gpu_available: bool) -> int:
    """Encode a SchedulingAction into a single integer index."""
    idx = action.core_assignment
    idx = idx * 3 + action.priority_adj
    idx = idx * 2 + (1 if action.preempt else 0)
    return idx


def decode_action(action_idx: int, num_cores: int,
                  gpu_available: bool) -> SchedulingAction:
    """Decode a single integer action index into a SchedulingAction."""
    preempt = bool(action_idx % 2)
    action_idx //= 2
    priority_adj = action_idx % 3
    action_idx //= 3
    core_assignment = action_idx
    return SchedulingAction(
        core_assignment=core_assignment,
        priority_adj=priority_adj,
        preempt=preempt,
    )


def _adjust_priority(current: int, adj: int) -> int:
    """Adjust priority using kernel-compatible levels.

    adj: 0=lower, 1=keep, 2=raise.
    Moves to the next valid kernel priority level (0, 2, 4, 6, 7).
    """
    if adj == 1:
        return current

    idx = 0
    for i, level in enumerate(PRIORITY_LEVELS):
        if level <= current:
            idx = i

    if adj == 0:
        # Lower: move to previous level
        idx = max(0, idx - 1)
    elif adj == 2:
        # Raise: move to next level
        idx = min(len(PRIORITY_LEVELS) - 1, idx + 1)

    return PRIORITY_LEVELS[idx]


def apply_action(engine: SimulatorEngine, task_id: int,
                 action: SchedulingAction) -> None:
    """Apply a scheduling action to the simulator state.

    Assigns the task to the chosen core (or GPU), adjusts priority,
    and optionally preempts the current task on the target core.
    """
    task = engine.tasks.get(task_id)
    if task is None or task.state != TaskState.READY:
        return

    num_cores = len(engine.cores)
    is_gpu_target = (action.core_assignment >= num_cores and
                     engine.gpu.available and task.can_use_gpu)

    # --- GPU assignment ---
    if is_gpu_target:
        task.effective_priority = _adjust_priority(
            task.effective_priority, action.priority_adj
        )
        task.state = TaskState.RUNNING
        gpu_req = GPURequest(
            task_id=task_id,
            ops=task.ops_per_inference,
            submit_time_ns=engine.clock_ns,
        )
        engine.gpu.queue.append(gpu_req)
        return

    # --- CPU assignment ---
    # Resolve core: if target is invalid (GPU slot but no GPU, or isolated),
    # fall back to least-loaded non-isolated core
    target_core_id = action.core_assignment
    if target_core_id >= num_cores:
        target_core_id = _find_least_loaded_core(engine)
    elif engine.cores[target_core_id].isolated:
        target_core_id = _find_least_loaded_core(engine)

    core = engine.cores[target_core_id]

    # Apply priority adjustment
    task.effective_priority = _adjust_priority(
        task.effective_priority, action.priority_adj
    )

    # Preemption logic
    if action.preempt and core.current_task is not None:
        current_task = engine.tasks.get(core.current_task)
        if (current_task is not None and
                current_task.state == TaskState.RUNNING and
                current_task.effective_priority < task.effective_priority):
            # Preempt: move current task back to ready queue
            current_task.state = TaskState.READY
            current_task.assigned_cpu = None
            _insert_in_run_queue(core, current_task.task_id, engine)
            core.context_switches += 1

            # Start the new task immediately
            task.state = TaskState.RUNNING
            task.assigned_cpu = target_core_id
            core.current_task = task_id
            core.quantum_remaining_ns = engine.platform.default_quantum_ns
            core.context_switches += 1
            return

    # No preemption (or preemption not possible): enqueue
    if core.current_task is None:
        # Core is idle — start immediately
        task.state = TaskState.RUNNING
        task.assigned_cpu = target_core_id
        core.current_task = task_id
        core.quantum_remaining_ns = engine.platform.default_quantum_ns
    else:
        # Core busy — add to run queue
        task.assigned_cpu = target_core_id
        _insert_in_run_queue(core, task_id, engine)


def _find_least_loaded_core(engine: SimulatorEngine) -> int:
    """Find the non-isolated core with the shortest run queue."""
    best_core = 0
    best_load = float("inf")
    for core in engine.cores:
        if core.isolated:
            continue
        load = len(core.run_queue) + (1 if core.current_task is not None else 0)
        if load < best_load:
            best_load = load
            best_core = core.core_id
    return best_core


def _insert_in_run_queue(core, task_id: int, engine: SimulatorEngine) -> None:
    """Insert task into core's run queue ordered by effective_priority (descending)."""
    task = engine.tasks[task_id]
    # Find insertion point: keep queue sorted highest priority first
    for i, queued_id in enumerate(core.run_queue):
        queued_task = engine.tasks.get(queued_id)
        if queued_task and queued_task.effective_priority < task.effective_priority:
            core.run_queue.insert(i, task_id)
            return
    core.run_queue.append(task_id)
