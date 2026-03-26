"""Discrete-event simulation engine for SLM-OS scheduling.

Implements the core simulation loop: event queue (min-heap), clock
advancement, event processing, and agent interaction. See plan
Section 1.6 for the event-driven architecture.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Optional

import numpy as np

from slm_sim.models import (
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_LEVELS,
    CoreState,
    GPUModel,
    GPURequest,
    MemorySubsystem,
    SimTask,
    TaskState,
)

if TYPE_CHECKING:
    from slm_sim.actions import SchedulingAction
    from slm_sim.experts import ExpertPolicy
    from slm_sim.platforms import PlatformProfile
    from slm_sim.workloads.scenarios import ScenarioComposer


class EventType(IntEnum):
    """Simulator event types ordered by processing priority."""
    TASK_ARRIVAL = 0
    TASK_COMPLETION = 1
    GPU_COMPLETION = 2
    QUANTUM_EXPIRED = 3
    DEADLINE_CHECK = 4
    SCHEDULING_DECISION = 5
    MIGRATION_CHECK = 6


@dataclass(order=True)
class Event:
    """A timestamped event in the simulation priority queue."""
    timestamp_ns: int
    event_type: EventType
    sequence: int = field(compare=True)
    task_id: Optional[int] = field(default=None, compare=False)
    core_id: Optional[int] = field(default=None, compare=False)
    data: dict = field(default_factory=dict, compare=False)


@dataclass
class EpisodeMetrics:
    """Metrics collected over one simulation episode."""
    total_tasks_arrived: int = 0
    total_tasks_completed: int = 0
    total_deadline_tasks: int = 0
    deadline_met: int = 0
    deadline_missed: int = 0
    total_latency_ns: int = 0
    total_context_switches: int = 0
    scheduling_decisions: int = 0

    @property
    def deadline_compliance_rate(self) -> float:
        if self.total_deadline_tasks == 0:
            return 1.0
        return self.deadline_met / self.total_deadline_tasks

    @property
    def mean_latency_ns(self) -> float:
        if self.total_tasks_completed == 0:
            return 0.0
        return self.total_latency_ns / self.total_tasks_completed


# Periodic event intervals
DEADLINE_CHECK_INTERVAL_NS = 1_000_000    # 1 ms
MIGRATION_CHECK_INTERVAL_NS = 10_000_000  # 10 ms

# Sliding window for utilization tracking
UTILIZATION_WINDOW_NS = 100_000_000  # 100 ms


class SimulatorEngine:
    """Discrete-event simulation engine for SLM-OS scheduling."""

    def __init__(
        self,
        platform: PlatformProfile,
        episode_duration_ns: int = 10_000_000_000,  # 10 seconds
        warmup_ns: int = 1_000_000_000,  # 1 second
        seed: int = 42,
    ):
        self.platform = platform
        self.episode_duration_ns = episode_duration_ns
        self.warmup_ns = warmup_ns
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # State — initialized in reset()
        self.clock_ns: int = 0
        self.event_queue: list[Event] = []
        self.event_sequence: int = 0
        self.tasks: dict[int, SimTask] = {}
        self.cores: list[CoreState] = []
        self.memory: MemorySubsystem = MemorySubsystem()
        self.gpu: GPUModel = GPUModel()
        self.metrics: EpisodeMetrics = EpisodeMetrics()
        self.next_task_id: int = 1

        # Per-core busy tracking for utilization calculation
        self._core_busy_ns: list[int] = []
        self._last_utilization_update_ns: int = 0

        # Completed tasks since last scheduling decision (for reward)
        self._recently_completed: list[dict] = []

        # Transition log
        self.transitions: list[dict] = []

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset simulator to initial state for a new episode."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.clock_ns = 0
        self.event_queue = []
        self.event_sequence = 0
        self.tasks = {}
        self.next_task_id = 1
        self.transitions = []
        self.metrics = EpisodeMetrics()
        self._recently_completed = []
        self._last_utilization_update_ns = 0

        # Deep copy cores from platform profile
        self.cores = [
            CoreState(
                core_id=c.core_id,
                core_type=c.core_type,
                max_freq_mhz=c.max_freq_mhz,
                power_weight=c.power_weight,
                cache_size_kb=c.cache_size_kb,
            )
            for c in self.platform.cores
        ]
        self._core_busy_ns = [0] * len(self.cores)

        # Reset memory subsystem
        self.memory = MemorySubsystem(
            total_ram_mb=self.platform.memory.total_ram_mb,
            weight_pool_mb=self.platform.memory.weight_pool_mb,
            workspace_pool_mb=self.platform.memory.workspace_pool_mb,
        )

        # Reset GPU
        self.gpu = GPUModel(
            available=self.platform.gpu.available,
            throughput_tflops=self.platform.gpu.throughput_tflops,
            dma_overhead_ns=self.platform.gpu.dma_overhead_ns,
            cache_flush_overhead_ns=self.platform.gpu.cache_flush_overhead_ns,
        )

        # Schedule periodic events
        self.push_event(EventType.DEADLINE_CHECK, DEADLINE_CHECK_INTERVAL_NS)
        self.push_event(EventType.MIGRATION_CHECK, MIGRATION_CHECK_INTERVAL_NS)

    def push_event(self, event_type: EventType, timestamp_ns: int,
                   task_id: Optional[int] = None,
                   core_id: Optional[int] = None,
                   data: Optional[dict] = None) -> None:
        """Add an event to the priority queue."""
        evt = Event(
            timestamp_ns=timestamp_ns,
            event_type=event_type,
            sequence=self.event_sequence,
            task_id=task_id,
            core_id=core_id,
            data=data or {},
        )
        self.event_sequence += 1
        heapq.heappush(self.event_queue, evt)

    def pop_event(self) -> Optional[Event]:
        """Remove and return the next event, or None if queue is empty."""
        if not self.event_queue:
            return None
        return heapq.heappop(self.event_queue)

    def allocate_task_id(self) -> int:
        """Get next unique task ID."""
        tid = self.next_task_id
        self.next_task_id += 1
        return tid

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def run_episode(self, agent: ExpertPolicy,
                    workload: ScenarioComposer) -> EpisodeMetrics:
        """Run a single simulation episode.

        Args:
            agent: Scheduling policy (expert or learned).
            workload: Scenario composer that seeds task arrivals.

        Returns:
            EpisodeMetrics after simulation completes.
        """
        from slm_sim.observation import extract_observation
        from slm_sim.reward import compute_reward

        self.reset()

        # Seed workload events
        workload.seed_all_events(self, 0, self.episode_duration_ns)

        prev_obs = None
        prev_action_idx = None

        while self.event_queue:
            event = self.pop_event()
            if event.timestamp_ns > self.episode_duration_ns:
                break

            # Advance clock
            old_clock = self.clock_ns
            self.clock_ns = event.timestamp_ns
            self._update_core_busy_time(old_clock, self.clock_ns)

            needs_decision = self._process_event(event)

            if not needs_decision:
                continue

            # --- Scheduling decision point ---
            # Find the highest-priority ready task
            ready_task = self._pick_highest_priority_ready_task()
            if ready_task is None:
                continue

            self.metrics.scheduling_decisions += 1

            # Extract state
            obs = extract_observation(self)

            # Log previous transition if we have one
            if prev_obs is not None:
                reward = compute_reward(self, self._recently_completed)
                self.transitions.append({
                    "state": prev_obs,
                    "action": prev_action_idx,
                    "reward": reward.total,
                    "reward_deadline": reward.deadline,
                    "reward_latency": reward.latency,
                    "reward_balance": reward.balance,
                    "reward_power": reward.power,
                    "next_state": obs,
                    "done": False,
                    "sim_time_ns": self.clock_ns,
                })
                self._recently_completed = []

            # Get agent's decision
            action = agent.decide(self, ready_task.task_id)

            from slm_sim.actions import apply_action, encode_action
            action_idx = encode_action(
                action, len(self.cores), self.gpu.available
            )
            apply_action(self, ready_task.task_id, action)

            # Schedule task completion if it's now running on a core
            self._schedule_completion_for_task(ready_task.task_id)

            prev_obs = obs
            prev_action_idx = action_idx

        # Final transition
        if prev_obs is not None:
            from slm_sim.reward import compute_reward
            obs = extract_observation(self)
            reward = compute_reward(self, self._recently_completed)
            self.transitions.append({
                "state": prev_obs,
                "action": prev_action_idx,
                "reward": reward.total,
                "reward_deadline": reward.deadline,
                "reward_latency": reward.latency,
                "reward_balance": reward.balance,
                "reward_power": reward.power,
                "next_state": obs,
                "done": True,
                "sim_time_ns": self.clock_ns,
            })

        return self.metrics

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    def _process_event(self, event: Event) -> bool:
        """Process a single event. Returns True if a scheduling decision is needed."""
        handlers = {
            EventType.TASK_ARRIVAL: self._handle_task_arrival,
            EventType.TASK_COMPLETION: self._handle_task_completion,
            EventType.GPU_COMPLETION: self._handle_gpu_completion,
            EventType.QUANTUM_EXPIRED: self._handle_quantum_expired,
            EventType.DEADLINE_CHECK: self._handle_deadline_check,
            EventType.SCHEDULING_DECISION: lambda e: True,
            EventType.MIGRATION_CHECK: self._handle_migration_check,
        }
        handler = handlers.get(event.event_type)
        if handler is None:
            return False
        return handler(event)

    def _handle_task_arrival(self, event: Event) -> bool:
        """Handle TASK_ARRIVAL: register task, trigger scheduling."""
        task_data = event.data
        task = SimTask(
            task_id=task_data["task_id"],
            name=task_data.get("name", f"task_{task_data['task_id']}"),
            state=TaskState.READY,
            priority=task_data.get("priority", 4),
            effective_priority=task_data.get("priority", 4),
            model_handle=task_data.get("model_handle"),
            model_size_mb=task_data.get("model_size_mb", 0.0),
            working_set_mb=task_data.get("working_set_mb", 0.0),
            ops_per_inference=task_data.get("ops_per_inference", 0.0),
            inference_duration_ns=task_data.get("inference_duration_ns", 0),
            can_use_gpu=task_data.get("can_use_gpu", False),
            deadline_ns=task_data.get("deadline_ns", 0),
            arrival_time_ns=self.clock_ns,
            remaining_work_ns=task_data.get("inference_duration_ns", 0),
            component_type=task_data.get("component_type", 4),
            cpu_affinity=task_data.get("cpu_affinity", -1),
        )
        self.tasks[task.task_id] = task
        self.metrics.total_tasks_arrived += 1

        # Track memory allocation for model-bearing tasks
        if task.model_size_mb > 0:
            self.memory.weight_pool_used_mb += task.model_size_mb
        if task.working_set_mb > 0:
            self.memory.workspace_pool_used_mb += task.working_set_mb

        return True  # Triggers scheduling decision

    def _handle_task_completion(self, event: Event) -> bool:
        """Handle TASK_COMPLETION: record metrics, free core, free memory."""
        task_id = event.task_id
        core_id = event.core_id
        task = self.tasks.get(task_id)
        if task is None:
            return False

        if task.state != TaskState.RUNNING:
            return False

        task.state = TaskState.TERMINATED
        completion_ns = self.clock_ns
        latency_ns = completion_ns - task.arrival_time_ns

        # Update metrics
        self.metrics.total_tasks_completed += 1
        self.metrics.total_latency_ns += latency_ns

        if task.deadline_ns > 0:
            self.metrics.total_deadline_tasks += 1
            if completion_ns <= task.deadline_ns:
                self.metrics.deadline_met += 1
            else:
                self.metrics.deadline_missed += 1

        # Record for reward computation
        self._recently_completed.append({
            "task_id": task_id,
            "arrival_time_ns": task.arrival_time_ns,
            "completion_time_ns": completion_ns,
            "deadline_ns": task.deadline_ns,
            "component_type": int(task.component_type),
        })

        # Free core
        if core_id is not None and core_id < len(self.cores):
            core = self.cores[core_id]
            if core.current_task == task_id:
                core.current_task = None
                # Start next task from run queue if available
                self._start_next_from_queue(core)

        # Free memory
        if task.model_size_mb > 0:
            self.memory.weight_pool_used_mb = max(
                0.0, self.memory.weight_pool_used_mb - task.model_size_mb
            )
        if task.working_set_mb > 0:
            self.memory.workspace_pool_used_mb = max(
                0.0, self.memory.workspace_pool_used_mb - task.working_set_mb
            )

        return True  # Triggers scheduling decision

    def _handle_quantum_expired(self, event: Event) -> bool:
        """Handle QUANTUM_EXPIRED: preempt and re-enqueue running task."""
        core_id = event.core_id
        if core_id is None or core_id >= len(self.cores):
            return False

        core = self.cores[core_id]
        if core.current_task is None:
            return False

        task = self.tasks.get(core.current_task)
        if task is None or task.state != TaskState.RUNNING:
            core.current_task = None
            return True

        # Preempt: update remaining work and move back to ready
        elapsed_ns = self.platform.default_quantum_ns
        task.remaining_work_ns = max(0, task.remaining_work_ns - elapsed_ns)

        if task.remaining_work_ns <= 0:
            # Task actually finished during this quantum
            self.push_event(
                EventType.TASK_COMPLETION,
                self.clock_ns,
                task_id=task.task_id,
                core_id=core_id,
            )
            return False

        # Re-enqueue the preempted task
        task.state = TaskState.READY
        task.assigned_cpu = None
        core.current_task = None
        core.context_switches += 1
        self.metrics.total_context_switches += 1

        # Add context switch overhead
        cs_cost = self.rng.integers(
            self.platform.context_switch_min_ns,
            self.platform.context_switch_max_ns + 1,
        )
        self.clock_ns += cs_cost

        # Start next from queue
        self._start_next_from_queue(core)

        return True  # Triggers scheduling for the preempted task

    def _handle_deadline_check(self, event: Event) -> bool:
        """Handle DEADLINE_CHECK: update deadline boosts for all tasks."""
        for task in self.tasks.values():
            if task.state not in (TaskState.READY, TaskState.RUNNING):
                continue
            if task.deadline_ns == 0:
                continue

            remaining_ns = task.deadline_ns - self.clock_ns
            if remaining_ns < 10_000_000:       # < 10 ms
                task.effective_priority = PRIORITY_CRITICAL
            elif remaining_ns < 50_000_000:     # < 50 ms
                task.effective_priority = max(task.effective_priority, PRIORITY_HIGH)
            elif remaining_ns < 100_000_000:    # < 100 ms
                boosted = min(task.priority + 1, PRIORITY_CRITICAL)
                task.effective_priority = max(task.effective_priority, boosted)

        # Schedule next deadline check
        next_check = self.clock_ns + DEADLINE_CHECK_INTERVAL_NS
        if next_check <= self.episode_duration_ns:
            self.push_event(EventType.DEADLINE_CHECK, next_check)

        return False  # Doesn't trigger scheduling by itself

    def _handle_gpu_completion(self, event: Event) -> bool:
        """Handle GPU_COMPLETION: mark task complete, start next GPU job."""
        task_id = event.task_id
        task = self.tasks.get(task_id)
        if task is not None and task.state == TaskState.RUNNING:
            task.state = TaskState.TERMINATED
            completion_ns = self.clock_ns
            latency_ns = completion_ns - task.arrival_time_ns

            self.metrics.total_tasks_completed += 1
            self.metrics.total_latency_ns += latency_ns

            if task.deadline_ns > 0:
                self.metrics.total_deadline_tasks += 1
                if completion_ns <= task.deadline_ns:
                    self.metrics.deadline_met += 1
                else:
                    self.metrics.deadline_missed += 1

            self._recently_completed.append({
                "task_id": task_id,
                "arrival_time_ns": task.arrival_time_ns,
                "completion_time_ns": completion_ns,
                "deadline_ns": task.deadline_ns,
                "component_type": int(task.component_type),
            })

            # Free memory
            if task.model_size_mb > 0:
                self.memory.weight_pool_used_mb = max(
                    0.0, self.memory.weight_pool_used_mb - task.model_size_mb
                )
            if task.working_set_mb > 0:
                self.memory.workspace_pool_used_mb = max(
                    0.0, self.memory.workspace_pool_used_mb - task.working_set_mb
                )

        self.gpu.current_job = None
        self._start_next_gpu_job()

        return True

    def _handle_migration_check(self, event: Event) -> bool:
        """Handle MIGRATION_CHECK: rebalance load across cores."""
        # Update utilization from busy time
        self._update_utilization_stats()

        # Find most- and least-loaded non-isolated cores
        non_isolated = [c for c in self.cores if not c.isolated]
        if len(non_isolated) < 2:
            # Schedule next check
            next_check = self.clock_ns + MIGRATION_CHECK_INTERVAL_NS
            if next_check <= self.episode_duration_ns:
                self.push_event(EventType.MIGRATION_CHECK, next_check)
            return False

        max_core = max(non_isolated, key=lambda c: len(c.run_queue))
        min_core = min(non_isolated, key=lambda c: len(c.run_queue))

        # Migrate if imbalance > 2 tasks
        if len(max_core.run_queue) - len(min_core.run_queue) > 2:
            if max_core.run_queue:
                # Move last task from overloaded core to underloaded
                migrated_id = max_core.run_queue.pop()
                task = self.tasks.get(migrated_id)
                if task is not None:
                    task.assigned_cpu = min_core.core_id
                    min_core.run_queue.append(migrated_id)

        # Schedule next
        next_check = self.clock_ns + MIGRATION_CHECK_INTERVAL_NS
        if next_check <= self.episode_duration_ns:
            self.push_event(EventType.MIGRATION_CHECK, next_check)

        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pick_highest_priority_ready_task(self) -> Optional[SimTask]:
        """Find the highest effective_priority READY task."""
        best = None
        for task in self.tasks.values():
            if task.state != TaskState.READY:
                continue
            if best is None or task.effective_priority > best.effective_priority:
                best = task
            elif (task.effective_priority == best.effective_priority and
                  task.arrival_time_ns < best.arrival_time_ns):
                best = task  # FIFO tie-break
        return best

    def _start_next_from_queue(self, core: CoreState) -> None:
        """Pop the next task from a core's run queue and start it."""
        while core.run_queue:
            next_id = core.run_queue.pop(0)
            task = self.tasks.get(next_id)
            if task is not None and task.state == TaskState.READY:
                task.state = TaskState.RUNNING
                task.assigned_cpu = core.core_id
                core.current_task = next_id
                core.quantum_remaining_ns = self.platform.default_quantum_ns
                core.context_switches += 1
                self.metrics.total_context_switches += 1
                self._schedule_completion_for_task(next_id)
                return
        # Queue empty — core goes idle
        core.current_task = None

    def _schedule_completion_for_task(self, task_id: int) -> None:
        """Schedule a TASK_COMPLETION or QUANTUM_EXPIRED event for a running task."""
        task = self.tasks.get(task_id)
        if task is None or task.state != TaskState.RUNNING:
            return

        if task.assigned_cpu is None:
            # GPU task — handled separately
            return

        core = self.cores[task.assigned_cpu]
        quantum_ns = self.platform.default_quantum_ns

        # Scale duration by core speed relative to reference (1500 MHz perf core)
        ref_freq = 1500
        speed_scale = core.max_freq_mhz / ref_freq
        if speed_scale <= 0:
            speed_scale = 1.0
        adjusted_duration = int(task.remaining_work_ns / speed_scale)

        if adjusted_duration <= quantum_ns:
            # Task finishes within quantum
            completion_time = self.clock_ns + adjusted_duration
            self.push_event(
                EventType.TASK_COMPLETION,
                completion_time,
                task_id=task_id,
                core_id=task.assigned_cpu,
            )
        else:
            # Task needs more than one quantum — schedule preemption
            self.push_event(
                EventType.QUANTUM_EXPIRED,
                self.clock_ns + quantum_ns,
                core_id=task.assigned_cpu,
            )

    def _start_next_gpu_job(self) -> None:
        """Start the next GPU job from the queue if the GPU is idle."""
        if not self.gpu.available or self.gpu.current_job is not None:
            return
        if not self.gpu.queue:
            return

        req = self.gpu.queue.pop(0)
        self.gpu.current_job = req

        latency_ns = self.gpu.compute_latency_ns(req.ops)
        completion_time = self.clock_ns + latency_ns
        self.push_event(
            EventType.GPU_COMPLETION,
            completion_time,
            task_id=req.task_id,
        )

    def _update_core_busy_time(self, old_ns: int, new_ns: int) -> None:
        """Track how long each core has been busy (for utilization)."""
        delta = new_ns - old_ns
        if delta <= 0:
            return
        for i, core in enumerate(self.cores):
            if core.current_task is not None:
                self._core_busy_ns[i] += delta

    def _update_utilization_stats(self) -> None:
        """Recompute sliding-window utilization for all cores."""
        window = UTILIZATION_WINDOW_NS
        elapsed = self.clock_ns
        if elapsed <= 0:
            return

        # Use total busy time relative to elapsed time (simplified sliding window)
        for i, core in enumerate(self.cores):
            denominator = min(elapsed, window)
            if denominator > 0:
                core.utilization_pct = min(
                    1.0, self._core_busy_ns[i] / max(elapsed, 1)
                )
            else:
                core.utilization_pct = 0.0

            # Update cache pressure
            total_ws = 0.0
            if core.current_task is not None:
                t = self.tasks.get(core.current_task)
                if t:
                    total_ws += t.working_set_mb
            for tid in core.run_queue:
                t = self.tasks.get(tid)
                if t:
                    total_ws += t.working_set_mb
            cache_mb = core.cache_size_kb / 1024.0
            core.cache_pressure = total_ws / cache_mb if cache_mb > 0 else 0.0
