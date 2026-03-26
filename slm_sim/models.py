"""Core data models for the SLM-OS scheduling simulator.

Defines CoreState, SimTask, MemorySubsystem, GPUModel, and supporting
enumerations. These mirror the SLM-OS kernel's struct slm_task and the
Rust runtime's SlmTaskInfo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class TaskState(IntEnum):
    """Task lifecycle states matching kernel task_state_t."""
    READY = 0
    RUNNING = 1
    BLOCKED = 2
    TERMINATED = 3


class CoreType(IntEnum):
    """CPU core type for heterogeneous scheduling."""
    EFFICIENCY = 0
    PERFORMANCE = 1


class ComponentType(IntEnum):
    """SLM-OS inference component types."""
    ANOMALY_DETECTOR = 0
    PRED_MAINT = 1
    SECURITY_MON = 2
    SYSTEM = 3
    OTHER = 4


# Priority levels matching the real SLM-OS kernel (with gaps)
PRIORITY_IDLE = 0
PRIORITY_LOW = 2
PRIORITY_NORMAL = 4
PRIORITY_HIGH = 6
PRIORITY_CRITICAL = 7

PRIORITY_MIN = PRIORITY_IDLE
PRIORITY_MAX = PRIORITY_CRITICAL

# All valid priority levels in the kernel
PRIORITY_LEVELS = [
    PRIORITY_IDLE,
    PRIORITY_LOW,
    PRIORITY_NORMAL,
    PRIORITY_HIGH,
    PRIORITY_CRITICAL,
]

CPU_AFFINITY_ANY = -1


@dataclass
class SimTask:
    """A simulated task mirroring SLM-OS's struct slm_task + SlmTaskInfo.

    Fields correspond to Section 1.5 of the plan, with priority values
    matching the real kernel (IDLE=0, LOW=2, NORMAL=4, HIGH=6, CRITICAL=7).
    """
    task_id: int
    name: str
    state: TaskState = TaskState.READY

    # Standard scheduling fields
    priority: int = PRIORITY_NORMAL
    effective_priority: int = PRIORITY_NORMAL
    assigned_cpu: Optional[int] = None
    cpu_affinity: int = CPU_AFFINITY_ANY

    # SLM-specific fields (from Phase 3 scheduler)
    model_handle: Optional[int] = None
    model_size_mb: float = 0.0
    working_set_mb: float = 0.0
    ops_per_inference: float = 0.0
    inference_duration_ns: int = 0
    can_use_gpu: bool = False

    # Deadline fields
    deadline_ns: int = 0  # 0 = no deadline
    arrival_time_ns: int = 0
    remaining_work_ns: int = 0

    # Component info
    component_type: ComponentType = ComponentType.OTHER


@dataclass
class CoreState:
    """Per-core state tracked by the simulator (Section 1.4)."""
    core_id: int
    core_type: CoreType = CoreType.PERFORMANCE
    max_freq_mhz: int = 1500
    current_task: Optional[int] = None  # task_id or None
    quantum_remaining_ns: int = 0
    isolated: bool = False
    run_queue: list[int] = field(default_factory=list)  # ordered by effective_priority
    utilization_pct: float = 0.0  # sliding window, 0.0-1.0
    cache_pressure: float = 0.0  # working_set_sum / cache_size
    power_weight: float = 1.0
    idle_time_ns: int = 0
    context_switches: int = 0
    cache_size_kb: int = 256  # L2 cache per core


@dataclass
class ModelInfo:
    """Metadata for a loaded ML model in memory."""
    model_handle: int
    name: str
    weight_size_mb: float
    refcount: int = 1


@dataclass
class MemorySubsystem:
    """Memory pool tracking mirroring Phase 3 model memory allocator (Section 1.7)."""
    total_ram_mb: int = 8192
    weight_pool_mb: int = 16
    workspace_pool_mb: int = 8
    weight_pool_used_mb: float = 0.0
    workspace_pool_used_mb: float = 0.0
    loaded_models: dict[int, ModelInfo] = field(default_factory=dict)
    model_refcounts: dict[int, int] = field(default_factory=dict)

    @property
    def weight_pool_pressure(self) -> float:
        """Fraction of weight pool used."""
        if self.weight_pool_mb == 0:
            return 0.0
        return self.weight_pool_used_mb / self.weight_pool_mb

    @property
    def workspace_pool_pressure(self) -> float:
        """Fraction of workspace pool used."""
        if self.workspace_pool_mb == 0:
            return 0.0
        return self.workspace_pool_used_mb / self.workspace_pool_mb


@dataclass
class GPURequest:
    """A pending GPU inference request."""
    task_id: int
    ops: float  # GFLOPS for this inference
    submit_time_ns: int = 0


@dataclass
class GPUModel:
    """Simplified GPU model for Jetson platform (Section 1.8)."""
    available: bool = False
    throughput_tflops: float = 8.0
    queue: list[GPURequest] = field(default_factory=list)
    current_job: Optional[GPURequest] = None
    dma_overhead_ns: int = 50_000
    cache_flush_overhead_ns: int = 10_000

    def compute_latency_ns(self, ops_gflops: float) -> int:
        """Estimate GPU inference latency for a given workload."""
        if not self.available or self.throughput_tflops <= 0:
            return 0
        # ops in GFLOPS, throughput in TFLOPS = 1000 GFLOPS
        compute_ns = int((ops_gflops / (self.throughput_tflops * 1000)) * 1e9)
        return compute_ns + self.dma_overhead_ns + self.cache_flush_overhead_ns
