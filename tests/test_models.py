"""Tests for slm_sim.models — data classes, enums, and memory subsystem."""

from __future__ import annotations

from slm_sim.models import (
    CPU_AFFINITY_ANY,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_IDLE,
    PRIORITY_LEVELS,
    PRIORITY_LOW,
    PRIORITY_MAX,
    PRIORITY_MIN,
    PRIORITY_NORMAL,
    ComponentType,
    CoreState,
    CoreType,
    GPUModel,
    GPURequest,
    MemorySubsystem,
    ModelInfo,
    SimTask,
    TaskState,
)


class TestPriorityLevels:
    """Verify priority values match real SLM-OS kernel."""

    def test_kernel_priority_values(self):
        assert PRIORITY_IDLE == 0
        assert PRIORITY_LOW == 2
        assert PRIORITY_NORMAL == 4
        assert PRIORITY_HIGH == 6
        assert PRIORITY_CRITICAL == 7

    def test_priority_bounds(self):
        assert PRIORITY_MIN == PRIORITY_IDLE
        assert PRIORITY_MAX == PRIORITY_CRITICAL

    def test_priority_levels_sorted(self):
        assert PRIORITY_LEVELS == sorted(PRIORITY_LEVELS)

    def test_priority_ordering(self):
        assert PRIORITY_IDLE < PRIORITY_LOW < PRIORITY_NORMAL < PRIORITY_HIGH < PRIORITY_CRITICAL


class TestSimTask:
    """Tests for SimTask dataclass."""

    def test_default_construction(self):
        task = SimTask(task_id=1, name="test")
        assert task.state == TaskState.READY
        assert task.priority == PRIORITY_NORMAL
        assert task.effective_priority == PRIORITY_NORMAL
        assert task.cpu_affinity == CPU_AFFINITY_ANY
        assert task.deadline_ns == 0
        assert task.model_handle is None

    def test_full_construction(self, sample_task):
        assert sample_task.task_id == 1
        assert sample_task.model_size_mb == 12.0
        assert sample_task.component_type == ComponentType.ANOMALY_DETECTOR


class TestCoreState:
    def test_default_core(self):
        core = CoreState(core_id=0)
        assert core.core_type == CoreType.PERFORMANCE
        assert core.current_task is None
        assert core.utilization_pct == 0.0
        assert core.run_queue == []

    def test_efficiency_core(self):
        core = CoreState(core_id=2, core_type=CoreType.EFFICIENCY, power_weight=0.3)
        assert core.core_type == CoreType.EFFICIENCY
        assert core.power_weight == 0.3


class TestMemorySubsystem:
    def test_initial_pressure(self):
        mem = MemorySubsystem()
        assert mem.weight_pool_pressure == 0.0
        assert mem.workspace_pool_pressure == 0.0

    def test_pressure_calculation(self):
        mem = MemorySubsystem(weight_pool_mb=100, workspace_pool_mb=50)
        mem.weight_pool_used_mb = 75.0
        mem.workspace_pool_used_mb = 25.0
        assert mem.weight_pool_pressure == 0.75
        assert mem.workspace_pool_pressure == 0.5

    def test_zero_pool_pressure(self):
        mem = MemorySubsystem(weight_pool_mb=0, workspace_pool_mb=0)
        assert mem.weight_pool_pressure == 0.0
        assert mem.workspace_pool_pressure == 0.0


class TestGPUModel:
    def test_unavailable_gpu(self):
        gpu = GPUModel(available=False)
        assert gpu.compute_latency_ns(1.0) == 0

    def test_latency_calculation(self):
        gpu = GPUModel(
            available=True,
            throughput_tflops=8.0,
            dma_overhead_ns=50_000,
            cache_flush_overhead_ns=10_000,
        )
        latency = gpu.compute_latency_ns(4.8)  # 4.8 GFLOPS
        # 4.8 / (8.0 * 1000) * 1e9 = 600_000 ns compute + 60_000 overhead
        assert latency > 60_000  # at least overhead
        assert latency == 600_000 + 50_000 + 10_000  # 660_000 ns
