"""Hardware platform profiles for the simulator.

Defines named presets for Jetson Orin Nano, Raspberry Pi 5, and a
hypothetical big.LITTLE configuration. Each profile specifies core
count/types, cache sizes, memory pools, GPU availability, and
context switch costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from slm_sim.models import CoreState, CoreType, GPUModel, MemorySubsystem


@dataclass
class PlatformProfile:
    """Complete hardware platform specification."""
    name: str
    cores: list[CoreState]
    memory: MemorySubsystem
    gpu: GPUModel
    context_switch_min_ns: int  # minimum context switch cost
    context_switch_max_ns: int  # maximum context switch cost
    default_quantum_ns: int = 10_000_000  # 10 ms default quantum

    @property
    def num_cores(self) -> int:
        return len(self.cores)

    @property
    def num_performance_cores(self) -> int:
        return sum(1 for c in self.cores if c.core_type == CoreType.PERFORMANCE)

    @property
    def num_efficiency_cores(self) -> int:
        return sum(1 for c in self.cores if c.core_type == CoreType.EFFICIENCY)


def make_jetson_orin_nano() -> PlatformProfile:
    """Jetson Orin Nano: 6x Cortex-A78AE, 8 GB, Ampere GPU."""
    cores = [
        CoreState(
            core_id=i,
            core_type=CoreType.PERFORMANCE,
            max_freq_mhz=1500,
            power_weight=1.0,
            cache_size_kb=256,  # 256 KB L2 per core
        )
        for i in range(6)
    ]
    memory = MemorySubsystem(
        total_ram_mb=8192,
        weight_pool_mb=512,   # scaled up from QEMU's 16 MB
        workspace_pool_mb=256,  # scaled up from QEMU's 8 MB
    )
    gpu = GPUModel(
        available=True,
        throughput_tflops=8.0,
        dma_overhead_ns=50_000,
        cache_flush_overhead_ns=10_000,
    )
    return PlatformProfile(
        name="jetson_orin_nano",
        cores=cores,
        memory=memory,
        gpu=gpu,
        context_switch_min_ns=2_000,  # 2 us
        context_switch_max_ns=5_000,  # 5 us
    )


def make_raspberry_pi5() -> PlatformProfile:
    """Raspberry Pi 5: 4x Cortex-A76, 4 GB, no usable GPU."""
    cores = [
        CoreState(
            core_id=i,
            core_type=CoreType.PERFORMANCE,
            max_freq_mhz=2400,
            power_weight=0.7,
            cache_size_kb=512,  # 512 KB L2 per core
        )
        for i in range(4)
    ]
    memory = MemorySubsystem(
        total_ram_mb=4096,
        weight_pool_mb=256,
        workspace_pool_mb=128,
    )
    gpu = GPUModel(available=False)
    return PlatformProfile(
        name="raspberry_pi5",
        cores=cores,
        memory=memory,
        gpu=gpu,
        context_switch_min_ns=3_000,  # 3 us
        context_switch_max_ns=6_000,  # 6 us
    )


def make_big_little() -> PlatformProfile:
    """Hypothetical big.LITTLE: 2x A78 (perf) + 4x A55 (efficiency)."""
    perf_cores = [
        CoreState(
            core_id=i,
            core_type=CoreType.PERFORMANCE,
            max_freq_mhz=2000,
            power_weight=1.0,
            cache_size_kb=256,
        )
        for i in range(2)
    ]
    eff_cores = [
        CoreState(
            core_id=i + 2,
            core_type=CoreType.EFFICIENCY,
            max_freq_mhz=1000,
            power_weight=0.3,
            cache_size_kb=128,
        )
        for i in range(4)
    ]
    memory = MemorySubsystem(
        total_ram_mb=4096,
        weight_pool_mb=256,
        workspace_pool_mb=128,
    )
    gpu = GPUModel(available=False)
    return PlatformProfile(
        name="big_little",
        cores=perf_cores + eff_cores,
        memory=memory,
        gpu=gpu,
        context_switch_min_ns=2_000,
        context_switch_max_ns=6_000,
    )


# Registry of available platform profiles
PLATFORM_REGISTRY: dict[str, callable] = {
    "jetson_orin_nano": make_jetson_orin_nano,
    "raspberry_pi5": make_raspberry_pi5,
    "big_little": make_big_little,
}


def get_platform(name: str) -> PlatformProfile:
    """Look up a platform profile by name."""
    if name not in PLATFORM_REGISTRY:
        raise ValueError(
            f"Unknown platform '{name}'. "
            f"Available: {list(PLATFORM_REGISTRY.keys())}"
        )
    return PLATFORM_REGISTRY[name]()
