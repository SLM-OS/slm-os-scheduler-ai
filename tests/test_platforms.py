"""Tests for slm_sim.platforms — hardware profile definitions."""

from __future__ import annotations

import pytest

from slm_sim.models import CoreType
from slm_sim.platforms import (
    get_platform,
    make_big_little,
    make_jetson_orin_nano,
    make_raspberry_pi5,
)


class TestJetsonProfile:
    def test_core_count(self, jetson_platform):
        assert jetson_platform.num_cores == 6

    def test_all_performance_cores(self, jetson_platform):
        assert jetson_platform.num_performance_cores == 6
        assert jetson_platform.num_efficiency_cores == 0

    def test_gpu_available(self, jetson_platform):
        assert jetson_platform.gpu.available is True
        assert jetson_platform.gpu.throughput_tflops == 8.0

    def test_memory(self, jetson_platform):
        assert jetson_platform.memory.total_ram_mb == 8192

    def test_context_switch_cost(self, jetson_platform):
        assert jetson_platform.context_switch_min_ns == 2_000
        assert jetson_platform.context_switch_max_ns == 5_000


class TestPi5Profile:
    def test_core_count(self, pi5_platform):
        assert pi5_platform.num_cores == 4

    def test_all_performance_cores(self, pi5_platform):
        assert pi5_platform.num_performance_cores == 4

    def test_no_gpu(self, pi5_platform):
        assert pi5_platform.gpu.available is False

    def test_memory(self, pi5_platform):
        assert pi5_platform.memory.total_ram_mb == 4096

    def test_lower_power_weight(self, pi5_platform):
        for core in pi5_platform.cores:
            assert core.power_weight == 0.7


class TestBigLittleProfile:
    def test_core_count(self, biglittle_platform):
        assert biglittle_platform.num_cores == 6

    def test_heterogeneous_cores(self, biglittle_platform):
        assert biglittle_platform.num_performance_cores == 2
        assert biglittle_platform.num_efficiency_cores == 4

    def test_core_types_ordered(self, biglittle_platform):
        # First 2 are performance, next 4 are efficiency
        for core in biglittle_platform.cores[:2]:
            assert core.core_type == CoreType.PERFORMANCE
        for core in biglittle_platform.cores[2:]:
            assert core.core_type == CoreType.EFFICIENCY

    def test_power_weights(self, biglittle_platform):
        for core in biglittle_platform.cores[:2]:
            assert core.power_weight == 1.0
        for core in biglittle_platform.cores[2:]:
            assert core.power_weight == 0.3


class TestPlatformRegistry:
    def test_get_known_platform(self):
        platform = get_platform("jetson_orin_nano")
        assert platform.name == "jetson_orin_nano"

    def test_get_unknown_platform(self):
        with pytest.raises(ValueError, match="Unknown platform"):
            get_platform("nonexistent")

    def test_all_platforms_constructible(self):
        for name in ["jetson_orin_nano", "raspberry_pi5", "big_little"]:
            platform = get_platform(name)
            assert platform.num_cores > 0
            assert platform.memory.total_ram_mb > 0
