"""Tests for workload profiles — verify event generation."""

from __future__ import annotations

import numpy as np
import pytest

from slm_sim.engine import EventType, SimulatorEngine
from slm_sim.platforms import make_jetson_orin_nano
from slm_sim.workloads.anomaly import AnomalyDetectorWorkload
from slm_sim.workloads.predmaint import PredMaintWorkload
from slm_sim.workloads.security import SecurityMonitorWorkload
from slm_sim.workloads.system import SystemTaskWorkload


@pytest.fixture
def engine():
    platform = make_jetson_orin_nano()
    eng = SimulatorEngine(platform=platform, episode_duration_ns=1_000_000_000)
    eng.reset()
    return eng


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def count_arrivals(engine):
    """Count TASK_ARRIVAL events in the queue."""
    return sum(1 for e in engine.event_queue if e.event_type == EventType.TASK_ARRIVAL)


class TestAnomalyWorkload:
    def test_generates_events(self, engine, rng):
        w = AnomalyDetectorWorkload(rng=rng)
        w.seed_events(engine, 0, 1_000_000_000)
        assert count_arrivals(engine) > 50  # ~100 Hz for 1 sec

    def test_intensity_scaling(self, engine, rng):
        w1 = AnomalyDetectorWorkload(rng=np.random.default_rng(42), intensity=1.0)
        w1.seed_events(engine, 0, 1_000_000_000)
        count1 = count_arrivals(engine)

        engine.reset()
        w2 = AnomalyDetectorWorkload(rng=np.random.default_rng(42), intensity=2.0)
        w2.seed_events(engine, 0, 1_000_000_000)
        count2 = count_arrivals(engine)

        assert count2 > count1 * 1.5  # ~2x more events


class TestPredMaintWorkload:
    def test_generates_events(self, engine, rng):
        w = PredMaintWorkload(rng=rng)
        w.seed_events(engine, 0, 1_000_000_000)
        n = count_arrivals(engine)
        assert n > 0  # At least periodic arrivals

    def test_has_gpu_eligible_tasks(self, engine, rng):
        w = PredMaintWorkload(rng=rng)
        w.seed_events(engine, 0, 1_000_000_000)
        gpu_tasks = [e for e in engine.event_queue
                     if e.event_type == EventType.TASK_ARRIVAL
                     and e.data.get("can_use_gpu", False)]
        assert len(gpu_tasks) > 0


class TestSecurityWorkload:
    def test_generates_events(self, engine, rng):
        w = SecurityMonitorWorkload(rng=rng)
        w.seed_events(engine, 0, 1_000_000_000)
        assert count_arrivals(engine) > 5  # ~10 Hz

    def test_low_priority(self, engine, rng):
        w = SecurityMonitorWorkload(rng=rng)
        w.seed_events(engine, 0, 1_000_000_000)
        arrivals = [e for e in engine.event_queue
                    if e.event_type == EventType.TASK_ARRIVAL]
        for e in arrivals:
            assert e.data["priority"] == 2  # PRIORITY_LOW


class TestSystemWorkload:
    def test_generates_events(self, engine, rng):
        w = SystemTaskWorkload(rng=rng)
        w.seed_events(engine, 0, 1_000_000_000)
        assert count_arrivals(engine) > 50  # ~100 Hz

    def test_no_deadline(self, engine, rng):
        w = SystemTaskWorkload(rng=rng)
        w.seed_events(engine, 0, 1_000_000_000)
        arrivals = [e for e in engine.event_queue
                    if e.event_type == EventType.TASK_ARRIVAL]
        for e in arrivals:
            assert e.data["deadline_ns"] == 0

    def test_no_model(self, engine, rng):
        w = SystemTaskWorkload(rng=rng)
        w.seed_events(engine, 0, 1_000_000_000)
        arrivals = [e for e in engine.event_queue
                    if e.event_type == EventType.TASK_ARRIVAL]
        for e in arrivals:
            assert e.data["model_size_mb"] == 0.0
