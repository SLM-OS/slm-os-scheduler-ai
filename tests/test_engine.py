"""Tests for slm_sim.engine — event queue and simulation engine."""

from __future__ import annotations

import pytest

from slm_sim.engine import Event, EventType, SimulatorEngine
from slm_sim.platforms import make_jetson_orin_nano


class TestEventOrdering:
    def test_events_ordered_by_timestamp(self):
        e1 = Event(timestamp_ns=100, event_type=EventType.TASK_ARRIVAL, sequence=0)
        e2 = Event(timestamp_ns=50, event_type=EventType.TASK_ARRIVAL, sequence=1)
        assert e2 < e1

    def test_same_timestamp_ordered_by_type(self):
        e1 = Event(timestamp_ns=100, event_type=EventType.SCHEDULING_DECISION, sequence=0)
        e2 = Event(timestamp_ns=100, event_type=EventType.TASK_ARRIVAL, sequence=1)
        assert e2 < e1  # TASK_ARRIVAL(0) < SCHEDULING_DECISION(5)

    def test_same_timestamp_same_type_ordered_by_sequence(self):
        e1 = Event(timestamp_ns=100, event_type=EventType.TASK_ARRIVAL, sequence=1)
        e2 = Event(timestamp_ns=100, event_type=EventType.TASK_ARRIVAL, sequence=0)
        assert e2 < e1


class TestSimulatorEngine:
    def test_construction(self, jetson_platform):
        engine = SimulatorEngine(platform=jetson_platform)
        assert engine.clock_ns == 0
        assert engine.episode_duration_ns == 10_000_000_000

    def test_reset(self, jetson_platform):
        engine = SimulatorEngine(platform=jetson_platform)
        engine.reset()
        assert len(engine.cores) == 6
        assert engine.clock_ns == 0
        assert len(engine.tasks) == 0
        # reset() seeds periodic events (deadline check + migration check)
        assert len(engine.event_queue) == 2

    def test_push_pop_event(self, jetson_platform):
        engine = SimulatorEngine(platform=jetson_platform)
        engine.push_event(EventType.TASK_ARRIVAL, 1000, task_id=1)
        engine.push_event(EventType.TASK_ARRIVAL, 500, task_id=2)
        evt = engine.pop_event()
        assert evt.timestamp_ns == 500
        assert evt.task_id == 2

    def test_pop_empty_queue(self, jetson_platform):
        engine = SimulatorEngine(platform=jetson_platform)
        assert engine.pop_event() is None

    def test_reset_clears_state(self, jetson_platform):
        engine = SimulatorEngine(platform=jetson_platform)
        engine.push_event(EventType.TASK_ARRIVAL, 1000)
        engine.clock_ns = 5000
        engine.reset()
        assert engine.clock_ns == 0
        # Only periodic events remain after reset
        assert len(engine.event_queue) == 2

    def test_gpu_initialized_from_platform(self, jetson_platform):
        engine = SimulatorEngine(platform=jetson_platform)
        engine.reset()
        assert engine.gpu.available is True

    def test_memory_initialized_from_platform(self, jetson_platform):
        engine = SimulatorEngine(platform=jetson_platform)
        engine.reset()
        assert engine.memory.total_ram_mb == 8192
