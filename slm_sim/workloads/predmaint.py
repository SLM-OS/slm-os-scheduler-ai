"""Predictive Maintenance workload profile.

Bursty, medium model: 1 Hz periodic + 10 Hz burst every 30s, 45 MB model,
50 ms deadline, GPU-eligible. See plan Section 3.1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from slm_sim.engine import EventType
from slm_sim.models import PRIORITY_HIGH, ComponentType
from slm_sim.workloads.base import WorkloadProfile

if TYPE_CHECKING:
    from slm_sim.engine import SimulatorEngine

# Predictive maintenance parameters (Section 3.1)
BASE_RATE_HZ = 1.0
BURST_RATE_HZ = 10.0
BURST_DURATION_NS = 500_000_000     # 500 ms burst duration
BURST_INTERVAL_NS = 30_000_000_000  # every 30 seconds
MODEL_SIZE_MB = 45.0
WORKING_SET_MB = 16.0
INFERENCE_MEAN_NS = 25_000_000   # 25 ms
INFERENCE_STD_NS = 5_000_000     # 5 ms
OPS_PER_INFERENCE = 4.8          # GFLOPS
DEADLINE_NS = 50_000_000         # 50 ms (hard)
PRIORITY = PRIORITY_HIGH         # 6
JITTER_FRACTION = 0.10


class PredMaintWorkload(WorkloadProfile):
    """Generates predictive maintenance inference requests."""

    @property
    def component_name(self) -> str:
        return "pred_maint"

    def seed_events(self, engine: SimulatorEngine, start_ns: int,
                    end_ns: int) -> None:
        """Generate periodic + bursty predictive maintenance arrivals.

        Base: 1 Hz periodic (scaled by intensity).
        Burst: Every 30s, a 500ms window at 10 Hz.
        """
        rate_hz = BASE_RATE_HZ * self.intensity
        if rate_hz <= 0:
            return
        interval_ns = int(1e9 / rate_hz)

        # Periodic arrivals
        t = start_ns + self.rng.integers(0, interval_ns)
        while t < end_ns:
            jitter = int(self.rng.uniform(-JITTER_FRACTION, JITTER_FRACTION) * interval_ns)
            arrival = t + jitter
            if start_ns <= arrival < end_ns:
                self._emit_arrival(engine, arrival)
            t += interval_ns

        # Burst windows
        burst_interval = int(BURST_INTERVAL_NS / max(self.intensity, 1.0))
        burst_start = start_ns + burst_interval
        while burst_start < end_ns:
            burst_end = min(burst_start + BURST_DURATION_NS, end_ns)
            burst_rate = BURST_RATE_HZ * self.intensity
            burst_step = int(1e9 / burst_rate) if burst_rate > 0 else BURST_DURATION_NS
            bt = burst_start
            while bt < burst_end:
                jitter = int(self.rng.uniform(-JITTER_FRACTION, JITTER_FRACTION) * burst_step)
                arrival = bt + jitter
                if start_ns <= arrival < end_ns:
                    self._emit_arrival(engine, arrival)
                bt += burst_step
            burst_start += burst_interval

    def _emit_arrival(self, engine: SimulatorEngine, time_ns: int) -> None:
        """Push a single TASK_ARRIVAL event."""
        task_id = engine.allocate_task_id()
        duration_ns = max(
            1_000_000,  # floor at 1 ms
            int(self.rng.normal(INFERENCE_MEAN_NS, INFERENCE_STD_NS)),
        )
        engine.push_event(
            EventType.TASK_ARRIVAL,
            time_ns,
            task_id=task_id,
            data={
                "task_id": task_id,
                "name": f"predmaint_{task_id}",
                "priority": PRIORITY,
                "model_handle": 2,
                "model_size_mb": MODEL_SIZE_MB,
                "working_set_mb": WORKING_SET_MB,
                "ops_per_inference": OPS_PER_INFERENCE,
                "inference_duration_ns": duration_ns,
                "can_use_gpu": True,
                "deadline_ns": time_ns + DEADLINE_NS,
                "component_type": int(ComponentType.PRED_MAINT),
            },
        )
