"""Anomaly Detector workload profile.

High-frequency, small model: ~100 Hz arrivals, 12 MB model, 5 ms deadline.
See plan Section 3.1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from slm_sim.engine import EventType
from slm_sim.models import PRIORITY_NORMAL, ComponentType
from slm_sim.workloads.base import WorkloadProfile

if TYPE_CHECKING:
    from slm_sim.engine import SimulatorEngine

# Anomaly detector parameters (Section 3.1)
BASE_RATE_HZ = 100.0
MODEL_SIZE_MB = 12.0
WORKING_SET_MB = 4.0
INFERENCE_MEAN_NS = 1_000_000   # 1 ms
INFERENCE_STD_NS = 200_000      # 0.2 ms
OPS_PER_INFERENCE = 1.2         # GFLOPS
DEADLINE_NS = 5_000_000         # 5 ms
PRIORITY = PRIORITY_NORMAL      # 4
BURST_PROBABILITY = 0.05
BURST_MIN = 5
BURST_MAX = 20


class AnomalyDetectorWorkload(WorkloadProfile):
    """Generates anomaly detection inference requests."""

    @property
    def component_name(self) -> str:
        return "anomaly_detector"

    def seed_events(self, engine: SimulatorEngine, start_ns: int,
                    end_ns: int) -> None:
        """Generate high-frequency anomaly detection arrivals.

        Base rate: 100 Hz (exponential inter-arrival), scaled by intensity.
        Burst mode: 5% chance at each arrival of a Poisson burst (5-20 requests).
        """
        rate_hz = BASE_RATE_HZ * self.intensity
        mean_interval_ns = int(1e9 / rate_hz) if rate_hz > 0 else int(1e9)
        jitter_fraction = 0.10  # ±10% jitter

        current_ns = start_ns
        while current_ns < end_ns:
            # Exponential inter-arrival time
            interval = self.rng.exponential(mean_interval_ns)
            # Add jitter
            jitter = self.rng.uniform(-jitter_fraction, jitter_fraction) * interval
            interval = max(1, int(interval + jitter))
            current_ns += interval

            if current_ns >= end_ns:
                break

            # Generate one task arrival
            self._emit_arrival(engine, current_ns)

            # Burst check
            if self.rng.random() < BURST_PROBABILITY:
                burst_count = self.rng.integers(BURST_MIN, BURST_MAX + 1)
                for b in range(burst_count):
                    burst_offset = self.rng.integers(0, 1_000_000)  # within 1 ms
                    burst_time = current_ns + burst_offset
                    if burst_time < end_ns:
                        self._emit_arrival(engine, burst_time)

    def _emit_arrival(self, engine: SimulatorEngine, time_ns: int) -> None:
        """Push a single TASK_ARRIVAL event."""
        task_id = engine.allocate_task_id()
        # Sample inference duration from Normal distribution
        duration_ns = max(
            100_000,  # floor at 0.1 ms
            int(self.rng.normal(INFERENCE_MEAN_NS, INFERENCE_STD_NS)),
        )

        engine.push_event(
            EventType.TASK_ARRIVAL,
            time_ns,
            task_id=task_id,
            data={
                "task_id": task_id,
                "name": f"anomaly_{task_id}",
                "priority": PRIORITY,
                "model_handle": 1,
                "model_size_mb": MODEL_SIZE_MB,
                "working_set_mb": WORKING_SET_MB,
                "ops_per_inference": OPS_PER_INFERENCE,
                "inference_duration_ns": duration_ns,
                "can_use_gpu": False,
                "deadline_ns": time_ns + DEADLINE_NS,  # absolute deadline
                "component_type": int(ComponentType.ANOMALY_DETECTOR),
            },
        )
