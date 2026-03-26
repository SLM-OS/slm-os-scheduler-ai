"""System Tasks workload profile.

OS overhead: 100 Hz periodic, no model, ~50 us duration, high priority.
See plan Section 3.1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from slm_sim.engine import EventType
from slm_sim.models import PRIORITY_HIGH, ComponentType
from slm_sim.workloads.base import WorkloadProfile

if TYPE_CHECKING:
    from slm_sim.engine import SimulatorEngine

# System task parameters (Section 3.1)
BASE_RATE_HZ = 100.0
INFERENCE_MEAN_NS = 50_000      # 50 us
INFERENCE_STD_NS = 10_000       # 10 us
WORKING_SET_MB = 0.5
PRIORITY = PRIORITY_HIGH        # 6
JITTER_FRACTION = 0.10


class SystemTaskWorkload(WorkloadProfile):
    """Generates system/OS overhead tasks."""

    @property
    def component_name(self) -> str:
        return "system"

    def seed_events(self, engine: SimulatorEngine, start_ns: int,
                    end_ns: int) -> None:
        """Generate periodic system task arrivals at 100 Hz."""
        rate_hz = BASE_RATE_HZ * self.intensity
        if rate_hz <= 0:
            return
        interval_ns = int(1e9 / rate_hz)

        t = start_ns + self.rng.integers(0, interval_ns)
        while t < end_ns:
            jitter = int(self.rng.uniform(-JITTER_FRACTION, JITTER_FRACTION) * interval_ns)
            arrival = t + jitter
            if start_ns <= arrival < end_ns:
                self._emit_arrival(engine, arrival)
            t += interval_ns

    def _emit_arrival(self, engine: SimulatorEngine, time_ns: int) -> None:
        """Push a single TASK_ARRIVAL event."""
        task_id = engine.allocate_task_id()
        duration_ns = max(
            10_000,  # floor at 10 us
            int(self.rng.normal(INFERENCE_MEAN_NS, INFERENCE_STD_NS)),
        )
        engine.push_event(
            EventType.TASK_ARRIVAL,
            time_ns,
            task_id=task_id,
            data={
                "task_id": task_id,
                "name": f"system_{task_id}",
                "priority": PRIORITY,
                "model_handle": None,
                "model_size_mb": 0.0,
                "working_set_mb": WORKING_SET_MB,
                "ops_per_inference": 0.0,
                "inference_duration_ns": duration_ns,
                "can_use_gpu": False,
                "deadline_ns": 0,  # no deadline, best-effort
                "component_type": int(ComponentType.SYSTEM),
            },
        )
