"""Security Monitor workload profile.

Background, continuous: 10 Hz periodic, 8 MB model, 20 ms deadline.
See plan Section 3.1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from slm_sim.engine import EventType
from slm_sim.models import PRIORITY_LOW, ComponentType
from slm_sim.workloads.base import WorkloadProfile

if TYPE_CHECKING:
    from slm_sim.engine import SimulatorEngine

# Security monitor parameters (Section 3.1)
BASE_RATE_HZ = 10.0
MODEL_SIZE_MB = 8.0
WORKING_SET_MB = 3.0
INFERENCE_MEAN_NS = 3_000_000   # 3 ms
INFERENCE_STD_NS = 1_000_000    # 1 ms
OPS_PER_INFERENCE = 0.8         # GFLOPS
DEADLINE_NS = 20_000_000        # 20 ms (soft)
PRIORITY = PRIORITY_LOW         # 2
JITTER_FRACTION = 0.10


class SecurityMonitorWorkload(WorkloadProfile):
    """Generates security monitoring inference requests."""

    @property
    def component_name(self) -> str:
        return "security_monitor"

    def seed_events(self, engine: SimulatorEngine, start_ns: int,
                    end_ns: int) -> None:
        """Generate periodic security monitoring arrivals at 10 Hz."""
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
            200_000,  # floor at 0.2 ms
            int(self.rng.normal(INFERENCE_MEAN_NS, INFERENCE_STD_NS)),
        )
        engine.push_event(
            EventType.TASK_ARRIVAL,
            time_ns,
            task_id=task_id,
            data={
                "task_id": task_id,
                "name": f"security_{task_id}",
                "priority": PRIORITY,
                "model_handle": 3,
                "model_size_mb": MODEL_SIZE_MB,
                "working_set_mb": WORKING_SET_MB,
                "ops_per_inference": OPS_PER_INFERENCE,
                "inference_duration_ns": duration_ns,
                "can_use_gpu": False,
                "deadline_ns": time_ns + DEADLINE_NS,
                "component_type": int(ComponentType.SECURITY_MON),
            },
        )
