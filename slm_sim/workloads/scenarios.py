"""Scenario compositions combining multiple workload profiles.

Each scenario defines which component workloads are active and at
what intensity. See plan Section 3.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from slm_sim.models import CPU_AFFINITY_ANY
from slm_sim.workloads.anomaly import AnomalyDetectorWorkload
from slm_sim.workloads.base import WorkloadProfile
from slm_sim.workloads.predmaint import PredMaintWorkload
from slm_sim.workloads.security import SecurityMonitorWorkload
from slm_sim.workloads.system import SystemTaskWorkload

if TYPE_CHECKING:
    from slm_sim.engine import SimulatorEngine


@dataclass
class ScenarioConfig:
    """Configuration for a workload scenario."""
    name: str
    description: str
    profiles: list[tuple[type, float]] = field(default_factory=list)
    # Each entry: (WorkloadProfileClass, intensity_multiplier)

    # Special scenario flags
    burst_overlay: bool = False         # burst_storm: periodic burst of all components
    affinity_map: Optional[dict] = None  # asymmetric: component -> core list


# Scenario definitions from Section 3.2
SCENARIOS: dict[str, ScenarioConfig] = {
    "light_single": ScenarioConfig(
        name="light_single",
        description="Anomaly only, ~10% load",
        profiles=[(AnomalyDetectorWorkload, 1.0)],
    ),
    "light_mixed": ScenarioConfig(
        name="light_mixed",
        description="Anomaly + Security, ~25% load",
        profiles=[
            (AnomalyDetectorWorkload, 1.0),
            (SecurityMonitorWorkload, 1.0),
        ],
    ),
    "medium_mixed": ScenarioConfig(
        name="medium_mixed",
        description="All three + System, ~50% load",
        profiles=[
            (AnomalyDetectorWorkload, 1.0),
            (PredMaintWorkload, 1.0),
            (SecurityMonitorWorkload, 1.0),
            (SystemTaskWorkload, 1.0),
        ],
    ),
    "heavy_inference": ScenarioConfig(
        name="heavy_inference",
        description="All three at 2x rate + System, ~80% load",
        profiles=[
            (AnomalyDetectorWorkload, 2.0),
            (PredMaintWorkload, 2.0),
            (SecurityMonitorWorkload, 2.0),
            (SystemTaskWorkload, 1.0),
        ],
    ),
    "burst_storm": ScenarioConfig(
        name="burst_storm",
        description="Normal + periodic burst of all three, spikes to ~95%",
        profiles=[
            (AnomalyDetectorWorkload, 1.0),
            (PredMaintWorkload, 1.0),
            (SecurityMonitorWorkload, 1.0),
            (SystemTaskWorkload, 1.0),
        ],
        burst_overlay=True,
    ),
    "deadline_pressure": ScenarioConfig(
        name="deadline_pressure",
        description="PredMaint at 3x rate, ~60% heavy deadlines",
        profiles=[
            (PredMaintWorkload, 3.0),
            (SystemTaskWorkload, 1.0),
        ],
    ),
    "memory_pressure": ScenarioConfig(
        name="memory_pressure",
        description="4 PredMaint instances, pool near-exhaustion",
        profiles=[
            (PredMaintWorkload, 4.0),
            (SystemTaskWorkload, 1.0),
        ],
    ),
    "asymmetric": ScenarioConfig(
        name="asymmetric",
        description="Anomaly on cores 0-1, PredMaint on cores 4-5, ~40% unbalanced",
        profiles=[
            (AnomalyDetectorWorkload, 1.0),
            (PredMaintWorkload, 1.0),
        ],
        affinity_map={
            "anomaly_detector": [0, 1],
            "pred_maint": [4, 5],
        },
    ),
}

# Burst overlay parameters
BURST_OVERLAY_INTERVAL_NS = 5_000_000_000  # every 5 seconds
BURST_OVERLAY_DURATION_NS = 500_000_000    # 500 ms burst window
BURST_OVERLAY_MULTIPLIER = 3.0             # 3x rate during burst


class ScenarioComposer:
    """Instantiates and runs multiple workload profiles for a scenario."""

    def __init__(self, scenario_name: str, rng: np.random.Generator):
        if scenario_name not in SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. "
                f"Available: {list(SCENARIOS.keys())}"
            )
        self.config = SCENARIOS[scenario_name]
        self.rng = rng
        self.profiles: list[WorkloadProfile] = [
            cls(rng=rng, intensity=intensity)
            for cls, intensity in self.config.profiles
        ]

    def seed_all_events(self, engine: SimulatorEngine, start_ns: int,
                        end_ns: int) -> None:
        """Seed events from all active workload profiles."""
        for profile in self.profiles:
            profile.seed_events(engine, start_ns, end_ns)

        # Burst overlay: add extra burst windows of all components
        if self.config.burst_overlay:
            self._seed_burst_overlay(engine, start_ns, end_ns)

        # Asymmetric: patch CPU affinity on already-seeded arrival events
        if self.config.affinity_map:
            self._apply_affinity_map(engine)

    def _seed_burst_overlay(self, engine: SimulatorEngine, start_ns: int,
                            end_ns: int) -> None:
        """Add periodic high-intensity burst windows for burst_storm scenario."""
        burst_start = start_ns + BURST_OVERLAY_INTERVAL_NS
        while burst_start < end_ns:
            burst_end = min(burst_start + BURST_OVERLAY_DURATION_NS, end_ns)
            # Create temporary high-intensity profiles for the burst window
            for cls, base_intensity in self.config.profiles:
                burst_profile = cls(
                    rng=self.rng,
                    intensity=base_intensity * BURST_OVERLAY_MULTIPLIER,
                )
                burst_profile.seed_events(engine, burst_start, burst_end)
            burst_start += BURST_OVERLAY_INTERVAL_NS

    def _apply_affinity_map(self, engine: SimulatorEngine) -> None:
        """Set CPU affinity on arrival events based on component type.

        Patches the 'cpu_affinity' field in event data dicts for events
        whose component matches the affinity map.
        """
        amap = self.config.affinity_map
        if not amap:
            return

        # Build component_name -> core list lookup
        # Map component names to their workload classes
        name_to_cores: dict[str, list[int]] = {}
        for profile in self.profiles:
            name = profile.component_name
            if name in amap:
                name_to_cores[name] = amap[name]

        if not name_to_cores:
            return

        # Build reverse mapping: component_type int -> core list
        from slm_sim.models import ComponentType
        comp_name_to_int = {
            "anomaly_detector": int(ComponentType.ANOMALY_DETECTOR),
            "pred_maint": int(ComponentType.PRED_MAINT),
            "security_monitor": int(ComponentType.SECURITY_MON),
            "system": int(ComponentType.SYSTEM),
        }
        type_to_cores: dict[int, list[int]] = {}
        for name, cores in name_to_cores.items():
            if name in comp_name_to_int:
                type_to_cores[comp_name_to_int[name]] = cores

        # Patch events in the queue
        for event in engine.event_queue:
            if event.data and "component_type" in event.data:
                ct = event.data["component_type"]
                if ct in type_to_cores:
                    cores = type_to_cores[ct]
                    # Round-robin assignment using task_id
                    tid = event.data.get("task_id", 0)
                    event.data["cpu_affinity"] = cores[tid % len(cores)]
