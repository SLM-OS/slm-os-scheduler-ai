"""Tests for slm_sim.workloads.scenarios — scenario definitions."""

from __future__ import annotations

import pytest

from slm_sim.workloads.scenarios import SCENARIOS, ScenarioComposer


class TestScenarioDefinitions:
    def test_all_eight_scenarios_defined(self):
        expected = {
            "light_single", "light_mixed", "medium_mixed",
            "heavy_inference", "burst_storm", "deadline_pressure",
            "memory_pressure", "asymmetric",
        }
        assert set(SCENARIOS.keys()) == expected

    def test_each_scenario_has_profiles(self):
        for name, config in SCENARIOS.items():
            assert len(config.profiles) > 0, f"{name} has no profiles"

    def test_scenario_names_match_keys(self):
        for key, config in SCENARIOS.items():
            assert config.name == key


class TestScenarioComposer:
    def test_construction_valid_scenario(self, rng):
        composer = ScenarioComposer("light_single", rng)
        assert len(composer.profiles) == 1

    def test_construction_invalid_scenario(self, rng):
        with pytest.raises(ValueError, match="Unknown scenario"):
            ScenarioComposer("nonexistent", rng)

    def test_medium_mixed_has_four_profiles(self, rng):
        composer = ScenarioComposer("medium_mixed", rng)
        assert len(composer.profiles) == 4
