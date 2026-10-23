from __future__ import annotations

import pytest

from carbon_twin.data.schemas import FuelRecord, FuelType
from carbon_twin.regulations.imo_nzf import IMONZFScenarioEngine


@pytest.fixture
def imo_engine() -> IMONZFScenarioEngine:
    return IMONZFScenarioEngine(scenario="MEPC83_draft")


class TestIMONZFEngine:
    def test_basic_evaluation(self, imo_engine, sample_bulk_carrier, sample_fuel_records_hfo):
        total_energy = sum(r.energy_mj or 0 for r in sample_fuel_records_hfo)
        result = imo_engine.evaluate(
            vessel=sample_bulk_carrier,
            annual_energy_mj=total_energy,
            fuel_records=sample_fuel_records_hfo,
            emission_factors={},
            year=2030,
            ghg_price_per_tco2e=100.0,
        )
        assert result.scenario == "MEPC83_draft"
        assert result.vessel_imo == sample_bulk_carrier.imo_number

    def test_higher_ghg_price_higher_penalty(
        self, imo_engine, sample_bulk_carrier, sample_fuel_records_hfo
    ):
        total_energy = sum(r.energy_mj or 0 for r in sample_fuel_records_hfo)
        results = imo_engine.evaluate_scenario_sweep(
            vessel=sample_bulk_carrier,
            energy=total_energy,
            fuels=sample_fuel_records_hfo,
            factors={},
            year=2030,
            ghg_prices=[50, 100, 200],
        )
        assert len(results) == 3
        penalties = [r.penalty_or_reward_eur for r in results]
        assert penalties[-1] >= penalties[0]

    def test_green_fuel_reward(self, imo_engine, sample_bulk_carrier):
        green_records = [
            FuelRecord(
                vessel_imo=sample_bulk_carrier.imo_number,
                fuel_type=FuelType.AMMONIA_GREEN,
                mass_tonnes=50,
                energy_mj=50 * 18.6 * 1000,
                purpose="main_engine",
            ),
        ]
        total_energy = sum(r.energy_mj or 0 for r in green_records)
        result = imo_engine.evaluate(
            vessel=sample_bulk_carrier,
            annual_energy_mj=total_energy,
            fuel_records=green_records,
            emission_factors={},
            year=2030,
            ghg_price_per_tco2e=100.0,
        )
        assert result.gfi_attained is not None
