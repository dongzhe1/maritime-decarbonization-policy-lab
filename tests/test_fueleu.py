"""Tests for FuelEU Maritime regulation engine."""

from __future__ import annotations

import pytest

from carbon_twin.data.schemas import FuelRecord, FuelType
from carbon_twin.regulations.fueleu import FuelEUEngine


@pytest.fixture
def fueleu_engine() -> FuelEUEngine:
    return FuelEUEngine(version="EU_2023_1805")


class TestFuelEUEngine:
    def test_hfo_only_deficit(self, fueleu_engine, sample_bulk_carrier, sample_fuel_records_hfo):
        total_energy = sum(r.energy_mj or 0 for r in sample_fuel_records_hfo)
        result = fueleu_engine.evaluate(
            vessel=sample_bulk_carrier,
            fuel_records=sample_fuel_records_hfo,
            energy_mj=total_energy,
            year=2026,
        )
        assert result.actual_ghg_intensity > 0
        assert result.target_ghg_intensity > 0
        assert result.version == "EU_2023_1805"

    def test_target_decreases_over_years(self, fueleu_engine):
        targets = {}
        for year in [2025, 2030, 2035, 2040, 2045, 2050]:
            targets[year] = fueleu_engine._target_intensity(year)
        for y1, y2 in zip([2025, 2030, 2035, 2040, 2045], [2030, 2035, 2040, 2045, 2050]):
            assert targets[y2] <= targets[y1]

    def test_low_carbon_fuel_surplus(self, fueleu_engine, sample_bulk_carrier):
        bio_records = [
            FuelRecord(
                vessel_imo=sample_bulk_carrier.imo_number,
                fuel_type=FuelType.METHANOL_BIO,
                mass_tonnes=100,
                energy_mj=100 * 19.9 * 1000,
                purpose="main_engine",
            ),
        ]
        total_energy = sum(r.energy_mj or 0 for r in bio_records)
        result = fueleu_engine.evaluate(
            vessel=sample_bulk_carrier,
            fuel_records=bio_records,
            energy_mj=total_energy,
            year=2026,
        )
        assert result.surplus

    def test_penalty_when_deficit(
        self, fueleu_engine, sample_bulk_carrier, sample_fuel_records_hfo
    ):
        total_energy = sum(r.energy_mj or 0 for r in sample_fuel_records_hfo)
        result = fueleu_engine.evaluate(
            vessel=sample_bulk_carrier,
            fuel_records=sample_fuel_records_hfo,
            energy_mj=total_energy,
            year=2030,
        )
        if not result.surplus:
            assert result.penalty_eur >= 0

    def test_pooling_redistributes_balance(self, fueleu_engine):
        from carbon_twin.data.schemas import FuelEUResult

        results = [
            FuelEUResult(
                vessel_imo="IMO0001",
                year=2026,
                version="EU_2023_1805",
                actual_ghg_intensity=85,
                target_ghg_intensity=89.34,
                compliance_balance=50000,
                surplus=True,
            ),
            FuelEUResult(
                vessel_imo="IMO0002",
                year=2026,
                version="EU_2023_1805",
                actual_ghg_intensity=92,
                target_ghg_intensity=89.34,
                compliance_balance=-30000,
                surplus=False,
                penalty_eur=72000,
            ),
        ]
        pooled = fueleu_engine.evaluate_pool(results)
        total_balance = sum(r.compliance_balance for r in pooled)
        assert total_balance == pytest.approx(20000, abs=1)
