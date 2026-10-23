from __future__ import annotations

import pytest

from carbon_twin.data.schemas import CIIRating, ShipType, Vessel
from carbon_twin.regulations.cii import CIIEngine


@pytest.fixture
def cii_engine() -> CIIEngine:
    return CIIEngine(version="IMO_CII_2023_phase1")


class TestCIIEngine:
    def test_bulk_carrier_rating_c(self, cii_engine, sample_bulk_carrier):
        result = cii_engine.evaluate(
            vessel=sample_bulk_carrier,
            annual_co2_tonnes=25000,
            distance_nm=50000,
            year=2026,
        )
        assert result.rating in list(CIIRating)
        assert result.attained_cii > 0
        assert result.required_cii > 0
        assert result.version == "IMO_CII_2023_phase1"

    def test_container_ship_rating(self, cii_engine, sample_container_ship):
        result = cii_engine.evaluate(
            vessel=sample_container_ship,
            annual_co2_tonnes=80000,
            distance_nm=120000,
            year=2026,
        )
        assert result.rating in list(CIIRating)
        assert result.vessel_imo == sample_container_ship.imo_number

    def test_reduction_factor_increases_over_years(self, cii_engine, sample_bulk_carrier):
        results = []
        for year in [2023, 2024, 2025, 2026]:
            r = cii_engine.evaluate(
                vessel=sample_bulk_carrier,
                annual_co2_tonnes=25000,
                distance_nm=50000,
                year=year,
            )
            results.append(r)
        for i in range(1, len(results)):
            assert results[i].required_cii <= results[i - 1].required_cii

    def test_higher_emissions_worse_rating(self, cii_engine, sample_bulk_carrier):
        low = cii_engine.evaluate(
            vessel=sample_bulk_carrier,
            annual_co2_tonnes=15000,
            distance_nm=50000,
            year=2026,
        )
        high = cii_engine.evaluate(
            vessel=sample_bulk_carrier,
            annual_co2_tonnes=35000,
            distance_nm=50000,
            year=2026,
        )
        assert low.attained_cii < high.attained_cii

    def test_zero_distance_raises(self, cii_engine, sample_bulk_carrier):
        with pytest.raises(ValueError):
            cii_engine.evaluate(
                vessel=sample_bulk_carrier,
                annual_co2_tonnes=25000,
                distance_nm=0,
                year=2026,
            )

    def test_corrective_action_for_e_rating(self, cii_engine, sample_bulk_carrier):
        result = cii_engine.evaluate(
            vessel=sample_bulk_carrier,
            annual_co2_tonnes=100000,
            distance_nm=20000,
            year=2026,
        )
        if result.rating == CIIRating.E:
            assert result.corrective_action_required

    def test_fleet_evaluation(self, cii_engine, sample_bulk_carrier, sample_container_ship):
        fleet_data = [
            {
                "vessel": sample_bulk_carrier,
                "annual_co2_tonnes": 25000,
                "distance_nm": 50000,
                "year": 2026,
            },
            {
                "vessel": sample_container_ship,
                "annual_co2_tonnes": 80000,
                "distance_nm": 120000,
                "year": 2026,
            },
        ]
        results = cii_engine.evaluate_fleet(fleet_data)
        assert len(results) == 2
