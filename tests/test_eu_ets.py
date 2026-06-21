"""Tests for EU ETS regulation engine."""

from __future__ import annotations

import pytest

from carbon_twin.data.schemas import EEAScope, Voyage
from carbon_twin.regulations.eu_ets import ETSEngine


@pytest.fixture
def ets_engine() -> ETSEngine:
    return ETSEngine(version="EU_ETS_Maritime_2024")


class TestETSEngine:
    def test_intra_eea_full_coverage(
        self, ets_engine, sample_intra_eea_voyage, sample_emission_estimate
    ):
        results = ets_engine.evaluate(
            voyages=[sample_intra_eea_voyage],
            emissions=[sample_emission_estimate],
            eua_price_eur=80.0,
            year=2026,
        )
        assert len(results) >= 1
        result = results[0]
        assert result.ets_cost_eur > 0
        assert result.phase_in_factor == 1.0

    def test_phase_in_2024(self, ets_engine, sample_intra_eea_voyage, sample_emission_estimate):
        results = ets_engine.evaluate(
            voyages=[sample_intra_eea_voyage],
            emissions=[sample_emission_estimate],
            eua_price_eur=80.0,
            year=2024,
        )
        result = results[0]
        assert result.phase_in_factor == pytest.approx(0.40)

    def test_extra_eea_half_coverage(
        self, ets_engine, sample_into_eea_voyage, sample_emission_estimate
    ):
        emission = sample_emission_estimate.model_copy(
            update={
                "vessel_imo": sample_into_eea_voyage.vessel_imo,
                "voyage_id": sample_into_eea_voyage.voyage_id,
            }
        )
        results = ets_engine.evaluate(
            voyages=[sample_into_eea_voyage],
            emissions=[emission],
            eua_price_eur=80.0,
            year=2026,
        )
        result = results[0]
        assert result.scope_breakdown.get("into_eea", 0) > 0 or result.covered_emissions_tco2e > 0

    def test_non_eea_zero_cost(self, ets_engine, sample_emission_estimate):
        non_eea_voyage = Voyage(
            voyage_id="VOY_NON_EEA",
            vessel_imo=sample_emission_estimate.vessel_imo,
            departure_port_id="CNSHA",
            arrival_port_id="SGSIN",
            departure_time=sample_emission_estimate.co2_tonnes,
            arrival_time=sample_emission_estimate.co2_tonnes,
            distance_nm=2000,
            eea_scope=EEAScope.NON_EEA,
            ets_scope_factor=0.0,
        )
        emission = sample_emission_estimate.model_copy(update={"voyage_id": "VOY_NON_EEA"})
        results = ets_engine.evaluate(
            voyages=[non_eea_voyage],
            emissions=[emission],
            eua_price_eur=80.0,
            year=2026,
        )
        assert all(r.ets_cost_eur == 0 for r in results)

    def test_higher_eua_price_higher_cost(
        self, ets_engine, sample_intra_eea_voyage, sample_emission_estimate
    ):
        low = ets_engine.evaluate(
            voyages=[sample_intra_eea_voyage],
            emissions=[sample_emission_estimate],
            eua_price_eur=50.0,
            year=2026,
        )
        high = ets_engine.evaluate(
            voyages=[sample_intra_eea_voyage],
            emissions=[sample_emission_estimate],
            eua_price_eur=200.0,
            year=2026,
        )
        assert high[0].ets_cost_eur > low[0].ets_cost_eur
