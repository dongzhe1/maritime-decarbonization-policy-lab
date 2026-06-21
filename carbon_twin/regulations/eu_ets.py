"""EU ETS for maritime (Directive 2023/959) compliance engine."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from carbon_twin.data.schemas import (
    EEAScope,
    EmissionEstimate,
    ETSResult,
    PortCall,
    Voyage,
)
from carbon_twin.regulations.versioning import RegulationVersionManager

# Mapping from EEA scope to the fraction of emissions covered
_SCOPE_FACTORS: dict[EEAScope, float] = {
    EEAScope.INTRA_EEA: 1.0,
    EEAScope.INTO_EEA: 0.5,
    EEAScope.OUT_OF_EEA: 0.5,
    EEAScope.PORT: 1.0,
    EEAScope.NON_EEA: 0.0,
}


class ETSEngine:
    """EU ETS maritime allowance and cost calculation engine."""

    def __init__(
        self,
        version: str = "EU_ETS_Maritime_2024",
        config_path: str | None = None,
    ) -> None:
        self._version = version
        mgr = RegulationVersionManager(
            config_path if config_path else "configs/regulation_versions.yaml"
        )
        self._params: dict[str, Any] = mgr.get_ets_params(version)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def evaluate(
        self,
        voyages: list[Voyage],
        emissions: list[EmissionEstimate],
        eua_price_eur: float,
        year: int,
        port_calls: list[PortCall] | None = None,
    ) -> list[ETSResult]:
        """Evaluate EU ETS obligations for a set of voyages and their emissions.

        Returns one ETSResult per unique vessel IMO in the input.
        """
        emission_map: dict[str, EmissionEstimate] = {}
        for em in emissions:
            if em.voyage_id:
                emission_map[em.voyage_id] = em

        # Compute per-voyage covered emissions
        vessel_voyage_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for voy in voyages:
            em = emission_map.get(voy.voyage_id)
            if em is None:
                continue
            covered = self._voyage_covered_emissions(voy, em, year)
            vessel_voyage_results[voy.vessel_imo].append(covered)

        # Aggregate per vessel
        results: list[ETSResult] = []
        for imo, voy_results in vessel_voyage_results.items():
            results.append(self._aggregate_vessel_annual(imo, voy_results, year, eua_price_eur))
        return results

    def evaluate_fleet(
        self,
        fleet_voyages: list[Voyage],
        fleet_emissions: list[EmissionEstimate],
        eua_price: float,
        year: int,
    ) -> list[ETSResult]:
        return self.evaluate(fleet_voyages, fleet_emissions, eua_price, year)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _voyage_covered_emissions(
        self,
        voyage: Voyage,
        emission: EmissionEstimate,
        year: int,
    ) -> dict[str, Any]:
        scope = voyage.eea_scope
        scope_factor = _SCOPE_FACTORS.get(scope, 0.0)

        # Override from voyage-level ets_scope_factor if set explicitly
        if voyage.ets_scope_factor > 0:
            scope_factor = voyage.ets_scope_factor

        phase_in = self._phase_in_factor(year)
        include_non_co2 = self._include_non_co2(year)

        # EU ETS covers tank-to-wake (combustion) emissions, not well-to-wake.
        # From 2026, CH4 and N2O are added on a CO2e basis -> use TtW CO2e.
        if include_non_co2:
            total_emissions = (
                emission.co2e_ttw_tonnes if emission.co2e_ttw_tonnes > 0 else emission.co2_tonnes
            )
        else:
            total_emissions = emission.co2_tonnes

        covered = total_emissions * scope_factor * phase_in

        return {
            "voyage_id": voyage.voyage_id,
            "vessel_imo": voyage.vessel_imo,
            "eea_scope": scope.value,
            "scope_factor": scope_factor,
            "phase_in_factor": phase_in,
            "total_emissions_tco2e": total_emissions,
            "covered_emissions_tco2e": covered,
            "include_non_co2": include_non_co2,
        }

    def _phase_in_factor(self, year: int) -> float:
        phase_in: dict[str, float] = self._params.get("phase_in", {})
        int_phase = {int(k): float(v) for k, v in phase_in.items()}
        if year in int_phase:
            return int_phase[year]
        sorted_years = sorted(int_phase.keys())
        if not sorted_years:
            return 1.0
        if year > sorted_years[-1]:
            return int_phase[sorted_years[-1]]
        if year < sorted_years[0]:
            return 0.0
        return 1.0

    def _include_non_co2(self, year: int) -> bool:
        """CH4 and N2O are included from 2026 onward."""
        gases_2026 = self._params.get("gases_2026", [])
        if len(gases_2026) > 1 and year >= 2026:
            return True
        return False

    def _aggregate_vessel_annual(
        self,
        vessel_imo: str,
        voyage_results: list[dict[str, Any]],
        year: int,
        eua_price_eur: float,
    ) -> ETSResult:
        total_emissions = sum(r["total_emissions_tco2e"] for r in voyage_results)
        covered_emissions = sum(r["covered_emissions_tco2e"] for r in voyage_results)

        scope_breakdown: dict[str, float] = defaultdict(float)
        for r in voyage_results:
            scope_breakdown[r["eea_scope"]] += r["covered_emissions_tco2e"]

        allowances = covered_emissions
        cost = self._compute_cost(allowances, eua_price_eur)

        phase_in = self._phase_in_factor(year)

        return ETSResult(
            vessel_imo=vessel_imo,
            year=year,
            version=self._version,
            total_emissions_tco2e=round(total_emissions, 4),
            covered_emissions_tco2e=round(covered_emissions, 4),
            scope_breakdown=dict(scope_breakdown),
            phase_in_factor=phase_in,
            allowances_required=round(allowances, 4),
            eua_price_eur=eua_price_eur,
            ets_cost_eur=round(cost, 2),
        )

    @staticmethod
    def _compute_cost(allowances: float, eua_price: float) -> float:
        return allowances * eua_price
