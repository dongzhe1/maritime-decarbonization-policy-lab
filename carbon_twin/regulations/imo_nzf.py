"""IMO Net-Zero Framework scenario analysis engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from carbon_twin.data.schemas import (
    FuelRecord,
    IMONZFResult,
    Vessel,
)
from carbon_twin.regulations.versioning import RegulationVersionManager

# Mapping from regulation-version scenario key to its scenario YAML file.
_SCENARIO_FILES: dict[str, str] = {
    "MEPC83_draft": "configs/scenarios/imo_nzf_mepc83.yaml",
    "MEPC84_negotiation": "configs/scenarios/imo_nzf_mepc84_uncertain.yaml",
}
# Fallback used only when the scenario key is unknown.
_DEFAULT_SCENARIO_PATH = "configs/scenarios/imo_nzf_mepc83.yaml"


class IMONZFScenarioEngine:
    """IMO Net-Zero Framework GHG Fuel Intensity (GFI) scenario engine.

    Supports multiple GHG price points and scenario-specific GFI targets that
    are still under negotiation (MEPC 83/84 drafts).
    """

    def __init__(
        self,
        scenario: str = "MEPC83_draft",
        config_path: str | None = None,
        scenario_config_path: str | None = None,
    ) -> None:
        self._scenario = scenario
        mgr = RegulationVersionManager(
            config_path if config_path else "configs/regulation_versions.yaml"
        )
        self._reg_params: dict[str, Any] = mgr.get_imo_nzf_params(scenario)

        # Load scenario-specific parameters (GFI targets, price assumptions).
        # The file is chosen from the selected scenario so that different
        # scenarios genuinely load different parameters.
        self._scenario_config_path = scenario_config_path
        self._scenario_params = self._load_scenario_params()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def evaluate(
        self,
        vessel: Vessel,
        annual_energy_mj: float,
        fuel_records: list[FuelRecord],
        emission_factors: dict[str, float],
        year: int,
        ghg_price_per_tco2e: float | None = None,
    ) -> IMONZFResult:
        gfi_att = self._gfi_attained(fuel_records, annual_energy_mj, emission_factors)
        gfi_tgt = self._gfi_target(year, self._scenario_params)

        if gfi_tgt is not None:
            # GFI reference baseline is fossil HFO WtW ~91.16 gCO2e/MJ
            reference = 91.16
            target_absolute = reference * (1.0 - gfi_tgt)
            deficit_or_surplus = target_absolute - gfi_att  # positive = good
            deficit_or_surplus_scaled = deficit_or_surplus * annual_energy_mj
        else:
            target_absolute = None
            deficit_or_surplus = None
            deficit_or_surplus_scaled = None

        penalty_or_reward = 0.0
        reward_units = 0.0

        if deficit_or_surplus_scaled is not None and ghg_price_per_tco2e is not None:
            # Convert gCO2e deficit to tCO2e: deficit_or_surplus_scaled is in gCO2e
            deficit_tco2e = deficit_or_surplus_scaled / 1e6
            penalty_or_reward = self._compute_penalty_or_reward(
                deficit_tco2e, annual_energy_mj, ghg_price_per_tco2e
            )
            if deficit_or_surplus_scaled > 0:
                reward_units = self._compute_reward_units(
                    deficit_or_surplus_scaled, annual_energy_mj
                )

        return IMONZFResult(
            vessel_imo=vessel.imo_number,
            year=year,
            scenario=self._scenario,
            gfi_attained=round(gfi_att, 4) if gfi_att is not None else None,
            gfi_target=round(target_absolute, 4) if target_absolute is not None else None,
            deficit_or_surplus=(
                round(deficit_or_surplus_scaled, 2)
                if deficit_or_surplus_scaled is not None
                else None
            ),
            ghg_price_per_tco2e=ghg_price_per_tco2e,
            penalty_or_reward_eur=round(penalty_or_reward, 2),
            reward_units=round(reward_units, 4),
        )

    def evaluate_scenario_sweep(
        self,
        vessel: Vessel,
        energy: float,
        fuels: list[FuelRecord],
        factors: dict[str, float],
        year: int,
        ghg_prices: list[float],
    ) -> list[IMONZFResult]:
        """Run the same vessel scenario across multiple GHG price points."""
        return [self.evaluate(vessel, energy, fuels, factors, year, price) for price in ghg_prices]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _gfi_attained(
        fuel_records: list[FuelRecord],
        energy_mj: float,
        factors: dict[str, float],
    ) -> float:
        """GHG Fuel Intensity: energy-weighted WtW gCO2e/MJ."""
        total_co2e_g = 0.0
        for rec in fuel_records:
            ef = factors.get(rec.fuel_type.value, factors.get(rec.fuel_type, 0.0))
            energy = rec.energy_mj if rec.energy_mj and rec.energy_mj > 0 else 0.0
            total_co2e_g += ef * energy
        if energy_mj <= 0:
            return 0.0
        return total_co2e_g / energy_mj

    @staticmethod
    def _gfi_target(year: int, scenario_params: dict) -> float | None:
        """Look up GFI reduction target for a given year; returns fraction or None."""
        targets: dict[str, float] = scenario_params.get("gfi_reduction_target", {})
        int_targets = {int(k): v for k, v in targets.items() if v is not None}
        if not int_targets:
            return None
        applicable = sorted(y for y in int_targets if y <= year)
        if not applicable:
            return None
        return float(int_targets[applicable[-1]])

    @staticmethod
    def _compute_penalty_or_reward(
        deficit_tco2e: float,
        energy_mj: float,
        ghg_price: float,
    ) -> float:
        """Positive deficit_tco2e = surplus (reward), negative = deficit (penalty).

        Returns EUR amount: negative for penalty, positive for reward.
        """
        return deficit_tco2e * ghg_price

    @staticmethod
    def _compute_reward_units(surplus_gco2e: float, energy_mj: float) -> float:
        """Reward units proportional to surplus.  1 unit per tonne CO2e surplus."""
        if surplus_gco2e <= 0:
            return 0.0
        return surplus_gco2e / 1e6

    def _load_scenario_params(self) -> dict:
        """Load detailed scenario params from the YAML file for this scenario.

        The file is selected from ``scenario_config_path`` if provided, else from
        the scenario name via ``_SCENARIO_FILES``.  Parameters live under
        ``scenario.imo_nzf_params`` (e.g. ``gfi_reduction_target`` for MEPC 83,
        or scenario-keyed stringency levels for MEPC 84).  Returns ``{}`` when no
        file/params are found.
        """
        path_str = self._scenario_config_path or _SCENARIO_FILES.get(
            self._scenario, _DEFAULT_SCENARIO_PATH
        )
        scenario_path = Path(path_str)
        if not scenario_path.exists():
            return {}
        with open(scenario_path) as f:
            raw = yaml.safe_load(f) or {}

        return raw.get("scenario", {}).get("imo_nzf_params", {})
