"""FuelEU Maritime (Regulation EU 2023/1805) compliance engine."""

from __future__ import annotations

from typing import Any

from carbon_twin.data.schemas import (
    FuelEUResult,
    FuelRecord,
    FuelType,
    Vessel,
)
from carbon_twin.regulations.versioning import RegulationVersionManager

# Fuel types classified as RFNBO (Renewable Fuels of Non-Biological Origin)
_RFNBO_FUEL_TYPES: set[FuelType] = {
    FuelType.METHANOL_E,
    FuelType.AMMONIA_GREEN,
    FuelType.HYDROGEN_GREEN,
}

# Energy content of one tonne of VLSFO-equivalent (MJ), used in the FuelEU
# penalty formula (Regulation (EU) 2023/1805, Annex IV).
_VLSFO_ENERGY_MJ_PER_TONNE: float = 41000.0


class FuelEUEngine:
    """FuelEU Maritime GHG intensity compliance engine."""

    def __init__(
        self,
        version: str = "EU_2023_1805",
        config_path: str | None = None,
    ) -> None:
        self._version = version
        mgr = RegulationVersionManager(
            config_path if config_path else "configs/regulation_versions.yaml"
        )
        self._params: dict[str, Any] = mgr.get_fueleu_params(version)

    # ------------------------------------------------------------------
    # Default WtW emission factors (gCO2e/MJ) used when caller provides none
    # ------------------------------------------------------------------

    _DEFAULT_FACTORS: dict[str, float] = {
        "HFO": 91.16,
        "VLSFO": 90.0,
        "MGO": 88.0,
        "LNG": 80.0,
        "LPG": 85.0,
        "methanol_fossil": 98.0,
        "methanol_bio": 14.0,
        "methanol_e": 5.0,
        "ammonia_fossil": 120.0,
        "ammonia_green": 8.0,
        "biodiesel_fame": 30.0,
        "hydrogen_green": 4.0,
        "shore_power": 0.0,
    }

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def evaluate(
        self,
        vessel: Vessel,
        fuel_records: list[FuelRecord],
        energy_mj: float,
        year: int,
        emission_factors: dict[str, float] | None = None,
        pooling_group: str | None = None,
    ) -> FuelEUResult:
        factors = emission_factors if emission_factors is not None else self._DEFAULT_FACTORS

        rfnbo_bonus = self._apply_rfnbo_multiplier(fuel_records, factors)
        actual = self._compute_ghg_intensity(fuel_records, energy_mj, factors, rfnbo_bonus)
        target = self._target_intensity(year)
        balance = self._compliance_balance(actual, target, energy_mj)
        surplus = balance >= 0
        penalty = self._compute_penalty(balance, actual, year) if not surplus else 0.0

        return FuelEUResult(
            vessel_imo=vessel.imo_number,
            year=year,
            version=self._version,
            actual_ghg_intensity=round(actual, 4),
            target_ghg_intensity=round(target, 4),
            compliance_balance=round(balance, 2),
            surplus=surplus,
            penalty_eur=round(penalty, 2),
            rfnbo_bonus_applied=rfnbo_bonus > 0,
            pooling_group=pooling_group,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_ghg_intensity(
        self,
        fuel_records: list[FuelRecord],
        total_energy_mj: float,
        factors: dict[str, float],
        rfnbo_bonus_energy: float = 0.0,
    ) -> float:
        """Weighted average GHG intensity in gCO2e/MJ.

        RFNBO bonus energy is added to the denominator (energy counted 2x) but
        emissions stay the same, effectively lowering the intensity.
        """
        total_co2e_g = 0.0

        for rec in fuel_records:
            ef = factors.get(rec.fuel_type.value, factors.get(rec.fuel_type, 0.0))
            energy = rec.energy_mj if rec.energy_mj and rec.energy_mj > 0 else 0.0
            total_co2e_g += ef * energy

        effective_energy = total_energy_mj + rfnbo_bonus_energy
        if effective_energy <= 0:
            return 0.0
        return total_co2e_g / effective_energy

    def _target_intensity(self, year: int) -> float:
        ref = float(self._params.get("reference_value_gCO2e_per_MJ", 91.16))
        targets: dict[str, float] = self._params.get("reduction_targets", {})

        # Find the applicable reduction target for the given year
        int_targets = {int(k): float(v) for k, v in targets.items()}
        applicable_years = sorted(y for y in int_targets if y <= year)
        if not applicable_years:
            return ref
        reduction = int_targets[applicable_years[-1]]
        return ref * (1.0 - reduction)

    @staticmethod
    def _compliance_balance(actual: float, target: float, energy_mj: float) -> float:
        """Positive = surplus, negative = deficit."""
        return (target - actual) * energy_mj

    def _compute_penalty(self, deficit_balance: float, actual_intensity: float, year: int) -> float:
        """Penalty for non-compliance per Regulation (EU) 2023/1805, Annex IV.

        ``deficit_balance`` is the compliance balance in gCO2e (negative when in
        deficit).  The regulation defines the penalty as::

            penalty_EUR = |compliance_balance| / (GHGIE_actual * 41000) * 2400

        where 41000 MJ is the energy content of one tonne of VLSFO-equivalent
        and 2400 EUR is the price per tonne of VLSFO-equivalent energy deficit.
        """
        if deficit_balance >= 0:
            return 0.0
        if actual_intensity <= 0:
            return 0.0
        penalty_rate = float(self._params.get("penalty_per_gCO2e_per_MJ_deficit", 2400))
        vlsfoe_tonnes_deficit = abs(deficit_balance) / (
            actual_intensity * _VLSFO_ENERGY_MJ_PER_TONNE
        )
        return vlsfoe_tonnes_deficit * penalty_rate

    def _apply_rfnbo_multiplier(
        self,
        fuel_records: list[FuelRecord],
        factors: dict[str, float],
    ) -> float:
        """Return extra energy (MJ) from RFNBO fuels counted at 2x toward compliance."""
        multiplier = float(self._params.get("rfnbo_multiplier", 2.0))
        extra_energy = 0.0
        for rec in fuel_records:
            if rec.fuel_type in _RFNBO_FUEL_TYPES:
                energy = rec.energy_mj if rec.energy_mj and rec.energy_mj > 0 else 0.0
                # Extra credit = (multiplier - 1) * energy
                extra_energy += (multiplier - 1.0) * energy
        return extra_energy

    @staticmethod
    def apply_banking(
        current_balance: float,
        banked_balance: float,
    ) -> tuple[float, float]:
        """Apply banked surplus from previous year to current balance.

        Returns (adjusted_balance, remaining_bank).
        """
        if current_balance >= 0:
            return current_balance, banked_balance
        needed = abs(current_balance)
        applied = min(needed, banked_balance)
        return current_balance + applied, banked_balance - applied

    @staticmethod
    def apply_borrowing(
        current_deficit: float,
        year: int,
        max_borrow_pct: float = 0.02,
    ) -> tuple[float, float]:
        """Borrow against next year's compliance to reduce current deficit.

        Returns (adjusted_deficit, borrowing_obligation).
        """
        if current_deficit >= 0:
            return current_deficit, 0.0
        borrow_limit = abs(current_deficit) * max_borrow_pct
        borrow_amount = min(abs(current_deficit), borrow_limit)
        return current_deficit + borrow_amount, borrow_amount

    @staticmethod
    def evaluate_pool(pool_results: list[FuelEUResult]) -> list[FuelEUResult]:
        """Redistribute surplus/deficit within a pooling group.

        Vessels with surplus offset those with deficit.  Residual deficit is
        spread proportionally among deficit vessels.
        """
        if not pool_results:
            return []

        total_balance = sum(r.compliance_balance for r in pool_results)
        total_energy = sum(abs(r.compliance_balance) for r in pool_results) or 1.0

        adjusted: list[FuelEUResult] = []
        for r in pool_results:
            # Simplified: each vessel gets a share of the total pool balance
            # proportional to the magnitude of its own balance.
            share = abs(r.compliance_balance) / total_energy
            pooled_balance = total_balance * share
            adjusted.append(
                r.model_copy(
                    update={
                        "compliance_balance": round(pooled_balance, 2),
                        "surplus": pooled_balance >= 0,
                        "pooling_contribution": round(pooled_balance - r.compliance_balance, 2),
                        "penalty_eur": 0.0 if pooled_balance >= 0 else r.penalty_eur,
                    }
                )
            )
        return adjusted
