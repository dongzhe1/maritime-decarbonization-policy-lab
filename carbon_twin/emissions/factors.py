"""Fuel emission factors and CO2-equivalent calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from carbon_twin.data.schemas import FuelRecord, FuelType

GWP100: dict[str, float] = {"CO2": 1.0, "CH4": 28.0, "N2O": 265.0}


@dataclass(frozen=True, slots=True)
class FuelEmissionFactors:
    """Emission factors for a single fuel type."""

    lcv_mj_per_kg: float = 0.0
    density_kg_per_m3: float = 0.0
    ttw_co2_g_per_g_fuel: float = 0.0
    ttw_ch4_g_per_g_fuel: float = 0.0
    ttw_n2o_g_per_g_fuel: float = 0.0
    wtw_co2e_g_per_mj: float = 0.0
    fueleu_default_emission_factor_gCO2e_per_MJ: float = 0.0
    category: str = "fossil"
    extra: dict[str, Any] = field(default_factory=dict)


def load_emission_factors(config_path: str | Path) -> dict[FuelType, FuelEmissionFactors]:
    """Load per-fuel emission factors from a YAML config file."""
    config_path = Path(config_path)
    with open(config_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    fuels_raw: dict[str, Any] = raw.get("fuels", {})
    factors: dict[FuelType, FuelEmissionFactors] = {}
    for fuel_key, vals in fuels_raw.items():
        try:
            ft = FuelType(fuel_key)
        except ValueError:
            continue

        known_fields = {
            "lcv_mj_per_kg",
            "density_kg_per_m3",
            "ttw_co2_g_per_g_fuel",
            "ttw_ch4_g_per_g_fuel",
            "ttw_n2o_g_per_g_fuel",
            "wtw_co2e_g_per_mj",
            "fueleu_default_emission_factor_gCO2e_per_MJ",
            "category",
        }
        core = {k: v for k, v in vals.items() if k in known_fields}
        extra = {k: v for k, v in vals.items() if k not in known_fields and k != "name"}
        factors[ft] = FuelEmissionFactors(**core, extra=extra)

    return factors


_DEFAULT_TTW_CO2_G_PER_G: float = 3.114  # VLSFO / HFO-like fallback


def get_co2e_ttw(
    fuel_type: FuelType,
    mass_tonnes: float,
    factors: dict[FuelType, FuelEmissionFactors],
) -> float:
    """Tank-to-wake CO2-equivalent in tonnes, using GWP-100 weighting."""
    if fuel_type not in factors:
        return mass_tonnes * _DEFAULT_TTW_CO2_G_PER_G  # fallback: 1 tonne fuel → ~3.114 t CO2
    ef = factors[fuel_type]
    mass_g = mass_tonnes * 1e6
    co2_g = ef.ttw_co2_g_per_g_fuel * mass_g
    ch4_g = ef.ttw_ch4_g_per_g_fuel * mass_g
    n2o_g = ef.ttw_n2o_g_per_g_fuel * mass_g
    co2e_g = co2_g * GWP100["CO2"] + ch4_g * GWP100["CH4"] + n2o_g * GWP100["N2O"]
    return co2e_g / 1e6


def get_co2e_wtw(
    fuel_type: FuelType,
    energy_mj: float,
    factors: dict[FuelType, FuelEmissionFactors],
) -> float:
    """Well-to-wake CO2-equivalent in tonnes from energy consumed."""
    if fuel_type not in factors:
        raise KeyError(f"No emission factors for fuel type {fuel_type}")
    ef = factors[fuel_type]
    co2e_g = ef.wtw_co2e_g_per_mj * energy_mj
    return co2e_g / 1e6


def get_ghg_intensity_wtw(
    fuel_records: list[FuelRecord],
    factors: dict[FuelType, FuelEmissionFactors],
) -> float:
    """Energy-weighted average WTW GHG intensity in gCO2e/MJ (for FuelEU Maritime).

    Each fuel record contributes its energy (mass * LCV) and the corresponding
    WTW emissions.  Returns the total-energy-weighted average intensity.
    """
    total_energy_mj = 0.0
    total_co2e_g = 0.0

    for rec in fuel_records:
        if rec.fuel_type not in factors:
            continue
        ef = factors[rec.fuel_type]

        if rec.energy_mj is not None and rec.energy_mj > 0:
            energy = rec.energy_mj
        else:
            energy = rec.mass_tonnes * 1e3 * ef.lcv_mj_per_kg

        total_energy_mj += energy
        total_co2e_g += ef.wtw_co2e_g_per_mj * energy

    if total_energy_mj == 0.0:
        return 0.0
    return total_co2e_g / total_energy_mj
