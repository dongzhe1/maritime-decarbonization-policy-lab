"""Per-voyage proxy emissions from voyages + vessel particulars.

A physics estimate (admiralty law + SFOC) of fuel and CO2 per voyage, left
uncalibrated so the MRV calibrator can learn the correction to THETIS-MRV totals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from carbon_twin.data.schemas import FuelType, Vessel, Voyage
from carbon_twin.emissions.factors import (
    FuelEmissionFactors,
    get_co2e_ttw,
    get_co2e_wtw,
    load_emission_factors,
)
from carbon_twin.emissions.physics import (
    _estimate_displacement,
    voyage_fuel_consumption,
)

# Voyage model fields we pass through when reconstructing from a parquet row.
_VOYAGE_FIELDS = set(Voyage.model_fields.keys())

# Default fuel assumption for the proxy (tier-1: single fossil fuel per voyage).
# FuelEU fuel-mix modelling refines this later; for the physics proxy a uniform
# VLSFO assumption is standard and what MRV calibration corrects against.
_DEFAULT_FUEL = FuelType.VLSFO


def _row_to_voyage(row: dict[str, Any]) -> Voyage | None:
    """Reconstruct a Voyage from a parquet row dict, tolerating extra columns."""
    data = {k: v for k, v in row.items() if k in _VOYAGE_FIELDS}
    try:
        return Voyage.model_validate(data)
    except Exception:
        return None


def build_proxy_emissions(
    voyages_df: pd.DataFrame,
    vessels_by_imo: dict[str, Vessel],
    year: int,
    fuel_type: FuelType = _DEFAULT_FUEL,
    fuel_factors: dict[FuelType, FuelEmissionFactors] | None = None,
) -> pd.DataFrame:
    """Compute physics-based proxy emissions for each voyage.

    Parameters
    ----------
    voyages_df : DataFrame
        Output of ``sct build-voyages`` (one row per voyage).
    vessels_by_imo : dict
        Maps IMO string → Vessel (particulars). Voyages whose vessel is absent
        are skipped (logged in the return's ``skipped`` attribute via attrs).
    year : int
        Reporting year stamped on every proxy row.
    fuel_type : FuelType
        Assumed main fuel for the physics proxy.
    fuel_factors : dict, optional
        Loaded emission factors. If None, only the TtW fallback (3.114 t/t) is
        available and WtW falls back to the same value.

    Returns
    -------
    DataFrame with columns:
        vessel_imo, year, voyage_id, proxy_emission_tonnes,
        ship_type, dwt, distance_nm,                       (calibration features)
        speed_knots, displacement_tonnes, draught_m, age_years,
        weather_factor, ets_scope_factor,                 (surrogate features)
        fuel_tonnes, co2e_ttw_tonnes, co2e_wtw_tonnes      (surrogate targets)
    """
    rows: list[dict[str, Any]] = []
    skipped = 0

    for record in voyages_df.to_dict(orient="records"):
        imo = str(record.get("vessel_imo", "")).strip()
        vessel = vessels_by_imo.get(imo)
        if vessel is None:
            skipped += 1
            continue

        voyage = _row_to_voyage(record)
        if voyage is None:
            skipped += 1
            continue

        weather = float(record.get("weather_factor", 1.0) or 1.0)
        fc = voyage_fuel_consumption(vessel, voyage, weather_factor=weather)
        fuel_t = fc["total_tonnes"]
        energy_mj = fc["energy_mj"]

        # TtW CO2e (fallback 3.114 t/t when factors absent)
        co2e_ttw = get_co2e_ttw(fuel_type, fuel_t, fuel_factors or {})
        # WtW CO2e from energy when factors available, else mirror TtW
        if fuel_factors and fuel_type in fuel_factors:
            try:
                co2e_wtw = get_co2e_wtw(fuel_type, energy_mj, fuel_factors)
            except KeyError:
                co2e_wtw = co2e_ttw
        else:
            co2e_wtw = co2e_ttw

        # Effective speed used by physics (for surrogate feature)
        speed = voyage.avg_speed_knots
        if not speed or speed <= 0:
            sea = voyage.sea_hours
            speed = (
                (voyage.distance_nm / sea)
                if sea and sea > 0
                else (vessel.design_speed_knots or 14.0)
            )

        age = (year - vessel.year_built) if vessel.year_built else 0

        rows.append(
            {
                "vessel_imo": imo,
                "year": year,
                "voyage_id": voyage.voyage_id,
                "proxy_emission_tonnes": round(co2e_ttw, 4),
                # calibration features
                "ship_type": vessel.ship_type.value,
                "dwt": vessel.deadweight or 0.0,
                "distance_nm": voyage.distance_nm,
                # surrogate features
                "speed_knots": round(float(speed), 3),
                "displacement_tonnes": round(_estimate_displacement(vessel), 1),
                "draught_m": voyage.draught_m or vessel.draught_m or 0.0,
                "age_years": age,
                "weather_factor": weather,
                "ets_scope_factor": voyage.ets_scope_factor,
                "eea_scope": voyage.eea_scope.value,
                "energy_mj": energy_mj,
                # surrogate targets
                "fuel_tonnes": fuel_t,
                "co2e_ttw_tonnes": round(co2e_ttw, 4),
                "co2e_wtw_tonnes": round(co2e_wtw, 4),
            }
        )

    out = pd.DataFrame(rows)
    out.attrs["skipped"] = skipped
    out.attrs["matched_vessels"] = out["vessel_imo"].nunique() if len(out) else 0
    return out


def load_fuel_factors_safe(
    config_path: str | Path = "configs/fuel_factors.yaml",
) -> dict[FuelType, FuelEmissionFactors] | None:
    """Load emission factors, returning None if the config is unavailable."""
    try:
        return load_emission_factors(config_path)
    except Exception:
        return None


def vessels_to_index(vessels: list[Vessel]) -> dict[str, Vessel]:
    """Index a list of vessels by IMO string for fast lookup."""
    return {str(v.imo_number).strip(): v for v in vessels}
