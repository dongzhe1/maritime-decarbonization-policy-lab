from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from carbon_twin.data.schemas import FuelType, ShipType, Vessel, Voyage
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

_VOYAGE_FIELDS = set(Voyage.model_fields.keys())

_DEFAULT_FUEL = FuelType.VLSFO


def _row_to_voyage(row: dict[str, Any]) -> Voyage | None:
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

        co2e_ttw = get_co2e_ttw(fuel_type, fuel_t, fuel_factors or {})
        if fuel_factors and fuel_type in fuel_factors:
            try:
                co2e_wtw = get_co2e_wtw(fuel_type, energy_mj, fuel_factors)
            except KeyError:
                co2e_wtw = co2e_ttw
        else:
            co2e_wtw = co2e_ttw

        speed = voyage.avg_speed_knots
        if not speed or speed <= 0:
            sea = voyage.sea_hours
            speed = (voyage.distance_nm / sea) if sea and sea > 0 else (
                vessel.design_speed_knots or 14.0
            )

        age = (year - vessel.year_built) if vessel.year_built else 0

        rows.append({
            "vessel_imo": imo,
            "year": year,
            "voyage_id": voyage.voyage_id,
            "proxy_emission_tonnes": round(co2e_ttw, 4),
            "ship_type": vessel.ship_type.value,
            "dwt": vessel.deadweight or 0.0,
            "distance_nm": voyage.distance_nm,
            "speed_knots": round(float(speed), 3),
            "displacement_tonnes": round(_estimate_displacement(vessel), 1),
            "draught_m": voyage.draught_m or vessel.draught_m or 0.0,
            "age_years": age,
            "weather_factor": weather,
            "ets_scope_factor": voyage.ets_scope_factor,
            "eea_scope": voyage.eea_scope.value,
            "energy_mj": energy_mj,
            "fuel_tonnes": fuel_t,
            "co2e_ttw_tonnes": round(co2e_ttw, 4),
            "co2e_wtw_tonnes": round(co2e_wtw, 4),
        })

    out = pd.DataFrame(rows)
    out.attrs["skipped"] = skipped
    out.attrs["matched_vessels"] = out["vessel_imo"].nunique() if len(out) else 0
    return out


def load_fuel_factors_safe(
    config_path: str | Path = "configs/fuel_factors.yaml",
) -> dict[FuelType, FuelEmissionFactors] | None:
    try:
        return load_emission_factors(config_path)
    except Exception:
        return None


def vessels_to_index(vessels: list[Vessel]) -> dict[str, Vessel]:
    return {str(v.imo_number).strip(): v for v in vessels}
