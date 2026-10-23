from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import yaml

from carbon_twin.data.schemas import ShipType, Vessel


def load_yaml_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_fuel_factors(path: str | Path) -> dict[str, dict[str, Any]]:
    raw = load_yaml_config(path)
    return raw["fuels"]


def load_regulation_versions(path: str | Path) -> dict:
    return load_yaml_config(path)


def load_eua_prices(path: str | Path, scenario: str = "base") -> dict[int, float]:
    raw = load_yaml_config(path)
    prices = raw["eua_price_scenarios"][scenario]["prices"]
    return {int(yr): float(p) for yr, p in prices.items()}


def load_usd_to_eur(path: str | Path = "configs/currency.yaml") -> float:
    try:
        return float(load_yaml_config(path)["usd_to_eur"])
    except Exception:
        return 0.9239


def load_fuel_prices(path: str | Path, scenario: str = "base_2026") -> dict[str, float]:
    raw = load_yaml_config(path)
    prices = raw["price_scenarios"][scenario]["prices"]
    fx = load_usd_to_eur()
    return {str(k): float(v) * fx for k, v in prices.items()}


_MRV_COLUMNS = {
    "imo_number": str,
    "ship_name": str,
    "ship_type": str,
    "technical_efficiency": float,
    "port_of_registry": str,
    "year": int,
    "total_co2_tonnes": float,
    "total_fuel_tonnes": float,
    "distance_nm": float,
    "hours_at_sea": float,
    "avg_speed": float,
    "transport_work": float,
    "energy_efficiency": float,
}


def load_mrv_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"imo_number": str}, low_memory=False)
    present = {c for c in _MRV_COLUMNS if c in df.columns}
    for col in present:
        expected = _MRV_COLUMNS[col]
        if expected is float:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif expected is int:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        else:
            df[col] = df[col].astype(str)
    return df


def load_ais_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "mmsi" in df.columns:
        df["mmsi"] = df["mmsi"].astype(str)
    if "imo_number" in df.columns:
        df["imo_number"] = df["imo_number"].astype(str)
    return df


def load_ports_geojson(path: str | Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    return gdf


_SHIP_TYPE_MAP: dict[str, ShipType] = {
    "bulk carrier": ShipType.BULK_CARRIER,
    "bulk_carrier": ShipType.BULK_CARRIER,
    "tanker": ShipType.TANKER,
    "oil tanker": ShipType.TANKER,
    "chemical tanker": ShipType.TANKER,
    "container ship": ShipType.CONTAINER_SHIP,
    "container_ship": ShipType.CONTAINER_SHIP,
    "general cargo": ShipType.GENERAL_CARGO,
    "general_cargo": ShipType.GENERAL_CARGO,
    "gas carrier": ShipType.GAS_CARRIER,
    "gas_carrier": ShipType.GAS_CARRIER,
    "lng carrier": ShipType.LNG_CARRIER,
    "lng_carrier": ShipType.LNG_CARRIER,
    "ro-ro cargo": ShipType.RO_RO_CARGO,
    "ro_ro_cargo": ShipType.RO_RO_CARGO,
    "refrigerated cargo": ShipType.REFRIGERATED_CARGO,
    "refrigerated_cargo": ShipType.REFRIGERATED_CARGO,
    "cruise passenger": ShipType.CRUISE_PASSENGER,
    "cruise_passenger": ShipType.CRUISE_PASSENGER,
    "passenger ship": ShipType.CRUISE_PASSENGER,
    "ro-pax": ShipType.RO_PAX,
    "ro_pax": ShipType.RO_PAX,
    "vehicle carrier": ShipType.VEHICLE_CARRIER,
    "vehicle_carrier": ShipType.VEHICLE_CARRIER,
    "vehicles carrier": ShipType.VEHICLE_CARRIER,
    "offshore": ShipType.OFFSHORE,
}


def _parse_ship_type(raw: str) -> ShipType:
    key = raw.strip().lower()
    return _SHIP_TYPE_MAP.get(key, ShipType.OTHER)


def load_vessels_csv(path: str | Path) -> list[Vessel]:
    df = pd.read_csv(path, dtype={"imo_number": str, "mmsi": str})
    vessels: list[Vessel] = []
    for _, row in df.iterrows():
        ship_type = _parse_ship_type(str(row.get("ship_type", "other")))
        gt = pd.to_numeric(row.get("gross_tonnage", 0), errors="coerce")

        kwargs: dict[str, Any] = {
            "imo_number": str(row["imo_number"]),
            "ship_type": ship_type,
            "gross_tonnage": float(gt) if pd.notna(gt) else 0.0,
        }

        optional_str = ["mmsi", "name", "flag", "ice_class"]
        optional_float = [
            "deadweight", "net_tonnage", "design_speed_knots",
            "main_engine_power_kw", "aux_engine_power_kw",
            "length_m", "beam_m", "draught_m",
        ]
        optional_int = ["year_built"]

        for col in optional_str:
            val = row.get(col)
            if pd.notna(val):
                kwargs[col] = str(val)

        for col in optional_float:
            val = row.get(col)
            v = pd.to_numeric(val, errors="coerce")
            if pd.notna(v):
                kwargs[col] = float(v)

        for col in optional_int:
            val = row.get(col)
            v = pd.to_numeric(val, errors="coerce")
            if pd.notna(v):
                kwargs[col] = int(v)

        vessels.append(Vessel(**kwargs))
    return vessels
