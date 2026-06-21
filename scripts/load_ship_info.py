#!/usr/bin/env python3
"""Convert IHS Fairplay / Sea-web ship_info.csv to the standard fleet CSV
(imo_number, ship_type, gross_tonnage, deadweight, design_speed_knots,
main_engine_power_kw, year_built, ...) consumed by the regulation engines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall: pip install -e '.[all]'")

# IHS Fairplay shiptype_level5 → internal ShipType. Ordered specific-first.
_TYPE_MAP: list[tuple[str, str]] = [
    # LNG before general gas
    ("lng tanker", "lng_carrier"),
    ("lpg tanker", "gas_carrier"),
    # ro-pax before plain passenger
    ("passenger/ro-ro ship (vehicles/rail)", "ro_pax"),
    ("passenger/ro-ro ship (vehicles)", "ro_pax"),
    ("passenger/ro-ro", "ro_pax"),
    # cruise
    ("passenger/cruise", "cruise_passenger"),
    ("passenger ship", "cruise_passenger"),
    ("cruise", "cruise_passenger"),
    # vehicle carrier
    ("vehicles carrier", "vehicle_carrier"),
    # bulk
    ("bulk carrier, self-discharging", "bulk_carrier"),
    ("bulk carrier", "bulk_carrier"),
    ("ore carrier", "bulk_carrier"),
    ("open hatch cargo ship", "general_cargo"),
    ("cement carrier", "bulk_carrier"),
    # tanker
    ("crude oil tanker", "tanker"),
    ("crude/oil products tanker", "tanker"),
    ("products tanker", "tanker"),
    ("chemical/products tanker", "tanker"),
    ("chemical tanker", "tanker"),
    ("asphalt/bitumen tanker", "tanker"),
    ("shuttle tanker", "tanker"),
    ("tanker", "tanker"),
    # container
    ("container ship", "container_ship"),
    # ro-ro
    ("ro-ro cargo ship", "ro_ro_cargo"),
    # reefer
    ("refrigerated cargo ship", "refrigerated_cargo"),
    # general cargo
    ("general cargo ship", "general_cargo"),
    ("general dry cargo", "general_cargo"),
    ("heavy load carrier", "general_cargo"),
    ("livestock carrier", "general_cargo"),
    # other commercial
    ("gas carrier", "gas_carrier"),
]

# Types relevant to CII / EU MRV (vessels ≥ 5,000 GT calling EU ports).
# We export all mapped types; caller can filter by GT if needed.
_KEEP_TYPES = {
    "bulk_carrier",
    "tanker",
    "container_ship",
    "general_cargo",
    "vehicle_carrier",
    "ro_pax",
    "ro_ro_cargo",
    "cruise_passenger",
    "gas_carrier",
    "lng_carrier",
    "refrigerated_cargo",
}


def _map_type(raw: str | None) -> str | None:
    if not raw:
        return None
    low = str(raw).strip().lower()
    for pat, st in _TYPE_MAP:
        if pat in low:
            return st
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert IHS ship_info.csv → standard fleet CSV")
    ap.add_argument("--ship-info", required=True, type=Path, help="IHS Fairplay ship_info.csv")
    ap.add_argument(
        "--imo-file",
        type=Path,
        default=None,
        help="Plain IMO list (one per line); if given, output is filtered to these IMOs only",
    )
    ap.add_argument(
        "--out", required=True, type=Path, help="Output CSV path (e.g. fleet_ship_info.csv)"
    )
    ap.add_argument(
        "--min-gt", type=float, default=0, help="Drop vessels below this GT (default 0 = keep all)"
    )
    args = ap.parse_args()

    print(f"[load_ship_info] reading {args.ship_info} ...")
    df = pd.read_csv(
        args.ship_info,
        low_memory=False,
        usecols=[
            "lrnoimo_ship_no",
            "ship_name",
            "shiptype_level5",
            "deadweight",
            "gross_tonnage",
            "speedservice",
            "speedmax",
            "total_kilowattsof_main_engines",
            "year_of_build",
            "flag_name",
        ],
    )
    print(f"  raw rows: {len(df):,}")

    # Map ship type
    df["ship_type"] = df["shiptype_level5"].map(_map_type)
    mapped = df["ship_type"].notna().sum()
    print(f"  type-mapped: {mapped:,} / {len(df):,}")

    # Keep only commercial types
    df = df[df["ship_type"].isin(_KEEP_TYPES)].copy()
    print(f"  after type filter: {len(df):,}")

    # GT filter
    df["gross_tonnage"] = pd.to_numeric(df["gross_tonnage"], errors="coerce")
    if args.min_gt > 0:
        df = df[df["gross_tonnage"] >= args.min_gt].copy()
        print(f"  after GT≥{args.min_gt:.0f} filter: {len(df):,}")

    # Filter to target IMO list if supplied
    if args.imo_file:
        target = {
            int(ln.strip()) for ln in args.imo_file.read_text().splitlines() if ln.strip().isdigit()
        }
        print(f"  target IMO list: {len(target):,}")
        before = len(df)
        df = df[df["lrnoimo_ship_no"].isin(target)].copy()
        print(
            f"  matched: {len(df):,} / {before:,} "
            f"({100 * len(df) / max(len(target), 1):.1f}% of target)"
        )
        unmatched = len(target) - len(df)
        if unmatched:
            print(f"  unmatched IMOs (not in ship_info): {unmatched:,}")

    # Build design_speed_knots: prefer speedservice > 0, fall back to speedmax
    df["speedservice"] = pd.to_numeric(df["speedservice"], errors="coerce")
    df["speedmax"] = pd.to_numeric(df["speedmax"], errors="coerce")
    df["design_speed_knots"] = df["speedservice"].where(df["speedservice"] > 0)
    df["design_speed_knots"] = df["design_speed_knots"].fillna(df["speedmax"])

    # Fill remaining nulls with per-type fallbacks (same as estimate_fleet)
    _SPEED_DEFAULTS = {
        "bulk_carrier": 13.5,
        "tanker": 13.5,
        "container_ship": 19.0,
        "general_cargo": 13.0,
        "gas_carrier": 16.5,
        "lng_carrier": 19.5,
        "ro_ro_cargo": 18.0,
        "refrigerated_cargo": 18.0,
        "cruise_passenger": 20.0,
        "ro_pax": 21.0,
        "vehicle_carrier": 19.0,
    }
    for st, spd in _SPEED_DEFAULTS.items():
        mask = df["ship_type"].eq(st) & df["design_speed_knots"].isna()
        df.loc[mask, "design_speed_knots"] = spd

    # Rename / select output columns. Column names must match load_vessels_csv's
    # schema (main_engine_power_kw / year_built / flag) so the physics engine
    # actually consumes the real engine power and build year.
    df["deadweight"] = pd.to_numeric(df["deadweight"], errors="coerce")
    df["main_engine_power_kw"] = pd.to_numeric(
        df["total_kilowattsof_main_engines"], errors="coerce"
    )
    df["year_built"] = pd.to_numeric(df["year_of_build"], errors="coerce").astype("Int64")

    out = df.rename(
        columns={
            "lrnoimo_ship_no": "imo_number",
            "ship_name": "name",
            "flag_name": "flag",
        }
    )[
        [
            "imo_number",
            "name",
            "ship_type",
            "gross_tonnage",
            "deadweight",
            "design_speed_knots",
            "main_engine_power_kw",
            "year_built",
            "flag",
        ]
    ].copy()

    # Remove duplicate IMOs (keep first occurrence)
    before = len(out)
    out = out.drop_duplicates(subset=["imo_number"])
    if len(out) < before:
        print(f"  dropped {before - len(out)} duplicate IMO rows")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n[done] {len(out):,} vessels → {args.out}")

    # Summary stats
    print("\n=== Coverage by type ===")
    for st in sorted(out["ship_type"].unique()):
        sub = out[out["ship_type"] == st]
        gt_ok = sub["gross_tonnage"].notna().sum()
        dwt_ok = sub["deadweight"].notna().sum()
        spd_ok = sub["design_speed_knots"].notna().sum()
        kw_ok = sub["main_engine_power_kw"].notna().sum()
        n = len(sub)
        print(
            f"  {st:20s}: n={n:4,}  GT={gt_ok / n * 100:.0f}%  "
            f"DWT={dwt_ok / n * 100:.0f}%  spd={spd_ok / n * 100:.0f}%  kW={kw_ok / n * 100:.0f}%"
        )


if __name__ == "__main__":
    main()
