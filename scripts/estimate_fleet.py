#!/usr/bin/env python3
"""Tier-1 vessel particulars back-estimated from MRV when IHS data is absent.

Produces ESTIMATES, not measured particulars: deadweight from transport work,
GT and design speed from ship-type typicals. Replace with IHS/Equasis for final
results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall: pip install -e '.[all]'")


# MRV ship-type string (lowercased substring) → internal ShipType value
_SHIP_TYPE_PATTERNS: list[tuple[str, str]] = [
    ("bulk", "bulk_carrier"),
    ("oil tanker", "tanker"),
    ("chemical", "tanker"),
    ("tanker", "tanker"),
    ("container", "container_ship"),
    ("lng", "lng_carrier"),
    ("gas", "gas_carrier"),
    ("ro-pax", "ro_pax"),
    ("ropax", "ro_pax"),
    ("passenger", "cruise_passenger"),
    ("vehicle", "vehicle_carrier"),
    ("ro-ro", "ro_ro_cargo"),
    ("roro", "ro_ro_cargo"),
    ("refrigerated", "refrigerated_cargo"),
    ("reefer", "refrigerated_cargo"),
    ("general cargo", "general_cargo"),
    ("combination", "bulk_carrier"),
]

# Per ship type: (median_dwt, gt_per_dwt, load_factor, design_speed_knots)
# Rough industry typicals — tier-1 only.
_TYPE_PARAMS: dict[str, tuple[float, float, float, float]] = {
    "bulk_carrier": (75000, 0.52, 0.90, 13.5),
    "tanker": (80000, 0.54, 0.88, 13.5),
    "container_ship": (50000, 1.00, 0.65, 19.0),
    "general_cargo": (12000, 0.65, 0.70, 13.0),
    "gas_carrier": (45000, 1.10, 0.85, 16.5),
    "lng_carrier": (75000, 1.55, 0.85, 19.5),
    "ro_ro_cargo": (15000, 1.60, 0.55, 18.0),
    "refrigerated_cargo": (10000, 0.80, 0.65, 18.0),
    "cruise_passenger": (8000, 4.00, 0.40, 20.0),
    "ro_pax": (6000, 2.20, 0.40, 21.0),
    "vehicle_carrier": (18000, 2.40, 0.45, 19.0),
    "offshore": (5000, 1.50, 0.50, 13.0),
    "other": (20000, 0.80, 0.65, 14.0),
}


def _map_ship_type(raw: str) -> str:
    key = str(raw).strip().lower()
    for pat, st in _SHIP_TYPE_PATTERNS:
        if pat in key:
            return st
    return "other"


def _estimate_dwt(transport_work, distance_nm, ship_type: str) -> float:
    median_dwt, _, load_factor, _ = _TYPE_PARAMS[ship_type]
    tw = pd.to_numeric(transport_work, errors="coerce")
    dist = pd.to_numeric(distance_nm, errors="coerce")
    if pd.notna(tw) and pd.notna(dist) and dist > 0 and tw > 0:
        avg_cargo = tw / dist  # tonnes carried on average
        dwt = avg_cargo / max(load_factor, 0.1)
        # Clamp to a plausible band around the type median (0.15×–4×)
        lo, hi = 0.15 * median_dwt, 4.0 * median_dwt
        return float(min(max(dwt, lo), hi))
    return float(median_dwt)


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate tier-1 fleet particulars from MRV")
    ap.add_argument("--mrv", required=True, type=Path, help="Renamed MRV CSV (<year>_renamed.csv)")
    ap.add_argument("--out", required=True, type=Path, help="Output directory")
    ap.add_argument("--year", required=True, type=int)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.mrv, dtype={"imo_number": str})
    if "imo_number" not in df.columns:
        sys.exit("[ERROR] MRV CSV has no imo_number column — run --only-mrv-rename first")

    # One row per vessel (MRV is already annual per-vessel, but dedup to be safe)
    df = df.drop_duplicates(subset=["imo_number"]).copy()

    records: list[dict] = []
    for _, r in df.iterrows():
        imo = str(r["imo_number"]).strip()
        if not imo.isdigit():
            continue
        st = _map_ship_type(r.get("ship_type", "other"))
        _, gt_per_dwt, _, design_speed = _TYPE_PARAMS[st]
        dwt = _estimate_dwt(r.get("transport_work"), r.get("distance_nm"), st)
        gt = dwt * gt_per_dwt
        records.append(
            {
                "imo_number": imo,
                "ship_type": st,
                "gross_tonnage": round(gt, 0),
                "deadweight": round(dwt, 0),
                "design_speed_knots": design_speed,
            }
        )

    fleet = pd.DataFrame(records)
    out_path = args.out / f"fleet_{args.year}.csv"
    fleet.to_csv(out_path, index=False)

    print(f"[done] {len(fleet):,} vessels estimated -> {out_path}")
    print("[note] TIER-1 ESTIMATES. Replace with Equasis/GISIS particulars for final results.")
    print("\nShip-type breakdown:")
    print(fleet["ship_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
