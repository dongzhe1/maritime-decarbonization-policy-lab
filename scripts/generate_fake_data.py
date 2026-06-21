#!/usr/bin/env python3
"""Generate synthetic MRV + fleet CSVs so the pipeline can be run end-to-end
without the (commercial / registration-gated) real inputs.

The numbers are physically plausible — admiralty-law fuel burn, realistic GT/DWT
and transport work — so CII ratings and compliance costs come out in a sensible
range, but they are random and must NOT be used for analysis. For real results
use THETIS-MRV + IHS particulars (see README).

    python scripts/generate_fake_data.py --n-vessels 500 --year 2024
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Per ship type: (median_dwt, gt_per_dwt, load_factor, design_speed, main_engine_kw)
_TYPE_PARAMS: dict[str, tuple[float, float, float, float, float]] = {
    "bulk_carrier": (75000, 0.52, 0.90, 13.5, 10000),
    "tanker": (80000, 0.54, 0.88, 13.5, 12000),
    "container_ship": (50000, 1.00, 0.65, 19.0, 40000),
    "general_cargo": (12000, 0.65, 0.70, 13.0, 6000),
    "gas_carrier": (45000, 1.10, 0.85, 16.5, 14000),
    "lng_carrier": (75000, 1.55, 0.85, 19.5, 28000),
    "ro_ro_cargo": (15000, 1.60, 0.55, 18.0, 16000),
    "refrigerated_cargo": (10000, 0.80, 0.65, 18.0, 12000),
    "cruise_passenger": (8000, 4.00, 0.40, 20.0, 40000),
    "ro_pax": (6000, 2.20, 0.40, 21.0, 22000),
    "vehicle_carrier": (18000, 2.40, 0.45, 19.0, 14000),
}

# Rough fleet shares (sum need not be 1; used as sampling weights).
_TYPE_WEIGHTS: dict[str, float] = {
    "bulk_carrier": 0.29,
    "tanker": 0.26,
    "container_ship": 0.16,
    "general_cargo": 0.11,
    "gas_carrier": 0.025,
    "lng_carrier": 0.027,
    "ro_ro_cargo": 0.021,
    "refrigerated_cargo": 0.009,
    "cruise_passenger": 0.017,
    "ro_pax": 0.034,
    "vehicle_carrier": 0.038,
}

_SFOC_G_PER_KWH = 180.0  # main-engine specific fuel consumption
_CO2_PER_FUEL_T = 3.114  # tCO2 per t VLSFO (IMO)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic MRV + fleet CSVs")
    ap.add_argument("--n-vessels", type=int, default=500)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    types = list(_TYPE_WEIGHTS)
    weights = np.array([_TYPE_WEIGHTS[t] for t in types])
    weights = weights / weights.sum()
    chosen = rng.choice(types, size=args.n_vessels, p=weights)

    fleet_rows: list[dict] = []
    mrv_rows: list[dict] = []
    for i, st in enumerate(chosen):
        imo = str(9000000 + i)  # synthetic 7-digit IMO-like number
        median_dwt, gt_per_dwt, load_factor, design_speed, me_kw = _TYPE_PARAMS[st]

        dwt = float(
            np.clip(rng.lognormal(np.log(median_dwt), 0.5), 0.2 * median_dwt, 4 * median_dwt)
        )
        gt = dwt * gt_per_dwt * rng.uniform(0.9, 1.1)
        kw = me_kw * (dwt / median_dwt) ** 0.5 * rng.uniform(0.85, 1.15)
        year_built = int(rng.integers(1998, args.year - 1))

        fleet_rows.append(
            {
                "imo_number": imo,
                "name": f"FAKE {st.upper()} {i}",
                "ship_type": st,
                "gross_tonnage": round(gt),
                "deadweight": round(dwt),
                "design_speed_knots": round(design_speed * rng.uniform(0.95, 1.05), 1),
                "main_engine_power_kw": round(kw),
                "year_built": year_built,
                "flag": "XX",
            }
        )

        # Operating profile: slow-steam at ~0.7 design, power ∝ speed³.
        op_speed = design_speed * rng.uniform(0.62, 0.82)
        distance = float(rng.uniform(45000, 115000))  # annual nm
        hours = distance / op_speed
        power_used = kw * (op_speed / design_speed) ** 3 * rng.uniform(0.95, 1.15)
        me_fuel_t = power_used * _SFOC_G_PER_KWH / 1e6 * hours
        fuel_t = me_fuel_t * 1.1  # +aux
        co2_t = fuel_t * _CO2_PER_FUEL_T
        transport_work = dwt * load_factor * distance  # tonne-nm

        mrv_rows.append(
            {
                "imo_number": imo,
                "ship_name": f"FAKE {st.upper()} {i}",
                "ship_type": st,
                "year": args.year,
                "total_co2_tonnes": round(co2_t, 1),
                "total_fuel_tonnes": round(fuel_t, 1),
                "distance_nm": round(distance, 1),
                "hours_at_sea": round(hours, 1),
                "avg_speed": round(op_speed, 2),
                "transport_work": round(transport_work, 0),
            }
        )

    mrv_dir = args.out_dir / "mrv"
    fleet_dir = args.out_dir / "vessel_static"
    mrv_dir.mkdir(parents=True, exist_ok=True)
    fleet_dir.mkdir(parents=True, exist_ok=True)

    mrv_path = mrv_dir / f"{args.year}_renamed.csv"
    fleet_path = fleet_dir / "fleet_ship_info.csv"
    pd.DataFrame(mrv_rows).to_csv(mrv_path, index=False)
    pd.DataFrame(fleet_rows).to_csv(fleet_path, index=False)

    print(f"[fake-data] {args.n_vessels} vessels (SYNTHETIC — not for analysis)")
    print(f"  MRV   -> {mrv_path}")
    print(f"  fleet -> {fleet_path}")


if __name__ == "__main__":
    main()
