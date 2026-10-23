from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

MRV_COLUMNS = [
    'imo_number',
    'ship_name',
    'ship_type',
    'year',
    'technical_efficiency',
    'port_of_registry',
    'Home Port',
    'Ice Class',
    'IMO Number.1',
    'Name.1',
    'DoC issue date',
    'DoC expiry date',
    'Verifier Name',
    'Verifier Address',
    'Verifier City',
    'Verifier Country',
    'Verifier Accreditation number',
    'Verifier NAB',
    'A',
    'B',
    'C',
    'D',
    'D.1',
    'total_fuel_tonnes',
    'Total fuel benefitting from a derogation in accordance with Part C, point 1.2, of Annex II to Regulation (EU) 2015/757 (voluntary) [m tonnes]',
    'Fuel consumptions assigned to On laden [m tonnes]',
    'Fuel consumptions assigned to Cargo heating [m tonnes]',
    'Fuel consumptions assigned to Dynamic positioning [m tonnes]',
    'total_co2_tonnes',
    'CO₂ emissions from all voyages between ports under a MS jurisdiction [m tonnes]',
    'CO₂ emissions from all voyages which departed from ports under a MS jurisdiction [m tonnes]',
    'CO₂ emissions from all voyages to ports under a MS jurisdiction [m tonnes]',
    'CO₂ emissions which occurred within ports under a MS jurisdiction at berth [m tonnes]',
    'CO₂ emissions which occurred within ports under a MS jurisdiction [m tonnes]',
    'CO₂ emissions assigned to On laden [m tonnes]',
    'CO₂ emissions assigned to Passenger transport [m tonnes]',
    'CO₂ emissions assigned to Freight transport [m tonnes]',
    'co2_ets_tonnes',
    'total_ch4_tonnes',
    'CH₄ emissions from all voyages between ports under a MS jurisdiction [m tonnes]',
    'CH₄ emissions from all voyages which departed from ports under a MS jurisdiction [m tonnes]',
    'CH₄ emissions from all voyages to ports under a MS jurisdiction [m tonnes]',
    'CH₄ emissions which occurred within ports under a MS jurisdiction at berth [m tonnes]',
    'CH₄ emissions which occurred within ports under a MS jurisdiction [m tonnes]',
    'CH₄ emissions assigned to On laden [m tonnes]',
    'CH₄ emissions assigned to Passenger transport [m tonnes]',
    'CH₄ emissions assigned to Freight transport [m tonnes]',
    'CH₄ emissions to be reported under Directive 2003/87/EC [m tonnes]',
    'total_n2o_tonnes',
    'N₂O emissions from all voyages between ports under a MS jurisdiction [m tonnes]',
    'N₂O emissions from all voyages which departed from ports under a MS jurisdiction [m tonnes]',
    'N₂O emissions from all voyages to ports under a MS jurisdiction [m tonnes]',
    'N₂O emissions which occurred within ports under a MS jurisdiction at berth [m tonnes]',
    'N₂O emissions which occurred within ports under a MS jurisdiction [m tonnes]',
    'N₂O emissions assigned to On laden [m tonnes]',
    'N₂O emissions assigned to Passenger transport [m tonnes]',
    'N₂O emissions assigned to Freight transport [m tonnes]',
    'N₂O emissions to be reported under Directive 2003/87/EC [m tonnes]',
    'total_co2eq_tonnes',
    'CO₂eq emissions from all voyages between ports under a MS jurisdiction [m tonnes]',
    'CO₂eq emissions from all voyages which departed from ports under a MS jurisdiction [m tonnes]',
    'CO₂eq emissions from all voyages to ports under a MS jurisdiction [m tonnes]',
    'CO₂eq emissions which occurred within ports under a MS jurisdiction at berth [m tonnes]',
    'CO₂eq emissions which occurred within ports under a MS jurisdiction [m tonnes]',
    'CO₂eq emissions assigned to On laden [m tonnes]',
    'CO₂eq emissions assigned to Passenger transport [m tonnes]',
    'CO₂eq emissions assigned to Freight transport [m tonnes]',
    'CO₂eq emissions to be reported under Directive 2003/87/EC [m tonnes]',
    'CO₂eq emissions benefitting from a derogation in accordance with Part C, point 1.2, of Annex II to Regulation (EU) 2015/757 (voluntary) [m tonnes]',
    'Distance through ice [n miles]',
    'hours_at_sea',
    'Time spent at sea through ice [hours]',
    'fuel_per_nm_kg',
    'Fuel consumption per distance on laden voyages [kg / n mile]',
    'Fuel consumption per transport work (mass) [g / m tonnes · n miles]',
    'Fuel consumption per transport work (mass) on laden voyages [g / m tonnes · n miles]',
    'Fuel consumption per transport work (volume) [g / m³ · n miles]',
    'Fuel consumption per transport work (volume) on laden voyages [g / m³ · n miles]',
    'Fuel consumption per transport work (dwt) [g / dwt carried · n miles]',
    'Fuel consumption per transport work (dwt) on laden voyages [g / dwt carried · n miles]',
    'Fuel consumption per transport work (pax) [g / pax · n miles]',
    'Fuel consumption per transport work (pax) on laden voyages [g / pax · n miles]',
    'Fuel consumption per transport work (freight) [g / m tonnes · n miles]',
    'Fuel consumption per transport work (freight) on laden voyages [g / m tonnes · n miles]',
    'Fuel consumption per time spent at sea [m tonnes / hour]',
    'co2_per_nm_kg',
    'CO₂ emissions per distance on laden voyages [kg CO₂ / n mile]',
    'CO₂ emissions per transport work (mass) [g CO₂ / m tonnes · n miles]',
    'CO₂ emissions per transport work (mass) on laden voyages [g CO₂ / m tonnes · n miles]',
    'CO₂ emissions per transport work (volume) [g CO₂ / m³ · n miles]',
    'CO₂ emissions per transport work (volume) on laden voyages [g CO₂ / m³ · n miles]',
    'CO₂ emissions per transport work (dwt) [g CO₂ / dwt carried · n miles]',
    'CO₂ emissions per transport work (dwt) on laden voyages [g CO₂ / dwt carried · n miles]',
    'CO₂ emissions per transport work (pax) [g CO₂ / pax · n miles]',
    'CO₂ emissions per transport work (pax) on laden voyages [g CO₂ / pax · n miles]',
    'CO₂ emissions per transport work (freight) [g CO₂ / m tonnes · n miles]',
    'CO₂ emissions per transport work (freight) on laden voyages [g CO₂ / m tonnes · n miles]',
    'CO₂ emissions per time spent at sea [m tonnes CO₂ / hour]',
    'CO₂eq emissions per distance [kg CO₂eq / n mile]',
    'CO₂eq emissions per distance on laden voyages [kg CO₂eq / n mile]',
    'CO₂eq emissions per transport work (mass) [g CO₂eq / m tonnes · n miles]',
    'CO₂eq emissions per transport work (mass) on laden voyages [g CO₂eq / m tonnes · n miles]',
    'CO₂eq emissions per transport work (volume) [g CO₂eq / m³ · n miles]',
    'CO₂eq emissions per transport work (volume) on laden voyages [g CO₂eq / m³ · n miles]',
    'CO₂eq emissions per transport work (dwt) [g CO₂eq / dwt carried · n miles]',
    'CO₂eq emissions per transport work (dwt) on laden voyages [g CO₂eq / dwt carried · n miles]',
    'CO₂eq emissions per transport work (pax) [g CO₂eq / pax · n miles]',
    'CO₂eq emissions per transport work (pax) on laden voyages [g CO₂eq / pax · n miles]',
    'CO₂eq emissions per transport work (freight) [g CO₂eq / m tonnes · n miles]',
    'CO₂eq emissions per transport work (freight) on laden voyages [g CO₂eq / m tonnes · n miles]',
    'CO₂eq emissions per time spent at sea [m tonnes CO₂eq / m tonnes · n miles]',
    'Additional information to facilitate the understanding of the reported average operational energy efficiency indicators',
    'Average density of the cargo transported [m tonnes / m³]',
]

_TYPES = [
    ("Bulk carrier", "bulk_carrier", 0.30, (10_000, 180_000)),
    ("Container ship", "container_ship", 0.15, (8_000, 150_000)),
    ("Oil tanker", "tanker", 0.14, (5_000, 160_000)),
    ("Chemical tanker", "tanker", 0.10, (4_000, 50_000)),
    ("General cargo ship", "general_cargo", 0.10, (2_000, 20_000)),
    ("Vehicle carrier", "vehicle_carrier", 0.04, (10_000, 25_000)),
    ("Ro-pax ship", "ro_pax", 0.04, (3_000, 12_000)),
    ("Gas carrier", "gas_carrier", 0.04, (3_000, 60_000)),
    ("LNG carrier", "lng_carrier", 0.05, (60_000, 100_000)),
    ("Ro-ro ship", "ro_ro_cargo", 0.02, (5_000, 20_000)),
    ("Refrigerated cargo carrier", "refrigerated_cargo", 0.02, (3_000, 15_000)),
]
_CO2_PER_T_FUEL = 3.15
_REGISTER_TYPE = {"bulk_carrier": "Bulk Carrier", "container_ship": "Container Ship (Fully Cellular)",
                  "tanker": "Crude Oil Tanker", "general_cargo": "General Cargo Ship",
                  "vehicle_carrier": "Vehicles Carrier", "ro_pax": "Passenger/Ro-Ro Ship (Vehicles)",
                  "gas_carrier": "LPG Tanker", "lng_carrier": "LNG Tanker",
                  "ro_ro_cargo": "Ro-Ro Cargo Ship", "refrigerated_cargo": "Refrigerated Cargo Ship"}
_EXPORT_NAME = {"imo_number": "IMO Number", "ship_name": "Name", "ship_type": "Ship type",
                "year": "Reporting Period", "technical_efficiency": "Technical efficiency",
                "port_of_registry": "Port of Registry",
                "total_fuel_tonnes": "Total fuel consumption [m tonnes]",
                "total_co2_tonnes": "Total CO₂ emissions [m tonnes]",
                "co2_ets_tonnes": "CO₂ emissions to be reported under Directive 2003/87/EC [m tonnes]",
                "total_ch4_tonnes": "Total CH₄ emissions [m tonnes]",
                "total_n2o_tonnes": "Total N₂O emissions [m tonnes]",
                "total_co2eq_tonnes": "Total CO₂eq emissions [m tonnes]",
                "hours_at_sea": "Time spent at sea [hours]",
                "fuel_per_nm_kg": "Fuel consumption per distance [kg / n mile]",
                "co2_per_nm_kg": "CO₂ emissions per distance [kg CO₂ / n mile]",
                "IMO Number.1": "IMO Number", "Name.1": "Name", "D.1": "D"}


def imo_numbers(n: int, rng: np.random.Generator) -> list[str]:
    out, seen = [], set()
    while len(out) < n:
        body = int(rng.integers(900_000, 999_999))
        digits = [int(c) for c in str(body)]
        imo = f"{body}{sum(d * (7 - i) for i, d in enumerate(digits)) % 10}"
        if imo not in seen:
            seen.add(imo)
            out.append(imo)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--n-vessels", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    n = args.n_vessels

    pick = rng.choice(len(_TYPES), size=n, p=np.array([t[2] for t in _TYPES]) / sum(t[2] for t in _TYPES))
    mrv_type = [_TYPES[k][0] for k in pick]
    fleet_type = [_TYPES[k][1] for k in pick]
    dwt = np.array([rng.uniform(*_TYPES[k][3]) for k in pick]).round(0)
    gt = (dwt * rng.uniform(0.55, 0.75, n)).round(0)
    design_speed = rng.uniform(12.0, 20.0, n).round(1)
    power = (10.0 * dwt ** 0.65 * (design_speed / 14.5) ** 2).round(0)
    year_built = rng.integers(1996, 2022, n)
    imo = imo_numbers(n, rng)

    distance = rng.uniform(8_000, 70_000, n)
    speed = design_speed * rng.uniform(0.6, 0.85, n)
    hours = distance / speed
    fuel_per_nm = (power * (speed / design_speed) ** 3 * 0.18 / speed
                   * rng.uniform(0.85, 1.15, n)).clip(10, 600)
    fuel = distance * fuel_per_nm / 1000.0
    co2 = fuel * _CO2_PER_T_FUEL
    short = rng.random(n) < 0.03
    distance[short] = rng.uniform(50, 400, short.sum())
    fuel[short] = distance[short] * fuel_per_nm[short] / 1000.0
    co2[short] = fuel[short] * _CO2_PER_T_FUEL
    no_fpd = rng.random(n) < 0.03

    port = co2 * rng.uniform(0.03, 0.15, n)
    berth = port * rng.uniform(0.4, 1.0, n)
    swap = rng.random(n) < 0.15
    port, berth = np.where(swap, berth, port), np.where(swap, port, berth)
    voyages = co2 - np.maximum(port, berth)
    q = rng.uniform(0.0, 1.0, n)
    between = voyages * q
    departed = voyages * (1 - q) * rng.uniform(0.3, 0.7, n)
    arrived = voyages - between - departed
    ets = between + np.maximum(port, berth) + 0.5 * (departed + arrived)
    no_breakdown = rng.random(n) < 0.02

    df = pd.DataFrame("", index=range(n), columns=MRV_COLUMNS)
    df["imo_number"] = imo
    df["ship_name"] = [f"TEST VESSEL {k + 1}" for k in range(n)]
    df["ship_type"] = mrv_type
    df["year"] = 2024
    df["technical_efficiency"] = [f"EEDI ({v:.2f} gCO₂/t·nm)" for v in rng.uniform(2.5, 12.0, n)]
    df["port_of_registry"] = "Testport"
    df["total_fuel_tonnes"] = fuel.round(2)
    df["total_co2_tonnes"] = co2.round(2)
    cats = {"between ports": between, "departed from ports": departed, "to ports": arrived,
            "within ports under a MS jurisdiction at berth": berth}
    for col in MRV_COLUMNS:
        if col.startswith("CO₂ emissions") and col.endswith("[m tonnes]") and "assigned" not in col:
            for key, val in cats.items():
                if key in col:
                    df[col] = np.where(no_breakdown, "", val.round(2))
            if col == "CO₂ emissions which occurred within ports under a MS jurisdiction [m tonnes]":
                df[col] = np.where(no_breakdown, "", port.round(2))
    df["co2_ets_tonnes"] = np.where(no_breakdown, 0.0, ets.round(2))
    df["total_ch4_tonnes"] = (co2 * 2e-5).round(4)
    df["total_n2o_tonnes"] = (co2 * 5e-5).round(4)
    df["total_co2eq_tonnes"] = (co2 * 1.01).round(2)
    df["hours_at_sea"] = hours.round(1)
    df["fuel_per_nm_kg"] = np.where(no_fpd, "", fuel_per_nm.round(2))
    df["co2_per_nm_kg"] = np.where(no_fpd, "", (fuel_per_nm * _CO2_PER_T_FUEL).round(2))

    in_register = rng.random(n) >= 0.04
    lng = np.array([t == "lng_carrier" for t in fleet_type]) | (rng.random(n) < 0.05)
    meoh = rng.random(n) < 0.01
    info = pd.DataFrame({
        "lrnoimo_ship_no": imo, "ship_name": df["ship_name"],
        "shiptype_level5": [_REGISTER_TYPE[t] for t in fleet_type],
        "deadweight": dwt, "gross_tonnage": gt,
        "speedservice": np.where(rng.random(n) < 0.03, 0.0, design_speed),
        "speedmax": (design_speed * 1.08).round(1),
        "total_kilowattsof_main_engines": np.where(rng.random(n) < 0.02, np.nan, power),
        "year_of_build": year_built, "flag_name": "Testflag",
        "fuel_type1first": np.where(lng, "LNG", np.where(meoh, "Methanol", "Residual Fuel")),
        "fuel_type2second": np.where(rng.random(n) < 0.5, "Distillate Fuel", ""),
        "main_engine_model": np.where(lng & (rng.random(n) < 0.5), "6X72DF", "6S60MC-C"),
    })[in_register]

    mrv_dir = args.out_dir / "data" / "raw" / "mrv"
    mrv_dir.mkdir(parents=True, exist_ok=True)
    header = [_EXPORT_NAME.get(c, c) for c in df.columns]
    with open(mrv_dir / "thetis_mrv_2024.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Ship"] + [""] * (len(header) - 1))
        w.writerow(header)
        w.writerows(df.astype(str).replace("nan", "").values.tolist())
    info.to_csv(args.out_dir / "data" / "raw" / "ship_info.csv", index=False)
    print(f"{n} MRV records, {len(info)} register records -> {args.out_dir / 'data' / 'raw'}")


if __name__ == "__main__":
    main()
