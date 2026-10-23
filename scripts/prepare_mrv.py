from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MRV_RENAME: dict[str, str] = {
    "IMO Number": "imo_number",
    "IMO number": "imo_number",
    "Name": "ship_name",
    "Ship type": "ship_type",
    "Technical efficiency": "technical_efficiency",
    "Port of Registry": "port_of_registry",
    "Reporting Period": "year",
    "Total fuel consumption [m tonnes]": "total_fuel_tonnes",
    "Annual Total fuel consumption [m tonnes]": "total_fuel_tonnes",
    "Total CO₂ emissions [m tonnes]": "total_co2_tonnes",
    "Total CO2 emissions [m tonnes]": "total_co2_tonnes",
    "CO₂ emissions to be reported under Directive 2003/87/EC [m tonnes]": "co2_ets_tonnes",
    "CO2 emissions to be reported under Directive 2003/87/EC [m tonnes]": "co2_ets_tonnes",
    "Total CO₂eq emissions [m tonnes]": "total_co2eq_tonnes",
    "Total CO2eq emissions [m tonnes]": "total_co2eq_tonnes",
    "Total CH₄ emissions [m tonnes]": "total_ch4_tonnes",
    "Total CH4 emissions [m tonnes]": "total_ch4_tonnes",
    "Total N₂O emissions [m tonnes]": "total_n2o_tonnes",
    "Total N2O emissions [m tonnes]": "total_n2o_tonnes",
    "Distance travelled [n miles]": "distance_nm",
    "Annual distance travelled [n miles]": "distance_nm",
    "Total distance travelled [n miles]": "distance_nm",
    "Fuel consumption per distance [kg / n mile]": "fuel_per_nm_kg",
    "CO₂ emissions per distance [kg CO₂ / n mile]": "co2_per_nm_kg",
    "CO2 emissions per distance [kg CO2 / n mile]": "co2_per_nm_kg",
    "Time spent at sea [hours]": "hours_at_sea",
    "Annual Time spent at sea [hours]": "hours_at_sea",
    "Total time spent at sea [hours]": "hours_at_sea",
    "Transport work [m.t.*n miles]": "transport_work",
    "Annual transport work [m.t.*n miles]": "transport_work",
    "Total transport work [m.t.*n miles]": "transport_work",
    "Gross tonnage": "gross_tonnage",
    "Deadweight": "deadweight",
}


def read_export(path: Path) -> pd.DataFrame:
    reader = pd.read_excel if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv
    head = reader(path, header=None, nrows=10, dtype=str)
    rows = [i for i in range(len(head))
            if head.iloc[i].astype(str).str.strip().isin(["IMO Number", "IMO number"]).any()]
    if not rows:
        raise SystemExit(f"{path}: no row with the field 'IMO Number' among the first ten")
    df = reader(path, header=rows[0], dtype=str)
    df.columns = [str(c).lstrip("﻿").strip() for c in df.columns]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export", type=Path)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imo-list", type=Path, default=None)
    args = ap.parse_args()
    df = read_export(args.export)
    df = df.rename(columns={c: MRV_RENAME[c] for c in df.columns if c in MRV_RENAME})
    if "year" not in df.columns:
        df["year"] = str(args.year)
    df = df[df["imo_number"].notna()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df):,} records -> {args.out}")
    if args.imo_list:
        args.imo_list.parent.mkdir(parents=True, exist_ok=True)
        args.imo_list.write_text("".join(f"{i.strip()}\n" for i in df["imo_number"]))
        print(f"IMO list -> {args.imo_list}")


if __name__ == "__main__":
    main()
