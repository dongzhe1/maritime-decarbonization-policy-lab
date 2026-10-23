from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall: pip install -e '.[all]'")

_COLS = ["lrnoimo_ship_no", "fuel_type1first", "fuel_type2second", "main_engine_model"]
_LNG_FUEL1 = {"lng", "gas boil off"}
_LNG_FUEL2 = {"lng", "liquefied biogas"}
_LNG_ENGINE = r"DF\b|DF$|-GI\b|-GI$|-GA\b|-GA$"
_METHANOL_ENGINE = r"LGIM"
_GAS_PROXY = {"gas_carrier", "lng_carrier"}


def capability_table(ship_info: pd.DataFrame) -> pd.DataFrame:
    f1 = ship_info["fuel_type1first"].fillna("").str.strip().str.lower()
    f2 = ship_info["fuel_type2second"].fillna("").str.strip().str.lower()
    eng = ship_info["main_engine_model"].fillna("").str.strip()
    lng_fuel = f1.isin(_LNG_FUEL1) | f2.isin(_LNG_FUEL2)
    lng_engine = eng.str.contains(_LNG_ENGINE, regex=True)
    meoh_fuel = f1.eq("methanol")
    meoh_engine = eng.str.contains(_METHANOL_ENGINE, regex=True)
    evidence = []
    for lf, le, mf, me in zip(lng_fuel, lng_engine, meoh_fuel, meoh_engine):
        tags = [t for t, flag in (("lng_fuel_field", lf), ("lng_dual_fuel_engine", le),
                                  ("methanol_fuel_field", mf), ("methanol_engine", me)) if flag]
        evidence.append(";".join(tags))
    return pd.DataFrame({
        "imo_number": ship_info["lrnoimo_ship_no"].astype(str).str.strip(),
        "lng_capable": (lng_fuel | lng_engine).astype(int),
        "methanol_capable": (meoh_fuel | meoh_engine).astype(int),
        "evidence": evidence,
        "fuel_type1": ship_info["fuel_type1first"].fillna(""),
        "fuel_type2": ship_info["fuel_type2second"].fillna(""),
        "main_engine_model": eng,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ship-info", required=True, type=Path)
    ap.add_argument("--fleet", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--summary", type=Path, default=None,
                    help="ship-type summary CSV (default: <out stem>_summary.csv)")
    ap.add_argument("--no-lngc-rule", action="store_true",
                    help="do not mark LNG carriers LNG-capable without field evidence")
    args = ap.parse_args()

    info = pd.read_csv(args.ship_info, usecols=_COLS, dtype=str, low_memory=False)
    fleet = pd.read_csv(args.fleet, dtype={"imo_number": str})
    fleet["imo_number"] = fleet["imo_number"].astype(str).str.strip()
    cap = capability_table(info).drop_duplicates("imo_number")
    cap = fleet[["imo_number", "ship_type"]].merge(cap, on="imo_number", how="left")
    cap[["lng_capable", "methanol_capable"]] = (
        cap[["lng_capable", "methanol_capable"]].fillna(0).astype(int))
    cap["evidence"] = cap["evidence"].fillna("no_sea_web_record")
    lngc = cap["ship_type"].eq("lng_carrier") & cap["lng_capable"].eq(0) & (not args.no_lngc_rule)
    cap.loc[lngc, "lng_capable"] = 1
    cap.loc[lngc, "evidence"] = cap.loc[lngc, "evidence"].str.cat(
        ["lng_carrier_boil_off"] * int(lngc.sum()), sep=";").str.strip(";")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cap.drop(columns="ship_type").to_csv(args.out, index=False)

    cap["lng_proxy"] = cap["ship_type"].isin(_GAS_PROXY).astype(int)
    summary = cap.groupby("ship_type").agg(
        n=("imo_number", "size"), lng_capable=("lng_capable", "sum"),
        methanol_capable=("methanol_capable", "sum"), lng_proxy=("lng_proxy", "sum"))
    summary.loc["ALL"] = summary.sum()
    summary["lng_capable_share"] = (summary["lng_capable"] / summary["n"]).round(4)
    summary["lng_proxy_share"] = (summary["lng_proxy"] / summary["n"]).round(4)
    out_summary = args.summary or args.out.with_name(args.out.stem + "_summary.csv")
    summary.reset_index().to_csv(out_summary, index=False)
    print(summary.to_string())
    print(f"wrote {args.out} ({len(cap)} vessels) and {out_summary}")


if __name__ == "__main__":
    main()
