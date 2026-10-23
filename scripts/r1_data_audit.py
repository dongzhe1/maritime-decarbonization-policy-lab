from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall: pip install -e '.[all]'")

from carbon_twin.cli import _load_scenario
from carbon_twin.data.loaders import load_mrv_csv, load_vessels_csv
from carbon_twin.emissions.proxy import vessels_to_index
from carbon_twin.model_variants import load_variant
from carbon_twin.pipeline import (
    _MIN_CII_DISTANCE_NM,
    _mrv_distance,
    _num,
    _scope_weight,
    assemble_compliance_from_mrv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_DICTIONARY = [
    ("imo_number", "IMO ship identification number", "key for matching to vessel particulars"),
    ("ship_type", "MRV ship type", "descriptive only; analysis uses the particulars ship type"),
    ("total_co2_tonnes", "Total CO2 emissions [m tonnes], all EEA-related voyages and berth",
     "CII proxy numerator; scope-weight denominator; MRV calibration anchor"),
    ("co2_ets_tonnes", "CO2 emissions to be reported under Directive 2003/87/EC [m tonnes]",
     "ETS-scoped CO2; numerator of the ETS/FuelEU scope weight"),
    ("total_fuel_tonnes", "Total fuel consumption [m tonnes]",
     "FuelEU audit energy; distance back-calculation"),
    ("distance_nm", "Distance travelled [n miles] (absent from 2022+ exports)", "annual distance"),
    ("fuel_per_nm_kg", "Fuel consumption per distance [kg / n mile]",
     "distance = fuel x 1000 / fuel_per_nm when distance_nm is absent"),
    ("co2_per_nm_kg", "CO2 emissions per distance [kg CO2 / n mile]",
     "fallback distance back-calculation"),
    ("hours_at_sea", "Time spent at sea [hours]", "MRV-implied operating speed = distance / hours"),
    ("transport_work", "Transport work [m.t.*n miles]", "descriptive only"),
]


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(x) else f"{x:,.4g}"
                                       if isinstance(x, float) else str(x)
                                       for x in row) + " |")
    return "\n".join(lines)


def _describe(df: pd.DataFrame, cols: list[str]) -> dict:
    out = {"n": int(len(df))}
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(dtype=float)
        out[f"{c}_median"] = round(float(x.median()), 2) if x.notna().any() else np.nan
        out[f"{c}_mean"] = round(float(x.mean()), 2) if x.notna().any() else np.nan
        out[f"{c}_missing_share"] = round(float(x.isna().mean()), 4) if len(df) else np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample/data/scope audit")
    ap.add_argument("--mrv", required=True, type=Path)
    ap.add_argument("--fleet", required=True, type=Path)
    ap.add_argument("--scenario", required=True, type=Path)
    ap.add_argument("--year", type=int, default=2024, help="MRV reporting year")
    ap.add_argument("--landscape-year", type=int, default=2026)
    ap.add_argument("--eua", type=float, default=80.0)
    ap.add_argument("--variants", default="v23,r1_lng_v23")
    ap.add_argument("--out-dir", type=Path, default=Path("results/r1/data_audit"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mrv = load_mrv_csv(args.mrv)
    fleet = vessels_to_index(load_vessels_csv(args.fleet))
    regs = _load_scenario(str(args.scenario))["regulations"]

    rows = []
    for r in mrv.to_dict(orient="records"):
        imo = str(r.get("imo_number", "")).strip()
        v = fleet.get(imo)
        dist = _mrv_distance(r)
        hours = _num(r.get("hours_at_sea"))
        total = _num(r.get("total_co2_tonnes"))
        w, src = _scope_weight(r, "mrv")
        rows.append({
            "imo": imo,
            "imo_missing": imo in ("", "nan", "None"),
            "mrv_ship_type": r.get("ship_type"),
            "matched": v is not None,
            "ship_type": v.ship_type.value if v else None,
            "gross_tonnage": v.gross_tonnage if v else np.nan,
            "deadweight": v.deadweight if v else np.nan,
            "design_speed_knots": v.design_speed_knots if v else np.nan,
            "main_engine_power_kw": v.main_engine_power_kw if v else np.nan,
            "age_years": (args.year - v.year_built) if (v and v.year_built) else np.nan,
            "total_co2_tonnes": total,
            "total_fuel_tonnes": _num(r.get("total_fuel_tonnes")),
            "distance_nm": dist,
            "distance_source": ("direct" if _num(r.get("distance_nm")) else
                                "fuel_per_nm" if (_num(r.get("fuel_per_nm_kg")) or 0) > 0 else
                                "co2_per_nm" if dist else "missing"),
            "hours_at_sea": hours,
            "scope_weight": w if src == "mrv_ets_share" else np.nan,
            "scope_weight_source": src,
        })
    df = pd.DataFrame(rows)
    df["duplicate_imo"] = df.duplicated("imo", keep=False) & ~df["imo_missing"]
    df["opt_ok"] = df["matched"] & df["distance_nm"].fillna(0).ge(_MIN_CII_DISTANCE_NM)
    df["exclusion_reason"] = np.select(
        [~df["matched"], df["distance_nm"].isna(),
         df["distance_nm"].lt(_MIN_CII_DISTANCE_NM)],
        ["no_particulars_match", "distance_not_derivable",
         f"distance_below_{int(_MIN_CII_DISTANCE_NM)}nm"],
        default="included")

    comp = assemble_compliance_from_mrv(
        mrv, fleet, args.landscape_year, args.eua,
        cii_version=regs["cii"]["version"], fueleu_version=regs["fueleu"]["version"],
        ets_version=regs["eu_ets"]["version"])
    cii_valid = set(comp.loc[comp["cii_rating"].notna(), "vessel_imo"])
    fueleu_ok = set(comp.loc[comp["fueleu_balance"].notna(), "vessel_imo"])
    df["cii_valid"] = df["imo"].isin(cii_valid)
    df["fueleu_audit"] = df["imo"].isin(fueleu_ok)
    df["speed_sample"] = df["opt_ok"] & df["design_speed_knots"].gt(0)

    steps = [
        ("mrv_rows", "rows in the MRV file", len(df)),
        ("mrv_unique_imo", "unique IMO numbers", df.loc[~df["imo_missing"], "imo"].nunique()),
        ("duplicate_imo_rows", "rows sharing an IMO with another row",
         int(df["duplicate_imo"].sum())),
        ("matched_rows", "rows matched to vessel particulars (IMO)", int(df["matched"].sum())),
        ("matched_unique_imo", "unique matched IMO numbers",
         df.loc[df["matched"], "imo"].nunique()),
        ("optimisation_rows", f"matched and annual distance >= {int(_MIN_CII_DISTANCE_NM)} nm",
         int(df["opt_ok"].sum())),
        ("optimisation_unique_imo", "unique IMO numbers in the optimisation sample",
         df.loc[df["opt_ok"], "imo"].nunique()),
        ("cii_valid", "optimisation filter + CO2 > 0 + attained CII <= 100",
         int(df["cii_valid"].sum())),
        ("fueleu_audit", "matched + MRV fuel > 0", int(df["fueleu_audit"].sum())),
        ("speed_sample", "optimisation sample with a recorded design speed",
         int(df["speed_sample"].sum())),
    ]
    pd.DataFrame(steps, columns=["step", "definition", "n"]).to_csv(
        args.out_dir / "sample_waterfall.csv", index=False)
    (df.loc[df["matched"] & ~df["opt_ok"]]
       .groupby(["exclusion_reason", "distance_source"]).size().rename("n").reset_index()
       .to_csv(args.out_dir / "sample_exclusion_reasons.csv", index=False))

    groups = {
        "optimisation_sample": df[df["opt_ok"]],
        "matched_excluded": df[df["matched"] & ~df["opt_ok"]],
        "unmatched": df[~df["matched"]],
    }
    num_cols = ["total_co2_tonnes", "total_fuel_tonnes", "distance_nm", "hours_at_sea",
                "scope_weight", "gross_tonnage", "deadweight", "design_speed_knots",
                "main_engine_power_kw", "age_years"]
    pd.DataFrame([{"group": g, **_describe(d, num_cols)} for g, d in groups.items()]).to_csv(
        args.out_dir / "sample_comparison.csv", index=False)
    by_type = []
    for g, d in groups.items():
        col = "ship_type" if g != "unmatched" else "mrv_ship_type"
        for t, n in d[col].fillna("unknown").value_counts().items():
            by_type.append({"group": g, "type_source": col, "ship_type": t, "n": int(n),
                            "share": round(n / max(len(d), 1), 4)})
    pd.DataFrame(by_type).to_csv(args.out_dir / "sample_comparison_by_type.csv", index=False)

    pd.DataFrame([{
        "internal_name": c, "definition": d, "role": role, "present": c in mrv.columns,
        "non_null_share": round(float(pd.to_numeric(mrv[c], errors="coerce").notna().mean()
                                      if c not in ("imo_number", "ship_type")
                                      else mrv[c].notna().mean()), 4) if c in mrv.columns else 0.0,
    } for c, d, role in _DICTIONARY]).to_csv(args.out_dir / "mrv_data_dictionary.csv", index=False)

    opt = df[df["opt_ok"]]
    sw = []
    for t, g in [("ALL", opt)] + list(opt.groupby("ship_type")):
        x = g["scope_weight"].dropna()
        sw.append({"ship_type": t, "n": int(len(g)), "n_with_weight": int(len(x)),
                   "fallback_share": round(float(g["scope_weight"].isna().mean()), 4),
                   **{f"p{q}": round(float(x.quantile(q / 100)), 4) if len(x) else np.nan
                      for q in (10, 25, 50, 75, 90)},
                   "mean": round(float(x.mean()), 4) if len(x) else np.nan,
                   "co2_weighted_mean": round(float(
                       (g["scope_weight"] * g["total_co2_tonnes"]).sum()
                       / g.loc[g["scope_weight"].notna(), "total_co2_tonnes"].sum()), 4)
                   if len(x) else np.nan})
    pd.DataFrame(sw).to_csv(args.out_dir / "scope_weight_by_type.csv", index=False)

    land = []
    for vname in [v.strip() for v in args.variants.split(",") if v.strip()]:
        v = load_variant(vname)
        c = assemble_compliance_from_mrv(
            mrv, fleet, args.landscape_year, args.eua,
            cii_version=regs["cii"]["version"],
            fueleu_version=v.fueleu_version_or(regs["fueleu"]["version"]),
            ets_version=regs["eu_ets"]["version"], **v.compliance_kwargs())
        fe = c.dropna(subset=["fueleu_balance"])
        land.append({
            "variant": vname, "year": args.landscape_year, "eua": args.eua,
            "n_vessels": int(len(c)),
            "ets_cost_total_eur": round(float(c["ets_cost_eur"].sum()), 0),
            "ets_covered_tco2": round(float(c["ets_covered_tco2e"].sum()), 0),
            "fueleu_n_audited": int(len(fe)),
            "fueleu_deficit_share": round(float((fe["fueleu_balance"] < 0).mean()), 4),
            "fueleu_deficit_cost_total_eur": round(
                float(fe["fueleu_penalty_eur"].fillna(0).sum()), 0),
            "cii_proxy_de_share": round(float(c["cii_rating"].isin(["D", "E"]).sum()
                                              / max(c["cii_rating"].notna().sum(), 1)), 4),
        })
    pd.DataFrame(land).to_csv(args.out_dir / "compliance_landscape.csv", index=False)

    md = ["# Data audit", "",
          f"MRV file: `{args.mrv}`; fleet file: `{args.fleet}`", "",
          "## Sample waterfall", "", "| step | n | definition |", "|---|---:|---|"]
    md += [f"| {s} | {n:,} | {d} |" for s, d, n in steps]
    md += ["", "## Scope weight (optimisation sample)", "",
           md_table(pd.DataFrame(sw)), "",
           "## Compliance landscape", "", md_table(pd.DataFrame(land))]
    (args.out_dir / "data_audit.md").write_text("\n".join(md) + "\n")
    log.info("data audit written to %s", args.out_dir)


if __name__ == "__main__":
    main()
