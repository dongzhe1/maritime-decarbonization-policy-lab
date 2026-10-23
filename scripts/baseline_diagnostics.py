from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall: pip install -e '.[all]'")

from carbon_twin.data.loaders import load_mrv_csv, load_vessels_csv
from carbon_twin.data.schemas import EEAScope, FuelType, Voyage
from carbon_twin.emissions.factors import get_co2e_ttw
from carbon_twin.emissions.proxy import load_fuel_factors_safe, vessels_to_index
from carbon_twin.optimization.speed import _fuel_at_speed

from carbon_twin.pipeline import _mrv_distance

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _num(x):
    v = pd.to_numeric(x, errors="coerce")
    return float(v) if pd.notna(v) else None


def _pct(x, lo, hi):
    return round(float(((x >= lo) & (x <= hi)).mean()), 4)


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline model diagnostics")
    ap.add_argument("--mrv", required=True, type=Path)
    ap.add_argument("--fleet", required=True, type=Path)
    ap.add_argument("--compliance", required=True, type=Path)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--out-dir", type=Path, default=Path("results/diagnostics"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mrv = load_mrv_csv(args.mrv)
    vessels = load_vessels_csv(args.fleet)
    fleet = vessels_to_index(vessels)
    factors = load_fuel_factors_safe()
    comp = pd.read_parquet(args.compliance)
    cii_valid = set(comp.loc[comp["cii_rating"].notna(), "vessel_imo"].astype(str))
    used = set(comp["vessel_imo"].astype(str))

    cal_rows, spd_rows = [], []
    n = len(mrv)
    for i, r in enumerate(mrv.to_dict(orient="records")):
        if i % 2000 == 0:
            log.info("physics pass %d/%d", i, n)
        imo = str(r.get("imo_number", "")).strip()
        v = fleet.get(imo)
        if v is None:
            continue
        dist = _mrv_distance(r)
        hours = _num(r.get("hours_at_sea"))
        mrv_co2 = _num(r.get("total_co2_tonnes"))
        if not dist or dist < 500 or not mrv_co2 or mrv_co2 <= 0:
            continue
        op_speed = (dist / hours) if hours and hours > 0 else (v.design_speed_knots or 14.0)
        op_speed = float(min(max(op_speed, 6.0), 25.0))
        if v.design_speed_knots:
            spd_rows.append(
                {
                    "imo": imo,
                    "ship_type": v.ship_type.value,
                    "op_speed_kn": op_speed,
                    "design_speed_kn": float(v.design_speed_knots),
                    "op_to_design": op_speed / float(v.design_speed_knots),
                }
            )
        leg_nm = float(min(dist, 2000.0))
        leg_hours = leg_nm / max(op_speed, 0.1)
        try:
            voy = Voyage(
                voyage_id=f"{imo}_d",
                vessel_imo=imo,
                departure_port_id="A",
                arrival_port_id="B",
                departure_time=dt.datetime(args.year, 1, 1),
                arrival_time=dt.datetime(args.year, 1, 1) + dt.timedelta(hours=leg_hours),
                distance_nm=leg_nm,
                avg_speed_knots=op_speed,
                sea_hours=leg_hours,
                eea_scope=EEAScope.INTRA_EEA,
                ets_scope_factor=1.0,
            )
            fc = _fuel_at_speed(v, voy, op_speed, FuelType.VLSFO, factors)
            annual_scale = dist / max(leg_nm, 1.0)
            modelled_co2 = get_co2e_ttw(FuelType.VLSFO, fc["total_tonnes"] * annual_scale, factors)
        except Exception:
            continue
        if modelled_co2 > 0:
            cal_rows.append(
                {
                    "imo": imo,
                    "ship_type": v.ship_type.value,
                    "mrv_co2_t": mrv_co2,
                    "modelled_co2_t": round(modelled_co2, 1),
                    "model_to_mrv_ratio": modelled_co2 / mrv_co2,
                }
            )
    log.info("physics pass done: %d calibration rows", len(cal_rows))
    if not cal_rows:
        log.error("0 calibration rows — check MRV column mapping (distance/CO2/fleet match)")

    cal = pd.DataFrame(cal_rows, columns=["imo", "ship_type", "mrv_co2_t",
                                          "modelled_co2_t", "model_to_mrv_ratio"])
    rows = []
    for st, g in list(cal.groupby("ship_type")) + [("ALL", cal)]:
        ratio = g["model_to_mrv_ratio"]
        rows.append(
            {
                "ship_type": st,
                "n": len(g),
                "ratio_mean": round(float(ratio.mean()), 3),
                "ratio_median": round(float(ratio.median()), 3),
                "ratio_p10": round(float(ratio.quantile(0.10)), 3),
                "ratio_p90": round(float(ratio.quantile(0.90)), 3),
                "within_20pct_share": _pct(ratio, 0.8, 1.2),
                "within_50pct_share": _pct(ratio, 0.5, 1.5),
            }
        )
    pd.DataFrame(rows).to_csv(args.out_dir / "baseline_co2_calibration.csv", index=False)

    spd = pd.DataFrame(spd_rows, columns=["imo", "ship_type", "op_speed_kn",
                                          "design_speed_kn", "op_to_design"])
    rows = []
    for st, g in list(spd.groupby("ship_type")) + [("ALL", spd)]:
        rows.append(
            {
                "ship_type": st,
                "n": len(g),
                "speed_median_kn": round(float(g["op_speed_kn"].median()), 2),
                "speed_p10_kn": round(float(g["op_speed_kn"].quantile(0.10)), 2),
                "speed_p90_kn": round(float(g["op_speed_kn"].quantile(0.90)), 2),
                "design_speed_mean_kn": round(float(g["design_speed_kn"].mean()), 2),
                "op_to_design_median": round(float(g["op_to_design"].median()), 3),
            }
        )
    pd.DataFrame(rows).to_csv(args.out_dir / "mrv_implied_speed_distribution.csv", index=False)

    mrv2 = mrv.copy()
    mrv2["imo"] = mrv2["imo_number"].astype(str).str.strip()
    mrv2["co2"] = pd.to_numeric(mrv2["total_co2_tonnes"], errors="coerce")
    mrv2["dist"] = mrv2.apply(lambda row: _mrv_distance(row.to_dict()), axis=1)

    def _group(imos):
        return mrv2[mrv2["imo"].isin(imos)]

    fleet_imos = set(fleet)
    groups = {
        "matched_cii_valid": _group(used & cii_valid),
        "matched_not_cii_valid": _group(used - cii_valid),
        "unmatched": mrv2[~mrv2["imo"].isin(fleet_imos)],
    }
    rows = []
    for name, g in groups.items():
        rows.append(
            {
                "group": name,
                "n": len(g),
                "total_mrv_co2_t": round(float(g["co2"].sum()), 1),
                "mean_co2_t": round(float(g["co2"].mean()), 1) if len(g) else 0.0,
                "median_co2_t": round(float(g["co2"].median()), 1) if len(g) else 0.0,
                "mean_distance_nm": round(float(g["dist"].mean()), 1) if len(g) else 0.0,
            }
        )
    pd.DataFrame(rows).to_csv(args.out_dir / "matched_unmatched_diagnostics.csv", index=False)

    ets = (
        comp.groupby("ship_type")
        .agg(
            n=("vessel_imo", "size"),
            mrv_reported_co2_t=("annual_co2_tonnes", "sum"),
            ets_scoped_co2e_t=("ets_covered_tco2e", "sum"),
        )
        .reset_index()
    )
    ets["share_of_fleet_ets_emissions"] = (
        ets["ets_scoped_co2e_t"] / ets["ets_scoped_co2e_t"].sum()
    ).round(4)
    ets.to_csv(args.out_dir / "ets_scope_emissions_by_ship_type.csv", index=False)

    fe = comp.dropna(subset=["fueleu_balance"]).copy()
    rows = []
    for st, g in list(fe.groupby("ship_type")) + [("ALL", fe)]:
        bal = g["fueleu_balance"]
        pen = g["fueleu_penalty_eur"].fillna(0)
        rows.append(
            {
                "ship_type": st,
                "n": len(g),
                "deficit_vessel_share": round(float((bal < 0).mean()), 4),
                "surplus_vessel_share": round(float((bal > 0).mean()), 4),
                "penalty_total_eur": round(float(pen.sum()), 2),
                "surplus_penalised_check_eur": round(float(pen[bal > 0].sum()), 2),
            }
        )
    pd.DataFrame(rows).to_csv(args.out_dir / "fueleu_scope_and_balance_audit.csv", index=False)

    n_mrv = int(mrv["imo_number"].astype(str).str.strip().nunique())
    n_matched = int(sum(1 for imo in mrv2["imo"].unique() if imo in fleet_imos))
    n_cii_valid = len(cii_valid)
    n_fueleu = int(comp["fueleu_balance"].notna().sum())
    n_speed = len(spd)
    opt_path = args.compliance.parent.parent / "optimization" / "optimized.parquet"
    n_rci = int(len(pd.read_parquet(opt_path))) if opt_path.exists() else -1
    denom = pd.DataFrame([
        {"sample_name": "mrv_reporting", "n": n_mrv,
         "used_for": "EU-MRV reporting population", "exclusion_reason": "-"},
        {"sample_name": "matched_to_particulars", "n": n_matched,
         "used_for": "fleet cost totals", "exclusion_reason": "no vessel particulars match"},
        {"sample_name": "cii_proxy_valid", "n": n_cii_valid,
         "used_for": "CII proxy rating mix", "exclusion_reason": "missing/implausible CII inputs"},
        {"sample_name": "rci_optimisation", "n": n_rci,
         "used_for": "speed/fuel RCI", "exclusion_reason": "fails optimisation-input filters"},
        {"sample_name": "fueleu_audit", "n": n_fueleu,
         "used_for": "FuelEU scope/balance audit", "exclusion_reason": "no FuelEU balance"},
        {"sample_name": "mrv_implied_speed", "n": n_speed,
         "used_for": "operating-speed validation", "exclusion_reason": "no design speed/hours"},
    ])
    denom.to_csv(args.out_dir / "sample_denominators.csv", index=False)

    log.info("done → %s (6 CSVs)", args.out_dir)


if __name__ == "__main__":
    main()
