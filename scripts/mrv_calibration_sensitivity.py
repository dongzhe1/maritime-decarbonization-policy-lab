from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall: pip install -e '.[all]'")

from carbon_twin.cli import _load_scenario
from carbon_twin.data.loaders import load_fuel_prices, load_mrv_csv, load_vessels_csv
from carbon_twin.emissions.proxy import load_fuel_factors_safe, vessels_to_index
from carbon_twin.model_variants import (
    add_variant_argument,
    load_variant,
    write_variant_manifest,
)
from carbon_twin.parallel import row_block, run_blocked
from carbon_twin.pipeline import run_fuel_optimization_from_mrv

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_WRK: dict = {}
_CASES = {"C0_uncalibrated_proxy": False, "C1_MRV_calibrated_baseline": True}
_KEEP = ["vessel_imo", "ship_type", "year", "rci", "rci_joint", "delta_cost_eur",
         "joint_fuel", "siloed_fuel", "energy_scale"]


def _init_worker(mrv_pkl, fleet_pkl, factors_pkl):
    _WRK["mrv_df"] = pickle.loads(mrv_pkl)
    _WRK["fleet_idx"] = pickle.loads(fleet_pkl)
    _WRK["factors"] = pickle.loads(factors_pkl)


def _run_cell(task: dict) -> list[dict]:
    return run_fuel_optimization_from_mrv(
        row_block(_WRK["mrv_df"], task), _WRK["fleet_idx"], task["year"], task["eua"],
        task["prices"],
        cii_version=task["cii_v"], fueleu_version=task["fueleu_v"], ets_version=task["ets_v"],
        fuel_factors=_WRK["factors"], extended_fuels=True, blend=True,
        calibrate_to_mrv=_CASES[task["case"]],
        **task.get("vkw", {}),
        **task.get("ckw", {}),
        return_rows=True,
    )


def _finish(task: dict, opt: pd.DataFrame) -> pd.DataFrame | None:
    if opt is None or opt.empty:
        return None
    keep = [c for c in _KEEP if c in opt.columns]
    opt = opt[keep].copy()
    opt["case"] = task["case"]
    opt["fuel_price_scenario"] = task["ps"]
    return opt


def _agg(g: pd.DataFrame) -> dict:
    return {
        "n_vessels": int(len(g)),
        "mean_rci_silo": round(float(g["rci"].mean()), 5),
        "median_rci_silo": round(float(g["rci"].median()), 5),
        "p90_rci_silo": round(float(g["rci"].quantile(0.90)), 5),
        "fleet_delta_cost_eur": round(float(g["delta_cost_eur"].sum()), 2),
        "mean_delta_cost_eur": round(float(g["delta_cost_eur"].mean()), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="MRV baseline-calibration sensitivity")
    ap.add_argument("--mrv", required=True, type=Path)
    ap.add_argument("--fleet", required=True, type=Path)
    ap.add_argument("--scenario", required=True, type=Path)
    ap.add_argument("--years", default="2026,2035,2045")
    ap.add_argument("--price-scenarios", default="base_2026")
    ap.add_argument("--eua", type=float, default=80.0)
    ap.add_argument("--out", type=Path,
                    default=Path("results/optimization/mrv_calibration_sensitivity.csv"))
    ap.add_argument("--out-ratios", type=Path,
                    default=Path("results/optimization/mrv_calibration_ratios.csv"))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    add_variant_argument(ap)
    args = ap.parse_args()
    variant = load_variant(args.variant)
    write_variant_manifest(getattr(args, "out", None)
                           or Path(args.out_dir) / "run", variant)

    years = [int(y) for y in args.years.split(",") if y.strip()]
    price_scenarios = [s.strip() for s in args.price_scenarios.split(",") if s.strip()]
    cfg = _load_scenario(str(args.scenario))
    regs = cfg["regulations"]
    mrv_df = load_mrv_csv(args.mrv)
    fleet_idx = vessels_to_index(load_vessels_csv(args.fleet))
    factors = variant.fuel_factors(load_fuel_factors_safe())
    prices_by = {ps: load_fuel_prices("configs/fuel_prices.yaml", scenario=ps)
                 for ps in price_scenarios}

    base = {"cii_v": regs["cii"]["version"],
            "fueleu_v": variant.fueleu_version_or(regs["fueleu"]["version"]),
            "ets_v": regs["eu_ets"]["version"], "eua": args.eua,
            "vkw": variant.fuel_optimizer_kwargs(),
            "ckw": variant.cii_kwargs()}
    tasks = [
        {**base, "case": case, "ps": ps, "year": yr, "prices": prices_by[ps]}
        for case in _CASES for ps in price_scenarios for yr in years
    ]

    n, nw = len(tasks), args.workers
    log.info("MRV calibration: %d runs (cases=%d years=%s) | workers=%d",
             n, len(_CASES), years, nw)

    pkls = (pickle.dumps(mrv_df), pickle.dumps(fleet_idx), pickle.dumps(factors))
    frames, done = [], 0
    for t, opt in run_blocked(tasks, _run_cell, workers=args.workers, n_rows=len(mrv_df),
                              initializer=_init_worker, initargs=pkls):
        if isinstance(opt, Exception):
            log.error("case=%s ps=%s yr=%s: %s", t["case"], t["ps"], t["year"], opt)
            continue
        df = _finish(t, opt)
        if df is not None:
            frames.append(df)
            done += 1
            log.info("[%d/%d] %-28s ps=%-14s yr=%d  rci_mean=%.4f",
                     done, n, t["case"], t["ps"], t["year"], df["rci"].mean())

    if not frames:
        log.error("no results")
        sys.exit(1)
    big = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for (case, ps, yr), g in big.groupby(["case", "fuel_price_scenario", "year"]):
        rows.append({"case": case, "year": yr, "decision_dimension": "fuel_and_speed",
                     "fuel_price_scenario": ps, "eua_price_eur_per_tco2": args.eua, **_agg(g)})
    pd.DataFrame(rows).sort_values(["fuel_price_scenario", "year", "case"]).to_csv(
        args.out, index=False)

    c1 = big[(big["case"] == "C1_MRV_calibrated_baseline") & (big["year"] == years[0])]
    ratio_rows = []
    for st, g in c1.groupby("ship_type"):
        mod_mrv = (1.0 / g["energy_scale"]).replace([float("inf")], pd.NA).dropna()
        if mod_mrv.empty:
            continue
        ratio_rows.append({
            "ship_type": st, "n_vessels": int(len(g)),
            "modelled_to_MRV_CO2_median_uncalibrated": round(float(mod_mrv.median()), 4),
            "modelled_to_MRV_CO2_iqr_uncalibrated": round(
                float(mod_mrv.quantile(0.75) - mod_mrv.quantile(0.25)), 4),
            "modelled_to_MRV_CO2_median_calibrated": 1.0,
            "modelled_to_MRV_CO2_iqr_calibrated": 0.0,
            "notes": "calibrated = 1.0 by construction (E anchored to MRV total CO2)",
        })
    pd.DataFrame(ratio_rows).to_csv(args.out_ratios, index=False)
    log.info("done → %s (+ ratios)", args.out)


if __name__ == "__main__":
    main()
