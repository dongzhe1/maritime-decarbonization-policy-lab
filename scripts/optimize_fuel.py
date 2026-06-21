#!/usr/bin/env python3
"""Fuel-dimension RCI across FuelEU stringency (joint vs ETS-siloed fuel choice).

Sweeps years in parallel; writes a per-vessel parquet plus RCI-by-year and
RCI-by-type summary CSVs.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall: pip install -e '.[all]'")

from carbon_twin.cli import _load_scenario, _resolve_eua_price
from carbon_twin.data.loaders import load_fuel_prices, load_mrv_csv, load_vessels_csv
from carbon_twin.emissions.proxy import load_fuel_factors_safe, vessels_to_index
from carbon_twin.pipeline import run_fuel_optimization_from_mrv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_WRK: dict = {}


def _init_worker(mrv_pkl: bytes, fleet_pkl: bytes, factors_pkl: bytes) -> None:
    _WRK["mrv_df"] = pickle.loads(mrv_pkl)
    _WRK["fleet_idx"] = pickle.loads(fleet_pkl)
    _WRK["factors"] = pickle.loads(factors_pkl)


def _run_year(task: dict) -> tuple[int, pd.DataFrame]:
    opt = run_fuel_optimization_from_mrv(
        _WRK["mrv_df"],
        _WRK["fleet_idx"],
        task["yr"],
        task["eua_price"],
        task["fuel_prices"],
        cii_version=task["cii_v"],
        fueleu_version=task["fueleu_v"],
        ets_version=task["ets_v"],
        fuel_factors=_WRK["factors"],
        extended_fuels=task["extended"],
        blend=task["blend"],
    )
    return task["yr"], opt


def _summary(opt: pd.DataFrame, year: int) -> dict:
    rci = opt["rci"].dropna()
    s = {
        "year": year,
        "n": int(len(opt)),
        "rci_mean": round(float(rci.mean()), 5),
        "rci_median": round(float(rci.median()), 5),
        "frac_gt_5pct": round(float((rci > 0.05).mean()), 4),
        "frac_gt_10pct": round(float((rci > 0.10).mean()), 4),
        "rci_p90": round(float(rci.quantile(0.90)), 5),
        "rci_max": round(float(rci.max()), 5),
    }
    if "joint_blend_pct" in opt:
        s["frac_blends"] = round(float(opt["blends"].mean()), 4)
        s["mean_blend_pct"] = round(float(opt["joint_blend_pct"].mean()), 4)
    else:
        s["frac_switch"] = round(float(opt["fuel_switch"].mean()), 4)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Fuel-dimension RCI vs FuelEU stringency")
    ap.add_argument("--mrv", required=True, type=Path)
    ap.add_argument("--fleet", required=True, type=Path)
    ap.add_argument("--scenario", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--out-sweep", type=Path, default=None)
    ap.add_argument("--out-by-type", type=Path, default=None)
    ap.add_argument("--years", default="2026,2030,2035,2040,2045")
    ap.add_argument("--price-scenario", default="base_2026")
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--blend", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(",") if y.strip()]
    scenario_cfg = _load_scenario(str(args.scenario))
    regs = scenario_cfg["regulations"]
    fuel_prices = load_fuel_prices("configs/fuel_prices.yaml", scenario=args.price_scenario)

    mrv_df = load_mrv_csv(args.mrv)
    fleet_idx = vessels_to_index(load_vessels_csv(args.fleet))
    factors = load_fuel_factors_safe()

    n_workers = min(args.workers, len(years))
    log.info(
        "prices=%s | extended=%s | blend=%s | years=%s | workers=%d",
        args.price_scenario,
        args.extended,
        args.blend,
        years,
        n_workers,
    )

    mrv_pkl = pickle.dumps(mrv_df)
    fleet_pkl = pickle.dumps(fleet_idx)
    factors_pkl = pickle.dumps(factors)

    tasks = [
        {
            "yr": yr,
            "eua_price": _resolve_eua_price(scenario_cfg, yr),
            "fuel_prices": fuel_prices,
            "cii_v": regs["cii"]["version"],
            "fueleu_v": regs["fueleu"]["version"],
            "ets_v": regs["eu_ets"]["version"],
            "extended": args.extended,
            "blend": args.blend,
        }
        for yr in years
    ]

    results: dict[int, pd.DataFrame] = {}
    done = 0
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(mrv_pkl, fleet_pkl, factors_pkl),
    ) as exe:
        future_to_yr = {exe.submit(_run_year, t): t["yr"] for t in tasks}
        for future in as_completed(future_to_yr):
            yr = future_to_yr[future]
            try:
                yr_ret, opt = future.result()
            except Exception as exc:
                log.error("yr=%d: %s", yr, exc)
                continue
            if opt.empty:
                log.warning("yr=%d: no vessels", yr_ret)
                continue
            results[yr_ret] = opt
            s = _summary(opt, yr_ret)
            done += 1
            act = (
                f"blend={s['mean_blend_pct']:.0%}"
                if "mean_blend_pct" in s
                else f"switch={s['frac_switch']:.1%}"
            )
            log.info(
                "[%d/%d] yr=%d  RCI mean=%.4f %s >10%%=%.1f%%  max=%.3f",
                done,
                len(years),
                yr_ret,
                s["rci_mean"],
                act,
                s["frac_gt_10pct"] * 100,
                s["rci_max"],
            )

    if not results:
        log.error("no results produced")
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Per-vessel parquet: primary (earliest) year.
    primary_yr = min(yr for yr in years if yr in results)
    results[primary_yr].to_parquet(args.out, index=False)
    log.info("per-vessel yr=%d → %s", primary_yr, args.out)

    # Year sweep CSV: sorted by year.
    if args.out_sweep is not None:
        sweep_rows = [_summary(results[yr], yr) for yr in sorted(results)]
        pd.DataFrame(sweep_rows).to_csv(args.out_sweep, index=False)
        log.info("year sweep → %s", args.out_sweep)

    if args.out_by_type is not None:
        by_type_rows = []
        for yr in sorted(results):
            for st, g in results[yr].groupby("ship_type"):
                r = g["rci"]
                by_type_rows.append(
                    {
                        "year": yr,
                        "ship_type": st,
                        "n": int(len(g)),
                        "rci_mean": round(float(r.mean()), 5),
                        "rci_max": round(float(r.max()), 5),
                    }
                )
        pd.DataFrame(by_type_rows).to_csv(args.out_by_type, index=False)
        log.info("by-type sweep → %s", args.out_by_type)


if __name__ == "__main__":
    main()
