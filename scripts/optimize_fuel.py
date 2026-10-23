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

from carbon_twin.cli import _load_scenario, _resolve_eua_price
from carbon_twin.data.loaders import load_fuel_prices, load_mrv_csv, load_vessels_csv
from carbon_twin.emissions.proxy import load_fuel_factors_safe, vessels_to_index
from carbon_twin.model_variants import (
    add_variant_argument,
    load_variant,
    write_variant_manifest,
)
from carbon_twin.parallel import row_block, run_blocked
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


def _run_year(task: dict) -> list[dict]:
    return run_fuel_optimization_from_mrv(
        row_block(_WRK["mrv_df"], task), _WRK["fleet_idx"],
        task["yr"], task["eua_price"], task["fuel_prices"],
        cii_version=task["cii_v"],
        fueleu_version=task["fueleu_v"],
        ets_version=task["ets_v"],
        fuel_factors=_WRK["factors"],
        extended_fuels=task["extended"],
        blend=task["blend"],
        **task.get("vkw", {}),
        **task.get("ckw", {}),
        return_rows=True,
    )


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
    if "delta_cost_eur" in opt:
        d = opt["delta_cost_eur"]
        s["delta_cost_total_eur"] = round(float(d.sum()), 2)
        s["delta_cost_median_eur"] = round(float(d.median()), 2)
        s["delta_cost_p90_eur"] = round(float(d.quantile(0.90)), 2)
        s["rci_joint_mean"] = round(float(opt["rci_joint"].mean()), 5)
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
    ap.add_argument("--eua", type=float, default=None,
                    help="fixed EUA EUR/t held across all years (default: scenario "
                         "reference-year price). Held fixed so only FuelEU stringency varies.")
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--blend", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    add_variant_argument(ap)
    args = ap.parse_args()
    variant = load_variant(args.variant)
    write_variant_manifest(getattr(args, "out", None)
                           or Path(args.out_dir) / "run", variant)

    years = [int(y) for y in args.years.split(",") if y.strip()]
    scenario_cfg = _load_scenario(str(args.scenario))
    regs = scenario_cfg["regulations"]
    fuel_prices = load_fuel_prices("configs/fuel_prices.yaml", scenario=args.price_scenario)

    eua_price = args.eua if args.eua is not None else _resolve_eua_price(scenario_cfg, years[0])

    mrv_df = load_mrv_csv(args.mrv)
    fleet_idx = vessels_to_index(load_vessels_csv(args.fleet))
    factors = variant.fuel_factors(load_fuel_factors_safe())

    n_workers = args.workers
    log.info("prices=%s | EUA=%.1f (fixed) | extended=%s | blend=%s | years=%s | workers=%d",
             args.price_scenario, eua_price, args.extended, args.blend, years, n_workers)

    mrv_pkl = pickle.dumps(mrv_df)
    fleet_pkl = pickle.dumps(fleet_idx)
    factors_pkl = pickle.dumps(factors)

    tasks = [
        {
            "yr": yr,
            "eua_price": eua_price,
            "fuel_prices": fuel_prices,
            "cii_v": regs["cii"]["version"],
            "fueleu_v": variant.fueleu_version_or(regs["fueleu"]["version"]),
            "vkw": variant.fuel_optimizer_kwargs(),
            "ckw": variant.cii_kwargs(),
            "ets_v": regs["eu_ets"]["version"],
            "extended": args.extended,
            "blend": args.blend,
        }
        for yr in years
    ]

    results: dict[int, pd.DataFrame] = {}
    done = 0
    for t, opt in run_blocked(tasks, _run_year, workers=args.workers, n_rows=len(mrv_df),
                              initializer=_init_worker,
                              initargs=(mrv_pkl, fleet_pkl, factors_pkl)):
        yr_ret = t["yr"]
        if isinstance(opt, Exception):
            log.error("yr=%d: %s", yr_ret, opt)
            continue
        if opt.empty:
            log.warning("yr=%d: no vessels", yr_ret)
            continue
        results[yr_ret] = opt
        s = _summary(opt, yr_ret)
        done += 1
        act = (f"blend={s['mean_blend_pct']:.0%}" if "mean_blend_pct" in s
               else f"switch={s['frac_switch']:.1%}")
        log.info("[%d/%d] yr=%d  RCI mean=%.4f %s >10%%=%.1f%%  max=%.3f",
                 done, len(years), yr_ret,
                 s["rci_mean"], act, s["frac_gt_10pct"] * 100, s["rci_max"])

    if not results:
        log.error("no results produced")
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    allyears = pd.concat([results[yr] for yr in sorted(results)], ignore_index=True)
    allyears.to_parquet(args.out, index=False)
    log.info("per-vessel (all years, n=%d) → %s", len(allyears), args.out)

    if args.out_sweep is not None:
        sweep_rows = [_summary(results[yr], yr) for yr in sorted(results)]
        pd.DataFrame(sweep_rows).to_csv(args.out_sweep, index=False)
        log.info("year sweep → %s", args.out_sweep)

    if args.out_by_type is not None:
        by_type_rows = []
        for yr in sorted(results):
            for st, g in results[yr].groupby("ship_type"):
                r = g["rci"]
                rec = {
                    "year": yr, "ship_type": st, "n": int(len(g)),
                    "rci_mean": round(float(r.mean()), 5),
                    "rci_max": round(float(r.max()), 5),
                }
                if "delta_cost_eur" in g:
                    rec["delta_cost_total_eur"] = round(float(g["delta_cost_eur"].sum()), 2)
                    rec["delta_cost_median_eur"] = round(float(g["delta_cost_eur"].median()), 2)
                    rec["delta_cost_p90_eur"] = round(float(g["delta_cost_eur"].quantile(0.90)), 2)
                by_type_rows.append(rec)
        pd.DataFrame(by_type_rows).to_csv(args.out_by_type, index=False)
        log.info("by-type sweep → %s", args.out_by_type)


if __name__ == "__main__":
    main()
