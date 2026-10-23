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


def _run_cell(task: dict) -> list[dict]:
    return run_fuel_optimization_from_mrv(
        row_block(_WRK["mrv_df"], task), _WRK["fleet_idx"],
        task["yr"], task["eua"], task["prices"],
        cii_version=task["cii_v"],
        fueleu_version=task["fueleu_v"],
        ets_version=task["ets_v"],
        fuel_factors=_WRK["factors"],
        extended_fuels=task["extended"],
        blend=task["blend"],
        charter_multiplier=task["cm"],
        **task.get("vkw", {}),
        **task.get("ckw", {}),
        return_rows=True,
    )


def _finish(task: dict, opt: pd.DataFrame) -> dict | None:
    if opt is None or opt.empty:
        return None
    rci = opt["rci"]
    rec = {
        "year": task["yr"],
        "price_scenario": task["ps"],
        "eua": task["eua"],
        "charter_mult": task["cm"],
        "n": int(len(opt)),
        "rci_mean": round(float(rci.mean()), 5),
        "rci_median": round(float(rci.median()), 5),
        "frac_gt_10pct": round(float((rci > 0.10).mean()), 4),
        "rci_p90": round(float(rci.quantile(0.90)), 5),
    }
    if "joint_blend_pct" in opt:
        rec["mean_blend_pct"] = round(float(opt["joint_blend_pct"].mean()), 4)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description="Fuel-RCI sensitivity to prices/EUA")
    ap.add_argument("--mrv", required=True, type=Path)
    ap.add_argument("--fleet", required=True, type=Path)
    ap.add_argument("--scenario", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--years", default="2035,2045")
    ap.add_argument("--price-scenarios",
                    default="base_2026,high_fossil_2026,low_alternative_2030")
    ap.add_argument("--euas", default="50,80,120,160,200")
    ap.add_argument("--charter-mults", default="1.0",
                    help="charter-rate multipliers to sweep (e.g. 0.5,1.0,2.0)")
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--blend", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="parallel worker processes (default: all CPUs)")
    add_variant_argument(ap)
    args = ap.parse_args()
    variant = load_variant(args.variant)
    write_variant_manifest(getattr(args, "out", None)
                           or Path(args.out_dir) / "run", variant)

    years = [int(y) for y in args.years.split(",") if y.strip()]
    price_scenarios = [s.strip() for s in args.price_scenarios.split(",") if s.strip()]
    euas = [float(e) for e in args.euas.split(",") if e.strip()]
    charter_mults = [float(c) for c in args.charter_mults.split(",") if c.strip()]

    scenario_cfg = _load_scenario(str(args.scenario))
    regs = scenario_cfg["regulations"]
    mrv_df = load_mrv_csv(args.mrv)
    fleet_idx = vessels_to_index(load_vessels_csv(args.fleet))
    factors = variant.fuel_factors(load_fuel_factors_safe())

    prices_by_scenario = {
        ps: load_fuel_prices("configs/fuel_prices.yaml", scenario=ps)
        for ps in price_scenarios
    }

    tasks = [
        {
            "yr": yr, "eua": eua, "ps": ps, "cm": cm,
            "prices": prices_by_scenario[ps],
            "cii_v": regs["cii"]["version"],
            "fueleu_v": variant.fueleu_version_or(regs["fueleu"]["version"]),
            "vkw": variant.fuel_optimizer_kwargs(),
            "ckw": variant.cii_kwargs(),
            "ets_v": regs["eu_ets"]["version"],
            "extended": args.extended,
            "blend": args.blend,
        }
        for ps in price_scenarios
        for eua in euas
        for cm in charter_mults
        for yr in years
    ]

    n_runs = len(tasks)
    n_workers = args.workers
    log.info("years=%s | scenarios=%s | EUA=%s | charter×%s | blend=%s extended=%s "
             "→ %d fleet runs | workers=%d",
             years, price_scenarios, euas, charter_mults,
             args.blend, args.extended, n_runs, n_workers)

    mrv_pkl = pickle.dumps(mrv_df)
    fleet_pkl = pickle.dumps(fleet_idx)
    factors_pkl = pickle.dumps(factors)

    rows: list[dict] = []
    done = 0
    for t, opt in run_blocked(tasks, _run_cell, workers=args.workers, n_rows=len(mrv_df),
                              initializer=_init_worker,
                              initargs=(mrv_pkl, fleet_pkl, factors_pkl)):
        if isinstance(opt, Exception):
            log.error("ps=%s EUA=%s cm=%s yr=%s: %s", t["ps"], t["eua"], t["cm"], t["yr"], opt)
            continue
        rec = _finish(t, opt)
        if rec is not None:
            rows.append(rec)
            done += 1
            log.info("[%d/%d] %-22s EUA=%5.0f chx%4.1f yr=%d  "
                     "RCI mean=%.4f >10%%=%.0f%%",
                     done, n_runs, t["ps"], t["eua"], t["cm"], t["yr"],
                     rec["rci_mean"], rec["frac_gt_10pct"] * 100)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(
        ["price_scenario", "eua", "charter_mult", "year"]
    ).reset_index(drop=True)
    df.to_csv(args.out, index=False)
    log.info("done — %d cells → %s", len(rows), args.out)


if __name__ == "__main__":
    main()
