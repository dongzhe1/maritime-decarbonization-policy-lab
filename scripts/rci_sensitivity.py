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

from carbon_twin.cli import _load_scenario, _resolve_eua_price, _resolve_fuel_price
from carbon_twin.data.loaders import load_mrv_csv, load_vessels_csv
from carbon_twin.emissions.proxy import load_fuel_factors_safe, vessels_to_index
from carbon_twin.model_variants import (
    add_variant_argument,
    load_variant,
    write_variant_manifest,
)
from carbon_twin.parallel import row_block, run_blocked
from carbon_twin.pipeline import run_optimization_from_mrv

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


def _run_pressure(task: dict) -> list[dict]:
    return run_optimization_from_mrv(
        pd.DataFrame(),
        row_block(_WRK["mrv_df"], task), _WRK["fleet_idx"],
        task["year"], task["eua_price"], task["fuel_price"],
        cii_version=task["cii_v"],
        fueleu_version=task["fueleu_v"],
        ets_version=task["ets_v"],
        fuel_factors=_WRK["factors"],
        cii_pressure=task["k"],
        **task.get("vkw", {}),
        return_rows=True,
    )


def _summarise(opt: pd.DataFrame, k: float) -> dict:
    rci = opt["rci"].dropna()
    pos = rci[rci > 0]
    s = {
        "cii_pressure": k,
        "n": int(len(rci)),
        "rci_mean": round(float(rci.mean()), 5),
        "rci_median": round(float(rci.median()), 5),
        "frac_positive": round(float((rci > 0).mean()), 4),
        "frac_gt_5pct": round(float((rci > 0.05).mean()), 4),
        "frac_gt_10pct": round(float((rci > 0.10).mean()), 4),
        "rci_p90": round(float(rci.quantile(0.90)), 5),
        "rci_p99": round(float(rci.quantile(0.99)), 5),
        "rci_max": round(float(rci.max()), 5),
        "mean_rci_when_positive": round(float(pos.mean()), 5) if len(pos) else 0.0,
    }
    if "delta_cost_eur" in opt:
        d = opt["delta_cost_eur"]
        s["absolute_loss_total_eur"] = round(float(d.sum()), 2)
        s["absolute_loss_median_eur"] = round(float(d.median()), 2)
        s["absolute_loss_p90_eur"] = round(float(d.quantile(0.90)), 2)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="RCI sensitivity to CII pressure")
    ap.add_argument("--case", type=Path, default=None)
    ap.add_argument("--mrv", required=True, type=Path)
    ap.add_argument("--fleet", required=True, type=Path)
    ap.add_argument("--scenario", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--out-by-type", type=Path, default=None)
    ap.add_argument("--pressures", default="0,0.05,0.1,0.2,0.5,1.0,2.0")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    add_variant_argument(ap)
    args = ap.parse_args()
    variant = load_variant(args.variant)
    write_variant_manifest(getattr(args, "out", None)
                           or Path(args.out_dir) / "run", variant)

    pressures = [float(x) for x in args.pressures.split(",") if x.strip()]

    scenario_cfg = _load_scenario(str(args.scenario))
    year = int(scenario_cfg["year"])
    eua_price = _resolve_eua_price(scenario_cfg, year)
    fuel_price = _resolve_fuel_price(scenario_cfg)
    regs = scenario_cfg["regulations"]

    mrv_df = load_mrv_csv(args.mrv)
    fleet_idx = vessels_to_index(load_vessels_csv(args.fleet))
    factors = variant.fuel_factors(load_fuel_factors_safe())

    n_workers = args.workers
    log.info("year=%d EUA=%.1f fuel=%.0f | pressures=%s | workers=%d",
             year, eua_price, fuel_price, pressures, n_workers)

    mrv_pkl = pickle.dumps(mrv_df)
    fleet_pkl = pickle.dumps(fleet_idx)
    factors_pkl = pickle.dumps(factors)

    tasks = [
        {
            "k": k, "year": year, "eua_price": eua_price, "fuel_price": fuel_price,
            "cii_v": regs["cii"]["version"],
            "fueleu_v": variant.fueleu_version_or(regs["fueleu"]["version"]),
            "vkw": variant.speed_optimizer_kwargs(),
            "ets_v": regs["eu_ets"]["version"],
        }
        for k in pressures
    ]

    results: dict[float, pd.DataFrame] = {}
    done = 0
    for t, opt in run_blocked(tasks, _run_pressure, workers=args.workers, n_rows=len(mrv_df),
                              initializer=_init_worker,
                              initargs=(mrv_pkl, fleet_pkl, factors_pkl)):
        k_ret = t["k"]
        if isinstance(opt, Exception):
            log.error("k=%.2f: %s", k_ret, opt)
            continue
        results[k_ret] = opt
        s = _summarise(opt, k_ret)
        done += 1
        log.info("[%d/%d] k=%-5.2f  mean=%.4f median=%.4f "
                 "pos=%.1f%%  >10%%=%.1f%%  max=%.3f",
                 done, len(pressures), k_ret,
                 s["rci_mean"], s["rci_median"],
                 s["frac_positive"] * 100, s["frac_gt_10pct"] * 100, s["rci_max"])

    overall_rows = [_summarise(results[k], k) for k in sorted(results)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(overall_rows).to_csv(args.out, index=False)
    log.info("overall sweep → %s", args.out)

    if args.out_by_type is not None:
        by_type_rows = []
        for k in sorted(results):
            for st, g in results[k].groupby("ship_type"):
                r = g["rci"].dropna()
                by_type_rows.append({
                    "cii_pressure": k, "ship_type": st, "n": int(len(r)),
                    "rci_mean": round(float(r.mean()), 5),
                    "frac_positive": round(float((r > 0).mean()), 4),
                    "rci_max": round(float(r.max()), 5),
                })
        pd.DataFrame(by_type_rows).to_csv(args.out_by_type, index=False)
        log.info("per-type sweep → %s", args.out_by_type)


if __name__ == "__main__":
    main()
