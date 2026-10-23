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
from carbon_twin.data.schemas import FuelType
from carbon_twin.emissions.proxy import load_fuel_factors_safe, vessels_to_index
from carbon_twin.model_variants import (
    add_variant_argument,
    load_variant,
    write_variant_manifest,
)
from carbon_twin.parallel import row_block, run_blocked
from carbon_twin.pipeline import (
    assemble_compliance_from_mrv,
    run_optimization_from_mrv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_WRK: dict = {}

_CASES = {
    "SA0_uncalibrated":   {"calibrate_to_mrv": False},
    "SA1_mrv_calibrated": {"calibrate_to_mrv": True},
}
_CII_PRESSURES = [0.0, 0.2, 2.0]


def _init_worker(mrv_pkl, fleet_pkl, factors_pkl, comp_pkl):
    _WRK["mrv_df"] = pickle.loads(mrv_pkl)
    _WRK["fleet_idx"] = pickle.loads(fleet_pkl)
    _WRK["factors"] = pickle.loads(factors_pkl)
    _WRK["comp_df"] = pickle.loads(comp_pkl)


def _run_cell(task: dict) -> list[dict]:
    calib = _CASES[task["case"]]["calibrate_to_mrv"]
    return run_optimization_from_mrv(
        _WRK["comp_df"], row_block(_WRK["mrv_df"], task), _WRK["fleet_idx"],
        task["year"], task["eua"], task["fuel_price"],
        cii_version=task["cii_v"], fueleu_version=task["fueleu_v"],
        ets_version=task["ets_v"],
        fuel_factors=_WRK["factors"],
        cii_pressure=task["k"],
        calibrate_to_mrv=calib,
        **task.get("vkw", {}),
        return_rows=True,
    )


def _finish(task: dict, opt: pd.DataFrame) -> pd.DataFrame | None:
    if opt is None or opt.empty:
        return None
    keep = [c for c in [
        "vessel_imo", "ship_type", "rci", "rci_joint", "delta_cost_eur",
        "cost_joint_eur", "cost_siloed_eur",
        "joint_optimal_speed", "siloed_optimal_speed",
    ] if c in opt.columns]
    out = opt[keep].copy()
    out["case"] = task["case"]
    out["cii_pressure_k"] = task["k"]
    out["year"] = task["year"]
    out["eua_price_eur_per_tco2"] = task["eua"]
    return out


def _agg(g: pd.DataFrame) -> dict:
    return {
        "n_vessels": int(len(g)),
        "mean_rci_silo": round(float(g["rci"].mean()), 5),
        "median_rci_silo": round(float(g["rci"].median()), 5),
        "p90_rci_silo": round(float(g["rci"].quantile(0.90)), 5),
        "pct_rci_gt_10pct": round(float((g["rci"] > 0.10).mean()), 4),
        "fleet_delta_cost_eur": round(float(g["delta_cost_eur"].sum()), 2),
        "mean_delta_cost_eur": round(float(g["delta_cost_eur"].mean()), 2),
        "mean_joint_speed": round(float(g["joint_optimal_speed"].mean()), 2)
            if "joint_optimal_speed" in g else float("nan"),
        "mean_silo_speed": round(float(g["siloed_optimal_speed"].mean()), 2)
            if "siloed_optimal_speed" in g else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MRV-calibrated speed RCI")
    ap.add_argument("--mrv", required=True, type=Path)
    ap.add_argument("--fleet", required=True, type=Path)
    ap.add_argument("--scenario", required=True, type=Path)
    ap.add_argument("--year", type=int, default=2035)
    ap.add_argument("--eua", type=float, default=80.0)
    ap.add_argument("--fuel-price", type=float, default=650.0,
                    help="VLSFO price EUR/t (used for speed optimizer)")
    ap.add_argument(
        "--out", type=Path,
        default=Path("results/optimization/speed_rci_mrv_calibration.csv"))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    add_variant_argument(ap)
    args = ap.parse_args()
    variant = load_variant(args.variant)
    write_variant_manifest(getattr(args, "out", None)
                           or Path(args.out_dir) / "run", variant)

    cfg = _load_scenario(str(args.scenario))
    regs = cfg["regulations"]
    mrv_df = load_mrv_csv(args.mrv)
    fleet_idx = vessels_to_index(load_vessels_csv(args.fleet))
    factors = variant.fuel_factors(load_fuel_factors_safe())
    comp_df = assemble_compliance_from_mrv(
        mrv_df, fleet_idx, args.year, args.eua,
        cii_version=regs["cii"]["version"],
        fueleu_version=variant.fueleu_version_or(regs["fueleu"]["version"]),
        ets_version=regs["eu_ets"]["version"],
        **variant.compliance_kwargs(),
    )

    base = {
        "cii_v": regs["cii"]["version"],
        "fueleu_v": variant.fueleu_version_or(regs["fueleu"]["version"]),
        "vkw": variant.speed_optimizer_kwargs(),
        "ets_v": regs["eu_ets"]["version"],
        "eua": args.eua,
        "fuel_price": args.fuel_price,
        "year": args.year,
    }
    tasks = [
        {**base, "case": case, "k": k}
        for case in _CASES for k in _CII_PRESSURES
    ]

    n, nw = len(tasks), args.workers
    log.info("speed RCI MRV calibration: %d runs | workers=%d", n, nw)

    pkls = (pickle.dumps(mrv_df), pickle.dumps(fleet_idx),
            pickle.dumps(factors), pickle.dumps(comp_df))
    frames, done = [], 0
    for t, opt in run_blocked(tasks, _run_cell, workers=args.workers, n_rows=len(mrv_df),
                              initializer=_init_worker, initargs=pkls):
        if isinstance(opt, Exception):
            log.error("case=%s k=%s: %s", t["case"], t["k"], opt)
            continue
        df = _finish(t, opt)
        if df is not None:
            frames.append(df)
            done += 1
            log.info("[%d/%d] %-26s k=%.1f  rci_mean=%.4f",
                     done, n, t["case"], t["k"], df["rci"].mean())

    if not frames:
        log.error("no results")
        sys.exit(1)
    big = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for (case, k), g in big.groupby(["case", "cii_pressure_k"]):
        rows.append({
            "case": case, "cii_pressure_k": k,
            "year": args.year,
            "eua_price_eur_per_tco2": args.eua,
            **_agg(g),
        })

    by_type_rows = []
    for (case, k, st), g in big.groupby(
            ["case", "cii_pressure_k", "ship_type"]):
        by_type_rows.append({
            "case": case, "cii_pressure_k": k, "ship_type": st,
            "n_vessels": int(len(g)),
            "mean_rci_silo": round(float(g["rci"].mean()), 5),
            "p90_rci_silo": round(float(g["rci"].quantile(0.90)), 5),
            "fleet_delta_cost_eur": round(float(g["delta_cost_eur"].sum()), 2),
        })

    summary_df = pd.DataFrame(rows).sort_values(["case", "cii_pressure_k"])
    summary_df.to_csv(args.out, index=False)

    by_type_path = args.out.with_name(
        args.out.stem + "_by_type" + args.out.suffix)
    pd.DataFrame(by_type_rows).sort_values(
        ["case", "cii_pressure_k", "ship_type"]).to_csv(
        by_type_path, index=False)
    log.info("done → %s (+ by_type)", args.out)


if __name__ == "__main__":
    main()
