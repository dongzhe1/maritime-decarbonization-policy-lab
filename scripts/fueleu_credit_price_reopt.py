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
from carbon_twin.pipeline import run_fuel_optimization_from_mrv

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_WRK: dict = {}

_CASES: dict[str, float] = {
    "CR100_statutory_equivalent": 1.00,
    "CR75_credit_75pct":          0.75,
    "CR50_credit_50pct":          0.50,
    "CR25_credit_25pct":          0.25,
    "CR0_credit_0pct":            0.00,
}

_KEEP = [
    "vessel_imo", "ship_type", "year",
    "rci", "rci_joint", "delta_cost_eur",
    "cost_joint_eur", "cost_siloed_eur",
    "joint_fuel", "siloed_fuel",
    "low_carbon_share_joint", "low_carbon_share_siloed",
    "biofuel_share_joint", "biofuel_share_siloed",
    "fuel_cost_joint_eur", "ets_cost_joint_eur",
    "fueleu_penalty_joint_eur", "cii_consequence_joint_eur",
    "fuel_cost_siloed_eur", "ets_cost_siloed_eur",
    "fueleu_penalty_siloed_eur",
]


def _init_worker(mrv_pkl, fleet_pkl, factors_pkl):
    _WRK["mrv_df"] = pickle.loads(mrv_pkl)
    _WRK["fleet_idx"] = pickle.loads(fleet_pkl)
    _WRK["factors"] = pickle.loads(factors_pkl)


def _run_cell(task: dict) -> list[dict]:
    frac = _CASES[task["case"]]
    return run_fuel_optimization_from_mrv(
        row_block(_WRK["mrv_df"], task), _WRK["fleet_idx"], task["year"], task["eua"],
        task["prices"],
        cii_version=task["cii_v"], fueleu_version=task["fueleu_v"],
        ets_version=task["ets_v"],
        fuel_factors=_WRK["factors"], extended_fuels=True, blend=True,
        fueleu_price_fraction=frac,
        **task.get("vkw", {}),
        **task.get("ckw", {}),
        return_rows=True,
    )


def _finish(task: dict, opt: pd.DataFrame) -> pd.DataFrame | None:
    frac = _CASES[task["case"]]
    if opt is None or opt.empty:
        return None
    keep = [c for c in _KEEP if c in opt.columns]
    opt = opt[keep].copy()
    opt["case"] = task["case"]
    opt["fueleu_price_fraction"] = frac
    opt["fuel_price_scenario"] = task["ps"]
    opt["eua_price_eur_per_tco2"] = task["eua"]
    return opt


def _pct_fuel(s: pd.Series, val: str) -> float:
    return round(float((s == val).mean()), 4)


def _agg(g: pd.DataFrame) -> dict:
    dom_s, dom_j = g["siloed_fuel"].mode(), g["joint_fuel"].mode()
    frac = float(g["fueleu_price_fraction"].iloc[0])
    mean_rci = round(float(g["rci"].mean()), 5)
    agg_rci = (round(float(g["delta_cost_eur"].sum() /
                          g["cost_siloed_eur"].sum()), 5)
               if g["cost_siloed_eur"].sum() > 0 else 0.0)
    return {
        "fueleu_price_fraction": round(frac, 2),
        "n_vessels": int(len(g)),
        "mean_rci_silo": mean_rci,
        "aggregate_rci_silo": agg_rci,
        "median_rci_silo": round(float(g["rci"].median()), 5),
        "p90_rci_silo": round(float(g["rci"].quantile(0.90)), 5),
        "fleet_delta_cost_eur": round(float(g["delta_cost_eur"].sum()), 2),
        "mean_delta_cost_eur": round(float(g["delta_cost_eur"].mean()), 2),
        "dominant_fuel_silo": str(dom_s.iloc[0]) if len(dom_s) else "",
        "dominant_fuel_joint": str(dom_j.iloc[0]) if len(dom_j) else "",
        "mean_low_carbon_share_silo": round(
            float(g["low_carbon_share_siloed"].mean()), 4)
            if "low_carbon_share_siloed" in g else 0.0,
        "mean_low_carbon_share_joint": round(
            float(g["low_carbon_share_joint"].mean()), 4)
            if "low_carbon_share_joint" in g else 0.0,
        "mean_biofuel_share_joint": round(
            float(g["biofuel_share_joint"].mean()), 4)
            if "biofuel_share_joint" in g else 0.0,
        "mean_fueleu_penalty_silo_eur": round(
            float(g["fueleu_penalty_siloed_eur"].mean()), 2)
            if "fueleu_penalty_siloed_eur" in g else 0.0,
        "mean_fueleu_penalty_joint_eur": round(
            float(g["fueleu_penalty_joint_eur"].mean()), 2)
            if "fueleu_penalty_joint_eur" in g else 0.0,
        "mean_ets_cost_silo_eur": round(
            float(g["ets_cost_siloed_eur"].mean()), 2)
            if "ets_cost_siloed_eur" in g else 0.0,
        "mean_ets_cost_joint_eur": round(
            float(g["ets_cost_joint_eur"].mean()), 2)
            if "ets_cost_joint_eur" in g else 0.0,
        "mean_fuel_cost_silo_eur": round(
            float(g["fuel_cost_siloed_eur"].mean()), 2)
            if "fuel_cost_siloed_eur" in g else 0.0,
        "mean_fuel_cost_joint_eur": round(
            float(g["fuel_cost_joint_eur"].mean()), 2)
            if "fuel_cost_joint_eur" in g else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="FuelEU credit-price re-optimization")
    ap.add_argument("--mrv", required=True, type=Path)
    ap.add_argument("--fleet", required=True, type=Path)
    ap.add_argument("--scenario", required=True, type=Path)
    ap.add_argument("--years", default="2035,2040,2045")
    ap.add_argument("--price-scenarios", default="base_2026")
    ap.add_argument("--eua", type=float, default=80.0)
    ap.add_argument(
        "--out", type=Path,
        default=Path("results/optimization/fueleu_credit_price_reopt.csv"))
    ap.add_argument(
        "--out-by-type", type=Path,
        default=Path(
            "results/optimization/fueleu_credit_price_reopt_by_type.csv"))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    add_variant_argument(ap)
    args = ap.parse_args()
    variant = load_variant(args.variant)
    write_variant_manifest(getattr(args, "out", None)
                           or Path(args.out_dir) / "run", variant)

    years = [int(y) for y in args.years.split(",") if y.strip()]
    price_scenarios = [s.strip() for s in args.price_scenarios.split(",")
                       if s.strip()]
    cfg = _load_scenario(str(args.scenario))
    regs = cfg["regulations"]
    mrv_df = load_mrv_csv(args.mrv)
    fleet_idx = vessels_to_index(load_vessels_csv(args.fleet))
    factors = variant.fuel_factors(load_fuel_factors_safe())
    prices_by = {ps: load_fuel_prices("configs/fuel_prices.yaml", scenario=ps)
                 for ps in price_scenarios}

    base = {
        "cii_v": regs["cii"]["version"],
        "fueleu_v": variant.fueleu_version_or(regs["fueleu"]["version"]),
        "vkw": variant.fuel_optimizer_kwargs(),
        "ckw": variant.cii_kwargs(),
        "ets_v": regs["eu_ets"]["version"],
        "eua": args.eua,
    }
    tasks = [
        {**base, "case": case, "ps": ps, "year": yr, "prices": prices_by[ps]}
        for case in _CASES for ps in price_scenarios for yr in years
    ]

    n, nw = len(tasks), args.workers
    log.info(
        "FuelEU credit-price reopt: %d runs (cases=%d years=%s) | workers=%d",
        n, len(_CASES), years, nw)

    pkls = (pickle.dumps(mrv_df), pickle.dumps(fleet_idx), pickle.dumps(factors))
    frames, done = [], 0
    for t, opt in run_blocked(tasks, _run_cell, workers=args.workers, n_rows=len(mrv_df),
                              initializer=_init_worker, initargs=pkls):
        if isinstance(opt, Exception):
            log.error("case=%s ps=%s yr=%s: %s",
                      t["case"], t["ps"], t["year"], opt)
            continue
        df = _finish(t, opt)
        if df is not None:
            frames.append(df)
            done += 1
            log.info(
                "[%d/%d] %-32s yr=%d  rci_mean=%.4f",
                done, n, t["case"], t["year"], df["rci"].mean())

    if not frames:
        log.error("no results")
        sys.exit(1)
    big = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for (case, ps, yr), g in big.groupby(
            ["case", "fuel_price_scenario", "year"]):
        rows.append({
            "case": case, "year": yr,
            "eua_price_eur_per_tco2": args.eua,
            "fuel_price_scenario": ps,
            **_agg(g),
        })
    pd.DataFrame(rows).sort_values(
        ["fuel_price_scenario", "year", "fueleu_price_fraction"]).to_csv(
        args.out, index=False)

    rows_t = []
    for (case, yr, st), g in big.groupby(["case", "year", "ship_type"]):
        dom_j = g["joint_fuel"].mode()
        rows_t.append({
            "case": case, "year": yr, "ship_type": st,
            "fueleu_price_fraction": float(g["fueleu_price_fraction"].iloc[0]),
            "n_vessels": int(len(g)),
            "mean_rci_silo": round(float(g["rci"].mean()), 5),
            "fleet_delta_cost_eur": round(float(g["delta_cost_eur"].sum()), 2),
            "dominant_fuel_joint": str(dom_j.iloc[0]) if len(dom_j) else "",
        })
    pd.DataFrame(rows_t).to_csv(args.out_by_type, index=False)
    log.info("done → %s (+ by_type)", args.out)


if __name__ == "__main__":
    main()
