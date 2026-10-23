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

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_WRK: dict = {}

_BIOFUELS = {FuelType.BIODIESEL_FAME, FuelType.METHANOL_BIO}
_FAME = {FuelType.BIODIESEL_FAME}

_CASES = {
    "B0_current_fuel_menu": {"exclude": set(), "cap": None, "price_mult": 1.0},
    "B1_no_FAME_no_biodiesel": {"exclude": _FAME, "cap": None, "price_mult": 1.0},
    "B2_no_biofuel": {"exclude": _BIOFUELS, "cap": None, "price_mult": 1.0},
    "B3_biofuel_cap_10": {"exclude": set(), "cap": 0.10, "price_mult": 1.0},
    "B3_biofuel_cap_30": {"exclude": set(), "cap": 0.30, "price_mult": 1.0},
    "B3_biofuel_cap_50": {"exclude": set(), "cap": 0.50, "price_mult": 1.0},
    "B4_biofuel_price_1p5x": {"exclude": set(), "cap": None, "price_mult": 1.5},
    "B4_biofuel_price_2p0x": {"exclude": set(), "cap": None, "price_mult": 2.0},
}

_KEEP = [
    "vessel_imo", "ship_type", "year", "rci", "rci_joint", "delta_cost_eur",
    "joint_fuel", "siloed_fuel", "low_carbon_share_joint", "low_carbon_share_siloed",
    "biofuel_share_joint", "biofuel_share_siloed",
]


def _init_worker(mrv_pkl, fleet_pkl, factors_pkl):
    _WRK["mrv_df"] = pickle.loads(mrv_pkl)
    _WRK["fleet_idx"] = pickle.loads(fleet_pkl)
    _WRK["factors"] = pickle.loads(factors_pkl)


def _scaled_prices(prices: dict, mult: float) -> dict:
    if mult == 1.0:
        return prices
    out = dict(prices)
    for ft in _BIOFUELS:
        if ft.value in out:
            out[ft.value] = out[ft.value] * mult
    return out


def _run_cell(task: dict) -> list[dict]:
    spec = _CASES[task["case"]]
    return run_fuel_optimization_from_mrv(
        row_block(_WRK["mrv_df"], task), _WRK["fleet_idx"], task["year"], task["eua"],
        _scaled_prices(task["prices"], spec["price_mult"]),
        cii_version=task["cii_v"], fueleu_version=task["fueleu_v"], ets_version=task["ets_v"],
        fuel_factors=_WRK["factors"], extended_fuels=True, blend=True,
        exclude_fuels=spec["exclude"] or None, max_biofuel_share=spec["cap"],
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
    opt["eua_price_eur_per_tco2"] = task["eua"]
    return opt


def _uses_fame(s: pd.Series) -> float:
    return round(float((s == FuelType.BIODIESEL_FAME.value).mean()), 4)


def _uses_biofuel(g: pd.DataFrame, col: str) -> float:
    return round(float(g[col].isin([f.value for f in _BIOFUELS]).mean()), 4)


def _agg(g: pd.DataFrame) -> dict:
    dom_s, dom_j = g["siloed_fuel"].mode(), g["joint_fuel"].mode()
    return {
        "n_vessels": int(len(g)),
        "mean_rci_silo": round(float(g["rci"].mean()), 5),
        "median_rci_silo": round(float(g["rci"].median()), 5),
        "p90_rci_silo": round(float(g["rci"].quantile(0.90)), 5),
        "fleet_delta_cost_eur": round(float(g["delta_cost_eur"].sum()), 2),
        "mean_delta_cost_eur": round(float(g["delta_cost_eur"].mean()), 2),
        "median_delta_cost_eur": round(float(g["delta_cost_eur"].median()), 2),
        "p90_delta_cost_eur": round(float(g["delta_cost_eur"].quantile(0.90)), 2),
        "mean_low_carbon_share_silo": round(float(g["low_carbon_share_siloed"].mean()), 4),
        "mean_low_carbon_share_joint": round(float(g["low_carbon_share_joint"].mean()), 4),
        "mean_biofuel_share_silo": round(float(g["biofuel_share_siloed"].mean()), 4),
        "mean_biofuel_share_joint": round(float(g["biofuel_share_joint"].mean()), 4),
        "dominant_fuel_silo": str(dom_s.iloc[0]) if len(dom_s) else "",
        "dominant_fuel_joint": str(dom_j.iloc[0]) if len(dom_j) else "",
        "pct_vessels_using_FAME_silo": _uses_fame(g["siloed_fuel"]),
        "pct_vessels_using_FAME_joint": _uses_fame(g["joint_fuel"]),
        "pct_vessels_using_any_biofuel_silo": _uses_biofuel(g, "siloed_fuel"),
        "pct_vessels_using_any_biofuel_joint": _uses_biofuel(g, "joint_fuel"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Biofuel/FAME availability sensitivity")
    ap.add_argument("--mrv", required=True, type=Path)
    ap.add_argument("--fleet", required=True, type=Path)
    ap.add_argument("--scenario", required=True, type=Path)
    ap.add_argument("--years", default="2035,2045")
    ap.add_argument("--price-scenarios", default="base_2026")
    ap.add_argument("--eua", type=float, default=80.0)
    ap.add_argument("--out", type=Path,
                    default=Path("results/optimization/biofuel_sensitivity.csv"))
    ap.add_argument("--out-by-type", type=Path,
                    default=Path("results/optimization/biofuel_sensitivity_by_type.csv"))
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
    log.info("biofuel sensitivity: %d runs (cases=%d years=%s scenarios=%s) | workers=%d",
             n, len(_CASES), years, price_scenarios, nw)

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
            log.info("[%d/%d] %-26s ps=%-14s yr=%d  rci_mean=%.4f",
                     done, n, t["case"], t["ps"], t["year"], df["rci"].mean())

    if not frames:
        log.error("no results")
        sys.exit(1)
    big = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for (case, ps, yr), g in big.groupby(["case", "fuel_price_scenario", "year"]):
        rows.append({"case": case, "year": yr, "eua_price_eur_per_tco2": args.eua,
                     "fuel_price_scenario": ps, **_agg(g)})
    pd.DataFrame(rows).sort_values(["fuel_price_scenario", "year", "case"]).to_csv(
        args.out, index=False)

    rows = []
    for (case, yr, st), g in big.groupby(["case", "year", "ship_type"]):
        dom_j = g["joint_fuel"].mode()
        rows.append({"case": case, "year": yr, "ship_type": st, "n_vessels": int(len(g)),
                     "mean_rci_silo": round(float(g["rci"].mean()), 5),
                     "fleet_delta_cost_eur": round(float(g["delta_cost_eur"].sum()), 2),
                     "mean_biofuel_share_joint": round(float(g["biofuel_share_joint"].mean()), 4),
                     "dominant_fuel_joint": str(dom_j.iloc[0]) if len(dom_j) else ""})
    pd.DataFrame(rows).to_csv(args.out_by_type, index=False)
    log.info("done → %s (+ by_type)", args.out)


if __name__ == "__main__":
    main()
