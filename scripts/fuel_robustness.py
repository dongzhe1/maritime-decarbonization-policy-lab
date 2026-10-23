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
from carbon_twin.pipeline import load_fuel_capability, run_fuel_optimization_from_mrv

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_WRK: dict = {}
_KEEP = [
    "vessel_imo",
    "ship_type",
    "year",
    "rci",
    "rci_joint",
    "delta_cost_eur",
    "joint_fuel",
    "siloed_fuel",
    "joint_blend_pct",
    "siloed_blend_pct",
    "fuel_cost_siloed_eur",
    "fuel_cost_joint_eur",
    "ets_cost_siloed_eur",
    "ets_cost_joint_eur",
    "fueleu_penalty_siloed_eur",
    "fueleu_penalty_joint_eur",
    "cii_consequence_siloed_eur",
    "cii_consequence_joint_eur",
    "delay_cost_siloed_eur",
    "delay_cost_joint_eur",
]


def _init_worker(mrv_pkl, fleet_pkl, factors_pkl, valid_pkl):
    _WRK["mrv_df"] = pickle.loads(mrv_pkl)
    _WRK["fleet_idx"] = pickle.loads(fleet_pkl)
    _WRK["factors"] = pickle.loads(factors_pkl)
    _WRK["cii_valid"] = pickle.loads(valid_pkl)


def _run_cell(task: dict) -> list[dict]:
    return run_fuel_optimization_from_mrv(
        row_block(_WRK["mrv_df"], task),
        _WRK["fleet_idx"],
        task["year"],
        task["eua"],
        task["prices"],
        cii_version=task["cii_v"],
        fueleu_version=task["fueleu_v"],
        ets_version=task["ets_v"],
        fuel_factors=_WRK["factors"],
        extended_fuels=True,
        blend=True,
        cii_pressure=task["cii_pressure"],
        fuel_feasibility=task["feasibility"],
        fuel_capability=task.get("fuel_capability"),
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
    opt["feasibility"] = task["feasibility"]
    opt["cii_pressure"] = task["cii_pressure"]
    opt["price_scenario"] = task["ps"]
    opt["eua"] = task["eua"]
    opt["cii_valid"] = opt["vessel_imo"].isin(_WRK["cii_valid"])
    return opt


def _agg_rci(g: pd.DataFrame) -> dict:
    return {
        "n": int(len(g)),
        "rci_silo_mean": round(float(g["rci"].mean()), 5),
        "rci_silo_median": round(float(g["rci"].median()), 5),
        "rci_silo_p90": round(float(g["rci"].quantile(0.90)), 5),
        "rci_joint_mean": round(float(g["rci_joint"].mean()), 5),
        "rci_joint_p90": round(float(g["rci_joint"].quantile(0.90)), 5),
        "delta_cost_total_eur": round(float(g["delta_cost_eur"].sum()), 2),
        "delta_cost_median_eur": round(float(g["delta_cost_eur"].median()), 2),
        "delta_cost_p90_eur": round(float(g["delta_cost_eur"].quantile(0.90)), 2),
        "mean_lowcarbon_share_silo": round(float(g["siloed_blend_pct"].mean()), 4),
        "mean_lowcarbon_share_joint": round(float(g["joint_blend_pct"].mean()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fuel-side RCI robustness + CII dependence")
    ap.add_argument("--mrv", required=True, type=Path)
    ap.add_argument("--fleet", required=True, type=Path)
    ap.add_argument("--scenario", required=True, type=Path)
    ap.add_argument(
        "--compliance",
        required=True,
        type=Path,
        help="compliance.parquet, for the CII-valid common sample",
    )
    ap.add_argument("--years", default="2026,2035,2045")
    ap.add_argument("--price-scenarios", default="base_2026,high_fossil_2026,low_alternative_2030")
    ap.add_argument("--eua", type=float, default=80.0)
    ap.add_argument("--feasibilities", default="continuous_base,dropin_only,compatible_switch")
    ap.add_argument("--fuel-capability", type=Path, default=None,
                    help="IMO fuel-capability CSV; adds the engine_capability menu")
    ap.add_argument("--out-dir", type=Path, default=Path("results/fuel_robustness"))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    add_variant_argument(ap)
    args = ap.parse_args()
    variant = load_variant(args.variant)
    write_variant_manifest(getattr(args, "out", None)
                           or Path(args.out_dir) / "run", variant)

    years = [int(y) for y in args.years.split(",") if y.strip()]
    price_scenarios = [s.strip() for s in args.price_scenarios.split(",") if s.strip()]
    feasibilities = [s.strip() for s in args.feasibilities.split(",") if s.strip()]
    fuel_capability = load_fuel_capability(args.fuel_capability)
    if fuel_capability and "engine_capability" not in feasibilities:
        feasibilities.append("engine_capability")

    scenario_cfg = _load_scenario(str(args.scenario))
    regs = scenario_cfg["regulations"]
    mrv_df = load_mrv_csv(args.mrv)
    fleet_idx = vessels_to_index(load_vessels_csv(args.fleet))
    factors = variant.fuel_factors(load_fuel_factors_safe())
    prices_by = {
        ps: load_fuel_prices("configs/fuel_prices.yaml", scenario=ps) for ps in price_scenarios
    }

    comp = pd.read_parquet(args.compliance)
    cii_valid = set(comp.loc[comp["cii_rating"].notna(), "vessel_imo"].astype(str))

    from carbon_twin.optimization.speed import _DEFAULT_CII_PRESSURE

    k_def = _DEFAULT_CII_PRESSURE

    tasks = []
    base = {
        "cii_v": regs["cii"]["version"],
        "fueleu_v": variant.fueleu_version_or(regs["fueleu"]["version"]),
        "vkw": variant.fuel_optimizer_kwargs(),
        "ckw": variant.cii_kwargs(),
        "ets_v": regs["eu_ets"]["version"],
        "eua": args.eua,
        "fuel_capability": fuel_capability,
    }
    for ps in price_scenarios:
        for yr in years:
            common = {**base, "ps": ps, "year": yr, "prices": prices_by[ps]}
            for feas in feasibilities:
                tasks.append(
                    {**common, "feasibility": feas, "cii_pressure": k_def, "case": f"{feas}_full"}
                )
            tasks.append(
                {
                    **common,
                    "feasibility": "continuous_base",
                    "cii_pressure": 0.0,
                    "case": "no_cii_consequence_k0",
                }
            )

    n = len(tasks)
    nw = args.workers
    log.info(
        "fuel robustness: %d runs (years=%s scenarios=%s feasibilities=%s) | workers=%d",
        n,
        years,
        price_scenarios,
        feasibilities,
        nw,
    )

    pkls = (
        pickle.dumps(mrv_df),
        pickle.dumps(fleet_idx),
        pickle.dumps(factors),
        pickle.dumps(cii_valid),
    )
    _WRK["cii_valid"] = cii_valid
    frames = []
    done = 0
    for t, opt in run_blocked(tasks, _run_cell, workers=args.workers, n_rows=len(mrv_df),
                              initializer=_init_worker, initargs=pkls):
        if isinstance(opt, Exception):
            log.error("case=%s ps=%s yr=%s: %s", t["case"], t["ps"], t["year"], opt)
            continue
        df = _finish(t, opt)
        if df is not None:
            frames.append(df)
            done += 1
            log.info(
                "[%d/%d] %-26s ps=%-22s yr=%d  rci_mean=%.3f",
                done,
                n,
                t["case"],
                t["ps"],
                t["year"],
                df["rci"].mean(),
            )

    if not frames:
        log.error("no results")
        sys.exit(1)
    big = pd.concat(frames, ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fs = big[big["cii_pressure"] == k_def]
    rows = []
    for (feas, ps, yr), g in fs.groupby(["feasibility", "price_scenario", "year"]):
        rows.append(
            {
                "fuel_scenario": feas,
                "price_scenario": ps,
                "year": yr,
                "eua": args.eua,
                **_agg_rci(g),
            }
        )
    pd.DataFrame(rows).to_csv(args.out_dir / "fuel_rci_sensitivity.csv", index=False)

    dep = big[big["feasibility"] == "continuous_base"]
    rows = []
    for case, label in [(k_def, "full_policy_mix"), (0.0, "no_cii_consequence_k0")]:
        sub = dep[dep["cii_pressure"] == case]
        for scope, gg in [("all_valid", sub), ("common_cii_valid", sub[sub["cii_valid"]])]:
            for (ps, yr), g in gg.groupby(["price_scenario", "year"]):
                dom_s = g["siloed_fuel"].mode()
                dom_j = g["joint_fuel"].mode()
                rows.append(
                    {
                        "scenario": label,
                        "sample_scope": scope,
                        "year": yr,
                        "eua_price": args.eua,
                        "fuel_price_scenario": ps,
                        **_agg_rci(g),
                        "mean_fueleu_penalty_silo_eur": round(
                            float(g["fueleu_penalty_siloed_eur"].mean()), 2
                        ),
                        "mean_fueleu_penalty_joint_eur": round(
                            float(g["fueleu_penalty_joint_eur"].mean()), 2
                        ),
                        "mean_ets_cost_silo_eur": round(float(g["ets_cost_siloed_eur"].mean()), 2),
                        "mean_ets_cost_joint_eur": round(float(g["ets_cost_joint_eur"].mean()), 2),
                        "mean_cii_consequence_silo_eur": round(
                            float(g["cii_consequence_siloed_eur"].mean()), 2
                        ),
                        "mean_cii_consequence_joint_eur": round(
                            float(g["cii_consequence_joint_eur"].mean()), 2
                        ),
                        "dominant_low_carbon_fuel_silo": str(dom_s.iloc[0]) if len(dom_s) else "",
                        "dominant_low_carbon_fuel_joint": str(dom_j.iloc[0]) if len(dom_j) else "",
                    }
                )
    pd.DataFrame(rows).to_csv(args.out_dir / "fuel_rci_cii_dependence.csv", index=False)

    rows = []
    for (feas, ps, yr, st), g in fs.groupby(["feasibility", "price_scenario", "year", "ship_type"]):
        rows.append(
            {
                "fuel_scenario": feas,
                "price_scenario": ps,
                "year": yr,
                "ship_type": st,
                "n": int(len(g)),
                "delta_fuel_cost_eur": round(
                    float((g["fuel_cost_siloed_eur"] - g["fuel_cost_joint_eur"]).sum()), 2
                ),
                "delta_ets_cost_eur": round(
                    float((g["ets_cost_siloed_eur"] - g["ets_cost_joint_eur"]).sum()), 2
                ),
                "delta_fueleu_penalty_eur": round(
                    float((g["fueleu_penalty_siloed_eur"] - g["fueleu_penalty_joint_eur"]).sum()), 2
                ),
                "delta_cii_consequence_eur": round(
                    float((g["cii_consequence_siloed_eur"] - g["cii_consequence_joint_eur"]).sum()),
                    2,
                ),
                "delta_delay_cost_eur": round(
                    float((g["delay_cost_siloed_eur"] - g["delay_cost_joint_eur"]).sum()), 2
                ),
                "delta_total_eur": round(float(g["delta_cost_eur"].sum()), 2),
            }
        )
    pd.DataFrame(rows).to_csv(args.out_dir / "fuel_choice_decomposition.csv", index=False)

    log.info("done → %s (3 CSVs)", args.out_dir)


if __name__ == "__main__":
    main()
