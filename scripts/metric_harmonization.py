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

_CASES = {
    "M0_actual_full_policy_mix": (True, "ttw_ets"),
    "M1_actual_ETS_FuelEU_only": (False, "ttw_ets"),
    "M2_harmonized_WtW_ranking": (False, "wtw_proxy"),
    "M3_same_stringency_aligned_ranking": (True, "wtw_proxy"),
}

_KEEP = [
    "vessel_imo", "ship_type", "year", "rci", "rci_joint", "delta_cost_eur",
    "joint_fuel", "siloed_fuel", "joint_blend_pct", "siloed_blend_pct",
    "low_carbon_share_joint", "low_carbon_share_siloed",
    "fuel_cost_siloed_eur", "fuel_cost_joint_eur",
    "ets_cost_siloed_eur", "ets_cost_joint_eur",
    "fueleu_penalty_siloed_eur", "fueleu_penalty_joint_eur",
    "delay_cost_siloed_eur", "delay_cost_joint_eur",
    "cost_siloed_eur", "cost_joint_eur", "siloed_speed", "joint_speed",
    "joint_fuel_pathway", "siloed_fuel_pathway",
    "fueleu_energy_mj_joint", "fueleu_balance_joint", "fueleu_balance_siloed",
    "fueleu_actual_intensity_joint", "fueleu_actual_intensity_siloed",
    "biofuel_share_joint", "biofuel_share_siloed",
    "scope_weight", "annual_scale", "cost_basis",
]


def _legacy_pathway(alt: pd.Series, blend: pd.Series) -> pd.Series:
    out = "VLSFO+" + alt.astype(str)
    out = out.where(blend < 1.0 - 1e-9, alt.astype(str) + "_only")
    return out.where(blend > 1e-9, "VLSFO_only")


def _init_worker(mrv_pkl, fleet_pkl, factors_pkl, capability_pkl=None):
    _WRK["mrv_df"] = pickle.loads(mrv_pkl)
    _WRK["fleet_idx"] = pickle.loads(fleet_pkl)
    _WRK["factors"] = pickle.loads(factors_pkl)
    _WRK["fuel_capability"] = pickle.loads(capability_pkl) if capability_pkl else None


def _run_cell(task: dict) -> list[dict]:
    cii_obj, silo_basis = _CASES[task["case"]]
    return run_fuel_optimization_from_mrv(
        row_block(_WRK["mrv_df"], task), _WRK["fleet_idx"], task["year"], task["eua"],
        task["prices"],
        cii_version=task["cii_v"], fueleu_version=task["fueleu_v"], ets_version=task["ets_v"],
        fuel_factors=_WRK["factors"], extended_fuels=True, blend=True,
        include_cii_in_objective=cii_obj, silo_carbon_basis=silo_basis,
        fuel_feasibility=task.get("feasibility", "continuous_base"),
        fuel_capability=_WRK.get("fuel_capability"),
        **task.get("vkw", {}),
        return_rows=True,
    )


def _finish(task: dict, opt: pd.DataFrame) -> pd.DataFrame | None:
    if opt is None or opt.empty:
        return None
    keep = [c for c in _KEEP if c in opt.columns]
    opt = opt[keep].copy()
    if "joint_fuel_pathway" not in opt.columns:
        opt["joint_fuel_pathway"] = _legacy_pathway(opt["joint_fuel"], opt["joint_blend_pct"])
        opt["siloed_fuel_pathway"] = _legacy_pathway(opt["siloed_fuel"],
                                                     opt["siloed_blend_pct"])
    opt["case"] = task["case"]
    opt["fuel_price_scenario"] = task["ps"]
    opt["eua_price_eur_per_tco2"] = task["eua"]
    return opt


def _agg(g: pd.DataFrame) -> dict:
    dom_s, dom_j = g["siloed_fuel"].mode(), g["joint_fuel"].mode()
    return {
        "n_vessels": int(len(g)),
        "mean_rci_silo": round(float(g["rci"].mean()), 5),
        "median_rci_silo": round(float(g["rci"].median()), 5),
        "p90_rci_silo": round(float(g["rci"].quantile(0.90)), 5),
        "mean_rci_joint_norm": round(float(g["rci_joint"].mean()), 5),
        "fleet_delta_cost_eur": round(float(g["delta_cost_eur"].sum()), 2),
        "mean_delta_cost_eur": round(float(g["delta_cost_eur"].mean()), 2),
        "median_delta_cost_eur": round(float(g["delta_cost_eur"].median()), 2),
        "p90_delta_cost_eur": round(float(g["delta_cost_eur"].quantile(0.90)), 2),
        "mean_low_carbon_share_silo": round(float(g["low_carbon_share_siloed"].mean()), 4),
        "mean_low_carbon_share_joint": round(float(g["low_carbon_share_joint"].mean()), 4),
        "dominant_fuel_silo": str(dom_s.iloc[0]) if len(dom_s) else "",
        "dominant_fuel_joint": str(dom_j.iloc[0]) if len(dom_j) else "",
        "mean_ets_cost_silo_eur": round(float(g["ets_cost_siloed_eur"].mean()), 2),
        "mean_ets_cost_joint_eur": round(float(g["ets_cost_joint_eur"].mean()), 2),
        "mean_fueleu_penalty_silo_eur": round(float(g["fueleu_penalty_siloed_eur"].mean()), 2),
        "mean_fueleu_penalty_joint_eur": round(float(g["fueleu_penalty_joint_eur"].mean()), 2),
        "mean_fuel_cost_silo_eur": round(float(g["fuel_cost_siloed_eur"].mean()), 2),
        "mean_fuel_cost_joint_eur": round(float(g["fuel_cost_joint_eur"].mean()), 2),
        "mean_delay_cost_silo_eur": round(float(g["delay_cost_siloed_eur"].mean()), 2),
        "mean_delay_cost_joint_eur": round(float(g["delay_cost_joint_eur"].mean()), 2),
        **_distribution(g),
    }


def _distribution(g: pd.DataFrame) -> dict:
    r = g["rci"]
    path_s, path_j = g["siloed_fuel_pathway"].mode(), g["joint_fuel_pathway"].mode()
    d = {f"p{q}_rci_silo": round(float(r.quantile(q / 100)), 5)
         for q in (10, 25, 75, 95, 99)}
    d.update({
        "max_rci_silo": round(float(r.max()), 5),
        "share_rci_lt_0p01pct": round(float((r < 1e-4).mean()), 5),
        "share_rci_lt_1pct": round(float((r < 0.01).mean()), 5),
        "share_rci_gt_5pct": round(float((r > 0.05).mean()), 5),
        "share_rci_gt_10pct": round(float((r > 0.10).mean()), 5),
        "share_pathway_changed": round(float(
            (g["siloed_fuel_pathway"] != g["joint_fuel_pathway"]).mean()), 5),
        "modal_pathway_silo": str(path_s.iloc[0]) if len(path_s) else "",
        "modal_pathway_joint": str(path_j.iloc[0]) if len(path_j) else "",
        "aggregate_rci_silo": round(float(
            g["delta_cost_eur"].sum() / g["cost_siloed_eur"].sum()), 5),
        "cost_basis": str(g["cost_basis"].iloc[0]) if "cost_basis" in g else "leg",
    })
    if "fueleu_energy_mj_joint" in g:
        e = g["fueleu_energy_mj_joint"].fillna(0.0)
        d["fleet_fueleu_energy_joint_gj"] = round(float(e.sum() / 1e3), 1)
        d["fleet_biofuel_energy_joint_gj"] = round(float(
            (e * g["biofuel_share_joint"].fillna(0.0)).sum() / 1e3), 1)
    if "scope_weight" in g:
        d["mean_scope_weight"] = round(float(g["scope_weight"].mean()), 4)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description="Metric-harmonization counterfactual")
    ap.add_argument("--mrv", required=True, type=Path)
    ap.add_argument("--fleet", required=True, type=Path)
    ap.add_argument("--scenario", required=True, type=Path)
    ap.add_argument("--years", default="2026,2035,2045")
    ap.add_argument("--price-scenarios", default="base_2026")
    ap.add_argument("--eua", type=float, default=80.0)
    ap.add_argument("--out", type=Path,
                    default=Path("results/optimization/metric_harmonization.csv"))
    ap.add_argument("--out-by-type", type=Path,
                    default=Path("results/optimization/metric_harmonization_by_type.csv"))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out-vessel", type=Path, default=None,
                    help="optional vessel-level parquet (distribution analyses)")
    ap.add_argument("--fuel-feasibility", default="continuous_base",
                    help="fuel menu for both the shortcut and the joint plan "
                         "(continuous_base / compatible_switch / engine_capability / ...)")
    ap.add_argument("--fuel-capability", type=Path, default=None,
                    help="IMO fuel-capability CSV for --fuel-feasibility engine_capability")
    ap.add_argument("--min-scope-weight", type=float, default=None,
                    help="keep only vessels whose scope weight (the variant's scope mode) "
                         "is at least this value, e.g. 0.5 drops ships below the 100/50 floor")
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
    sample_scope = "all_valid"
    if args.min_scope_weight is not None:
        from carbon_twin.pipeline import _scope_weight

        keep = [_scope_weight(r, variant.scope_mode)[0] >= args.min_scope_weight - 1e-9
                for r in mrv_df.to_dict(orient="records")]
        log.info("scope weight >= %.3f: keeping %d of %d MRV rows",
                 args.min_scope_weight, sum(keep), len(keep))
        mrv_df = mrv_df[keep].reset_index(drop=True)
        sample_scope = f"scope_weight_ge_{args.min_scope_weight:g}"
    fleet_idx = vessels_to_index(load_vessels_csv(args.fleet))
    factors = variant.fuel_factors(load_fuel_factors_safe())
    prices_by = {ps: load_fuel_prices("configs/fuel_prices.yaml", scenario=ps)
                 for ps in price_scenarios}

    base = {"cii_v": regs["cii"]["version"],
            "fueleu_v": variant.fueleu_version_or(regs["fueleu"]["version"]),
            "ets_v": regs["eu_ets"]["version"], "eua": args.eua,
            "vkw": variant.fuel_optimizer_kwargs()}
    tasks = [
        {**base, "case": case, "ps": ps, "year": yr, "prices": prices_by[ps],
         "feasibility": args.fuel_feasibility}
        for case in _CASES for ps in price_scenarios for yr in years
    ]

    n = len(tasks)
    log.info("metric harmonization: %d runs (cases=%d years=%s scenarios=%s) | workers=%d",
             n, len(_CASES), years, price_scenarios, args.workers)

    pkls = (pickle.dumps(mrv_df), pickle.dumps(fleet_idx), pickle.dumps(factors),
            pickle.dumps(load_fuel_capability(args.fuel_capability)))
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
            log.info("[%d/%d] %-36s ps=%-14s yr=%d  rci_mean=%.4f",
                     done, n, t["case"], t["ps"], t["year"], df["rci"].mean())

    if not frames:
        log.error("no results")
        sys.exit(1)
    big = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out_vessel:
        big.assign(model_variant=variant.name).to_parquet(args.out_vessel, index=False)

    rows = []
    for (case, ps, yr), g in big.groupby(["case", "fuel_price_scenario", "year"]):
        rows.append({"case": case, "year": yr, "fuel_price_scenario": ps,
                     "eua_price_eur_per_tco2": args.eua, "sample_scope": sample_scope,
                     "fuel_feasibility": args.fuel_feasibility, **_agg(g)})
    pd.DataFrame(rows).sort_values(["fuel_price_scenario", "year", "case"]).to_csv(
        args.out, index=False)

    rows = []
    for (case, ps, yr, st), g in big.groupby(
            ["case", "fuel_price_scenario", "year", "ship_type"]):
        rows.append({"case": case, "year": yr, "fuel_price_scenario": ps, "ship_type": st,
                     "n_vessels": int(len(g)),
                     "mean_rci_silo": round(float(g["rci"].mean()), 5),
                     "fleet_delta_cost_eur": round(float(g["delta_cost_eur"].sum()), 2),
                     "mean_low_carbon_share_joint":
                         round(float(g["low_carbon_share_joint"].mean()), 4)})
    pd.DataFrame(rows).to_csv(args.out_by_type, index=False)
    log.info("done → %s (+ by_type)", args.out)


if __name__ == "__main__":
    main()
