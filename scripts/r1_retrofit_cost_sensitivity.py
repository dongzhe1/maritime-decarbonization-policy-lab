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
from carbon_twin.pipeline import load_fuel_capability, run_fuel_optimization_from_mrv

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_GAS_CAPABLE = {"gas_carrier", "lng_carrier"}
_WRK: dict = {}


def _init_worker(mrv_pkl, fleet_pkl, factors_pkl, capability_pkl=None):
    _WRK["mrv_by_group"] = pickle.loads(mrv_pkl)
    _WRK["fleet_idx"] = pickle.loads(fleet_pkl)
    _WRK["factors"] = pickle.loads(factors_pkl)
    _WRK["fuel_capability"] = pickle.loads(capability_pkl) if capability_pkl else None


def _premium_prices(prices: dict, factors: dict, fuels: list[str], premium: float) -> dict:
    out = dict(prices)
    for f in fuels:
        ef = factors.get(FuelType(f))
        if f in out and ef is not None:
            out[f] = out[f] + premium * ef.lcv_mj_per_kg
    return out


def _run_cell(task: dict) -> list[dict]:
    mrv_df = row_block(_WRK["mrv_by_group"][task["group"]], task)
    if mrv_df.empty:
        return []
    return run_fuel_optimization_from_mrv(
        mrv_df, _WRK["fleet_idx"], task["year"], task["eua"], task["prices"],
        cii_version=task["cii_v"], fueleu_version=task["fueleu_v"], ets_version=task["ets_v"],
        fuel_factors=_WRK["factors"], extended_fuels=True, blend=True,
        include_cii_in_objective=False, fuel_feasibility=task["menu"],
        **task["vkw"], **task.get("capex_kw", {}), return_rows=True,
    )


def _finish(task: dict, opt: pd.DataFrame) -> pd.DataFrame | None:
    if opt is None or opt.empty:
        return None
    for col in ("joint_fuel_pathway", "siloed_fuel_pathway"):
        if col not in opt:
            fuel, blend = ("joint_fuel", "joint_blend_pct") if col.startswith("joint") \
                else ("siloed_fuel", "siloed_blend_pct")
            lab = "VLSFO+" + opt[fuel].astype(str)
            lab = lab.where(opt[blend] < 1 - 1e-9, opt[fuel].astype(str) + "_only")
            opt[col] = lab.where(opt[blend] > 1e-9, "VLSFO_only")
    keep = ["vessel_imo", "ship_type", "rci", "delta_cost_eur", "cost_siloed_eur",
            "joint_fuel_pathway", "siloed_fuel_pathway", "low_carbon_share_joint"]
    opt = opt[[c for c in keep if c in opt]].copy()
    cap = _WRK.get("fuel_capability")
    imo = opt["vessel_imo"].astype(str).str.strip()
    if cap is not None:
        opt["lng_capable"] = imo.map(lambda i: bool(cap.get(i, {}).get("lng_capable")))
        opt["methanol_capable"] = imo.map(
            lambda i: bool(cap.get(i, {}).get("methanol_capable")))
    else:
        opt["lng_capable"] = opt["ship_type"].isin(_GAS_CAPABLE)
        opt["methanol_capable"] = False
    for plan in ("joint", "siloed"):
        fuels = opt[f"{plan}_fuel_pathway"].map(_pathway_fuels)
        opt[f"{plan}_uses_lng"] = fuels.map(lambda f: "LNG" in f)
        opt[f"{plan}_uses_methanol"] = fuels.map(
            lambda f: any(x.startswith("methanol") for x in f))
    opt[task.get("level_col", "premium_eur_per_gj")] = task["premium"]
    opt["year"] = task["year"]
    opt["menu"] = task["menu"]
    return opt


def _pathway_fuels(label: str) -> set[str]:
    return set(str(label).removesuffix("_only").split("+"))


def _share(mask: pd.Series) -> float:
    return round(float(mask.mean()), 5) if len(mask) else float("nan")


def _agg(g: pd.DataFrame) -> dict:
    ps, pj = g["siloed_fuel_pathway"].mode(), g["joint_fuel_pathway"].mode()
    usage = {}
    if "joint_uses_lng" in g:
        lc, mc = g["lng_capable"].astype(bool), g["methanol_capable"].astype(bool)
        usage = {
            "n_lng_capable": int(lc.sum()),
            "share_joint_uses_lng": _share(g["joint_uses_lng"]),
            "share_siloed_uses_lng": _share(g["siloed_uses_lng"]),
            "share_joint_uses_lng_capable": _share(g.loc[lc, "joint_uses_lng"]),
            "share_joint_uses_lng_noncapable": _share(g.loc[~lc, "joint_uses_lng"]),
            "share_siloed_uses_lng_noncapable": _share(g.loc[~lc, "siloed_uses_lng"]),
            "share_joint_uses_methanol": _share(g["joint_uses_methanol"]),
            "share_joint_uses_methanol_noncapable": _share(g.loc[~mc, "joint_uses_methanol"]),
        }
    return {
        "n_vessels": int(len(g)),
        "mean_rci_silo": round(float(g["rci"].mean()), 5),
        "median_rci_silo": round(float(g["rci"].median()), 5),
        "p90_rci_silo": round(float(g["rci"].quantile(0.9)), 5),
        "aggregate_rci_silo": round(
            float(g["delta_cost_eur"].sum() / g["cost_siloed_eur"].sum()), 5),
        "fleet_delta_cost_eur": round(float(g["delta_cost_eur"].sum()), 2),
        "share_pathway_changed": round(float(
            (g["siloed_fuel_pathway"] != g["joint_fuel_pathway"]).mean()), 5),
        "modal_pathway_silo": str(ps.iloc[0]) if len(ps) else "",
        "modal_pathway_joint": str(pj.iloc[0]) if len(pj) else "",
        "mean_low_carbon_share_joint": round(float(g["low_carbon_share_joint"].mean()), 4),
        **usage,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrofit-cost sensitivity")
    ap.add_argument("--mrv", required=True, type=Path)
    ap.add_argument("--fleet", required=True, type=Path)
    ap.add_argument("--scenario", required=True, type=Path)
    ap.add_argument("--years", default="2035,2045")
    ap.add_argument("--premiums", default="0,2,5,10,15", help="EUR per GJ of retrofit fuel")
    ap.add_argument("--mode", choices=["premium", "capex"], default="premium")
    ap.add_argument("--capex-per-kw", default="0,250,500,1000,1500",
                    help="--mode capex: retrofit capex levels, EUR per kW main-engine power")
    ap.add_argument("--capex-crf", type=float, default=0.15,
                    help="capital recovery factor (0.15 ≈ 10 years at 8%%)")
    ap.add_argument("--fuel-capability", type=Path, default=None)
    ap.add_argument("--retrofit-fuels", default="LNG,methanol_bio,methanol_e")
    ap.add_argument("--menus", default="continuous_base,compatible_switch")
    ap.add_argument("--price-scenario", default="base_2026")
    ap.add_argument("--eua", type=float, default=80.0)
    ap.add_argument("--out", type=Path, default=Path("results/r1/retrofit_cost.csv"))
    ap.add_argument("--out-by-type", type=Path, default=None)
    ap.add_argument("--out-vessel", type=Path, default=None,
                    help="optional vessel-level parquet (pathways and capability flags)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    add_variant_argument(ap)
    args = ap.parse_args()
    variant = load_variant(args.variant)
    write_variant_manifest(args.out, variant)

    regs = _load_scenario(str(args.scenario))["regulations"]
    mrv_df = load_mrv_csv(args.mrv)
    fleet_idx = vessels_to_index(load_vessels_csv(args.fleet))
    factors = variant.fuel_factors(load_fuel_factors_safe())
    base_prices = load_fuel_prices("configs/fuel_prices.yaml", scenario=args.price_scenario)
    retrofit = [f.strip() for f in args.retrofit_fuels.split(",") if f.strip()]

    imo = mrv_df["imo_number"].astype(str).str.strip()
    gas = imo.map(lambda i: fleet_idx[i].ship_type.value in _GAS_CAPABLE
                  if i in fleet_idx else False)
    groups = {"gas_capable": mrv_df[gas], "other": mrv_df[~gas]}
    capability = load_fuel_capability(args.fuel_capability)

    tasks = []
    if args.mode == "capex":
        groups = {"all": mrv_df}
        for level in [float(x) for x in args.capex_per_kw.split(",")]:
            for yr in [int(y) for y in args.years.split(",")]:
                for menu in [m.strip() for m in args.menus.split(",")]:
                    tasks.append({
                        "premium": level, "level_col": "capex_eur_per_kw", "group": "all",
                        "year": yr, "menu": menu, "prices": base_prices, "eua": args.eua,
                        "cii_v": regs["cii"]["version"],
                        "fueleu_v": variant.fueleu_version_or(regs["fueleu"]["version"]),
                        "ets_v": regs["eu_ets"]["version"],
                        "vkw": variant.fuel_optimizer_kwargs(),
                        "capex_kw": {
                            "retrofit_capex_eur_per_kw": {f: level for f in retrofit},
                            "capex_crf": args.capex_crf, "fuel_capability": capability},
                    })
    for premium in ([float(x) for x in args.premiums.split(",")]
                    if args.mode == "premium" else []):
        for group in groups:
            fuels = [f for f in retrofit if not (group == "gas_capable" and f == "LNG")]
            prices = _premium_prices(base_prices, factors, fuels, premium)
            for yr in [int(y) for y in args.years.split(",")]:
                for menu in [m.strip() for m in args.menus.split(",")]:
                    tasks.append({
                        "premium": premium, "group": group, "year": yr, "menu": menu,
                        "prices": prices, "eua": args.eua,
                        "cii_v": regs["cii"]["version"],
                        "fueleu_v": variant.fueleu_version_or(regs["fueleu"]["version"]),
                        "ets_v": regs["eu_ets"]["version"],
                        "vkw": variant.fuel_optimizer_kwargs(),
                    })

    pkls = (pickle.dumps(groups), pickle.dumps(fleet_idx), pickle.dumps(factors),
            pickle.dumps(capability))
    _WRK["fuel_capability"] = capability
    frames = []
    blocked = run_blocked(tasks, _run_cell, workers=args.workers, n_rows=len(mrv_df),
                          initializer=_init_worker, initargs=pkls)
    for i, (t, opt) in enumerate(blocked, 1):
        if isinstance(opt, Exception):
            raise opt
        df = _finish(t, opt)
        if df is not None:
            frames.append(df)
        log.info("[%d/%d] premium=%.1f group=%s yr=%d menu=%s", i, len(tasks),
                 t["premium"], t["group"], t["year"], t["menu"])
    if not frames:
        sys.exit("no results")
    big = pd.concat(frames, ignore_index=True)
    keys = ["menu", "year",
            "capex_eur_per_kw" if args.mode == "capex" else "premium_eur_per_gj"]
    rows = [{**dict(zip(keys, k)), "model_variant": variant.name, **_agg(g)}
            for k, g in big.groupby(keys)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    if args.out_vessel:
        big.assign(model_variant=variant.name).to_parquet(args.out_vessel, index=False)
    if args.out_by_type:
        rows = [{**dict(zip(keys + ["ship_type"], k)), **_agg(g)}
                for k, g in big.groupby(keys + ["ship_type"])]
        pd.DataFrame(rows).to_csv(args.out_by_type, index=False)
    log.info("done -> %s", args.out)


if __name__ == "__main__":
    main()
