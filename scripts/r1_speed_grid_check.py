from __future__ import annotations

import argparse
import functools
import os
import pickle
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall: pip install -e '.[all]'")

import carbon_twin.optimization.fuel_choice as fuel_choice
from carbon_twin.cli import _load_scenario, _resolve_eua_price
from carbon_twin.data.loaders import load_fuel_prices, load_mrv_csv, load_vessels_csv
from carbon_twin.emissions.proxy import load_fuel_factors_safe, vessels_to_index
from carbon_twin.model_variants import load_variant
from carbon_twin.parallel import row_block, run_blocked
from carbon_twin.pipeline import run_fuel_optimization_from_mrv

_ORIGINAL = fuel_choice.optimal_fuel_blend_speed_multi_regulation
_KEEP = ["vessel_imo", "rci", "siloed_speed", "joint_speed", "cost_siloed_eur", "cost_joint_eur"]
_WRK: dict = {}


def _init_worker(sample_pkl: bytes, fleet_pkl: bytes, common_pkl: bytes) -> None:
    _WRK["sample"] = pickle.loads(sample_pkl)
    _WRK["fleet"] = pickle.loads(fleet_pkl)
    _WRK["common"] = pickle.loads(common_pkl)


def _solve_block(task: dict) -> list[dict]:
    fuel_choice.optimal_fuel_blend_speed_multi_regulation = functools.partial(
        _ORIGINAL, n_points=task["n"])
    try:
        rows = run_fuel_optimization_from_mrv(row_block(_WRK["sample"], task), _WRK["fleet"],
                                              task["yr"], return_rows=True, **_WRK["common"])
    finally:
        fuel_choice.optimal_fuel_blend_speed_multi_regulation = _ORIGINAL
    return [{**{k: r.get(k) for k in _KEEP}, "n_points": task["n"], "year": task["yr"]}
            for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrv", required=True, type=Path)
    ap.add_argument("--fleet", required=True, type=Path)
    ap.add_argument("--scenario", required=True, type=Path)
    ap.add_argument("--variant", default="r1_lng_otto_ms_prio_fb_bd")
    ap.add_argument("--root", type=Path, default=Path("results/r1"))
    ap.add_argument("--years", default="2035,2045")
    ap.add_argument("--grids", default="15,29,57")
    ap.add_argument("--n-vessels", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--eua", type=float, default=80.0)
    ap.add_argument("--out-dir", type=Path, default=Path("results/r1/speed_grid_check"))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()

    variant = load_variant(args.variant)
    regs = _load_scenario(str(args.scenario))["regulations"]
    mrv = load_mrv_csv(args.mrv)
    fleet = vessels_to_index(load_vessels_csv(args.fleet))
    ref = pd.read_parquet(args.root / args.variant / "fuel_dimension.parquet",
                          columns=["vessel_imo", "year", "rci"])
    imo_col = next(c for c in ("imo", "imo_number", "IMO") if c in mrv.columns)
    pool = mrv[mrv[imo_col].astype(str).isin(set(ref.vessel_imo.astype(str)))]
    sample = pool.sample(n=min(args.n_vessels, len(pool)), random_state=args.seed)
    common = dict(
        eua_price=args.eua,
        fuel_prices=load_fuel_prices("configs/fuel_prices.yaml", scenario="base_2026"),
        cii_version=regs["cii"]["version"],
        fueleu_version=variant.fueleu_version_or(regs["fueleu"]["version"]),
        ets_version=regs["eu_ets"]["version"],
        fuel_factors=variant.fuel_factors(load_fuel_factors_safe()),
        extended_fuels=True, blend=True,
        **variant.fuel_optimizer_kwargs(), **variant.cii_kwargs())

    tasks = [{"yr": int(y), "n": int(g)} for y in args.years.split(",")
             for g in args.grids.split(",")]
    frames = []
    for t, out in run_blocked(tasks, _solve_block, workers=args.workers, n_rows=len(sample),
                              initializer=_init_worker,
                              initargs=(pickle.dumps(sample), pickle.dumps(fleet),
                                        pickle.dumps(common))):
        if isinstance(out, Exception):
            raise out
        frames.append(out)
        print(f"{t['yr']}: {t['n']} speed points done ({len(out)} vessels)", flush=True)
    res = pd.concat(frames, ignore_index=True)
    res["vessel_imo"] = res.vessel_imo.astype(str)
    ref["vessel_imo"] = ref.vessel_imo.astype(str)

    base = int(args.grids.split(",")[0])
    check = res[res.n_points.eq(base)].merge(ref, on=["vessel_imo", "year"],
                                             suffixes=("", "_reported"))
    max_dev = float((check.rci - check.rci_reported).abs().max())
    print(f"{len(check)} vessel-years at {base} points; largest deviation from the reported "
          f"regret: {max_dev:.2e}")

    wide = res.pivot_table(index=["vessel_imo", "year"], columns="n_points",
                           values=["rci", "siloed_speed", "joint_speed"])
    summary = []
    for yr in sorted(res.year.unique()):
        w = wide.xs(yr, level="year")
        for n in sorted(res.n_points.unique()):
            summary.append({
                "year": yr, "n_points": n, "n_vessels": int(w[("rci", n)].notna().sum()),
                "mean_regret_pct": 100 * float(w[("rci", n)].mean()),
                "max_abs_change_regret_pp": 100 * float((w[("rci", n)] - w[("rci", base)]).abs().max()),
                "mean_abs_speed_change_kn_joint": float((w[("joint_speed", n)]
                                                         - w[("joint_speed", base)]).abs().mean()),
            })
    summary = pd.DataFrame(summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out_dir / "speed_grid_vessels.csv", index=False)
    summary.to_csv(args.out_dir / "speed_grid_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"reproduction check: max |rci - reported| = {max_dev:.2e}")
    if not np.isfinite(max_dev) or max_dev > 1e-6:
        print("WARNING: the base grid does not reproduce the reported regret; check the inputs")


if __name__ == "__main__":
    main()
