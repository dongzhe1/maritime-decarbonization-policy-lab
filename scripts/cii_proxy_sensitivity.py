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
from carbon_twin.regulations.cii import CIIEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RATINGS = ["A", "B", "C", "D", "E"]
_NEAR = 0.05


def _bias_scenarios(path: Path) -> dict[str, dict[str, float]]:
    df = pd.read_csv(path)
    out: dict[str, dict[str, float]] = {}
    for _, r in df.iterrows():
        sc = str(r["scenario"])
        out.setdefault(sc, {})
        grp = str(r["ship_type_group"]).strip()
        types = ["__all__"] if grp == "all" else [t.strip() for t in grp.split(";")]
        for t in types:
            out[sc][t] = float(r["rho_multiplier"])
    return out


def _mult_for(scn: dict[str, float], ship_type: str) -> float:
    if ship_type in scn:
        return scn[ship_type]
    return scn.get("__all__", 1.0)


_WRK: dict = {}


def _init_worker(mrv_pkl, fleet_pkl, factors_pkl):
    _WRK["mrv_df"] = pickle.loads(mrv_pkl)
    _WRK["fleet_idx"] = pickle.loads(fleet_pkl)
    _WRK["factors"] = pickle.loads(factors_pkl)


def _run_speed(task: dict) -> list[dict]:
    return run_optimization_from_mrv(
        pd.DataFrame(),
        row_block(_WRK["mrv_df"], task),
        _WRK["fleet_idx"],
        task["year"],
        task["eua"],
        task["fuel_price"],
        cii_version=task["cii_v"],
        fueleu_version=task["fueleu_v"],
        ets_version=task["ets_v"],
        fuel_factors=_WRK["factors"],
        cii_pressure=task["k"],
        cii_rho_multiplier=task["mult"],
        **task.get("vkw", {}),
        return_rows=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="CII proxy scope-bias robustness")
    ap.add_argument("--mrv", required=True, type=Path)
    ap.add_argument("--fleet", required=True, type=Path)
    ap.add_argument("--scenario", required=True, type=Path)
    ap.add_argument("--compliance", required=True, type=Path)
    ap.add_argument(
        "--bias-config", type=Path, default=Path("configs/mrv_cii_proxy_bias_scenarios.csv")
    )
    ap.add_argument("--k-grid", default="0.2,2.0")
    ap.add_argument("--out-dir", type=Path, default=Path("results/cii_proxy"))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    add_variant_argument(ap)
    args = ap.parse_args()
    variant = load_variant(args.variant)
    write_variant_manifest(getattr(args, "out", None)
                           or Path(args.out_dir) / "run", variant)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scenario_cfg = _load_scenario(str(args.scenario))
    regs = scenario_cfg["regulations"]
    scenarios = _bias_scenarios(args.bias_config)

    vessels = load_vessels_csv(args.fleet)
    dwt = {str(v.imo_number): (v.deadweight or None) for v in vessels}

    comp = pd.read_parquet(args.compliance)
    rated = comp.dropna(subset=["cii_rating"]).copy()
    rated["dwt"] = rated["vessel_imo"].astype(str).map(dwt)
    eng = CIIEngine(version=regs["cii"]["version"])

    mix_rows, tail_rows = [], []
    for sc, scn in scenarios.items():
        rated["_mult"] = rated["ship_type"].map(lambda t: _mult_for(scn, t))
        rated["_rating_b"] = [
            eng._assign_rating(a * m, r, st, d).value
            for a, r, m, st, d in zip(
                rated["cii_attained"],
                rated["cii_required"],
                rated["_mult"],
                rated["ship_type"],
                rated["dwt"],
            )
        ]
        rated["_ratio_b"] = rated["cii_attained"] / rated["cii_required"] * rated["_mult"]
        for st, g in list(rated.groupby("ship_type")) + [("ALL", rated)]:
            n = len(g)
            share = {x: float((g["_rating_b"] == x).mean()) for x in RATINGS}
            mix_rows.append(
                {
                    "scenario": sc,
                    "ship_type": st,
                    "n": n,
                    **{f"{x}_share": round(share[x], 4) for x in RATINGS},
                    "DE_share": round(share["D"] + share["E"], 4),
                }
            )
            rcd = (
                g["ship_type"].map(lambda t: eng.cd_boundary_ratio(t, None))
                if st == "ALL"
                else eng.cd_boundary_ratio(st, None)
            )
            ratio = g["_ratio_b"]
            tail_rows.append(
                {
                    "scenario": sc,
                    "ship_type": st,
                    "n": n,
                    "rho_mean": round(float(ratio.mean()), 4),
                    "rho_p90": round(float(ratio.quantile(0.90)), 4),
                    "near_CD_share": round(float((abs(ratio - rcd) <= _NEAR).mean()), 4),
                    "above_CD_share": round(float((ratio > rcd).mean()), 4),
                }
            )
    pd.DataFrame(mix_rows).to_csv(args.out_dir / "cii_proxy_rating_mix.csv", index=False)
    pd.DataFrame(tail_rows).to_csv(args.out_dir / "cii_proxy_threshold_tail.csv", index=False)
    log.info("rating mix + threshold tail → %s (%d scenarios)", args.out_dir, len(scenarios))

    year = int(scenario_cfg["year"])
    eua = _resolve_eua_price(scenario_cfg, year)
    fuel_price = _resolve_fuel_price(scenario_cfg)
    ks = [float(x) for x in args.k_grid.split(",") if x.strip()]
    mrv_df = load_mrv_csv(args.mrv)
    fleet_idx = vessels_to_index(vessels)
    factors = variant.fuel_factors(load_fuel_factors_safe())

    tasks = []
    for sc, scn in scenarios.items():
        mult = {t: scn[t] for t in scn if t != "__all__"}
        if "__all__" in scn:
            mult = scn["__all__"]
        for k in ks:
            tasks.append(
                {
                    "scenario": sc,
                    "k": k,
                    "mult": mult,
                    "year": year,
                    "eua": eua,
                    "fuel_price": fuel_price,
                    "cii_v": regs["cii"]["version"],
                    "fueleu_v": variant.fueleu_version_or(regs["fueleu"]["version"]),
                    "vkw": variant.speed_optimizer_kwargs(),
                    "ets_v": regs["eu_ets"]["version"],
                }
            )

    nw = args.workers
    log.info("speed-RCI proxy sweep: %d runs (scenarios x k) | workers=%d", len(tasks), nw)
    pkls = (pickle.dumps(mrv_df), pickle.dumps(fleet_idx), pickle.dumps(factors))
    rows, done = [], 0
    for t, opt in run_blocked(tasks, _run_speed, workers=args.workers, n_rows=len(mrv_df),
                              initializer=_init_worker, initargs=pkls):
        if isinstance(opt, Exception):
            log.error("scenario=%s k=%s: %s", t["scenario"], t["k"], opt)
            continue
        sc, k = t["scenario"], t["k"]
        r = opt["rci"]
        rec = {
            "scenario": sc,
            "k": k,
            "n": int(len(r)),
            "rci_mean": round(float(r.mean()), 5),
            "rci_p90": round(float(r.quantile(0.90)), 5),
            "pct_rci_gt_10": round(float((r > 0.10).mean()), 4),
            "rci_max": round(float(r.max()), 5),
        }
        if "delta_cost_eur" in opt:
            rec["absolute_loss_total_eur"] = round(float(opt["delta_cost_eur"].sum()), 2)
        rows.append(rec)
        done += 1
        log.info(
            "[%d/%d] scenario=%-12s k=%.2f  rci_mean=%.5f  >10%%=%.2f%%",
            done,
            len(tasks),
            sc,
            k,
            rec["rci_mean"],
            rec["pct_rci_gt_10"] * 100,
        )
    pd.DataFrame(rows).sort_values(["scenario", "k"]).to_csv(
        args.out_dir / "speed_rci_cii_proxy_sensitivity.csv", index=False
    )
    log.info("done → %s", args.out_dir)


if __name__ == "__main__":
    main()
