from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall: pip install -e '.[all]'")

from carbon_twin.data.loaders import load_vessels_csv
from carbon_twin.data.schemas import FuelType
from carbon_twin.pipeline import feasible_fuel_options
from scripts.fueleu_internalization_sweep import (
    _CASE_SETTINGS,
    _file_sha256,
    _git_info,
    _write_atomic_csv,
    _write_atomic_parquet,
    _write_atomic_text,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent
_SWEEP_PY = Path(__file__).resolve().parent / "fueleu_internalization_sweep.py"

_VARIANT_ARGS: list[str] = []
_CAPABILITY_PATH: list[Path] = []

_LAST_CELL_INFO: dict = {}

_SCENARIOS: dict[str, dict] = {
    "CENTRAL_EXTENDED": {
        "years": [2035, 2045], "fuel_feasibility": "continuous_base",
        "exclude_fuels": None,
    },
    "COMPATIBILITY_CONSTRAINED_2035": {
        "years": [2035], "fuel_feasibility": "compatible_switch",
        "exclude_fuels": None,
    },
    "NO_FAME_2045": {
        "years": [2045], "fuel_feasibility": "continuous_base",
        "exclude_fuels": "biodiesel_fame",
    },
    "COMPATIBILITY_ENGINE": {
        "years": [2035, 2045], "fuel_feasibility": "engine_capability",
        "exclude_fuels": None,
    },
}

_FLEET_CONDITIONS: list[tuple[str, float, str]] = [
    ("fleet_regret_closed_fraction", 0.50, "ge"),
    ("fleet_regret_closed_fraction", 0.90, "ge"),
    ("fleet_regret_closed_fraction", 0.99, "ge"),
    ("fleet_regret_closed_fraction", 0.999, "ge"),
    ("fuel_pathway_label_agreement_share", 0.50, "ge"),
    ("fuel_pathway_label_agreement_share", 0.90, "ge"),
    ("fuel_pathway_label_agreement_share", 0.99, "ge"),
    ("fuel_plan_agreement_1pp_share", 0.50, "ge"),
    ("fuel_plan_agreement_1pp_share", 0.90, "ge"),
    ("fuel_plan_agreement_1pp_share", 0.99, "ge"),
    ("mean_rci_proxy", 0.05, "le"),
    ("mean_rci_proxy", 0.01, "le"),
    ("mean_rci_proxy", 0.005, "le"),
]

_VESSEL_CONDITIONS: list[tuple[str, str]] = [
    ("fuel_pathway_label_match", "fuel_pathway_label_match"),
    ("fuel_plan_match_1pp", "fuel_plan_agreement_1pp"),
    ("full_decision_exact", "full_decision_exact"),
    ("full_decision_1pp_0p01kn", "full_decision_1pp_0p01kn"),
]

CURVE_GROUP_KEYS = [
    "scenario_id", "case", "year", "blend_steps",
    "fuel_price_scenario", "eua_price_eur_per_tco2",
    "feasibility_mode", "exclude_fuels",
    "include_cii_in_planner", "include_cii_in_evaluation",
    "fueleu_price_fraction", "fueleu_evaluation_multiplier",
]
BY_TYPE_GROUP_KEYS = CURVE_GROUP_KEYS + ["ship_type"]

_BASELINE_ZERO_EUR = 1e-3
_CLOSURE_SNAP = 1e-12


def recompute_global_closure_metrics(
    df: pd.DataFrame,
    group_keys: list[str],
    delta_col: str = "fleet_delta_cost_eur",
) -> pd.DataFrame:
    if not len(df):
        return df.copy()
    missing_keys = [k for k in group_keys if k not in df.columns]
    if missing_keys:
        raise ValueError(
            f"recompute_global_closure_metrics: grouping columns absent from the "
            f"frame: {missing_keys}")

    parts = []
    for key, g in df.groupby(group_keys, dropna=False):
        g = g.sort_values("lambda_fueleu").copy()
        anchor = g[np.isclose(g["lambda_fueleu"].astype(float), 0.0, atol=1e-12)]
        if len(anchor) == 0:
            raise ValueError(
                f"recompute_global_closure_metrics: no lambda=0 anchor for group "
                f"{dict(zip(group_keys, key if isinstance(key, tuple) else (key,)))}"
                f" (tested lambdas: {sorted(g['lambda_fueleu'].unique())[:6]}...). "
                "Closure cannot be normalized without a same-resolution lambda=0 "
                "row -- run the anchor cell for this group before finalizing.")
        if len(anchor) > 1:
            raise ValueError(
                f"recompute_global_closure_metrics: {len(anchor)} lambda=0 rows for "
                f"group {dict(zip(group_keys, key if isinstance(key, tuple) else (key,)))}"
                " -- deduplicate before recomputing closure.")

        d0 = float(anchor[delta_col].iloc[0])
        if abs(d0) <= _BASELINE_ZERO_EUR:
            g["fleet_regret_remaining_fraction"] = np.nan
            g["fleet_regret_closed_fraction"] = np.nan
            g["baseline_regret_zero"] = True
        else:
            remaining = g[delta_col].astype(float) / d0
            closed = 1.0 - remaining
            remaining = remaining.mask(remaining.abs() < _CLOSURE_SNAP, 0.0)
            closed = closed.mask(closed.abs() < _CLOSURE_SNAP, 0.0)
            g["fleet_regret_remaining_fraction"] = remaining
            g["fleet_regret_closed_fraction"] = closed
            g["baseline_regret_zero"] = False
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _dedupe_cells(df: pd.DataFrame, extra_keys: tuple[str, ...] = ()) -> pd.DataFrame:
    df = df.copy()
    df["_lam_r"] = df["lambda_fueleu"].round(6)
    key = ["scenario_id", "case", "year", "blend_steps", "_lam_r", *extra_keys]
    return df.drop_duplicates(subset=key, keep="last").drop(columns="_lam_r")


def _window_lambdas(lo: float, hi: float, step: float) -> list[float]:
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + step * i, 6) for i in range(n)]


def _run_sweep_cell(
    *, mrv: Path, fleet: Path, scenario_config: Path, scenario_id: str,
    year: int, lambdas: list[float], blend_steps: int,
    fuel_feasibility: str, exclude_fuels: str | None,
    fuel_price_scenario: str, eua: float, case: str,
    workers: int, sample_limit: int, cell_dir: Path, cell_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir = cell_dir / cell_label
    cmd = [
        sys.executable, str(_SWEEP_PY),
        "--mrv", str(mrv), "--fleet", str(fleet), "--scenario", str(scenario_config),
        "--years", str(year),
        "--lambdas", ",".join(f"{x:.4f}" for x in lambdas),
        "--fuel-price-scenario", fuel_price_scenario,
        "--eua", str(eua), "--case", case,
        "--output-dir", str(out_dir), "--workers", str(workers),
        "--blend-steps", str(blend_steps),
        "--fuel-feasibility", fuel_feasibility,
        "--scenario-id", scenario_id,
        "--skip-m1-regression",
    ]
    if exclude_fuels:
        cmd += ["--exclude-fuels", exclude_fuels]
    if sample_limit:
        cmd += ["--sample-limit", str(sample_limit)]
    cmd += _VARIANT_ARGS
    if _CAPABILITY_PATH:
        cmd += ["--fuel-capability", str(_CAPABILITY_PATH[0])]
    log.info("cell %-40s year=%d lambdas=%d blend_steps=%d",
             cell_label, year, len(lambdas), blend_steps)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        log.error("cell %s FAILED (exit %d) in %.1fs\n%s", cell_label,
                  r.returncode, dt, (r.stdout[-2000:] + r.stderr[-2000:]))
        raise SystemExit(f"sweep cell failed: {cell_label}")
    all_curve = pd.read_csv(out_dir / "fueleu_internalization_all_curve.csv")
    vessel = pd.read_parquet(out_dir / "fueleu_internalization_vessel.parquet")
    by_type = pd.read_csv(out_dir / "fueleu_internalization_by_type.csv")
    _LAST_CELL_INFO.clear()
    _LAST_CELL_INFO.update({
        "cell_label": cell_label,
        "lambda_grid": [round(float(x), 6) for x in lambdas],
        "runtime_s": round(dt, 1),
        "output_sha256": {
            p.name: _file_sha256(p) for p in (
                out_dir / "fueleu_internalization_all_curve.csv",
                out_dir / "fueleu_internalization_vessel.parquet",
                out_dir / "fueleu_internalization_by_type.csv",
            ) if p.exists()
        },
    })
    log.info("cell %s done in %.1fs (%d curve rows)", cell_label, dt, len(all_curve))
    return all_curve, vessel, by_type


def _bracket_for_condition(g: pd.DataFrame, col: str, thr: float,
                           direction: str) -> tuple[float, float] | None:
    vals = g[col].to_numpy(dtype=float)
    lams = g["lambda_fueleu"].to_numpy(dtype=float)
    cond = (vals >= thr) if direction == "ge" else (vals <= thr)
    cond = np.where(np.isnan(vals), False, cond)
    if len(cond) == 0 or cond[0]:
        return None
    hit = np.where(cond)[0]
    if len(hit) == 0:
        return None
    i = int(hit[0])
    return (float(lams[i - 1]), float(lams[i]))


def _pathway_change_brackets(g: pd.DataFrame, col: str) -> list[tuple[float, float]]:
    vals = g[col].to_numpy()
    lams = g["lambda_fueleu"].to_numpy(dtype=float)
    return [(float(lams[i - 1]), float(lams[i]))
            for i in range(1, len(vals)) if vals[i] != vals[i - 1]]


def _merge_windows(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not windows:
        return []
    ws = sorted({(round(a, 6), round(b, 6)) for a, b in windows})
    merged = [list(ws[0])]
    for lo, hi in ws[1:]:
        if lo <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(a, b) for a, b in merged]


def _detect_brackets(curve_sorted: pd.DataFrame) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    for col, thr, direction in _FLEET_CONDITIONS:
        b = _bracket_for_condition(curve_sorted, col, thr, direction)
        if b:
            windows.append(b)
    windows += _pathway_change_brackets(curve_sorted, "fleet_modal_fuel_pathway_partial")
    return _merge_windows(windows)


_DOWNWARD_CONDITION_COL = "fuel_plan_agreement_1pp_share"
_DOWNWARD_CONDITION_THR = 0.99


def _condition_met(row) -> bool:
    return float(row[_DOWNWARD_CONDITION_COL]) >= _DOWNWARD_CONDITION_THR


def _first_unmet_to_met(curve: pd.DataFrame) -> tuple[float, float] | None:
    g = curve.sort_values("lambda_fueleu").reset_index(drop=True)
    return _bracket_for_condition(g, _DOWNWARD_CONDITION_COL,
                                  _DOWNWARD_CONDITION_THR, "ge")


def _ensure_resolution_anchors(args, scen, cell_dir: Path,
                               all_curves: list, all_vessel: list,
                               all_by_type: list, manifest_cells: list) -> None:
    if not all_curves:
        return
    have = pd.concat(all_curves, ignore_index=True)
    for (year, bsteps), g in have.groupby(["year", "blend_steps"]):
        lams = g["lambda_fueleu"].astype(float)
        missing = [x for x in (0.0, 1.0)
                   if not bool(np.any(np.isclose(lams, x, atol=1e-12)))]
        if not missing:
            continue
        log.info("year %s blend_steps=%s: adding missing anchor(s) %s",
                 year, bsteps, missing)
        c, v, bt = _run_sweep_cell(
            mrv=args.mrv, fleet=args.fleet, scenario_config=args.scenario_config,
            scenario_id=args.scenario, year=int(year), lambdas=missing,
            blend_steps=int(bsteps), fuel_feasibility=scen["fuel_feasibility"],
            exclude_fuels=scen["exclude_fuels"],
            fuel_price_scenario=args.fuel_price_scenario, eua=args.eua,
            case=args.case, workers=args.workers, sample_limit=args.sample_limit,
            cell_dir=cell_dir,
            cell_label=f"anchorfix{int(bsteps)}_{int(year)}")
        all_curves.append(c); all_vessel.append(v); all_by_type.append(bt)
        manifest_cells.append({"year": int(year), "kind": "anchor",
                               "purpose": "anchor", "blend_steps": int(bsteps),
                               "n_lambdas": len(missing), "lambdas": missing})


def _principal_window(curve_sorted: pd.DataFrame,
                      windows: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not windows:
        return None
    best, best_span = None, -1.0
    for lo, hi in windows:
        row_lo = curve_sorted.loc[curve_sorted["lambda_fueleu"].sub(lo).abs().idxmin()]
        row_hi = curve_sorted.loc[curve_sorted["lambda_fueleu"].sub(hi).abs().idxmin()]
        span = abs(float(row_hi["fleet_regret_closed_fraction"])
                  - float(row_lo["fleet_regret_closed_fraction"]))
        if span > best_span:
            best, best_span = (lo, hi), span
    return best


def stage_matrix(args: argparse.Namespace) -> None:
    scen = _SCENARIOS[args.scenario]
    out = args.out
    cell_dir = out / "cells" / args.scenario
    cell_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    all_curves: list[pd.DataFrame] = []
    all_vessel: list[pd.DataFrame] = []
    all_by_type: list[pd.DataFrame] = []
    manifest_cells: list[dict] = []
    coarse_lambdas = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.05)]

    for year in scen["years"]:
        coarse_curve, coarse_vessel, coarse_by_type = _run_sweep_cell(
            mrv=args.mrv, fleet=args.fleet, scenario_config=args.scenario_config,
            scenario_id=args.scenario, year=year, lambdas=coarse_lambdas,
            blend_steps=11, fuel_feasibility=scen["fuel_feasibility"],
            exclude_fuels=scen["exclude_fuels"], fuel_price_scenario=args.fuel_price_scenario,
            eua=args.eua, case=args.case, workers=args.workers,
            sample_limit=args.sample_limit, cell_dir=cell_dir,
            cell_label=f"coarse_{year}")
        all_curves.append(coarse_curve)
        all_vessel.append(coarse_vessel)
        all_by_type.append(coarse_by_type)
        manifest_cells.append({"scenario_id": args.scenario, "year": year,
                               "kind": "coarse", "purpose": "coarse",
                               "blend_steps": 11,
                               "n_lambdas": len(coarse_lambdas),
                               **dict(_LAST_CELL_INFO)})

        coarse_sorted = coarse_curve.sort_values("lambda_fueleu").reset_index(drop=True)
        windows = _detect_brackets(coarse_sorted)
        log.info("year %d: %d refinement window(s) detected: %s", year, len(windows), windows)

        refined_frames: list[pd.DataFrame] = []
        for lo, hi in windows:
            grid = _window_lambdas(lo, hi, args.refined_step)
            rc, rv, rbt = _run_sweep_cell(
                mrv=args.mrv, fleet=args.fleet, scenario_config=args.scenario_config,
                scenario_id=args.scenario, year=year, lambdas=grid, blend_steps=11,
                fuel_feasibility=scen["fuel_feasibility"], exclude_fuels=scen["exclude_fuels"],
                fuel_price_scenario=args.fuel_price_scenario, eua=args.eua, case=args.case,
                workers=args.workers, sample_limit=args.sample_limit, cell_dir=cell_dir,
                cell_label=f"refined_{year}_{lo:.3f}_{hi:.3f}")
            all_curves.append(rc); all_vessel.append(rv); all_by_type.append(rbt)
            refined_frames.append(rc)
            manifest_cells.append({"scenario_id": args.scenario, "year": year,
                                   "kind": "refined", "purpose": "refined",
                                   "blend_steps": 11,
                                   "window": [lo, hi], "step": args.refined_step,
                                   "n_lambdas": len(grid), **dict(_LAST_CELL_INFO)})

        principal = _principal_window(coarse_sorted, windows)
        if principal is not None:
            lo, hi = principal
            refined_for_window = pd.concat(
                [f for f in refined_frames
                 if f["lambda_fueleu"].between(lo, hi).any()], ignore_index=True
            ).sort_values("lambda_fueleu").reset_index(drop=True) if refined_frames else coarse_sorted
            sub = _bracket_for_condition(refined_for_window,
                                         "fleet_regret_closed_fraction", 0.99, "ge")
            if sub is not None and args.fallback_step < args.refined_step:
                sub_lo, sub_hi = sub
                grid = _window_lambdas(sub_lo, sub_hi, args.fallback_step)
                fc, fv, fbt = _run_sweep_cell(
                    mrv=args.mrv, fleet=args.fleet, scenario_config=args.scenario_config,
                    scenario_id=args.scenario, year=year, lambdas=grid, blend_steps=11,
                    fuel_feasibility=scen["fuel_feasibility"], exclude_fuels=scen["exclude_fuels"],
                    fuel_price_scenario=args.fuel_price_scenario, eua=args.eua, case=args.case,
                    workers=args.workers, sample_limit=args.sample_limit, cell_dir=cell_dir,
                    cell_label=f"fallback_{year}_{sub_lo:.4f}_{sub_hi:.4f}")
                all_curves.append(fc); all_vessel.append(fv); all_by_type.append(fbt)
                manifest_cells.append({"scenario_id": args.scenario, "year": year,
                                       "kind": "fallback", "purpose": "fallback",
                                       "blend_steps": 11,
                                       "window": [sub_lo, sub_hi], "step": args.fallback_step,
                                       "n_lambdas": len(grid), **dict(_LAST_CELL_INFO)})

            grid = _window_lambdas(lo, hi, args.refined_step)
            for bsteps in args.blend_steps_list:
                if bsteps == 11:
                    continue

                def _run_at(lams: list[float], label: str, kind: str,
                            window=None, step=None, _bs=bsteps):
                    c, v, bt = _run_sweep_cell(
                        mrv=args.mrv, fleet=args.fleet,
                        scenario_config=args.scenario_config,
                        scenario_id=args.scenario, year=year, lambdas=lams,
                        blend_steps=_bs, fuel_feasibility=scen["fuel_feasibility"],
                        exclude_fuels=scen["exclude_fuels"],
                        fuel_price_scenario=args.fuel_price_scenario, eua=args.eua,
                        case=args.case, workers=args.workers,
                        sample_limit=args.sample_limit, cell_dir=cell_dir,
                        cell_label=label)
                    all_curves.append(c); all_vessel.append(v); all_by_type.append(bt)
                    rec = {"scenario_id": args.scenario, "year": year, "kind": kind,
                           "blend_steps": _bs, "n_lambdas": len(lams),
                           "purpose": kind, **dict(_LAST_CELL_INFO)}
                    if window is not None:
                        rec["window"] = list(window)
                    if step is not None:
                        rec["step"] = step
                    manifest_cells.append(rec)
                    return c

                _run_at([0.0, 1.0], f"anchor{bsteps}_{year}", "anchor")

                blend_curve_cell = _run_at(
                    grid, f"blend{bsteps}_{year}_{lo:.3f}_{hi:.3f}",
                    "blend_resolution", window=(lo, hi), step=args.refined_step)

                cur = blend_curve_cell.sort_values("lambda_fueleu").reset_index(drop=True)
                if _condition_met(cur.iloc[0]):
                    lo_probe, hi_probe = None, float(cur["lambda_fueleu"].iloc[0])
                    probe = hi_probe
                    while probe > 1e-9:
                        probe = round(max(0.0, probe - 0.10), 4)
                        pc = _run_at([probe], f"down{bsteps}_{year}_{probe:.4f}",
                                     "downward_search")
                        if not _condition_met(pc.sort_values("lambda_fueleu").iloc[0]):
                            lo_probe = probe
                            break
                        hi_probe = probe
                    if lo_probe is None:
                        log.warning(
                            "year %d blend_steps=%d: condition already met at "
                            "lambda=0 -- threshold is exact_at_zero, no bracket",
                            year, bsteps)
                    else:
                        b_grid = _window_lambdas(lo_probe, hi_probe, args.refined_step)
                        bc2 = _run_at(b_grid,
                                      f"downref{bsteps}_{year}_{lo_probe:.3f}_{hi_probe:.3f}",
                                      "refined", window=(lo_probe, hi_probe),
                                      step=args.refined_step)
                        sub2 = _first_unmet_to_met(bc2)
                        if sub2 and args.fallback_step < args.refined_step:
                            s_lo, s_hi = sub2
                            _run_at(_window_lambdas(s_lo, s_hi, args.fallback_step),
                                    f"downfb{bsteps}_{year}_{s_lo:.4f}_{s_hi:.4f}",
                                    "fallback", window=(s_lo, s_hi),
                                    step=args.fallback_step)
        else:
            log.warning("year %d: no principal window detected (no bracket found on "
                        "the coarse grid) -- skipping blend-resolution sensitivity", year)

    _ensure_resolution_anchors(args, scen, cell_dir, all_curves, all_vessel,
                               all_by_type, manifest_cells)

    curve_all = _dedupe_cells(pd.concat(all_curves, ignore_index=True))
    vessel_all = _dedupe_cells(pd.concat(all_vessel, ignore_index=True), ("vessel_id",))
    by_type_all = _dedupe_cells(pd.concat(all_by_type, ignore_index=True), ("ship_type",))
    _write_atomic_csv(curve_all, out / f"matrix_{args.scenario}_all_curve.csv")
    _write_atomic_parquet(vessel_all, out / f"matrix_{args.scenario}_vessel.parquet")
    _write_atomic_csv(by_type_all, out / f"matrix_{args.scenario}_by_type.csv")
    _write_atomic_text(json.dumps({
        "scenario_id": args.scenario,
        "years": scen["years"],
        "fuel_feasibility": scen["fuel_feasibility"],
        "exclude_fuels": scen["exclude_fuels"],
        "runtime_s": round(time.perf_counter() - t_start, 1),
        "n_cells": len(manifest_cells),
        "cells": manifest_cells,
    }, indent=2, default=str), out / f"matrix_{args.scenario}_manifest.json")
    log.info("scenario %s matrix done: %d cells, %d curve rows, %d vessel rows",
             args.scenario, len(manifest_cells), len(curve_all), len(vessel_all))


def _condition_status(g: pd.DataFrame, col: str, thr: float,
                      direction: str) -> dict:
    lams = g["lambda_fueleu"].astype(float).to_numpy()
    vals = g[col].astype(float).to_numpy()
    rec = {
        "condition_name": col,
        "condition_threshold": thr,
        "condition_direction": direction,
        "n_tested_points": int(len(lams)),
        "grid_min_step": float(np.min(np.diff(lams))) if len(lams) > 1 else 0.0,
        "lambda_min_tested": float(lams.min()) if len(lams) else np.nan,
        "lambda_max_tested": float(lams.max()) if len(lams) else np.nan,
        "lambda_low": np.nan, "lambda_high": np.nan, "interval": "",
        "left_censored": False, "right_censored": False,
    }
    if np.all(np.isnan(vals)):
        rec["status"] = "baseline_undefined"
        rec["interval"] = "baseline_undefined"
        return rec

    cond = (vals >= thr) if direction == "ge" else (vals <= thr)
    cond = np.where(np.isnan(vals), False, cond)
    has_zero = bool(len(lams)) and bool(np.isclose(lams[0], 0.0, atol=1e-12))

    if cond[0] and has_zero:
        rec.update(status="exact_at_zero", lambda_low=0.0, lambda_high=0.0,
                   interval="[0, 0]")
        return rec
    if cond[0]:
        rec.update(status="left_censored", left_censored=True,
                   lambda_high=float(lams[0]),
                   interval=f"<= {lams[0]:.4f}")
        return rec

    hit = np.where(cond)[0]
    if len(hit):
        i = int(hit[0])
        rec.update(status="bracketed", lambda_low=float(lams[i - 1]),
                   lambda_high=float(lams[i]),
                   interval=f"({lams[i - 1]:.4f}, {lams[i]:.4f}]")
        return rec

    if np.isclose(lams[-1], 1.0, atol=1e-12):
        rec.update(status="not_reached_on_domain",
                   interval="not_reached_by_lambda=1")
    else:
        rec.update(status="right_censored", right_censored=True,
                   lambda_low=float(lams[-1]),
                   interval=f"> {lams[-1]:.4f}")
    return rec


def _threshold_long_table(curve_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in curve_all.groupby(["scenario_id", "year", "blend_steps"]):
        g = g.drop_duplicates(subset="lambda_fueleu").sort_values("lambda_fueleu")
        has_zero = bool(np.any(np.isclose(
            g["lambda_fueleu"].astype(float), 0.0, atol=1e-12)))
        for col, thr, direction in _FLEET_CONDITIONS:
            if col == "fleet_regret_closed_fraction" and not has_zero:
                raise ValueError(
                    f"threshold long table: closure condition {col}>={thr} for "
                    f"group {key} has no same-resolution lambda=0 anchor -- run "
                    "the anchor cell before finalizing.")
            rec = {"scenario_id": key[0], "year": key[1], "blend_steps": key[2]}
            rec.update(_condition_status(g, col, thr, direction))
            rows.append(rec)
    return pd.DataFrame(rows)


def _pathway_transition_intervals(curve_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in curve_all.groupby(["scenario_id", "year", "blend_steps"]):
        g = g.drop_duplicates(subset="lambda_fueleu").sort_values("lambda_fueleu")
        vals = g["fleet_modal_fuel_pathway_partial"].to_numpy()
        lams = g["lambda_fueleu"].astype(float).to_numpy()
        for i in range(1, len(vals)):
            if vals[i] != vals[i - 1]:
                rows.append({
                    "scenario_id": key[0], "year": key[1], "blend_steps": key[2],
                    "lambda_low": float(lams[i - 1]), "lambda_high": float(lams[i]),
                    "from_pathway": str(vals[i - 1]), "to_pathway": str(vals[i]),
                    "interval": f"({lams[i - 1]:.4f}, {lams[i]:.4f}]",
                    "status": "bracketed",
                })
        if len(vals) and not np.isclose(lams[0], 0.0, atol=1e-12):
            rows.append({
                "scenario_id": key[0], "year": key[1], "blend_steps": key[2],
                "lambda_low": np.nan, "lambda_high": float(lams[0]),
                "from_pathway": "", "to_pathway": str(vals[0]),
                "interval": f"<= {lams[0]:.4f}", "status": "left_censored",
            })
    return pd.DataFrame(rows)


def _global_threshold_rows(curve_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["scenario_id", "year", "blend_steps"]
    for key, g in curve_all.groupby(group_cols):
        g = g.drop_duplicates(subset="lambda_fueleu").sort_values("lambda_fueleu")
        lams = g["lambda_fueleu"].to_numpy(dtype=float)
        rec = dict(zip(group_cols, key))
        rec["n_tested_points"] = len(lams)
        rec["grid_min_step"] = float(np.min(np.diff(lams))) if len(lams) > 1 else 0.0
        for col, thr, direction in _FLEET_CONDITIONS:
            b = _bracket_for_condition(g, col, thr, direction)
            left_censored = bool(len(g) and
                                 ((g[col].iloc[0] >= thr) if direction == "ge"
                                  else (g[col].iloc[0] <= thr)))
            right_censored = b is None and not left_censored
            name = f"{col}_{direction}{thr}".replace(".", "p")
            rec[f"{name}_lambda_low"] = b[0] if b else np.nan
            rec[f"{name}_lambda_high"] = b[1] if b else np.nan
            rec[f"{name}_interval"] = (f"({b[0]:.4f}, {b[1]:.4f}]" if b
                                       else ("left_censored" if left_censored
                                             else "right_censored"))
            rec[f"{name}_left_censored"] = left_censored
            rec[f"{name}_right_censored"] = right_censored
        rows.append(rec)
    return pd.DataFrame(rows)


def _vessel_thresholds_by_resolution(vessel_all: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["scenario_id", "case", "year", "blend_steps", "vessel_id"]
    for key, g in vessel_all.groupby(keys):
        g = g.drop_duplicates(subset="lambda_fueleu").sort_values("lambda_fueleu")
        lams = g["lambda_fueleu"].astype(float).to_numpy()
        has_zero = bool(len(lams)) and bool(np.isclose(lams[0], 0.0, atol=1e-12))
        rec = dict(zip(keys, key))
        rec["ship_type"] = str(g["ship_type"].iloc[0])
        rec["lambda_min_tested"] = float(lams.min()) if len(lams) else np.nan
        rec["lambda_max_tested"] = float(lams.max()) if len(lams) else np.nan
        rec["has_lambda_zero_anchor"] = has_zero

        for col, name in _VESSEL_CONDITIONS:
            b = _bracket_for_condition(g, col, 0.5, "ge")
            already_true = bool(len(g) and bool(g[col].iloc[0]))
            left_c = already_true and not has_zero
            rec[f"first_lambda_{name}"] = (
                0.0 if (already_true and has_zero) else (b[1] if b else np.nan))
            rec[f"{name}_interval_low"] = b[0] if b else np.nan
            rec[f"{name}_interval_high"] = b[1] if b else np.nan
            rec[f"{name}_left_censored"] = left_c
            rec[f"{name}_right_censored"] = bool(
                b is None and not already_true)

        anchor = g[np.isclose(g["lambda_fueleu"].astype(float), 0.0, atol=1e-12)]
        base_delta = float(anchor["delta_cost_eur"].iloc[0]) if len(anchor) else np.nan
        for pct, name in ((0.90, "90pct_individual_regret_closed"),
                          (0.99, "99pct_individual_regret_closed")):
            if not has_zero or not np.isfinite(base_delta):
                rec[f"first_lambda_{name}"] = np.nan
                rec[f"{name}_left_censored"] = True
                rec[f"{name}_right_censored"] = False
                continue
            if base_delta > 1e-3:
                closed = 1.0 - g["delta_cost_eur"].astype(float) / base_delta
                b = _bracket_for_condition(g.assign(_regret_closed=closed),
                                           "_regret_closed", pct, "ge")
                already_true = bool(closed.iloc[0] >= pct)
                rec[f"first_lambda_{name}"] = (
                    0.0 if already_true else (b[1] if b else np.nan))
                rec[f"{name}_left_censored"] = False
                rec[f"{name}_right_censored"] = bool(b is None and not already_true)
            else:
                rec[f"first_lambda_{name}"] = np.nan
                rec[f"{name}_left_censored"] = False
                rec[f"{name}_right_censored"] = False
                rec[f"{name}_baseline_regret_zero"] = True
        rows.append(rec)
    return pd.DataFrame(rows)


def _vessel_threshold_summary(vessel_thr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in vessel_thr.groupby(["scenario_id", "case", "year", "blend_steps"]):
        rec = {"scenario_id": key[0], "case": key[1], "year": key[2],
               "blend_steps": key[3], "n_vessels": len(g)}
        for col in [c for c in g.columns if c.startswith("first_lambda_")]:
            vals = g[col].dropna()
            if len(vals):
                rec[f"{col}_median"] = float(vals.median())
                rec[f"{col}_p10"] = float(vals.quantile(0.10))
                rec[f"{col}_p90"] = float(vals.quantile(0.90))
                rec[f"{col}_reach_share"] = len(vals) / len(g)
        rows.append(rec)
    return pd.DataFrame(rows)


def _transition_matrix(vessel_all: pd.DataFrame) -> pd.DataFrame:
    key = ["scenario_id", "year", "blend_steps", "lambda_fueleu",
          "partial_fuel_pathway", "joint_fuel_pathway"]
    tm = (vessel_all.groupby(key).size().rename("n_vessels").reset_index())
    totals = tm.groupby(["scenario_id", "year", "blend_steps", "lambda_fueleu"]
                        )["n_vessels"].transform("sum")
    tm["share"] = tm["n_vessels"] / totals
    return tm.sort_values(key).reset_index(drop=True)


def _optimization_sample_ids(vessel_all: pd.DataFrame) -> set[str]:
    sel = vessel_all[
        (vessel_all["scenario_id"] == "CENTRAL_EXTENDED")
        & (vessel_all["blend_steps"] == 11)
        & (np.isclose(vessel_all["lambda_fueleu"].astype(float), 0.0, atol=1e-12))
    ]
    if not len(sel):
        sel = vessel_all[
            (vessel_all["blend_steps"] == 11)
            & (np.isclose(vessel_all["lambda_fueleu"].astype(float), 0.0, atol=1e-12))]
    return set(sel["vessel_id"].astype(str))


def _feasibility_audit(fleet_path: Path,
                       optimization_ids: set[str] | None = None,
                       vessel_id_fn=None) -> pd.DataFrame:
    from carbon_twin.optimization.fuel_choice import fuel_menu
    from carbon_twin.pipeline import load_fuel_capability

    vessels = load_vessels_csv(fleet_path)
    capability = load_fuel_capability(_CAPABILITY_PATH[0]) if _CAPABILITY_PATH else None
    if vessel_id_fn is None:
        from scripts.fueleu_internalization_sweep import _vessel_id as vessel_id_fn

    scopes: list[tuple[str, list]] = [("matched_fleet", vessels)]
    if optimization_ids:
        scopes.append(("optimization_sample",
                       [v for v in vessels
                        if vessel_id_fn(v.imo_number) in optimization_ids]))

    rows = []
    for sample_scope, scope_vessels in scopes:
        for scenario_id, scen in _SCENARIOS.items():
            if scen["fuel_feasibility"] == "engine_capability" and capability is None:
                continue
            exclude = ({FuelType(scen["exclude_fuels"])}
                       if scen["exclude_fuels"] else None)
            by_type: dict[str, dict] = {}
            for vessel in scope_vessels:
                st = by_type.setdefault(vessel.ship_type.value, {
                    "n_vessels": 0, "n_lng": 0, "n_fame": 0,
                    "n_methanol_bio": 0, "n_methanol_e": 0, "candidate_counts": [],
                })
                menu = feasible_fuel_options(
                    vessel, scen["fuel_feasibility"], True, exclude, capability)
                if menu is None:
                    menu = fuel_menu(vessel, extended=True)
                st["n_vessels"] += 1
                st["n_lng"] += int(FuelType.LNG in menu)
                st["n_fame"] += int(FuelType.BIODIESEL_FAME in menu)
                st["n_methanol_bio"] += int(FuelType.METHANOL_BIO in menu)
                st["n_methanol_e"] += int(FuelType.METHANOL_E in menu)
                st["candidate_counts"].append(len(menu))

            def _row(ship_type: str, st: dict) -> dict:
                n = max(st["n_vessels"], 1)
                return {
                    "sample_scope": sample_scope, "scenario_id": scenario_id,
                    "ship_type": ship_type, "n_vessels": st["n_vessels"],
                    "share_lng_feasible": st["n_lng"] / n,
                    "share_fame_feasible": st["n_fame"] / n,
                    "share_methanol_bio_feasible": st["n_methanol_bio"] / n,
                    "share_methanol_e_feasible": st["n_methanol_e"] / n,
                    "mean_candidate_count": float(np.mean(st["candidate_counts"]))
                        if st["candidate_counts"] else 0.0,
                    "final_feasible_vessel_count": st["n_vessels"],
                }

            for ship_type, st in sorted(by_type.items()):
                rows.append(_row(ship_type, st))
            total = {
                "n_vessels": sum(s["n_vessels"] for s in by_type.values()),
                "n_lng": sum(s["n_lng"] for s in by_type.values()),
                "n_fame": sum(s["n_fame"] for s in by_type.values()),
                "n_methanol_bio": sum(s["n_methanol_bio"] for s in by_type.values()),
                "n_methanol_e": sum(s["n_methanol_e"] for s in by_type.values()),
                "candidate_counts": [c for s in by_type.values()
                                     for c in s["candidate_counts"]],
            }
            rows.append(_row("ALL", total))
    return pd.DataFrame(rows)


def _check(name: str, scenario_id: str, year, blend_steps, status: str,
          observed, tolerance, message: str) -> dict:
    return {"check_name": name, "scenario_id": scenario_id, "year": year,
           "blend_steps": blend_steps, "status": status,
           "observed_value": observed, "tolerance": tolerance, "message": message}


def _run_validation(curve_all: pd.DataFrame, vessel_all: pd.DataFrame,
                    by_type_all: pd.DataFrame,
                    transition_matrix: pd.DataFrame,
                    thr_long: pd.DataFrame | None = None,
                    vessel_thr_all: pd.DataFrame | None = None,
                    feasibility: pd.DataFrame | None = None,
                    out: Path | None = None,
                    skip_manuscript: bool = False) -> list[dict]:
    checks: list[dict] = []

    tol = 1e-6
    bad = vessel_all[vessel_all["delta_cost_eur"] < -tol]
    checks.append(_check("non_negative_vessel_regret", "ALL", "ALL", "ALL",
                         "PASS" if bad.empty else "FAIL", len(bad), tol,
                         f"{len(bad)} vessel-row(s) below tolerance"))

    for key, g in curve_all.groupby(["scenario_id", "year", "blend_steps"]):
        n_anchor = int(np.sum(np.isclose(
            g["lambda_fueleu"].astype(float), 0.0, atol=1e-12)))
        checks.append(_check("lambda_zero_anchor_unique", key[0], key[1], key[2],
                             "PASS" if n_anchor == 1 else "FAIL", n_anchor, "exactly 1",
                             "same-resolution lambda=0 anchor required for closure"))
        checks.append(_check("same_resolution_anchor_present", key[0], key[1], key[2],
                             "PASS" if n_anchor >= 1 else "FAIL", n_anchor, ">=1",
                             "closure denominator must come from this resolution"))

        if "fleet_regret_closed_fraction" not in g.columns:
            continue
        gs = g.drop_duplicates(subset="lambda_fueleu").sort_values("lambda_fueleu")
        closed = gs["fleet_regret_closed_fraction"].astype(float).to_numpy()
        if np.all(np.isnan(closed)):
            checks.append(_check("closure_bounds", key[0], key[1], key[2], "PASS",
                                 "all-NaN", "baseline_regret_zero",
                                 "closure undefined (lambda=0 regret ~ 0)"))
        else:
            finite = closed[~np.isnan(closed)]
            in_bounds = bool(np.all(finite >= -1e-9) and np.all(finite <= 1.0 + 1e-9))
            checks.append(_check("closure_bounds", key[0], key[1], key[2],
                                 "PASS" if in_bounds else "FAIL",
                                 [float(finite.min()), float(finite.max())], "[0, 1]",
                                 "closed fraction must lie within [0,1] after global recompute"))
            drops = float(np.min(np.diff(finite))) if len(finite) > 1 else 0.0
            checks.append(_check("closure_monotonic", key[0], key[1], key[2],
                                 "PASS" if drops >= -1e-9 else "FAIL", drops, ">= -1e-9",
                                 "closed fraction must be non-decreasing in lambda"))

    for key, g in curve_all.groupby(["scenario_id", "year", "blend_steps"]):
        g1 = g[g["lambda_fueleu"].sub(1.0).abs() < 1e-9]
        if not len(g1):
            checks.append(_check("lambda_one_endpoint", key[0], key[1], key[2],
                                 "FAIL", "absent", "required",
                                 "no lambda=1 anchor at this resolution"))
            continue
        row = g1.iloc[0]
        ok = abs(float(row["fleet_delta_cost_eur"])) <= 0.01
        checks.append(_check("lambda_one_endpoint", key[0], key[1], key[2],
                             "PASS" if ok else "FAIL",
                             float(row["fleet_delta_cost_eur"]), 0.01,
                             "fleet delta cost at lambda=1"))

    for key, g in curve_all.groupby(["scenario_id", "year", "blend_steps"]):
        g = g.drop_duplicates(subset="lambda_fueleu").sort_values("lambda_fueleu")
        d = g["fleet_delta_cost_eur"].to_numpy()
        viol = float(np.max(d[1:] - d[:-1])) if len(d) > 1 else 0.0
        ok = viol <= 1e-6 * max(1.0, float(np.max(np.abs(d)))) if len(d) else True
        checks.append(_check("fleet_monotonic_regret", key[0], key[1], key[2],
                             "PASS" if ok else "FAIL", viol, "1e-6 rel",
                             "max increase in fleet delta cost across sorted lambda"))

    for key, g in vessel_all.groupby(["scenario_id", "year", "blend_steps"]):
        n_per_lambda = g.groupby("lambda_fueleu")["vessel_id"].nunique()
        ok = n_per_lambda.nunique() == 1
        checks.append(_check("sample_stability", key[0], key[1], key[2],
                             "PASS" if ok else "FAIL", n_per_lambda.to_dict(), "exact",
                             "vessel count must be constant across lambda"))

    curve_key = curve_all.set_index(["scenario_id", "year", "lambda_fueleu", "blend_steps"])
    for key, g in by_type_all.groupby(["scenario_id", "year", "lambda_fueleu", "blend_steps"]):
        if key not in curve_key.index:
            continue
        global_row = curve_key.loc[key]
        if isinstance(global_row, pd.DataFrame):
            global_row = global_row.iloc[0]
        n_sum = int(g["n_vessels"].sum())
        n_ok = n_sum == int(global_row["n_vessels"])
        delta_sum = float(g["fleet_delta_cost_eur"].sum())
        tol_d = 1e-6 + 1e-9 * abs(float(global_row["fleet_delta_cost_eur"]))
        d_ok = abs(delta_sum - float(global_row["fleet_delta_cost_eur"])) <= tol_d
        checks.append(_check("aggregate_reconciliation_n", *key[:1], key[1], key[3],
                             "PASS" if n_ok else "FAIL", n_sum, "exact",
                             f"by-type n sum vs global (key={key})"))
        checks.append(_check("aggregate_reconciliation_delta", *key[:1], key[1], key[3],
                             "PASS" if d_ok else "FAIL", delta_sum, tol_d,
                             f"by-type fleet delta cost sum vs global (key={key})"))
    if len(transition_matrix):
        for key, g in transition_matrix.groupby(["scenario_id", "year", "blend_steps",
                                                  "lambda_fueleu"]):
            if key not in curve_key.index:
                continue
            global_row = curve_key.loc[key]
            if isinstance(global_row, pd.DataFrame):
                global_row = global_row.iloc[0]
            ok = int(g["n_vessels"].sum()) == int(global_row["n_vessels"])
            checks.append(_check("transition_matrix_reconciliation", key[0], key[1], key[2],
                                 "PASS" if ok else "FAIL", int(g["n_vessels"].sum()), "exact",
                                 "transition matrix vessel count vs global"))

    if thr_long is not None and len(thr_long):
        valid_status = {"bracketed", "exact_at_zero", "left_censored",
                        "right_censored", "not_reached_on_domain", "baseline_undefined"}
        bad_status = thr_long[~thr_long["status"].isin(valid_status)]
        checks.append(_check("threshold_censoring_consistent", "ALL", "ALL", "ALL",
                             "PASS" if bad_status.empty else "FAIL",
                             sorted(set(thr_long["status"])), sorted(valid_status),
                             "every threshold row carries a recognized status"))
        inconsistent = thr_long[
            (thr_long["status"] == "bracketed")
            & (thr_long["lambda_low"].isna() | thr_long["lambda_high"].isna())]
        checks.append(_check("threshold_bracket_completeness", "ALL", "ALL", "ALL",
                             "PASS" if inconsistent.empty else "FAIL",
                             len(inconsistent), 0,
                             "bracketed rows must carry both interval endpoints"))
        nf = thr_long[(thr_long["scenario_id"] == "NO_FAME_2045")
                      & (thr_long["blend_steps"] == 51)
                      & (thr_long["condition_name"] == "fuel_plan_agreement_1pp_share")]
        if len(nf):
            st = set(nf["status"])
            ok = not (st & {"left_censored", "right_censored"})
            checks.append(_check("no_fame_51_lower_bracket_found", "NO_FAME_2045",
                                 2045, 51, "PASS" if ok else "FAIL", sorted(st),
                                 "bracketed|exact_at_zero|not_reached_on_domain",
                                 "51-point plan-agreement threshold must be dated, "
                                 "not censored at the window edge"))

    if vessel_thr_all is not None and len(vessel_thr_all):
        dup = vessel_thr_all.duplicated(
            subset=["scenario_id", "case", "year", "blend_steps", "vessel_id"]).sum()
        checks.append(_check("vessel_threshold_resolution_separation", "ALL", "ALL", "ALL",
                             "PASS" if dup == 0 else "FAIL", int(dup), 0,
                             "one vessel-threshold row per resolution, never pooled"))

    if len(vessel_all) and "partial_alternative_component_share" in vessel_all.columns:
        zero_blend = vessel_all[
            vessel_all["partial_alternative_component_share"].astype(float) <= 1e-9]
        if len(zero_blend):
            base = zero_blend["partial_base_fuel"].astype(str)
            expect = base + "_only"
            ok = bool((zero_blend["partial_fuel_pathway"].astype(str) == expect).all())
            checks.append(_check("zero_blend_pathway_consistency", "ALL", "ALL", "ALL",
                                 "PASS" if ok else "FAIL",
                                 int((zero_blend["partial_fuel_pathway"].astype(str)
                                      != expect).sum()), 0,
                                 "zero alternative share must render as <base>_only, "
                                 "never the alternative candidate's label"))

    if feasibility is not None and len(feasibility):
        for scope, expected in (("matched_fleet", 13544), ("optimization_sample", 12333)):
            sel = feasibility[(feasibility["sample_scope"] == scope)
                              & (feasibility["ship_type"] == "ALL")]
            if not len(sel):
                checks.append(_check(f"{scope}_scope_present", "ALL", "ALL", "ALL",
                                     "FAIL", "absent", "required",
                                     f"feasibility audit missing {scope} scope"))
                continue
            n = int(sel["n_vessels"].iloc[0])
            full_scale = int(feasibility[
                (feasibility["sample_scope"] == "matched_fleet")
                & (feasibility["ship_type"] == "ALL")]["n_vessels"].iloc[0]) >= 13000
            name = ("matched_fleet_count_13544" if scope == "matched_fleet"
                    else "optimization_sample_count_12333")
            if full_scale:
                checks.append(_check(name, "ALL", "ALL", "ALL",
                                     "PASS" if n == expected else "FAIL", n, expected,
                                     f"{scope} denominator"))
            else:
                checks.append(_check(name, "ALL", "ALL", "ALL", "SKIP", n, expected,
                                     f"{scope} denominator (sample-limited run; "
                                     "exact count asserted only at full scale)"))
        for key, g in feasibility.groupby(["sample_scope", "scenario_id"]):
            per_type = g[g["ship_type"] != "ALL"]["n_vessels"].sum()
            all_row = g[g["ship_type"] == "ALL"]["n_vessels"]
            ok = bool(len(all_row)) and int(per_type) == int(all_row.iloc[0])
            checks.append(_check("ship_type_sum_reconciliation", key[1], "ALL", "ALL",
                                 "PASS" if ok else "FAIL", int(per_type), "== ALL row",
                                 f"ship-type rows must sum to the ALL row ({key[0]})"))

    if out is not None:
        main_fig = out / "manuscript_lambda_main_figure_data.csv"
        if main_fig.exists():
            mf = pd.read_csv(main_fig)
            nan_lam = int(mf["lambda_fueleu"].isna().sum()) if "lambda_fueleu" in mf else -1
            checks.append(_check("figure_data_no_nan_lambda", "ALL", "ALL", "ALL",
                                 "PASS" if nan_lam == 0 else "FAIL", nan_lam, 0,
                                 "manuscript figure data must carry numeric lambda only"))
            if len(mf) and "fleet_regret_closed_fraction" in mf:
                merged = mf.merge(
                    curve_all[["scenario_id", "year", "blend_steps", "lambda_fueleu",
                               "fleet_regret_closed_fraction"]],
                    on=["scenario_id", "year", "blend_steps", "lambda_fueleu"],
                    how="left", suffixes=("_ms", "_auth"))
                diff = (merged["fleet_regret_closed_fraction_ms"]
                        - merged["fleet_regret_closed_fraction_auth"]).abs()
                worst = float(diff.max()) if len(diff) else 0.0
                checks.append(_check("manuscript_data_uses_corrected_closure",
                                     "ALL", "ALL", "ALL",
                                     "PASS" if worst <= 1e-9 or np.isnan(worst) else "FAIL",
                                     worst, 1e-9,
                                     "manuscript closure must equal the recomputed master"))
        for stem in ("fig_lambda_main", "fig_lambda_robustness", "fig_blend_resolution"):
            if skip_manuscript:
                checks.append(_check("figures_exist_png_pdf", "ALL", "ALL", "ALL", "SKIP",
                                     None, "both", f"{stem}: --skip-manuscript"))
                continue
            have = [(_REPO / "paper" / "figures" / f"{stem}.{ext}").exists()
                    for ext in ("png", "pdf")]
            checks.append(_check("figures_exist_png_pdf", "ALL", "ALL", "ALL",
                                 "PASS" if all(have) else "FAIL", dict(zip(("png", "pdf"), have)),
                                 "both", f"{stem} must exist as PNG and PDF"))
        summary = out / "manuscript_lambda_summary_table.csv"
        if summary.exists() and thr_long is not None and len(thr_long):
            sm = pd.read_csv(summary)
            mismatches = 0
            auth = thr_long[thr_long["condition_name"] == "fleet_regret_closed_fraction"]
            for _, r in sm.iterrows():
                for thr, col in ((0.90, "regret_closed_90pct_interval"),
                                 (0.99, "regret_closed_99pct_interval")):
                    a = auth[(auth["scenario_id"] == r["scenario_id"])
                             & (auth["year"] == r["year"])
                             & (auth["blend_steps"] == r["blend_steps"])
                             & (auth["condition_threshold"] == thr)]
                    if len(a) and col in r and str(r[col]) != str(a["interval"].iloc[0]):
                        mismatches += 1
            checks.append(_check("summary_table_thresholds_match_authoritative_long_table",
                                 "ALL", "ALL", "ALL",
                                 "PASS" if mismatches == 0 else "FAIL", mismatches, 0,
                                 "summary intervals must be copied from the long table"))

    return checks


def _capture_pytest(target: str, out_path: Path) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "pytest", "-q", target]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO))
    text = (f"command: {' '.join(cmd)}\nexit code: {r.returncode}\n\n"
           f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}\n")
    _write_atomic_text(text, out_path)
    return r.returncode == 0, text.splitlines()[-1] if r.stdout else ""


def stage_finalize(args: argparse.Namespace) -> None:
    out = args.out
    required = ["CENTRAL_EXTENDED", "COMPATIBILITY_CONSTRAINED_2035", "NO_FAME_2045"]
    scenario_ids = [s for s in _SCENARIOS
                    if s in required or (out / f"matrix_{s}_all_curve.csv").exists()]
    missing = [s for s in scenario_ids
              if not (out / f"matrix_{s}_all_curve.csv").exists()]
    if missing:
        sys.exit(f"finalize: missing matrix outputs for scenario(s) {missing} "
                 f"-- run --stage matrix for each first")

    curve_all = _dedupe_cells(pd.concat(
        [pd.read_csv(out / f"matrix_{s}_all_curve.csv") for s in scenario_ids],
        ignore_index=True))
    vessel_all = _dedupe_cells(pd.concat(
        [pd.read_parquet(out / f"matrix_{s}_vessel.parquet") for s in scenario_ids],
        ignore_index=True), ("vessel_id",))
    by_type_all = _dedupe_cells(pd.concat(
        [pd.read_csv(out / f"matrix_{s}_by_type.csv") for s in scenario_ids],
        ignore_index=True), ("ship_type",))

    curve_all = recompute_global_closure_metrics(curve_all, CURVE_GROUP_KEYS)
    by_type_all = recompute_global_closure_metrics(by_type_all, BY_TYPE_GROUP_KEYS)
    log.info("global closure recomputed: %d curve rows, %d by-type rows",
             len(curve_all), len(by_type_all))

    _write_atomic_csv(curve_all, out / "fueleu_internalization_all_curve.csv")
    from scripts.fueleu_internalization_sweep import _LEGACY_CURVE_COLUMNS
    _write_atomic_csv(curve_all[_LEGACY_CURVE_COLUMNS],
                      out / "fueleu_internalization_curve.csv")
    _write_atomic_csv(by_type_all, out / "fueleu_internalization_by_type.csv")
    _write_atomic_parquet(vessel_all, out / "fueleu_internalization_vessel.parquet")

    global_thr = _global_threshold_rows(curve_all)
    _write_atomic_csv(global_thr, out / "fueleu_internalization_thresholds_global.csv")
    thr_long = _threshold_long_table(curve_all)
    _write_atomic_csv(thr_long, out / "fueleu_internalization_thresholds_long.csv")
    transitions = _pathway_transition_intervals(curve_all)
    _write_atomic_csv(transitions, out / "fuel_pathway_transition_intervals.csv")

    vessel_thr_all = _vessel_thresholds_by_resolution(vessel_all)
    _write_atomic_parquet(vessel_thr_all,
                          out / "vessel_thresholds_by_resolution.parquet")
    _write_atomic_csv(_vessel_threshold_summary(vessel_thr_all),
                      out / "vessel_threshold_summary_by_resolution.csv")
    vessel_thr = vessel_thr_all[vessel_thr_all["blend_steps"] == 11].copy()
    _write_atomic_parquet(vessel_thr, out / "vessel_thresholds_global.parquet")
    summary_rows = []
    for (scen, yr), g in vessel_thr.groupby(["scenario_id", "year"]):
        rec = {"scenario_id": scen, "year": yr, "blend_steps": 11, "n_vessels": len(g)}
        for col in [c for c in g.columns if c.startswith("first_lambda_")]:
            vals = g[col].dropna()
            if len(vals):
                rec[f"{col}_median"] = float(vals.median())
                rec[f"{col}_p10"] = float(vals.quantile(0.10))
                rec[f"{col}_p90"] = float(vals.quantile(0.90))
                rec[f"{col}_reach_share"] = len(vals) / len(g)
        summary_rows.append(rec)
    _write_atomic_csv(pd.DataFrame(summary_rows),
                      out / "vessel_threshold_summary_global.csv")

    tm = _transition_matrix(vessel_all)
    _write_atomic_csv(tm, out / "fuel_pathway_transition_matrix.csv")
    fa = _feasibility_audit(args.fleet, _optimization_sample_ids(vessel_all))
    _write_atomic_csv(fa, out / "fuel_menu_feasibility_audit.csv")

    blend_rows = curve_all[curve_all["blend_steps"].isin([11, 21, 51])]
    principal_windows = []
    for s in scenario_ids:
        manifest_path = out / f"matrix_{s}_manifest.json"
        if manifest_path.exists():
            cells = json.loads(manifest_path.read_text(encoding="utf-8"))["cells"]
            for c in cells:
                if c["kind"] == "blend_resolution":
                    principal_windows.append((s, c["year"], c["window"][0], c["window"][1]))
    if principal_windows:
        mask = pd.Series(False, index=blend_rows.index)
        for s, yr, lo, hi in principal_windows:
            mask |= ((blend_rows["scenario_id"] == s) & (blend_rows["year"] == yr)
                     & (blend_rows["lambda_fueleu"] >= lo - 1e-9)
                     & (blend_rows["lambda_fueleu"] <= hi + 1e-9))
        blend_curve = blend_rows[mask].sort_values(
            ["scenario_id", "year", "blend_steps", "lambda_fueleu"])
    else:
        blend_curve = blend_rows.iloc[0:0]
    _write_atomic_csv(blend_curve, out / "blend_resolution_curve.csv")
    blend_thr = _global_threshold_rows(blend_curve) if len(blend_curve) else global_thr.iloc[0:0]
    _write_atomic_csv(blend_thr, out / "blend_resolution_thresholds.csv")

    refined_windows_all = []
    for s in scenario_ids:
        manifest_path = out / f"matrix_{s}_manifest.json"
        if manifest_path.exists():
            cells = json.loads(manifest_path.read_text(encoding="utf-8"))["cells"]
            for c in cells:
                if c["kind"] in ("refined", "fallback"):
                    refined_windows_all.append((s, c["year"], c["window"][0], c["window"][1]))
    canonical_rows = curve_all[curve_all["blend_steps"] == 11]
    if refined_windows_all:
        mask = pd.Series(False, index=canonical_rows.index)
        for s, yr, lo, hi in refined_windows_all:
            mask |= ((canonical_rows["scenario_id"] == s) & (canonical_rows["year"] == yr)
                     & (canonical_rows["lambda_fueleu"] >= lo - 1e-9)
                     & (canonical_rows["lambda_fueleu"] <= hi + 1e-9))
        refined_curve = canonical_rows[mask].sort_values(
            ["scenario_id", "year", "lambda_fueleu"])
    else:
        refined_curve = canonical_rows.iloc[0:0]
    _write_atomic_csv(refined_curve, out / "refined_lambda_curve.csv")

    menu_curve = curve_all[curve_all["blend_steps"] == 11]
    _write_atomic_csv(menu_curve, out / "fuel_menu_robustness_curve.csv")
    _write_atomic_csv(global_thr[global_thr["blend_steps"] == 11],
                      out / "fuel_menu_robustness_thresholds.csv")
    _write_atomic_csv(by_type_all[by_type_all["blend_steps"] == 11],
                      out / "fuel_menu_robustness_by_type.csv")

    if not args.skip_manuscript:
        r = subprocess.run(
            [sys.executable, "paper/scripts/build_lambda_manuscript_data.py",
             "--results-dir", str(out)], cwd=str(_REPO), capture_output=True, text=True)
        if r.returncode != 0:
            log.error("manuscript data build failed:\n%s",
                      r.stdout[-2000:] + r.stderr[-2000:])
            sys.exit(1)
        for fig in ("fig_lambda_main", "fig_lambda_robustness", "fig_blend_resolution"):
            r = subprocess.run([sys.executable, f"paper/scripts/{fig}.py"],
                               cwd=str(_REPO), capture_output=True, text=True)
            if r.returncode != 0:
                log.error("%s failed:\n%s", fig, r.stdout[-2000:] + r.stderr[-2000:])
                sys.exit(1)
        import shutil
        for stem in ("fig_lambda_main", "fig_lambda_robustness", "fig_blend_resolution"):
            for ext in ("png", "pdf"):
                src = _REPO / "paper" / "figures" / f"{stem}.{ext}"
                if src.exists():
                    shutil.copy2(src, out / f"{stem}.{ext}")

    checks = _run_validation(curve_all, vessel_all, by_type_all, tm,
                             thr_long=thr_long, vessel_thr_all=vessel_thr_all,
                             feasibility=fa, out=out,
                             skip_manuscript=args.skip_manuscript)
    _write_atomic_text(json.dumps(checks, indent=2, default=str),
                       out / "validation_results.json")
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    primary_lines = ["FuelEU internalization -- primary validation report",
                     "=" * 55,
                     f"Timestamp: {datetime.now(UTC).isoformat()}",
                     f"Scenarios: {scenario_ids}", "", "Checks:"]
    followups_lines = ["FuelEU internalization -- follow-up (refined/blend/menu) "
                       "validation report", "=" * 70, ""]
    for c in checks:
        line = (f"  [{c['status']}] {c['check_name']} scenario={c['scenario_id']} "
               f"year={c['year']} blend_steps={c['blend_steps']}: {c['message']} "
               f"(observed={c['observed_value']}, tol={c['tolerance']})")
        (followups_lines if c["blend_steps"] not in (11, "ALL") else primary_lines).append(line)
    primary_lines += ["", f"RESULT: {'PASS' if n_fail == 0 else 'FAIL'} ({n_fail} issue(s))"]
    followups_lines += ["", f"RESULT: {'PASS' if n_fail == 0 else 'FAIL'} ({n_fail} issue(s) "
                        "total across all checks, see validation_report_primary.txt)"]
    _write_atomic_text("\n".join(primary_lines), out / "validation_report_primary.txt")
    _write_atomic_text("\n".join(followups_lines), out / "validation_report_followups.txt")

    if args.skip_tests:
        dedicated_ok = full_ok = True
    else:
        dedicated_ok, _ = _capture_pytest("tests/test_fueleu_internalization.py",
                                          out / "pytest_internalization_output.txt")
        full_ok, _ = _capture_pytest(".", out / "pytest_full_suite_output.txt")

    git = _git_info()
    manifest_primary = {
        "experiment": "fueleu_internalization_final_primary", **git,
        "model_variant": _VARIANT_ARGS[1] if _VARIANT_ARGS else "v23",
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "scenarios": scenario_ids,
        "n_curve_rows": len(curve_all), "n_vessel_rows": len(vessel_all),
        "validation_result": "PASS" if n_fail == 0 else "FAIL",
        "n_validation_issues": n_fail,
    }
    manifest_followups = {
        "experiment": "fueleu_internalization_final_followups", **git,
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "n_blend_resolution_rows": len(blend_curve),
        "n_menu_robustness_rows": len(menu_curve),
    }
    _write_atomic_text(json.dumps(manifest_primary, indent=2, default=str),
                       out / "run_manifest_primary.json")
    _write_atomic_text(json.dumps(manifest_followups, indent=2, default=str),
                       out / "run_manifest_followups.json")
    _write_atomic_text(
        f"# Primary run log\n\nTimestamp: {manifest_primary['run_timestamp_utc']}\n"
        f"Scenarios: {scenario_ids}\nCurve rows: {len(curve_all)}\n"
        f"Vessel rows: {len(vessel_all)}\n", out / "run_log_primary.md")
    _write_atomic_text(
        f"# Follow-up run log\n\nTimestamp: {manifest_followups['run_timestamp_utc']}\n"
        f"Blend-resolution rows: {len(blend_curve)}\n"
        f"Menu-robustness rows: {len(menu_curve)}\n", out / "run_log_followups.md")

    _write_completion_report(out, scenario_ids, thr_long, n_fail, dedicated_ok,
                             full_ok, git, checks)
    _write_manuscript_result_note(out, curve_all, thr_long, transitions)
    _write_readme_results(out)

    if not args.skip_packaging:
        r = subprocess.run(
            [sys.executable, "scripts/package_fueleu_internalization_delivery.py",
             "--results-dir", str(out)], cwd=str(_REPO), capture_output=True, text=True)
        if r.returncode != 0:
            log.error("packaging failed:\n%s", r.stdout[-2000:] + r.stderr[-2000:])
            sys.exit(1)
        log.info(r.stdout[-1000:])

    log.info("finalize done: validation %s (%d issues), pytest dedicated=%s full=%s",
             "PASS" if n_fail == 0 else "FAIL", n_fail, dedicated_ok, full_ok)
    if n_fail or not dedicated_ok or not full_ok:
        sys.exit(1)


def _write_readme_results(out: Path) -> None:
    _write_atomic_text(
        "# FuelEU internalization -- corrected results\n\n"
        "## Authoritative file hierarchy\n\n"
        "1. `fueleu_internalization_all_curve.csv` -- master curve; closure columns\n"
        "   were recomputed globally against each same-resolution lambda=0 anchor\n"
        "   AFTER merging and deduping every cell.\n"
        "2. `fueleu_internalization_thresholds_long.csv` -- AUTHORITATIVE thresholds,\n"
        "   one row per scenario x year x blend_steps x condition, with an explicit\n"
        "   status (bracketed / exact_at_zero / left_censored / right_censored /\n"
        "   not_reached_on_domain / baseline_undefined). The wide\n"
        "   `*_thresholds_global.csv` is a compatibility view only.\n"
        "3. `fuel_pathway_transition_intervals.csv` -- fleet-modal pathway switches\n"
        "   as tested intervals.\n"
        "4. `manuscript_*` -- data behind the manuscript tables and figures.\n"
        "5. `validation_results.json` -- machine-readable per-check results.\n\n"
        "## Fuel identity -- read this before quoting any fuel\n\n"
        "Use ONLY the pathway fields: `fleet_modal_fuel_pathway_partial/joint`,\n"
        "`partial_fuel_pathway`/`joint_fuel_pathway`, and the\n"
        "`*_energy_dominant_fuel_*` fields.\n\n"
        "The legacy `dominant_fuel_partial`/`dominant_fuel_joint` columns (and their\n"
        "`legacy_alternative_candidate_label_*` aliases) are the ALTERNATIVE CANDIDATE\n"
        "LABEL, not the consumed fuel: a candidate carrying a zero blend share still\n"
        "reports e.g. `MGO` while the actual energy pathway is `VLSFO_only`. They are\n"
        "retained for backward compatibility and must not be used as fuel-choice\n"
        "evidence.\n\n"
        "## Closure caveat\n\n"
        "Per-cell closure written by `fueleu_internalization_sweep.py` is NOT\n"
        "authoritative: for a cell restricted to a local window it normalizes against\n"
        "that window's first row. Only the finalize stage's globally recomputed\n"
        "closure (in the files above) is valid.\n",
        out / "README_RESULTS.md")


def _write_completion_report(out: Path, scenario_ids: list[str],
                             thr_long: pd.DataFrame, n_fail: int,
                             dedicated_ok: bool, full_ok: bool, git: dict,
                             checks: list[dict]) -> None:
    reused, new_cells = [], []
    for s in scenario_ids:
        mp = out / f"matrix_{s}_manifest.json"
        if mp.exists():
            for c in json.loads(mp.read_text(encoding="utf-8")).get("cells", []):
                entry = (f"{s} {c.get('year')} blend_steps={c.get('blend_steps')} "
                         f"purpose={c.get('purpose', c.get('kind'))} "
                         f"n_lambdas={c.get('n_lambdas')}")
                (new_cells if c.get("kind") in ("anchor", "downward_search")
                 else reused).append(entry)

    closure = thr_long[thr_long["condition_name"] == "fleet_regret_closed_fraction"]
    plan = thr_long[thr_long["condition_name"] == "fuel_plan_agreement_1pp_share"]
    lines = [
        "# FuelEU internalization experiment -- completion report",
        "",
        "*Auto-drafted from the run manifests, the authoritative threshold long "
        "table and validation_results.json. The narrative sections are machine-"
        "summarized -- read them before treating this as manuscript-final.*",
        "",
        "## 1. Executive summary",
        f"- Scenarios: {scenario_ids}",
        f"- Validation: {'PASS' if n_fail == 0 else f'FAIL ({n_fail} issues)'} "
        f"over {len(checks)} checks",
        f"- Git commit: {git.get('git_commit')}",
        "",
        "## 2. Corrected closure bug",
        "- Closure is now recomputed by `recompute_global_closure_metrics()` AFTER "
        "the global merge/dedupe, against the row at exactly lambda=0 in each "
        "same-resolution group (located by value, never by position).",
        "- Per-cell closure from a local window (which normalized against that "
        "window's first row and therefore reset to 0 at e.g. lambda=0.080/0.800) is "
        "discarded and never reaches a delivered file.",
        "- A group without a same-resolution lambda=0 anchor is a hard error, not a "
        "silent fallback to the smallest tested lambda.",
        "",
        "## 3. Cells reused vs newly run",
        f"- Reused/standard cells: {len(reused)}",
        f"- New anchor / downward-search cells: {len(new_cells)}",
    ]
    lines += [f"  - {e}" for e in new_cells]
    lines += ["", "## 6. Corrected thresholds (tested intervals with status)"]
    for _, r in closure.iterrows():
        lines.append(
            f"- {r['scenario_id']} {r['year']} blend_steps={r['blend_steps']} "
            f"closure>={r['condition_threshold']}: {r['interval']} [{r['status']}]")
    lines += ["", "## 7. Fuel-plan agreement thresholds"]
    for _, r in plan.iterrows():
        lines.append(
            f"- {r['scenario_id']} {r['year']} blend_steps={r['blend_steps']} "
            f"plan agreement>={r['condition_threshold']}: {r['interval']} "
            f"[{r['status']}]")
    lines += [
        "",
        "## 10. Validation and test results",
        f"- validation_results.json: {n_fail} FAIL of {len(checks)} checks",
        f"- pytest tests/test_fueleu_internalization.py: {'PASS' if dedicated_ok else 'FAIL'}",
        f"- pytest -q (full repository suite): {'PASS' if full_ok else 'FAIL'}",
        "",
        "## 11. Deviations from the specification",
        "- `fuel_pathway`/`energy_dominant_fuel` are derived from the existing "
        "fixed-`base_fuel` architecture (base_fuel is one constant per optimizer "
        "call), not a new multi-base-fuel candidate search.",
        "- CII mode is selected via the existing `--case` flag rather than separate "
        "planner/evaluation CLI flags; every mandatory scenario ran "
        "FI_PRIMARY_NO_CII, so the dimension is not exercised.",
        "- `blend_resolution_*` / `fuel_menu_robustness_*` are filtered VIEWS over "
        "the corrected master tables, not a second computation path.",
        "- The downward-search primary condition is fuel-plan agreement rather than "
        "regret closure, because closure inside a single cell is not trustworthy "
        "before the global recompute.",
        "",
        "## 14. Remaining scientific limitations",
        "- lambda is a reduced-form decision-stage weight, not observed behaviour.",
        "- Fixed 2024 fleet/activity baseline.",
        "- No endogenous FuelEU credit-market equilibrium.",
        "- Fuel prices and feasible menus are scenario assumptions.",
        "- The 2045 central coarse-grid step near lambda=0.70 is a blend-grid "
        "discretization artifact and is reported as such, not as a threshold.",
    ]
    _write_atomic_text("\n".join(lines), out / "COMPLETION_REPORT.md")


def _write_manuscript_result_note(out: Path, curve_all: pd.DataFrame,
                                  thr_long: pd.DataFrame,
                                  transitions: pd.DataFrame) -> None:
    lines = [
        "# Manuscript-safe result note (auto-drafted from corrected outputs)", "",
        "## 1. Lambda=0 left endpoints (blend_steps=11)",
    ]
    for scen in sorted(_SCENARIOS):
        for yr in _SCENARIOS[scen]["years"]:
            g0 = curve_all[(curve_all["scenario_id"] == scen) & (curve_all["year"] == yr)
                          & (curve_all["blend_steps"] == 11)
                          & (curve_all["lambda_fueleu"].sub(0.0).abs() < 1e-12)]
            if len(g0):
                r = g0.iloc[0]
                lines.append(
                    f"- {scen} {yr}: mean RCI = {r['mean_rci_proxy']:.12f}; "
                    f"fleet regret = EUR {r['fleet_delta_cost_eur']:,.3f}; "
                    f"partial pathway = {r['fleet_modal_fuel_pathway_partial']}; "
                    f"joint pathway = {r['fleet_modal_fuel_pathway_joint']}")

    lines += ["", "## 2. Fleet-modal pathway transition intervals"]
    if len(transitions):
        for _, r in transitions[transitions["status"] == "bracketed"].iterrows():
            lines.append(
                f"- {r['scenario_id']} {r['year']} blend_steps={r['blend_steps']}: "
                f"{r['from_pathway']} -> {r['to_pathway']} {r['interval']}")
    else:
        lines.append("- (none detected on the tested grids)")

    lines += ["", "## 3. Corrected fleet-regret closure intervals",
              "(globally recomputed against each same-resolution lambda=0 anchor)"]
    closure = thr_long[thr_long["condition_name"] == "fleet_regret_closed_fraction"]
    for _, r in closure.iterrows():
        lines.append(
            f"- {r['scenario_id']} {r['year']} blend_steps={r['blend_steps']} "
            f">={r['condition_threshold']:.3f}: {r['interval']} [{r['status']}]")

    lines += [
        "", "## 4. Blend-resolution finding",
        "The principal transition window of each scenario was re-swept at "
        "blend_steps=11/21/51, each with its own lambda=0 and lambda=1 anchors, so "
        "closure at 21/51 points is normalized within its own resolution. Compare "
        "`blend_resolution_curve.csv` / `blend_resolution_thresholds.csv` before "
        "citing any exact threshold; differences across resolutions are numerical, "
        "not economic.",
        "",
        "## 5. Compatibility-constrained (2035) and no-FAME (2045) findings",
        "Both robustness scenarios were run over the complete 0-to-1 grid, and the "
        "no-FAME 51-point case received an adaptive downward search so its boundary "
        "is dated by a real bracket rather than reported as censored at the window "
        "edge. See `fuel_menu_robustness_*` and the long threshold table.",
        "",
        "## 6. Claims this experiment supports",
        "- lambda_fueleu is a reduced-form decision-stage FuelEU internalization "
        "weight.",
        "- FuelEU internalization changes fuel planning through discrete pathway "
        "switching boundaries, not a smooth response.",
        "- Switching boundaries are conditional on the feasible fuel menu and on "
        "numerical blend resolution.",
        "- All thresholds are tested-grid intervals with an explicit censoring "
        "status, never interpolated point estimates.",
        "",
        "## 7. Claims this experiment does NOT support (do not write these)",
        "- \"operators need X% awareness\" -- lambda is not an awareness rate.",
        "- lambda is a probability, knowledge share, or organizational maturity "
        "score.",
        "- lambda=0.70 in the 2045 central case is a second economic threshold "
        "(it is an 11-point blend-grid discretization artifact).",
        "- \"the proxy selects MGO\" where the actual energy pathway is VLSFO_only "
        "-- the legacy `dominant_fuel_*` label is an alternative-candidate label, "
        "not the consumed fuel.",
        "- thresholds are universal across fleets or firms.",
        "- RCI is a welfare loss.",
    ]
    _write_atomic_text("\n".join(lines), out / "MANUSCRIPT_RESULT_NOTE.md")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="FuelEU internalization orchestrator")
    ap.add_argument("--stage", choices=["matrix", "finalize", "all"], default="all")
    ap.add_argument("--scenario", choices=sorted(_SCENARIOS),
                    help="required for --stage matrix; ignored for finalize/all")
    ap.add_argument("--mrv", type=Path)
    ap.add_argument("--fleet", type=Path)
    ap.add_argument("--scenario-config", type=Path,
                    default=Path("configs/scenarios/base_2026.yaml"))
    ap.add_argument("--out", type=Path, default=Path("results/fueleu_internalization"))
    ap.add_argument("--fuel-price-scenario", default="base_2026")
    ap.add_argument("--eua", type=float, default=80.0)
    ap.add_argument("--case", choices=sorted(_CASE_SETTINGS), default="FI_PRIMARY_NO_CII")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--sample-limit", type=int, default=0)
    ap.add_argument("--refined-step", type=float, default=0.005)
    ap.add_argument("--fallback-step", type=float, default=0.001)
    ap.add_argument("--blend-steps-list", default="11,21,51")
    ap.add_argument("--skip-manuscript", action="store_true")
    ap.add_argument("--skip-packaging", action="store_true")
    ap.add_argument("--skip-tests", action="store_true",
                    help="do not run the test suite after the finalize stage")
    ap.add_argument("--variant", default=None,
                    help="model variant from configs/model_variants.yaml, passed to "
                         "every sweep cell (default: submitted-manuscript behaviour)")
    ap.add_argument("--fuel-capability", type=Path, default=None,
                    help="IMO fuel-capability CSV for the COMPATIBILITY_ENGINE scenario")
    args = ap.parse_args()
    if args.fuel_capability:
        _CAPABILITY_PATH[:] = [args.fuel_capability]
    if args.variant:
        from carbon_twin.model_variants import load_variant

        load_variant(args.variant)
        _VARIANT_ARGS[:] = ["--variant", args.variant]
    args.blend_steps_list = [int(x) for x in args.blend_steps_list.split(",") if x.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    if args.stage in ("matrix", "all") and (args.mrv is None or args.fleet is None):
        ap.error("--mrv and --fleet are required for --stage matrix/all")
    if args.stage == "finalize" and args.fleet is None:
        ap.error("--fleet is required for --stage finalize (fuel-menu feasibility audit)")

    if args.stage == "matrix":
        if not args.scenario:
            ap.error("--scenario is required for --stage matrix")
        stage_matrix(args)
    elif args.stage == "finalize":
        stage_finalize(args)
    else:
        for scen in sorted(_SCENARIOS):
            if _SCENARIOS[scen]["fuel_feasibility"] == "engine_capability" \
                    and not _CAPABILITY_PATH:
                continue
            args.scenario = scen
            stage_matrix(args)
        stage_finalize(args)


if __name__ == "__main__":
    main()
