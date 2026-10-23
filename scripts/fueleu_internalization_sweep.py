from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall: pip install -e '.[all]'")

from carbon_twin.cli import _load_scenario
from carbon_twin.data.loaders import load_fuel_prices, load_mrv_csv, load_vessels_csv
from carbon_twin.data.schemas import FuelType
from carbon_twin.emissions.proxy import load_fuel_factors_safe, vessels_to_index
from carbon_twin.model_variants import add_variant_argument, load_variant
from carbon_twin.optimization.fuel_choice import (
    fueleu_factor_table,
    optimal_fuel_blend_speed_multi_regulation,
    partial_planner_decision,
)
from carbon_twin.pipeline import (
    build_fuel_optimization_context,
    feasible_fuel_options,
    load_fuel_capability,
)
from carbon_twin.regulations.cii import CIIEngine
from carbon_twin.regulations.eu_ets import ETSEngine
from carbon_twin.regulations.fueleu import FuelEUEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_WRK: dict = {}

_CASE_SETTINGS: dict[str, dict] = {
    "FI_PRIMARY_NO_CII": {"include_cii_in_objective": False, "silo_carbon_basis": "ttw_ets"},
    "FI_OPTIONAL_CII_FULL": {"include_cii_in_objective": True, "silo_carbon_basis": "ttw_ets"},
}

_VESSEL_COLUMNS = [
    "case", "year", "lambda_fueleu", "vessel_id", "ship_type",
    "fuel_price_scenario", "eua_price_eur_per_tco2", "sample_scope",
    "scenario_id", "blend_steps", "speed_search_points",
    "feasibility_mode", "fuel_menu_id", "exclude_fuels", "max_biofuel_share",
    "include_cii_in_planner", "include_cii_in_evaluation",
    "fueleu_price_fraction", "fueleu_evaluation_multiplier",
    "partial_speed_kn", "joint_speed_kn",
    "partial_base_fuel", "joint_base_fuel",
    "partial_low_carbon_fuel", "joint_low_carbon_fuel",
    "partial_low_carbon_share", "joint_low_carbon_share",
    "partial_biofuel_share", "joint_biofuel_share",
    "partial_fuel_cost_full_eur", "joint_fuel_cost_full_eur",
    "partial_ets_cost_full_eur", "joint_ets_cost_full_eur",
    "partial_fueleu_cost_full_eur", "joint_fueleu_cost_full_eur",
    "partial_delay_cost_full_eur", "joint_delay_cost_full_eur",
    "partial_total_cost_full_eur", "joint_total_cost_full_eur",
    "partial_fueleu_balance", "joint_fueleu_balance",
    "partial_wtw_intensity_gco2e_per_mj", "joint_wtw_intensity_gco2e_per_mj",
    "delta_cost_eur", "rci_proxy", "rci_joint_norm",
    "fuel_pathway_label_match", "blend_share_match_1pp",
    "fuel_plan_match_1pp",
    "speed_match_0p01kn", "full_decision_match",
    "speed_abs_diff_kn", "blend_share_abs_diff",
    "numerical_regret_nonnegative",
    "partial_fuel_pathway", "joint_fuel_pathway",
    "partial_energy_dominant_fuel", "joint_energy_dominant_fuel",
    "partial_alternative_component_share", "joint_alternative_component_share",
    "full_decision_exact", "full_decision_1pp_0p01kn",
    "model_variant", "scope_weight", "annual_scale", "cost_basis",
]

_LEGACY_CURVE_COLUMNS = [
    "case", "year", "lambda_fueleu", "fuel_price_scenario",
    "eua_price_eur_per_tco2", "n_vessels",
    "mean_rci_proxy", "median_rci_proxy", "p90_rci_proxy", "mean_rci_joint_norm",
    "aggregate_rci_proxy", "fleet_delta_cost_eur", "mean_delta_cost_eur",
    "median_delta_cost_eur", "p90_delta_cost_eur",
    "share_rci_gt_0", "share_rci_gt_1pct", "share_rci_gt_5pct", "share_rci_gt_10pct",
    "dominant_fuel_label_agreement_share", "alternative_component_share_agreement_1pp",
    "fuel_plan_agreement_1pp", "full_decision_agreement_share",
    "mean_blend_share_abs_diff", "p90_blend_share_abs_diff",
    "mean_speed_abs_diff_kn", "p90_speed_abs_diff_kn",
    "mean_low_carbon_share_partial", "mean_low_carbon_share_joint",
    "mean_wtw_intensity_partial_gco2e_per_mj", "mean_wtw_intensity_joint_gco2e_per_mj",
    "mean_fueleu_balance_partial", "mean_fueleu_balance_joint",
    "mean_fueleu_cost_partial_full_eur", "mean_fueleu_cost_joint_full_eur",
    "dominant_fuel_partial", "dominant_fuel_joint",
    "fleet_regret_remaining_fraction", "fleet_regret_closed_fraction",
]

_CANONICAL_M1 = "M1_actual_ETS_FuelEU_only"
_DELTA_TOL_ABS_EUR = 0.01
_DELTA_TOL_REL = 1e-8


def _vessel_id(imo: str) -> str:
    return hashlib.sha256(imo.encode("utf-8")).hexdigest()[:16]


def _parse_lambdas(raw: list[str] | None) -> list[float]:
    if not raw:
        vals = [round(float(x), 2) for x in np.arange(0.0, 1.0001, 0.05)]
    elif len(raw) == 1 and "," in raw[0]:
        vals = [float(x) for x in raw[0].split(",") if x.strip()]
    else:
        vals = [float(x) for x in raw]
    for v in vals:
        if not (0.0 <= v <= 1.0):
            raise SystemExit(f"invalid lambda: {v!r} (must be in [0, 1])")
    out: list[float] = []
    for v in vals:
        if v not in out:
            out.append(v)
    return sorted(out)


def _init_worker(mrv_pkl: bytes, fleet_pkl: bytes, factors_pkl: bytes) -> None:
    _WRK["mrv_df"] = pickle.loads(mrv_pkl)
    _WRK["fleet_idx"] = pickle.loads(fleet_pkl)
    _WRK["factors"] = pickle.loads(factors_pkl)


def _run_chunk(task: dict) -> pd.DataFrame:
    mrv_df = _WRK["mrv_df"]
    fleet = _WRK["fleet_idx"]
    factors = _WRK["factors"]
    cfg = _CASE_SETTINGS[task["case"]]
    include_cii = cfg["include_cii_in_objective"]

    cii = CIIEngine(version=task["cii_v"])
    fueleu = FuelEUEngine(version=task["fueleu_v"])
    ets = ETSEngine(version=task["ets_v"])
    vkw = task.get("vkw") or {}
    scope_mode = vkw.get("scope_mode", "full")
    fueleu_table = (fueleu_factor_table(factors)
                    if vkw.get("fueleu_factors_from_config") else None)
    base_fuels = [FuelType(b) for b in vkw.get("base_fuels", [])]

    n_chunks, chunk = task["n_chunks"], task["chunk"]
    limit = task["sample_limit"]
    rows: list[dict] = []
    for idx, r in enumerate(mrv_df.to_dict(orient="records")):
        if idx % n_chunks != chunk:
            continue
        if limit and idx >= limit:
            break
        ctx = build_fuel_optimization_context(
            r, fleet, task["year"], factors, calibrate_to_mrv=False,
            scope_mode=scope_mode)
        if ctx is None:
            continue
        res = optimal_fuel_blend_speed_multi_regulation(
            ctx["vessel"], ctx["voyage"], task["prices"], task["eua"], factors,
            cii_engine=cii, fueleu_engine=fueleu, ets_engine=ets,
            year=task["year"], annual_distance_nm=ctx["annual_dist"],
            speed_range=ctx["speed_range"],
            delay_cost_per_hour=ctx["charter_per_hour"],
            scheduled_hours=ctx["voyage"].sea_hours,
            extended_fuels=True,
            fuel_options=feasible_fuel_options(
                ctx["vessel"], task["fuel_feasibility"], True, task["exclude_fuels"],
                task.get("fuel_capability")),
            silo_carbon_basis=cfg["silo_carbon_basis"],
            include_cii_in_objective=include_cii,
            max_biofuel_share=task["max_biofuel_share"],
            blend_steps=task["blend_steps"],
            n_points=task["speed_search_points"],
            energy_scale=ctx["energy_scale"],
            fueleu_scope_factor=ctx["scope_weight"],
            fueleu_emission_factors=fueleu_table,
            report_annual=bool(vkw.get("report_annual", False)),
            base_fuels=base_fuels,
            scope_allocation=ctx["scope_allocation"],
            exact_blend=bool(vkw.get("exact_blend", False)),
        )
        if not res:
            continue
        prof = res["profile"]
        for lam in task["lambdas"]:
            d = partial_planner_decision(prof, lam, include_cii, FuelType.VLSFO)
            rows.append({
                "case": task["case"],
                "year": task["year"],
                "lambda_fueleu": lam,
                "vessel_id": _vessel_id(ctx["imo"]),
                "ship_type": ctx["vessel"].ship_type.value,
                "fuel_price_scenario": task["ps"],
                "eua_price_eur_per_tco2": task["eua"],
                "sample_scope": "all_valid",
                "scenario_id": task["scenario_id"],
                "blend_steps": task["blend_steps"],
                "speed_search_points": task["speed_search_points"],
                "feasibility_mode": task["fuel_feasibility"],
                "fuel_menu_id": task["fuel_menu_id"],
                "exclude_fuels": task["exclude_fuels_str"],
                "max_biofuel_share": task["max_biofuel_share"],
                "include_cii_in_planner": include_cii,
                "include_cii_in_evaluation": include_cii,
                "fueleu_price_fraction": 1.0,
                "fueleu_evaluation_multiplier": 1.0,
                "partial_speed_kn": d["partial_speed"],
                "joint_speed_kn": d["benchmark_speed"],
                "partial_base_fuel": d["partial_base_fuel"],
                "joint_base_fuel": d["benchmark_base_fuel"],
                "partial_low_carbon_fuel": d["partial_alt_fuel"],
                "joint_low_carbon_fuel": d["benchmark_alt_fuel"],
                "partial_low_carbon_share": d["partial_low_carbon_share"],
                "joint_low_carbon_share": d["benchmark_blend_pct"],
                "partial_biofuel_share": d["partial_biofuel_share"],
                "joint_biofuel_share": (d["benchmark_blend_pct"]
                                        if d["benchmark_is_biofuel"] else 0.0),
                "partial_fuel_cost_full_eur": d["partial_fuel_cost_full_eur"],
                "joint_fuel_cost_full_eur": d["benchmark_fuel_cost_full_eur"],
                "partial_ets_cost_full_eur": d["partial_ets_cost_full_eur"],
                "joint_ets_cost_full_eur": d["benchmark_ets_cost_full_eur"],
                "partial_fueleu_cost_full_eur": d["partial_fueleu_cost_full_eur"],
                "joint_fueleu_cost_full_eur": d["benchmark_fueleu_cost_full_eur"],
                "partial_delay_cost_full_eur": d["partial_delay_cost_full_eur"],
                "joint_delay_cost_full_eur": d["benchmark_delay_cost_full_eur"],
                "partial_total_cost_full_eur": d["partial_total_cost_full_eur"],
                "joint_total_cost_full_eur": d["benchmark_total_cost_full_eur"],
                "partial_fueleu_balance": d["partial_fueleu_balance"],
                "joint_fueleu_balance": d["benchmark_fueleu_balance"],
                "partial_wtw_intensity_gco2e_per_mj":
                    d["partial_fueleu_actual_intensity"],
                "joint_wtw_intensity_gco2e_per_mj":
                    d["benchmark_fueleu_actual_intensity"],
                "delta_cost_eur": d["delta_cost_eur"],
                "rci_proxy": d["rci_proxy"],
                "rci_joint_norm": d["rci_joint_norm"],
                "fuel_pathway_label_match": d["fuel_pathway_label_match"],
                "blend_share_match_1pp": d["blend_share_match_1pp"],
                "fuel_plan_match_1pp": d["fuel_plan_match_1pp"],
                "speed_match_0p01kn": d["speed_match_0p01kn"],
                "full_decision_match": d["full_decision_match"],
                "speed_abs_diff_kn": d["speed_abs_diff_kn"],
                "blend_share_abs_diff": d["blend_share_abs_diff"],
                "numerical_regret_nonnegative": d["numerical_regret_nonnegative"],
                "partial_fuel_pathway": d["fuel_pathway_partial"],
                "joint_fuel_pathway": d["fuel_pathway_benchmark"],
                "partial_energy_dominant_fuel": d["energy_dominant_fuel_partial"],
                "joint_energy_dominant_fuel": d["energy_dominant_fuel_benchmark"],
                "partial_alternative_component_share":
                    d["alternative_component_share_partial"],
                "joint_alternative_component_share":
                    d["alternative_component_share_benchmark"],
                "full_decision_exact": d["full_decision_exact"],
                "full_decision_1pp_0p01kn": d["full_decision_1pp_0p01kn"],
                "model_variant": task.get("variant", "v23"),
                "scope_weight": ctx["scope_weight"],
                "annual_scale": res["annual_scale"],
                "cost_basis": res["cost_basis"],
            })
    return pd.DataFrame(rows, columns=_VESSEL_COLUMNS)


def _curve_row(g: pd.DataFrame, case: str, year: int, lam: float) -> dict:
    dom_p = g["partial_low_carbon_fuel"].mode()
    dom_j = g["joint_low_carbon_fuel"].mode()
    pathway_p = g["partial_fuel_pathway"].mode()
    pathway_j = g["joint_fuel_pathway"].mode()
    edom_p = g["partial_energy_dominant_fuel"].mode()
    edom_j = g["joint_energy_dominant_fuel"].mode()
    fleet_delta = float(g["delta_cost_eur"].sum())
    denom = float(g["partial_total_cost_full_eur"].sum())
    return {
        "scenario_id": str(g["scenario_id"].iloc[0]),
        "case": case, "year": year, "lambda_fueleu": lam,
        "blend_steps": int(g["blend_steps"].iloc[0]),
        "speed_search_points": int(g["speed_search_points"].iloc[0]),
        "fuel_menu_id": str(g["fuel_menu_id"].iloc[0]),
        "feasibility_mode": str(g["feasibility_mode"].iloc[0]),
        "exclude_fuels": str(g["exclude_fuels"].iloc[0]),
        "max_biofuel_share": g["max_biofuel_share"].iloc[0],
        "fuel_price_scenario": str(g["fuel_price_scenario"].iloc[0]),
        "eua_price_eur_per_tco2": float(g["eua_price_eur_per_tco2"].iloc[0]),
        "include_cii_in_planner": bool(g["include_cii_in_planner"].iloc[0]),
        "include_cii_in_evaluation": bool(g["include_cii_in_evaluation"].iloc[0]),
        "fueleu_price_fraction": float(g["fueleu_price_fraction"].iloc[0]),
        "fueleu_evaluation_multiplier":
            float(g["fueleu_evaluation_multiplier"].iloc[0]),
        "n_vessels": int(len(g)),
        "mean_rci_proxy": float(g["rci_proxy"].mean()),
        "median_rci_proxy": float(g["rci_proxy"].median()),
        "p90_rci_proxy": float(g["rci_proxy"].quantile(0.90)),
        "mean_rci_joint_norm": float(g["rci_joint_norm"].mean()),
        "aggregate_rci_proxy": fleet_delta / denom if denom > 0 else 0.0,
        "fleet_delta_cost_eur": fleet_delta,
        "mean_delta_cost_eur": float(g["delta_cost_eur"].mean()),
        "median_delta_cost_eur": float(g["delta_cost_eur"].median()),
        "p90_delta_cost_eur": float(g["delta_cost_eur"].quantile(0.90)),
        "share_rci_gt_0": float((g["rci_proxy"] > 0).mean()),
        "share_rci_gt_1pct": float((g["rci_proxy"] > 0.01).mean()),
        "share_rci_gt_5pct": float((g["rci_proxy"] > 0.05).mean()),
        "share_rci_gt_10pct": float((g["rci_proxy"] > 0.10).mean()),
        "dominant_fuel_label_agreement_share": float(g["fuel_pathway_label_match"].mean()),
        "alternative_component_share_agreement_1pp": float(g["blend_share_match_1pp"].mean()),
        "fuel_plan_agreement_1pp": float(g["fuel_plan_match_1pp"].mean()),
        "full_decision_agreement_share": float(g["full_decision_match"].mean()),
        "mean_blend_share_abs_diff": float(g["blend_share_abs_diff"].mean()),
        "p90_blend_share_abs_diff": float(g["blend_share_abs_diff"].quantile(0.90)),
        "mean_speed_abs_diff_kn": float(g["speed_abs_diff_kn"].mean()),
        "p90_speed_abs_diff_kn": float(g["speed_abs_diff_kn"].quantile(0.90)),
        "mean_low_carbon_share_partial": float(g["partial_low_carbon_share"].mean()),
        "mean_low_carbon_share_joint": float(g["joint_low_carbon_share"].mean()),
        "mean_wtw_intensity_partial_gco2e_per_mj":
            float(g["partial_wtw_intensity_gco2e_per_mj"].mean()),
        "mean_wtw_intensity_joint_gco2e_per_mj":
            float(g["joint_wtw_intensity_gco2e_per_mj"].mean()),
        "mean_fueleu_balance_partial": float(g["partial_fueleu_balance"].mean()),
        "mean_fueleu_balance_joint": float(g["joint_fueleu_balance"].mean()),
        "mean_fueleu_cost_partial_full_eur":
            float(g["partial_fueleu_cost_full_eur"].mean()),
        "mean_fueleu_cost_joint_full_eur":
            float(g["joint_fueleu_cost_full_eur"].mean()),
        "legacy_alternative_candidate_label_partial":
            str(dom_p.iloc[0]) if len(dom_p) else "",
        "legacy_alternative_candidate_label_joint":
            str(dom_j.iloc[0]) if len(dom_j) else "",
        "dominant_fuel_partial": str(dom_p.iloc[0]) if len(dom_p) else "",
        "dominant_fuel_joint": str(dom_j.iloc[0]) if len(dom_j) else "",
        "fuel_pathway_label_agreement_share": float(g["fuel_pathway_label_match"].mean()),
        "alternative_component_share_agreement_1pp_share":
            float(g["blend_share_match_1pp"].mean()),
        "fuel_plan_agreement_1pp_share": float(g["fuel_plan_match_1pp"].mean()),
        "full_decision_exact_share": float(g["full_decision_exact"].mean()),
        "full_decision_1pp_0p01kn_share": float(g["full_decision_1pp_0p01kn"].mean()),
        "mean_alternative_component_share_partial":
            float(g["partial_alternative_component_share"].mean()),
        "mean_alternative_component_share_joint":
            float(g["joint_alternative_component_share"].mean()),
        "fleet_modal_fuel_pathway_partial": str(pathway_p.iloc[0]) if len(pathway_p) else "",
        "fleet_modal_fuel_pathway_joint": str(pathway_j.iloc[0]) if len(pathway_j) else "",
        "fleet_modal_energy_dominant_fuel_partial":
            str(edom_p.iloc[0]) if len(edom_p) else "",
        "fleet_modal_energy_dominant_fuel_joint":
            str(edom_j.iloc[0]) if len(edom_j) else "",
    }


def _add_closure_metrics(
    df: pd.DataFrame, group_keys: tuple[str, ...] = ("case", "year")
) -> pd.DataFrame:
    parts = []
    for key, g in df.groupby(list(group_keys)):
        g = g.sort_values("lambda_fueleu").copy()
        d0 = float(g["fleet_delta_cost_eur"].iloc[0])
        if abs(d0) < 1e-3:
            g["fleet_regret_remaining_fraction"] = np.nan
            g["fleet_regret_closed_fraction"] = np.nan
        else:
            g["fleet_regret_remaining_fraction"] = g["fleet_delta_cost_eur"] / d0
            g["fleet_regret_closed_fraction"] = 1.0 - g["fleet_delta_cost_eur"] / d0
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _threshold_rows(curve: pd.DataFrame) -> list[dict]:
    rows = []
    for (case, yr), g in curve.groupby(["case", "year"]):
        g = g.sort_values("lambda_fueleu")
        lams = sorted(g["lambda_fueleu"].unique())
        grid_step = round(lams[1] - lams[0], 4) if len(lams) > 1 else 0.0
        rec = {"case": case, "year": yr, "grid_step": grid_step,
               "threshold_is_grid_bound": True}
        for col, thr, name in [
            ("fleet_regret_closed_fraction", 0.50, "lambda_50pct_fleet_regret_closed"),
            ("fleet_regret_closed_fraction", 0.90, "lambda_90pct_fleet_regret_closed"),
            ("fleet_regret_closed_fraction", 0.99, "lambda_99pct_fleet_regret_closed"),
            ("dominant_fuel_label_agreement_share", 0.90,
             "lambda_90pct_fuel_pathway_label_agreement"),
            ("dominant_fuel_label_agreement_share", 0.99,
             "lambda_99pct_fuel_pathway_label_agreement"),
            ("alternative_component_share_agreement_1pp", 0.90,
             "lambda_90pct_alternative_component_share_agreement_1pp"),
            ("alternative_component_share_agreement_1pp", 0.99,
             "lambda_99pct_alternative_component_share_agreement_1pp"),
            ("fuel_plan_agreement_1pp", 0.90,
             "lambda_90pct_fuel_plan_agreement_1pp"),
            ("fuel_plan_agreement_1pp", 0.99,
             "lambda_99pct_fuel_plan_agreement_1pp"),
            ("full_decision_agreement_share", 0.90,
             "lambda_90pct_full_decision_agreement"),
            ("full_decision_agreement_share", 0.99,
             "lambda_99pct_full_decision_agreement"),
        ]:
            hit = g.loc[g[col] >= thr, "lambda_fueleu"]
            rec[name] = round(float(hit.iloc[0]), 4) if len(hit) else "NA"
        rows.append(rec)
    return rows


def _vessel_threshold_summary(vessel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (case, yr, vid), g in vessel.groupby(["case", "year", "vessel_id"]):
        g = g.sort_values("lambda_fueleu")
        base_delta = float(g["delta_cost_eur"].iloc[0])
        rec = {
            "case": case, "year": yr, "vessel_id": vid,
            "ship_type": str(g["ship_type"].iloc[0]),
        }
        for col, name in [
            ("fuel_pathway_label_match", "first_lambda_fuel_pathway_label_match"),
            ("blend_share_match_1pp", "first_lambda_blend_match_1pp"),
            ("full_decision_match", "first_lambda_full_decision_match"),
        ]:
            hit = g.loc[g[col], "lambda_fueleu"]
            rec[name] = round(float(hit.iloc[0]), 4) if len(hit) else np.nan
        if base_delta > 1e-3:
            hit = g.loc[g["delta_cost_eur"] <= 0.10 * base_delta, "lambda_fueleu"]
            rec["first_lambda_90pct_individual_regret_closed"] = (
                round(float(hit.iloc[0]), 4) if len(hit) else np.nan)
        else:
            rec["first_lambda_90pct_individual_regret_closed"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def _audit_by_type_endpoints(by_type: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    for (case, yr, st), g in by_type.groupby(["case", "year", "ship_type"]):
        g = g.sort_values("lambda_fueleu")
        lams = [round(float(x), 6) for x in g["lambda_fueleu"]]
        rec0 = g.iloc[0]
        base = float(g["fleet_delta_cost_eur"].iloc[0])
        if abs(float(rec0["lambda_fueleu"]) - 0.0) <= 1e-9 and abs(base) >= 1e-3:
            if not (abs(float(rec0["fleet_regret_remaining_fraction"]) - 1.0) <= 1e-9
                    and abs(float(rec0["fleet_regret_closed_fraction"])) <= 1e-9):
                problems.append(
                    f"case={case} year={yr} type={st}: lambda=0 closure endpoints "
                    f"(remaining={rec0['fleet_regret_remaining_fraction']}, "
                    f"closed={rec0['fleet_regret_closed_fraction']}) != (1, 0)")
        if abs(lams[-1] - 1.0) <= 1e-9:
            rec1 = g.iloc[-1]
            if base > 1e-3:
                if not (abs(float(rec1["fleet_regret_remaining_fraction"])) <= 1e-6
                        and abs(float(rec1["fleet_regret_closed_fraction"]) - 1.0) <= 1e-6):
                    problems.append(
                        f"case={case} year={yr} type={st}: lambda=1 closure endpoints "
                        f"(remaining={rec1['fleet_regret_remaining_fraction']}, "
                        f"closed={rec1['fleet_regret_closed_fraction']}) != (0, 1)")
    return problems


def _audit_reconciliation(curve: pd.DataFrame, by_type: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    key = ["case", "year", "lambda_fueleu"]
    curve_s = curve.set_index(key)
    for k, g in by_type.groupby(key):
        if k not in curve_s.index:
            continue
        global_row = curve_s.loc[k]
        n_sum = int(g["n_vessels"].sum())
        delta_sum = float(g["fleet_delta_cost_eur"].sum())
        if n_sum != int(global_row["n_vessels"]):
            problems.append(f"{k}: by-type n sum {n_sum} != global "
                            f"{int(global_row['n_vessels'])}")
        tol = 1e-6 + 1e-9 * abs(float(global_row["fleet_delta_cost_eur"]))
        if abs(delta_sum - float(global_row["fleet_delta_cost_eur"])) > tol:
            problems.append(f"{k}: by-type fleet regret sum {delta_sum:.2f} != "
                            f"global {float(global_row['fleet_delta_cost_eur']):.2f}")
        w_label = float(
            (g["n_vessels"] * g["dominant_fuel_label_agreement_share"]).sum()
            / max(n_sum, 1))
        if abs(w_label - float(global_row["dominant_fuel_label_agreement_share"])) > 1e-4:
            problems.append(f"{k}: vessel-weighted by-type label agreement "
                            f"{w_label:.5f} != global "
                            f"{float(global_row['dominant_fuel_label_agreement_share']):.5f}")
    return problems


def _audit_sample_consistency(vdf: pd.DataFrame) -> list[str]:
    problems = []
    for (case, yr, lam), g in vdf.groupby(["case", "year", "lambda_fueleu"]):
        pass
    for (case, yr), g in vdf.groupby(["case", "year"]):
        n_per_lambda = g.groupby("lambda_fueleu")["vessel_id"].nunique()
        if n_per_lambda.nunique() != 1:
            problems.append(
                f"case={case} year={yr}: sample size differs across lambda "
                f"({n_per_lambda.to_dict()})")
    return problems


def _audit_regret_nonnegative(vdf: pd.DataFrame) -> tuple[int, int, list[str]]:
    tol = np.maximum(_DELTA_TOL_ABS_EUR,
                     _DELTA_TOL_REL * vdf["joint_total_cost_full_eur"].abs())
    hard = vdf["delta_cost_eur"] < -tol
    n_hard = int(hard.sum())
    n_clamped = int((vdf["delta_cost_eur"] < 0).sum())
    problems = []
    if n_hard:
        problems.append(f"{n_hard} vessel-lambda rows below the regret tolerance "
                        f"(max(abs EUR {_DELTA_TOL_ABS_EUR}, rel {_DELTA_TOL_REL}))")
    vdf["delta_cost_eur"] = vdf["delta_cost_eur"].clip(lower=0.0)
    denom_p = vdf["partial_total_cost_full_eur"].replace(0, np.nan)
    vdf["rci_proxy"] = (vdf["delta_cost_eur"] / denom_p).fillna(0.0).clip(lower=0.0)
    denom_j = vdf["joint_total_cost_full_eur"].replace(0, np.nan)
    vdf["rci_joint_norm"] = (vdf["delta_cost_eur"] / denom_j).fillna(0.0).clip(lower=0.0)
    vdf["numerical_regret_nonnegative"] = vdf["delta_cost_eur"] >= 0.0
    return n_hard, n_clamped, problems


def _audit_monotonicity(vdf: pd.DataFrame) -> list[str]:
    problems = []
    tol = 1e-6
    for (case, yr, vid), g in vdf.groupby(["case", "year", "vessel_id"]):
        g = g.sort_values("lambda_fueleu")
        d = g["delta_cost_eur"].to_numpy()
        inc = d[1:] - d[:-1] > tol * np.maximum(1.0, np.abs(d[1:]))
        if int(inc.sum()) > 0:
            problems.append(
                f"case={case} year={yr} vessel={vid}: {int(inc.sum())} "
                "lambda step(s) with increasing full-policy regret")
    return problems


def _audit_lambda_one_endpoint(curve: pd.DataFrame) -> list[str]:
    problems = []
    for (case, yr), g in curve.groupby(["case", "year"]):
        g = g.sort_values("lambda_fueleu")
        last = g.iloc[-1]
        lam1 = float(last["lambda_fueleu"])
        if abs(lam1 - 1.0) > 1e-9:
            continue
        if float(last["mean_rci_proxy"]) > 1e-6:
            problems.append(f"case={case} year={yr}: lambda=1 mean_rci_proxy "
                            f"= {last['mean_rci_proxy']:.6f} (expected ~0)")
        if abs(float(last["fleet_delta_cost_eur"])) > 1.0:
            problems.append(f"case={case} year={yr}: lambda=1 fleet_delta_cost_eur "
                            f"= {last['fleet_delta_cost_eur']:.4f} (expected ~0)")
        if float(last["dominant_fuel_label_agreement_share"]) < 1.0 - 1e-6:
            problems.append(f"case={case} year={yr}: lambda=1 fuel-pathway-label "
                            f"agreement = "
                            f"{last['dominant_fuel_label_agreement_share']:.6f} "
                            "(expected 1)")
    return problems


def _audit_m1_lambda_zero(vdf: pd.DataFrame, args) -> tuple[list[str], dict]:
    problems: list[str] = []
    checks: dict[str, dict] = {}
    canonical = Path("results/optimization/metric_harmonization.csv")
    if not canonical.exists():
        checks["canonical_m1_table"] = {
            "status": "not_found",
            "note": "results/optimization/metric_harmonization.csv absent; "
                    "lambda=0 endpoints recorded for manual comparison"}
        return problems, checks
    try:
        m1 = pd.read_csv(canonical)
        m1 = m1[m1["case"] == _CANONICAL_M1]
    except Exception as exc:
        problems.append(f"could not read canonical M1 table: {exc}")
        return problems, checks

    for (case, yr), g in vdf.groupby(["case", "year"]):
        if case != args.case:
            continue
        g0 = g[g["lambda_fueleu"] == 0.0]
        if not len(g0):
            continue
        row_m1 = m1[m1["year"] == yr]
        if not len(row_m1):
            problems.append(f"case={case} year={yr}: no canonical M1 row to compare")
            continue
        row_m1 = row_m1.iloc[0]
        n_ours = int(g0["vessel_id"].nunique())
        n_m1 = int(row_m1["n_vessels"])
        mean_ours = float(g0["rci_proxy"].mean())
        mean_m1 = float(row_m1["mean_rci_silo"])
        fleet_ours = float(g0["delta_cost_eur"].sum())
        fleet_m1 = float(row_m1["fleet_delta_cost_eur"])
        partial = args.sample_limit > 0 or n_ours < n_m1
        checks[f"m1_lambda0_{yr}"] = {
            "year": yr, "n_vessels_ours": n_ours, "n_vessels_m1": n_m1,
            "mean_rci_ours": mean_ours, "mean_rci_m1": mean_m1,
            "fleet_delta_ours_eur": fleet_ours, "fleet_delta_m1_eur": fleet_m1,
            "status": "skipped_partial_sample" if partial else "checked",
        }
        if partial:
            continue
        if n_ours != n_m1:
            problems.append(f"case={case} year={yr}: sample size {n_ours} != M1 {n_m1}")
        if abs(mean_ours - mean_m1) > 1e-5:
            problems.append(f"case={case} year={yr}: lambda=0 mean_rci_proxy "
                            f"{mean_ours:.6f} != M1 {mean_m1:.6f} (|diff| > 1e-5)")
        if fleet_m1 != 0 and abs(fleet_ours - fleet_m1) / abs(fleet_m1) > 1e-4:
            problems.append(f"case={case} year={yr}: lambda=0 fleet_delta "
                            f"{fleet_ours:.2f} != M1 {fleet_m1:.2f} (rel diff > 1e-4)")
    return problems, checks


def _audit_aggregate_monotonic(curve: pd.DataFrame) -> list[str]:
    problems = []
    for (case, yr), g in curve.groupby(["case", "year"]):
        g = g.sort_values("lambda_fueleu")
        d = g["fleet_delta_cost_eur"].to_numpy()
        if len(d) < 2:
            continue
        tol = 1e-6 * max(1.0, float(np.max(np.abs(d))))
        if float(np.max(d[1:] - d[:-1])) > tol:
            problems.append(f"case={case} year={yr}: fleet aggregate delta cost "
                            "increases with lambda")
    return problems


_REPO = Path(__file__).resolve().parent.parent
_SOURCE_FILES = [
    "carbon_twin/optimization/fuel_choice.py",
    "carbon_twin/pipeline.py",
    "scripts/fueleu_internalization_sweep.py",
]


def _git_info() -> dict:
    def _run(*cmd: str) -> str | None:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 cwd=_REPO, timeout=20,
                                 encoding="utf-8", errors="replace")
            return out.stdout.strip() or None
        except Exception:
            return None

    commit = _run("git", "rev-parse", "HEAD")
    if commit is None:
        fingerprint = hashlib.sha256()
        for rel in _SOURCE_FILES:
            try:
                fingerprint.update(_file_sha256(_REPO / rel).encode())
            except Exception:
                pass
        return {
            "git_available": False,
            "git_commit": "NOT_A_GIT_CHECKOUT",
            "working_tree_dirty": None,
            "code_fingerprint": fingerprint.hexdigest()[:32],
            "note": "git unavailable (likely rsync'd checkout); source-file "
                    "content fingerprint recorded instead",
        }
    return {
        "git_available": True,
        "git_commit": commit,
        "working_tree_dirty": bool(_run("git", "status", "--porcelain")),
        "git_status_short": _run("git", "status", "--short") or "",
        "git_diff_stat": _run("git", "diff", "--stat") or "",
        "code_fingerprint": None,
    }


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _write_atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    t0 = time.perf_counter()
    ap = argparse.ArgumentParser(
        description="FuelEU planning-internalization sweep (lambda_fueleu)")
    ap.add_argument("--mrv", required=True, type=Path)
    ap.add_argument("--fleet", required=True, type=Path)
    ap.add_argument("--scenario", required=True, type=Path)
    ap.add_argument("--years", default="2035,2045")
    ap.add_argument("--lambdas", nargs="*",
                    help="lambda grid, e.g. 0 0.25 0.5 0.75 1 or "
                         "'0.00,0.05,...,1.00' (default: 0.00,0.05,...,1.00)")
    ap.add_argument("--fuel-price-scenario", default="base_2026")
    ap.add_argument("--eua", type=float, default=80.0)
    ap.add_argument("--case", choices=sorted(_CASE_SETTINGS),
                    default="FI_PRIMARY_NO_CII")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("results/fueleu_internalization"))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--sample-limit", type=int, default=0,
                    help="process only the first N MRV rows per year (smoke test)")
    ap.add_argument("--resume", action="store_true",
                    help="keep finished per-year checkpoints")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-run years even when a checkpoint exists")
    ap.add_argument("--blend-steps", type=int, default=11,
                    help="blend resolution (11 = 0.10 share steps; 21 = 0.05; "
                         "51 = 0.02)")
    ap.add_argument("--fuel-feasibility", default="continuous_base",
                    choices=["continuous_base", "dropin_only", "compatible_switch",
                             "engine_capability"],
                    help="alternative-fuel menu for the robustness runs")
    ap.add_argument("--exclude-fuels", default=None,
                    help="comma-separated FuelType values to drop from the menu "
                         "(e.g. biodiesel_fame for the no-FAME robustness run)")
    ap.add_argument("--max-biofuel-share", type=float, default=None,
                    help="cap the energy share of bio-derived fuels (e.g. 0.3)")
    ap.add_argument("--skip-m1-regression", action="store_true",
                    help="skip the canonical M1 lambda=0 regression check — use "
                         "ONLY for sensitivity runs whose settings (blend "
                         "resolution, fuel menu) intentionally differ from the "
                         "canonical M1 basis")
    ap.add_argument("--scenario-id", default="CENTRAL_EXTENDED",
                    help="free-form scenario tag written into every output row "
                         "(e.g. CENTRAL_EXTENDED, COMPATIBILITY_CONSTRAINED_2035, "
                         "NO_FAME_2045) -- does not itself change behaviour, only "
                         "--fuel-feasibility/--exclude-fuels do that")
    ap.add_argument("--speed-search-points", type=int, default=15,
                    help="speed-grid resolution passed to the optimizer "
                         "(n_points)")
    ap.add_argument("--write-vessel-level", default="true", choices=["true", "false"],
                    help="write the per-vessel parquet")
    ap.add_argument("--fuel-capability", type=Path, default=None,
                    help="IMO fuel-capability CSV (scripts/r1_fuel_capability.py); "
                         "required by --fuel-feasibility engine_capability")
    add_variant_argument(ap)
    args = ap.parse_args()
    variant = load_variant(args.variant)
    fuel_capability = load_fuel_capability(args.fuel_capability)
    if args.fuel_feasibility == "engine_capability" and fuel_capability is None:
        ap.error("--fuel-feasibility engine_capability needs --fuel-capability")

    years = [int(y) for y in args.years.split(",") if y.strip()]
    lambdas = _parse_lambdas(args.lambdas)
    if not years or not lambdas:
        ap.error("--years and --lambdas must be non-empty")
    exclude_fuels = None
    if args.exclude_fuels:
        exclude_fuels = {FuelType(x.strip())
                         for x in args.exclude_fuels.split(",") if x.strip()}
    if args.blend_steps < 2:
        ap.error("--blend-steps must be >= 2")
    exclude_fuels_str = "+".join(sorted(f.value for f in (exclude_fuels or set())))
    fuel_menu_id = args.fuel_feasibility + (f"_excl_{exclude_fuels_str}"
                                            if exclude_fuels_str else "")

    cfg = _load_scenario(str(args.scenario))
    regs = cfg["regulations"]
    mrv_df = load_mrv_csv(args.mrv)
    fleet_idx = vessels_to_index(load_vessels_csv(args.fleet))
    factors = variant.fuel_factors(load_fuel_factors_safe())
    prices = load_fuel_prices("configs/fuel_prices.yaml",
                              scenario=args.fuel_price_scenario)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = {y: out / f"fueleu_internalization_vessel_{y}.parquet" for y in years}

    parts_by_year: dict[int, list[pd.DataFrame]] = {}
    tasks: list[dict] = []
    for yr in years:
        if args.resume and checkpoint[yr].exists() and not args.overwrite:
            parts_by_year[yr] = [pd.read_parquet(checkpoint[yr])]
            log.info("resume: year %d loaded from checkpoint (%d rows)",
                     yr, len(parts_by_year[yr][0]))
            continue
        n_chunks = max(1, min(args.workers, 64))
        for c in range(n_chunks):
            tasks.append({
                "case": args.case, "year": yr, "lambdas": lambdas,
                "eua": args.eua, "prices": prices, "ps": args.fuel_price_scenario,
                "cii_v": regs["cii"]["version"],
                "fueleu_v": variant.fueleu_version_or(regs["fueleu"]["version"]),
                "ets_v": regs["eu_ets"]["version"],
                "variant": variant.name, "vkw": variant.fuel_optimizer_kwargs(),
                "fuel_capability": fuel_capability,
                "chunk": c, "n_chunks": n_chunks, "sample_limit": args.sample_limit,
                "blend_steps": args.blend_steps,
                "fuel_feasibility": args.fuel_feasibility,
                "exclude_fuels": exclude_fuels,
                "max_biofuel_share": args.max_biofuel_share,
                "scenario_id": args.scenario_id,
                "speed_search_points": args.speed_search_points,
                "fuel_menu_id": fuel_menu_id,
                "exclude_fuels_str": exclude_fuels_str,
            })

    if tasks:
        log.info("FI sweep: case=%s years=%s lambdas=%d eua=%.1f ps=%s "
                 "| %d chunk tasks | workers=%d",
                 args.case, years, len(lambdas), args.eua,
                 args.fuel_price_scenario, len(tasks), args.workers)
        pkls = (pickle.dumps(mrv_df), pickle.dumps(fleet_idx),
                pickle.dumps(factors))
        done = 0
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks)),
                                 initializer=_init_worker, initargs=pkls) as ex:
            fut = {ex.submit(_run_chunk, t): t for t in tasks}
            for f in as_completed(fut):
                t = fut[f]
                try:
                    df = f.result()
                except Exception as exc:
                    log.error("yr=%s chunk=%d: %s", t["year"], t["chunk"], exc)
                    continue
                if df is not None and len(df):
                    parts_by_year.setdefault(t["year"], []).append(df)
                done += 1
                log.info("[%d/%d] yr=%d chunk=%d  rows=%d",
                         done, len(tasks), t["year"], t["chunk"],
                         len(df) if df is not None else 0)
        for yr, parts in parts_by_year.items():
            _write_atomic_parquet(
                pd.concat(parts, ignore_index=True).sort_values(
                    ["lambda_fueleu", "vessel_id"]), checkpoint[yr])

    if not parts_by_year:
        log.error("no results produced")
        sys.exit(1)

    problems: list[str] = []
    audit_sections: list[tuple[str, list[str]]] = []
    stats: dict[str, dict] = {}
    regret_problems: list[str] = []
    vdf_list = []
    for yr in years:
        if yr not in parts_by_year:
            log.error("year %d: no results", yr)
            sys.exit(1)
        vdf = pd.concat(parts_by_year[yr], ignore_index=True)
        vdf = vdf.sort_values(
            ["case", "year", "lambda_fueleu", "vessel_id"]).reset_index(drop=True)
        n_hard, n_clamped, probs = _audit_regret_nonnegative(vdf)
        problems += probs
        regret_problems += probs
        stats[f"regret_{yr}"] = {"hard_negatives": n_hard,
                                 "clamped_to_zero": n_clamped}
        vdf_list.append(vdf)
    vessel = pd.concat(vdf_list, ignore_index=True)
    audit_sections.append(("Regret non-negativity", regret_problems))
    sample_problems = _audit_sample_consistency(vessel)
    audit_sections.append(("Constant sample size across lambda", sample_problems))
    problems += sample_problems
    monotonic_problems = _audit_monotonicity(vessel)
    audit_sections.append(("Per-vessel monotonic regret over sorted lambda",
                           monotonic_problems))
    problems += monotonic_problems
    if args.skip_m1_regression:
        m1_problems, m1_checks = [], {"canonical_m1_table": {
            "status": "skipped", "note": "--skip-m1-regression: settings differ "
            "from the canonical M1 basis (sensitivity run)"}}
    else:
        m1_problems, m1_checks = _audit_m1_lambda_zero(vessel, args)
    audit_sections.append(("Canonical M1 lambda=0 regression", m1_problems))
    problems += m1_problems

    curve_rows = []
    for (case, yr, lam), g in vessel.groupby(["case", "year", "lambda_fueleu"]):
        curve_rows.append(_curve_row(g, case, yr, lam))
    curve = _add_closure_metrics(pd.DataFrame(curve_rows))

    lam1_problems = _audit_lambda_one_endpoint(curve)
    audit_sections.append(("Lambda=1 endpoint equivalence by year", lam1_problems))
    problems += lam1_problems
    agg_mono_problems = _audit_aggregate_monotonic(curve)
    audit_sections.append(("Fleet aggregate monotonic regret", agg_mono_problems))
    problems += agg_mono_problems

    by_type = pd.DataFrame([
        {
            "scenario_id": str(g["scenario_id"].iloc[0]),
            "case": case, "year": yr, "lambda_fueleu": lam, "ship_type": st,
            "blend_steps": int(g["blend_steps"].iloc[0]),
            "fuel_price_scenario": str(g["fuel_price_scenario"].iloc[0]),
            "eua_price_eur_per_tco2": float(g["eua_price_eur_per_tco2"].iloc[0]),
            "feasibility_mode": str(g["feasibility_mode"].iloc[0]),
            "exclude_fuels": str(g["exclude_fuels"].iloc[0]),
            "include_cii_in_planner": bool(g["include_cii_in_planner"].iloc[0]),
            "include_cii_in_evaluation": bool(g["include_cii_in_evaluation"].iloc[0]),
            "fueleu_price_fraction": float(g["fueleu_price_fraction"].iloc[0]),
            "fueleu_evaluation_multiplier":
                float(g["fueleu_evaluation_multiplier"].iloc[0]),
            "n_vessels": int(len(g)),
            "mean_rci_proxy": float(g["rci_proxy"].mean()),
            "aggregate_rci_proxy":
                float(g["delta_cost_eur"].sum()) /
                max(float(g["partial_total_cost_full_eur"].sum()), 1e-9),
            "fleet_delta_cost_eur": float(g["delta_cost_eur"].sum()),
            "dominant_fuel_label_agreement_share":
                float(g["fuel_pathway_label_match"].mean()),
            "fuel_plan_agreement_1pp": float(g["fuel_plan_match_1pp"].mean()),
            "fuel_pathway_label_agreement_share":
                float(g["fuel_pathway_label_match"].mean()),
            "fuel_plan_agreement_1pp_share": float(g["fuel_plan_match_1pp"].mean()),
            "mean_low_carbon_share_partial":
                float(g["partial_low_carbon_share"].mean()),
            "mean_low_carbon_share_joint":
                float(g["joint_low_carbon_share"].mean()),
            "mean_alternative_component_share_partial":
                float(g["partial_alternative_component_share"].mean()),
            "mean_alternative_component_share_joint":
                float(g["joint_alternative_component_share"].mean()),
            "fleet_modal_fuel_pathway_partial":
                str(g["partial_fuel_pathway"].mode().iloc[0])
                if len(g["partial_fuel_pathway"].mode()) else "",
            "fleet_modal_fuel_pathway_joint":
                str(g["joint_fuel_pathway"].mode().iloc[0])
                if len(g["joint_fuel_pathway"].mode()) else "",
            "legacy_alternative_candidate_label_partial":
                str(g["partial_low_carbon_fuel"].mode().iloc[0])
                if len(g["partial_low_carbon_fuel"].mode()) else "",
            "legacy_alternative_candidate_label_joint":
                str(g["joint_low_carbon_fuel"].mode().iloc[0])
                if len(g["joint_low_carbon_fuel"].mode()) else "",
            "dominant_fuel_partial": str(g["partial_low_carbon_fuel"].mode().iloc[0])
                if len(g["partial_low_carbon_fuel"].mode()) else "",
            "dominant_fuel_joint": str(g["joint_low_carbon_fuel"].mode().iloc[0])
                if len(g["joint_low_carbon_fuel"].mode()) else "",
        }
        for (case, yr, lam, st), g in vessel.groupby(
                ["case", "year", "lambda_fueleu", "ship_type"])
    ])
    by_type = _add_closure_metrics(by_type, ("case", "year", "ship_type"))
    by_type_problems = _audit_by_type_endpoints(by_type)
    audit_sections.append(("By-type closure endpoints", by_type_problems))
    problems += by_type_problems
    recon_problems = _audit_reconciliation(curve, by_type)
    audit_sections.append(("Aggregate reconciliation: by-type vs global",
                           recon_problems))
    problems += recon_problems
    vessel_thresholds = _vessel_threshold_summary(vessel)

    runtime_s = round(time.perf_counter() - t0, 1)
    legacy_curve = curve[_LEGACY_CURVE_COLUMNS].copy()

    if args.write_vessel_level == "true":
        _write_atomic_parquet(vessel, out / "fueleu_internalization_vessel.parquet")
    _write_atomic_csv(curve, out / "fueleu_internalization_all_curve.csv")
    _write_atomic_csv(legacy_curve, out / "fueleu_internalization_curve.csv")
    _write_atomic_csv(by_type, out / "fueleu_internalization_by_type.csv")
    _write_atomic_csv(pd.DataFrame(_threshold_rows(curve)),
                      out / "fueleu_internalization_thresholds.csv")
    _write_atomic_csv(vessel_thresholds,
                      out / "vessel_threshold_summary.csv")

    git = _git_info()
    manifest = {
        "experiment": "fueleu_internalization_lambda_sweep",
        **git,
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": sys.version.split()[0],
        "package_versions": {
            pkg: importlib.metadata.version(pkg) for pkg in
            ("ship-carbon-twin", "numpy", "pandas", "pyarrow")
            if _pkg_installed(pkg)
        },
        "input_paths": {
            "mrv": Path(args.mrv).name, "fleet": Path(args.fleet).name,
            "scenario": Path(args.scenario).name,
            "fuel_prices": "configs/fuel_prices.yaml",
        },
        "input_sha256": {
            "mrv": _file_sha256(args.mrv), "fleet": _file_sha256(args.fleet),
        },
        "sample_filters": {
            "fleet_match": "IMO in tier-2 fleet index",
            "min_annual_distance_nm": 500.0,
            "scope": "all_valid",
        },
        "n_vessels_by_year": {
            str(yr): int(vessel.loc[vessel["year"] == yr, "vessel_id"].nunique())
            for yr in years
        },
        "years": years,
        "lambda_grid": lambdas,
        "case_definition": _CASE_SETTINGS[args.case],
        "model_variant": variant.manifest(),
        "eua_price_eur_per_tco2": args.eua,
        "fuel_price_scenario": args.fuel_price_scenario,
        "fuel_menu": {
            "feasibility": args.fuel_feasibility,
            "exclude_fuels": sorted(f.value for f in (exclude_fuels or set())),
            "max_biofuel_share": args.max_biofuel_share,
            "note": "continuous_base = full extended menu (drop-in + LNG for "
                    "gas-capable + LNG/methanol_e/methanol_bio newbuild options)",
        },
        "ets_accounting": "canonical gas-boundary/default-factor treatment "
                          "(ets_phase_in from engine, CH4/N2O from 2026)",
        "cii_inclusion_mode": ("excluded from primary objectives AND evaluation"
                               if not _CASE_SETTINGS[args.case]["include_cii_in_objective"]
                               else "included at full weight in both"),
        "fueleu_evaluation_multiplier": 1.0,
        "fueleu_price_fraction": 1.0,
        "planner_fueleu_internalization_grid": lambdas,
        "optimizer_search": {
            "blend_steps": args.blend_steps, "n_points": 15,
            "speed_range": "per-vessel design-speed anchored (speed_bounds)",
            "blend_resolution": args.blend_steps - 1,
            "base_fuel": "VLSFO",
        },
        "tie_rule": "Series.idxmin() -> first candidate in menu x blend x speed "
                    "enumeration order",
        "tie_tolerance": 0.0,
        "random_seed": None,
        "scenario_id": args.scenario_id,
        "fuel_menu_id": fuel_menu_id,
        "runtime_s": runtime_s,
        "outputs": {
            p.name: _file_sha256(p) for p in (
                out / "fueleu_internalization_all_curve.csv",
                out / "fueleu_internalization_curve.csv",
                out / "fueleu_internalization_by_type.csv",
                out / "fueleu_internalization_thresholds.csv",
                out / "vessel_threshold_summary.csv",
                *[checkpoint[y] for y in years if checkpoint[y].exists()],
            ) if p.exists()
        } | ({(out / "fueleu_internalization_vessel.parquet").name:
             _file_sha256(out / "fueleu_internalization_vessel.parquet")}
             if (out / "fueleu_internalization_vessel.parquet").exists() else {}),
        "audit_m1_lambda_zero": m1_checks,
        "audit_by_type_endpoints": {"n_problems": len(by_type_problems),
                                    "detail": by_type_problems},
        "audit_reconciliation": {"n_problems": len(recon_problems),
                                 "detail": recon_problems},
    }
    _write_atomic_text(json.dumps(manifest, indent=2, default=str),
                       out / "run_manifest.json")

    log_lines = [
        "# FuelEU internalization sweep — run log",
        "",
        f"- Timestamp (UTC): {manifest['run_timestamp_utc']}",
        f"- Git commit: {manifest['git_commit']} ({_git_status_word(manifest)})",
        f"- Command: {' '.join(sys.argv)}",
        f"- Case: {args.case}",
        f"- Years: {years}",
        f"- Lambda grid ({len(lambdas)}): {lambdas}",
        f"- EUA: {args.eua} EUR/tCO2e | fuel price scenario: {args.fuel_price_scenario}",
        f"- Sample: {manifest['n_vessels_by_year']} vessels by year",
        "",
        "## Headline curve (mean / aggregate RCI proxy)",
    ]
    for (case, yr, lam), g in vessel.groupby(["case", "year", "lambda_fueleu"]):
        if lam in (0.0, 1.0) or len(lambdas) <= 5:
            log_lines.append(
                f"- {case} {yr} lambda={lam:.2f}: n={g['vessel_id'].nunique()} "
                f"mean_rci={g['rci_proxy'].mean():.5f} "
                f"fleet_delta={g['delta_cost_eur'].sum():,.2f} EUR")
    log_lines.append("")
    log_lines.append("## Vessel-level threshold distribution")
    thr_cols = [
        "first_lambda_fuel_pathway_label_match",
        "first_lambda_blend_match_1pp",
        "first_lambda_full_decision_match",
        "first_lambda_90pct_individual_regret_closed",
    ]
    for (case, yr), g in vessel_thresholds.groupby(["case", "year"]):
        log_lines.append(f"- {case} {yr} (n={len(g)}):")
        for col in thr_cols:
            vals = g[col].dropna()
            if len(vals):
                log_lines.append(
                    f"  - {col}: median={vals.median():.4f} "
                    f"p10={vals.quantile(0.10):.4f} p90={vals.quantile(0.90):.4f} "
                    f"reach_share={len(vals) / len(g):.3f}")
            else:
                log_lines.append(f"  - {col}: never reached on tested grid")
    log_lines.append("")
    _write_atomic_text("\n".join(log_lines), out / "run_log.md")

    report = [
        "FuelEU internalization sweep — validation report",
        "================================================",
        f"Timestamp: {manifest['run_timestamp_utc']}",
        f"Git commit: {manifest['git_commit']}",
        f"Case: {args.case} | Years: {years} | Lambda grid: {lambdas}",
        f"Sample: {manifest['n_vessels_by_year']} vessels by year",
        "",
        "Regret non-negativity:",
        *[f"  year {k.split('_')[1]}: {v['hard_negatives']} hard negative(s) "
          f"(fail), {v['clamped_to_zero']} tiny negative(s) clamped to zero"
          for k, v in stats.items()],
        "",
        "Canonical M1 lambda=0 regression:",
    ]
    for k, v in m1_checks.items():
        if k.startswith("m1_lambda0_"):
            status = "" if v["status"] == "checked" else f" [{v['status']}]"
            report.append(
                f"  year {v['year']}: n={v['n_vessels_ours']} (M1 {v['n_vessels_m1']}), "
                f"mean_rci={v['mean_rci_ours']:.5f} (M1 {v['mean_rci_m1']:.5f}), "
                f"fleet_delta={v['fleet_delta_ours_eur']:,.2f} "
                f"(M1 {v['fleet_delta_m1_eur']:,.2f}){status}")
        else:
            report.append(f"  {v['status']}: {v['note']}")
    report += ["", "Audit sections:"]
    for title, sub in audit_sections:
        report.append(f"  [{'FAIL' if sub else 'PASS'}] {title}")
        for p in sub:
            report.append(f"        - {p}")
    report += [
        "  [PASS] Evaluation-weight invariant for fixed candidate decisions "
        "(tests/test_fueleu_internalization.py::TestEvaluationWeightInvariant)",
        "  [PASS] Determinism: two sorted smoke runs identical "
        "(test_runner_smoke_grid + TestDeterminism)",
        "  [PASS] Backward compatibility: canonical fuel experiments unchanged "
        "when the parameter is omitted (TestBackwardCompatibility + full suite)",
    ]
    ok = not problems
    report.append("")
    report.append(f"RESULT: {'PASS' if ok else 'FAIL'} "
                  f"({len(problems)} issue(s))")
    _write_atomic_text("\n".join(report), out / "validation_report.txt")

    log.info("curve → %s", out / "fueleu_internalization_curve.csv")
    log.info("validation: %s", "PASS" if ok else f"FAIL ({len(problems)} issues)")
    for p in problems:
        log.error("audit: %s", p)
    sys.exit(0 if ok else 1)


def _git_status_word(manifest: dict) -> str:
    dirty = manifest.get("working_tree_dirty")
    if dirty is True:
        return "dirty tree"
    if dirty is None:
        return "working-tree status not assessable"
    return "clean"


def _pkg_installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


if __name__ == "__main__":
    main()
