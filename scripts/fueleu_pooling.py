from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall: pip install -e '.[all]'")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_VLSFO_E = 41000.0
_PENALTY_RATE = 2400.0


def _pool_penalty(balance: pd.Series, actual_int: pd.Series, energy: pd.Series) -> float:
    net = float(balance.sum())
    if net >= 0:
        return 0.0
    tot_e = float(energy.sum())
    wtd_int = float((actual_int * energy).sum() / tot_e) if tot_e > 0 else 0.0
    if wtd_int <= 0:
        return 0.0
    return abs(net) / (wtd_int * _VLSFO_E) * _PENALTY_RATE


def _allocate(penalty_total: float, standalone: pd.Series) -> pd.Series:
    s = standalone.clip(lower=0.0)
    tot = float(s.sum())
    if tot <= 0:
        return pd.Series(0.0, index=standalone.index)
    return s / tot * penalty_total


def _rci_from_pool(g: pd.DataFrame, joint_pen: pd.Series, silo_pen: pd.Series,
                   clip: bool = True) -> pd.Series:
    cost_j = g["cost_joint_eur"] - g["fueleu_penalty_joint_eur"] + joint_pen
    cost_s = g["cost_siloed_eur"] - g["fueleu_penalty_siloed_eur"] + silo_pen
    r = (cost_s - cost_j) / cost_s
    return (r.clip(lower=0.0) if clip else r).where(cost_s > 0, 0.0)


def _delta_from_pool(g: pd.DataFrame, joint_pen: pd.Series, silo_pen: pd.Series) -> pd.Series:
    cost_j = g["cost_joint_eur"] - g["fueleu_penalty_joint_eur"] + joint_pen
    cost_s = g["cost_siloed_eur"] - g["fueleu_penalty_siloed_eur"] + silo_pen
    return cost_s - cost_j


def _stats(g: pd.DataFrame, rci: pd.Series, delta: pd.Series, pooled: float,
           signed: pd.Series | None = None) -> dict:
    bal = g["fueleu_balance_joint"]
    no_pool = float(g["fueleu_penalty_joint_eur"].sum())
    return {
        "n_vessels": int(len(g)),
        "total_positive_CB": round(float(bal[bal > 0].sum()), 1),
        "total_negative_CB": round(float(bal[bal < 0].sum()), 1),
        "net_CB": round(float(bal.sum()), 1),
        "no_pooling_penalty_eur": round(no_pool, 2),
        "pooled_penalty_eur": round(pooled, 2),
        "penalty_reduction_pct": round((1 - pooled / no_pool) * 100, 2) if no_pool > 0 else 0.0,
        "mean_rci_silo": round(float(rci.mean()), 5),
        "median_rci_silo": round(float(rci.median()), 5),
        "p90_rci_silo": round(float(rci.quantile(0.90)), 5),
        "fleet_delta_cost_eur": round(float(delta.sum()), 2),
        "mean_delta_cost_eur": round(float(delta.mean()), 2),
        "mean_rci_signed": None if signed is None else round(float(signed.mean()), 5),
        "share_rci_negative": None if signed is None else round(float((signed < 0).mean()), 5),
    }


def _case_rows(df: pd.DataFrame, case: str, scenario: str, eua: float) -> list[dict]:
    rows = []
    for yr, g in df.groupby("year"):
        no_pool_j = g["fueleu_penalty_joint_eur"]
        no_pool_s = g["fueleu_penalty_siloed_eur"]
        if case == "P0_no_pooling_vessel_year":
            jp, sp = no_pool_j, no_pool_s
        elif case == "P1_perfect_pooling_analytical_fleet":
            pj = _pool_penalty(g["fueleu_balance_joint"], g["fueleu_actual_intensity_joint"],
                               g["fueleu_energy_mj_joint"])
            ps = _pool_penalty(g["fueleu_balance_siloed"], g["fueleu_actual_intensity_siloed"],
                               g["fueleu_energy_mj_siloed"])
            jp, sp = _allocate(pj, no_pool_j), _allocate(ps, no_pool_s)
        elif case.startswith("P3_exogenous_FuelEU_credit_"):
            frac = float(case.rsplit("_", 1)[1]) / 100.0
            jp, sp = no_pool_j * frac, no_pool_s * frac
        else:
            raise ValueError(case)
        rci = _rci_from_pool(g, jp, sp)
        signed = _rci_from_pool(g, jp, sp, clip=False)
        delta = _delta_from_pool(g, jp, sp)
        rows.append({"case": case, "year": yr, "fuel_price_scenario": scenario,
                     "eua_price_eur_per_tco2": eua,
                     **_stats(g, rci, delta, float(jp.sum()), signed)})
    return rows


def _p2_rows(df: pd.DataFrame, scenario: str, eua: float) -> tuple[list[dict], list[dict]]:
    fleet_rows, type_rows = [], []
    for yr, gy in df.groupby("year"):
        jp = pd.Series(0.0, index=gy.index)
        sp = pd.Series(0.0, index=gy.index)
        for st, g in gy.groupby("ship_type"):
            pj = _pool_penalty(g["fueleu_balance_joint"], g["fueleu_actual_intensity_joint"],
                               g["fueleu_energy_mj_joint"])
            psi = _pool_penalty(g["fueleu_balance_siloed"], g["fueleu_actual_intensity_siloed"],
                                g["fueleu_energy_mj_siloed"])
            jp.loc[g.index] = _allocate(pj, g["fueleu_penalty_joint_eur"])
            sp.loc[g.index] = _allocate(psi, g["fueleu_penalty_siloed_eur"])
            no_pool = float(g["fueleu_penalty_joint_eur"].sum())
            bal = g["fueleu_balance_joint"]
            type_rows.append({
                "case": "P2_perfect_pooling_by_ship_type", "year": yr, "ship_type": st,
                "n_vessels": int(len(g)),
                "total_positive_CB": round(float(bal[bal > 0].sum()), 1),
                "total_negative_CB": round(float(bal[bal < 0].sum()), 1),
                "net_CB": round(float(bal.sum()), 1),
                "pooled_penalty_eur": round(pj, 2),
                "penalty_reduction_pct": round((1 - pj / no_pool) * 100, 2) if no_pool > 0 else 0.0,
            })
        rci = _rci_from_pool(gy, jp, sp)
        signed = _rci_from_pool(gy, jp, sp, clip=False)
        delta = _delta_from_pool(gy, jp, sp)
        fleet_rows.append({"case": "P2_perfect_pooling_by_ship_type", "year": yr,
                           "fuel_price_scenario": scenario, "eua_price_eur_per_tco2": eua,
                           **_stats(gy, rci, delta, float(jp.sum()), signed)})
    return fleet_rows, type_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="FuelEU pooling / pre-flexibility bound")
    ap.add_argument("--parquet", required=True, type=Path,
                    help="fuel_optimized_blend_ext.parquet (enriched with FuelEU balance cols)")
    ap.add_argument("--scenario-label", default="base_2026")
    ap.add_argument("--eua", type=float, default=80.0)
    ap.add_argument("--out", type=Path, default=Path("results/optimization/fueleu_pooling.csv"))
    ap.add_argument("--out-by-type", type=Path,
                    default=Path("results/optimization/fueleu_pooling_by_type.csv"))
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    need = ["fueleu_balance_joint", "fueleu_actual_intensity_joint", "fueleu_energy_mj_joint",
            "fueleu_balance_siloed", "fueleu_penalty_joint_eur", "fueleu_penalty_siloed_eur",
            "cost_joint_eur", "cost_siloed_eur", "year", "ship_type"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"parquet missing pooling columns {missing} — rerun 03d with the updated pipeline")

    log.info("pooling on %d vessel-years across %d years", len(df), df["year"].nunique())
    rows: list[dict] = []
    for case in ["P0_no_pooling_vessel_year", "P1_perfect_pooling_analytical_fleet",
                 "P3_exogenous_FuelEU_credit_25", "P3_exogenous_FuelEU_credit_50",
                 "P3_exogenous_FuelEU_credit_75"]:
        rows += _case_rows(df, case, args.scenario_label, args.eua)
    p2_fleet, p2_type = _p2_rows(df, args.scenario_label, args.eua)
    rows += p2_fleet

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["year", "case"]).to_csv(args.out, index=False)
    pd.DataFrame(p2_type).sort_values(["year", "ship_type"]).to_csv(args.out_by_type, index=False)
    for r in rows:
        log.info("%-34s yr=%d  reduction=%5.1f%%  rci_mean=%.4f",
                 r["case"], r["year"], r["penalty_reduction_pct"], r["mean_rci_silo"])
    log.info("done → %s (+ by_type)", args.out)


if __name__ == "__main__":
    main()
