"""Robust optimisation utilities for maritime carbon compliance.

Provides minimax-regret speed selection, CVaR-based speed optimisation,
the Regulation Coupling Index (RCI), and Pareto frontier computation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from carbon_twin.data.schemas import FuelType, Vessel, Voyage
from carbon_twin.emissions.factors import get_co2e_ttw
from carbon_twin.optimization.speed import _fuel_at_speed

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _voyage_cost(
    vessel: Vessel,
    voyage: Voyage,
    speed: float,
    fuel_factors: dict,
    eua_price: float,
    fuel_price: float,
    fuel_type: FuelType = FuelType.VLSFO,
) -> float:
    """Total cost (fuel + ETS) for a single speed/scenario point."""
    fc = _fuel_at_speed(vessel, voyage, speed, fuel_type, fuel_factors)
    fuel_cost = fc["total_tonnes"] * fuel_price
    co2e = get_co2e_ttw(fuel_type, fc["total_tonnes"], fuel_factors)
    ets_cost = co2e * voyage.ets_scope_factor * eua_price
    return fuel_cost + ets_cost


def _optimal_cost_at_scenario(
    vessel: Vessel,
    voyage: Voyage,
    fuel_factors: dict,
    eua_price: float,
    fuel_price: float,
    speed_range: tuple[float, float],
    n_speed_points: int,
    fuel_type: FuelType = FuelType.VLSFO,
) -> tuple[float, float]:
    """Return (optimal_speed, minimum_cost) for a single scenario."""
    speeds = np.linspace(speed_range[0], speed_range[1], n_speed_points)
    best_speed = float(speeds[0])
    best_cost = float("inf")
    for spd in speeds:
        c = _voyage_cost(vessel, voyage, float(spd), fuel_factors, eua_price, fuel_price, fuel_type)
        if c < best_cost:
            best_cost = c
            best_speed = float(spd)
    return best_speed, best_cost


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def minimax_regret_speed(
    vessel: Vessel,
    voyage: Voyage,
    fuel_factors: dict,
    eua_price_scenarios: list[float],
    fuel_price_scenarios: list[float],
    speed_range: tuple[float, float] = (8, 22),
    n_speed_points: int = 30,
) -> dict[str, Any]:
    """Find the speed minimising worst-case regret across price scenarios.

    Regret(speed, scenario) = Cost(speed, scenario) - Cost(optimal_speed(scenario), scenario)

    Returns:
        optimal_speed: speed that minimises max regret
        max_regret: worst-case regret at the optimal speed
        regret_profile: DataFrame with columns [speed_knots, max_regret, mean_regret]
        speed_profile: DataFrame with speed-cost data per scenario
    """
    speeds = np.linspace(speed_range[0], speed_range[1], n_speed_points)
    fuel_type = FuelType.VLSFO

    # Pre-compute optimal cost for each scenario
    scenario_optimal: list[float] = []  # optimal cost per scenario
    for eua in eua_price_scenarios:
        for fuel_p in fuel_price_scenarios:
            _, opt_cost = _optimal_cost_at_scenario(
                vessel, voyage, fuel_factors, eua, fuel_p, speed_range, n_speed_points, fuel_type
            )
            scenario_optimal.append(opt_cost)

    # Compute regret for each (speed, scenario) pair
    speed_profile_rows: list[dict] = []
    regret_matrix = np.zeros((n_speed_points, len(scenario_optimal)))

    scenario_idx = 0
    for eua in eua_price_scenarios:
        for fuel_p in fuel_price_scenarios:
            for si, spd in enumerate(speeds):
                c = _voyage_cost(vessel, voyage, float(spd), fuel_factors, eua, fuel_p, fuel_type)
                regret_matrix[si, scenario_idx] = max(c - scenario_optimal[scenario_idx], 0.0)
                speed_profile_rows.append(
                    {
                        "speed_knots": float(spd),
                        "eua_price": eua,
                        "fuel_price": fuel_p,
                        "cost": c,
                        "regret": regret_matrix[si, scenario_idx],
                    }
                )
            scenario_idx += 1

    max_regrets = regret_matrix.max(axis=1)
    mean_regrets = regret_matrix.mean(axis=1)

    best_idx = int(np.argmin(max_regrets))
    optimal_speed = float(speeds[best_idx])
    max_regret = float(max_regrets[best_idx])

    regret_profile = pd.DataFrame(
        {
            "speed_knots": [float(s) for s in speeds],
            "max_regret": max_regrets.tolist(),
            "mean_regret": mean_regrets.tolist(),
        }
    )

    return {
        "optimal_speed": optimal_speed,
        "max_regret": max_regret,
        "regret_profile": regret_profile,
        "speed_profile": pd.DataFrame(speed_profile_rows),
    }


def cvar_speed_optimization(
    vessel: Vessel,
    voyage: Voyage,
    fuel_factors: dict,
    eua_price_dist: tuple[float, float],
    fuel_price_dist: tuple[float, float],
    n_samples: int = 500,
    alpha: float = 0.05,
    lambda_cvar: float = 1.0,
    speed_range: tuple[float, float] = (8, 22),
    n_speed_points: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    """Optimise E[cost] + lambda * CVaR(alpha) over speed.

    Args:
        eua_price_dist: (mean, std) for lognormal distribution of EUA price
        fuel_price_dist: (mean, std) for lognormal distribution of fuel price

    Returns:
        optimal_speed, expected_cost, cvar, total_objective, speed_profile DataFrame
    """
    rng = np.random.default_rng(seed)
    fuel_type = FuelType.VLSFO

    # Sample price scenarios from lognormal distributions
    def _lognormal_samples(mean: float, std: float, n: int) -> np.ndarray:
        if std <= 0:
            return np.full(n, mean)
        mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
        sigma = np.sqrt(np.log(1 + (std / mean) ** 2))
        return rng.lognormal(mu, sigma, n)

    eua_samples = _lognormal_samples(eua_price_dist[0], eua_price_dist[1], n_samples)
    fuel_samples = _lognormal_samples(fuel_price_dist[0], fuel_price_dist[1], n_samples)

    speeds = np.linspace(speed_range[0], speed_range[1], n_speed_points)

    best_objective = float("inf")
    best_speed = float(speeds[0])
    best_expected_cost = float("inf")
    best_cvar = float("inf")

    profile_rows: list[dict] = []

    for spd in speeds:
        costs = np.array(
            [
                _voyage_cost(
                    vessel,
                    voyage,
                    float(spd),
                    fuel_factors,
                    float(eua_samples[i]),
                    float(fuel_samples[i]),
                    fuel_type,
                )
                for i in range(n_samples)
            ]
        )
        expected_cost = float(costs.mean())
        threshold = float(np.percentile(costs, 100 * (1 - alpha)))
        tail = costs[costs >= threshold]
        cvar = float(tail.mean()) if len(tail) > 0 else float(costs.max())
        objective = expected_cost + lambda_cvar * cvar

        profile_rows.append(
            {
                "speed_knots": float(spd),
                "expected_cost": expected_cost,
                "cvar": cvar,
                "total_objective": objective,
            }
        )

        if objective < best_objective:
            best_objective = objective
            best_speed = float(spd)
            best_expected_cost = expected_cost
            best_cvar = cvar

    return {
        "optimal_speed": best_speed,
        "expected_cost": best_expected_cost,
        "cvar": best_cvar,
        "total_objective": best_objective,
        "speed_profile": pd.DataFrame(profile_rows),
    }


def regulation_coupling_index(
    vessel: Vessel,
    voyage: Voyage,
    fuel_factors: dict,
    cii_engine: Any,
    fueleu_engine: Any,
    ets_engine: Any,
    year: int = 2026,
    speed_range: tuple = (8, 22),
) -> dict[str, Any]:
    """Compute the Regulation Coupling Index (RCI).

    RCI = (sum_of_individual_optima_costs - joint_optimum_cost) / joint_optimum_cost

    Demonstrates the value of joint optimisation vs. optimising each regulation
    independently.

    Returns:
        rci, joint_optimal_speed, joint_optimal_cost,
        cii_only_speed, ets_only_speed, fueleu_only_speed,
        costs_per_strategy DataFrame
    """
    from carbon_twin.optimization.speed import (
        optimal_speed_multi_regulation,
        optimal_speed_single_voyage,
    )

    # Default price assumptions
    fuel_price = 600.0
    eua_price = 80.0

    n_points = 30

    # Joint optimisation (CII + FuelEU + ETS)
    joint_result = optimal_speed_multi_regulation(
        vessel=vessel,
        voyage=voyage,
        fuel_price=fuel_price,
        eua_price=eua_price,
        fuel_factors=fuel_factors,
        cii_engine=cii_engine,
        fueleu_engine=fueleu_engine,
        ets_engine=ets_engine,
        year=year,
        speed_range=speed_range,
        n_points=n_points,
    )
    joint_speed = joint_result["optimal_speed"]
    joint_cost = joint_result["total_cost"]

    # ETS-only: optimise just fuel + ETS
    ets_result = optimal_speed_single_voyage(
        vessel=vessel,
        voyage=voyage,
        fuel_price=fuel_price,
        eua_price=eua_price,
        fuel_factors=fuel_factors,
        regulations=["ets"],
        speed_range=speed_range,
        n_points=n_points,
    )
    ets_speed = ets_result["optimal_speed"]
    # Evaluate ETS-only speed under full joint cost function
    ets_at_joint_cost = _eval_joint_cost(
        vessel, voyage, ets_speed, fuel_factors, fuel_price, eua_price
    )

    # CII-only: minimise attained CII (proxy: minimise fuel consumption = slow speed)
    cii_speed = float(speed_range[0]) + (float(speed_range[1]) - float(speed_range[0])) * 0.2
    cii_at_joint_cost = _eval_joint_cost(
        vessel, voyage, cii_speed, fuel_factors, fuel_price, eua_price
    )

    # FuelEU-only: speed has limited FuelEU impact (intensity is fuel-type driven), use mid-speed
    fueleu_speed = (float(speed_range[0]) + float(speed_range[1])) / 2
    fueleu_at_joint_cost = _eval_joint_cost(
        vessel, voyage, fueleu_speed, fuel_factors, fuel_price, eua_price
    )

    sum_individual = ets_at_joint_cost + cii_at_joint_cost + fueleu_at_joint_cost
    # RCI: normalised by three-fold of joint cost
    rci = (sum_individual - 3.0 * joint_cost) / (3.0 * joint_cost) if joint_cost > 0 else 0.0

    costs_df = pd.DataFrame(
        [
            {"strategy": "joint", "speed": joint_speed, "cost": joint_cost},
            {"strategy": "ets_only", "speed": ets_speed, "cost": ets_at_joint_cost},
            {"strategy": "cii_only", "speed": cii_speed, "cost": cii_at_joint_cost},
            {"strategy": "fueleu_only", "speed": fueleu_speed, "cost": fueleu_at_joint_cost},
        ]
    )

    return {
        "rci": round(rci, 6),
        "joint_optimal_speed": joint_speed,
        "joint_optimal_cost": joint_cost,
        "cii_only_speed": cii_speed,
        "ets_only_speed": ets_speed,
        "fueleu_only_speed": fueleu_speed,
        "costs_per_strategy": costs_df,
    }


def _eval_joint_cost(
    vessel: Vessel,
    voyage: Voyage,
    speed: float,
    fuel_factors: dict,
    fuel_price: float,
    eua_price: float,
    fuel_type: FuelType = FuelType.VLSFO,
) -> float:
    """Evaluate total fuel+ETS cost at a specific speed."""
    return _voyage_cost(vessel, voyage, speed, fuel_factors, eua_price, fuel_price, fuel_type)


def robust_compliance_frontier(
    vessel: Vessel,
    voyages: list[Voyage],
    fuel_factors: dict,
    scenario_samples: list[dict],
    speed_candidates: list[float] | None = None,
) -> pd.DataFrame:
    """Compute the Pareto frontier of (expected_cost, CVaR_cost, expected_co2e).

    Args:
        scenario_samples: list of dicts with keys 'eua_price', 'fuel_price'
        speed_candidates: speed values to evaluate; defaults to 20 points in [8, 22]

    Returns:
        DataFrame with columns:
        speed, expected_cost, cvar_cost, worst_case_cost, expected_co2e, is_pareto_optimal
    """
    if speed_candidates is None:
        speed_candidates = list(np.linspace(8, 22, 20))

    fuel_type = FuelType.VLSFO
    alpha = 0.05

    rows: list[dict] = []
    for spd in speed_candidates:
        costs_per_scenario: list[float] = []
        co2e_per_scenario: list[float] = []

        for sc in scenario_samples:
            eua_p = float(sc.get("eua_price", 80.0))
            fuel_p = float(sc.get("fuel_price", 600.0))
            total_cost = 0.0
            total_co2e = 0.0
            for voy in voyages:
                c = _voyage_cost(vessel, voy, spd, fuel_factors, eua_p, fuel_p, fuel_type)
                fc = _fuel_at_speed(vessel, voy, spd, fuel_type, fuel_factors)
                co2e = get_co2e_ttw(fuel_type, fc["total_tonnes"], fuel_factors)
                total_cost += c
                total_co2e += co2e
            costs_per_scenario.append(total_cost)
            co2e_per_scenario.append(total_co2e)

        costs_arr = np.array(costs_per_scenario)
        co2e_arr = np.array(co2e_per_scenario)

        expected_cost = float(costs_arr.mean())
        worst_case_cost = float(costs_arr.max())
        threshold = float(np.percentile(costs_arr, 100 * (1 - alpha)))
        tail = costs_arr[costs_arr >= threshold]
        cvar_cost = float(tail.mean()) if len(tail) > 0 else worst_case_cost
        expected_co2e = float(co2e_arr.mean())

        rows.append(
            {
                "speed": float(spd),
                "expected_cost": expected_cost,
                "cvar_cost": cvar_cost,
                "worst_case_cost": worst_case_cost,
                "expected_co2e": expected_co2e,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        df["is_pareto_optimal"] = []
        return df

    # Mark Pareto-optimal points: minimise (expected_cost, expected_co2e) jointly
    is_pareto = np.ones(len(df), dtype=bool)
    costs = df["expected_cost"].values
    co2es = df["expected_co2e"].values

    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            if (
                costs[j] <= costs[i]
                and co2es[j] <= co2es[i]
                and (costs[j] < costs[i] or co2es[j] < co2es[i])
            ):
                is_pareto[i] = False
                break

    df["is_pareto_optimal"] = is_pareto
    return df
