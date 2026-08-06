"""Multi-year fleet transition planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from carbon_twin.data.schemas import (
    Fleet,
    FuelType,
    ShipType,
    UnifiedComplianceResult,
    Vessel,
    Voyage,
)
from carbon_twin.emissions.surrogate import EmissionSurrogate


@dataclass
class RetrofitOption:
    name: str
    cost_usd: float
    fuel_saving_pct: float
    speed_impact_pct: float = 0.0
    applicable_ship_types: list[ShipType] = field(default_factory=list)
    lifetime_years: int = 15


class FleetTransitionOptimizer:
    """Multi-year fleet transition planning using MILP.

    Minimizes total cost over a planning horizon with decision variables for
    speed, fuel choice, and retrofit per vessel-year.
    """

    def __init__(
        self,
        fleet: Fleet,
        regulations: dict[str, Any] | None = None,
        scenarios: list[dict[str, Any]] | None = None,
        planning_horizon_years: int = 10,
        base_year: int = 2026,
        retrofit_options: list[RetrofitOption] | None = None,
        budget_per_year: float | None = None,
        surrogate: EmissionSurrogate | None = None,
    ) -> None:
        self._fleet = fleet
        self._regulations = regulations or {}
        self._scenarios = scenarios or [{}]
        self._horizon = planning_horizon_years
        self._base_year = base_year
        self._retrofit_options = retrofit_options or []
        self._budget_per_year = budget_per_year
        self._surrogate = surrogate
        self._model: Any = None
        self._solved = False
        self._results: dict[str, Any] = {}

    def build_model(self) -> Any:
        """Build MILP model: minimize total cost over planning horizon.

        Decision variables:
          - speed[v, y]: operating speed for vessel v in year y (continuous)
          - fuel[v, y, f]: fraction of fuel f used by vessel v in year y
          - retrofit[v, y, r]: binary, whether vessel v gets retrofit r in year y

        Constraints:
          - Sum of fuel fractions == 1 for each vessel-year
          - Budget constraint per year
          - Retrofit sequencing (can only retrofit once per option)
          - Speed bounds
        """
        try:
            import pyomo.environ as pyo
        except ImportError:
            raise ImportError("pyomo is required for fleet transition optimization")

        model = pyo.ConcreteModel("FleetTransition")

        vessels = [v.imo_number for v in self._fleet.vessels]
        years = list(range(self._base_year, self._base_year + self._horizon))
        fuels = [ft.value for ft in FuelType if ft in (
            FuelType.VLSFO, FuelType.LNG, FuelType.METHANOL_BIO,
            FuelType.METHANOL_E, FuelType.AMMONIA_GREEN, FuelType.BIODIESEL_FAME,
        )]
        retrofits = [r.name for r in self._retrofit_options]

        model.V = pyo.Set(initialize=vessels)
        model.Y = pyo.Set(initialize=years)
        model.F = pyo.Set(initialize=fuels)
        model.R = pyo.Set(initialize=retrofits)

        model.speed = pyo.Var(model.V, model.Y, bounds=(8.0, 22.0), within=pyo.NonNegativeReals)
        model.fuel_frac = pyo.Var(model.V, model.Y, model.F, bounds=(0.0, 1.0))
        model.retrofit = pyo.Var(model.V, model.Y, model.R, within=pyo.Binary)

        _fuel_prices: dict[str, float] = {
            FuelType.VLSFO.value: 600.0,
            FuelType.LNG.value: 700.0,
            FuelType.METHANOL_BIO.value: 1200.0,
            FuelType.METHANOL_E.value: 1800.0,
            FuelType.AMMONIA_GREEN.value: 1500.0,
            FuelType.BIODIESEL_FAME.value: 1400.0,
        }

        _fuel_co2_factor: dict[str, float] = {
            FuelType.VLSFO.value: 3.114,
            FuelType.LNG.value: 2.750,
            FuelType.METHANOL_BIO.value: 1.375,
            FuelType.METHANOL_E.value: 1.375,
            FuelType.AMMONIA_GREEN.value: 0.0,
            FuelType.BIODIESEL_FAME.value: 2.834,
        }

        _retrofit_cost: dict[str, float] = {r.name: r.cost_usd for r in self._retrofit_options}
        _retrofit_saving: dict[str, float] = {
            r.name: r.fuel_saving_pct / 100.0 for r in self._retrofit_options
        }

        vessel_map = {v.imo_number: v for v in self._fleet.vessels}
        _base_fuel_tonnes: dict[str, float] = {}
        for v in self._fleet.vessels:
            gt = v.gross_tonnage
            _base_fuel_tonnes[v.imo_number] = gt * 0.05

        eua_base = 80.0
        eua_growth = 0.05

        def fuel_fraction_sum_rule(m: Any, v: str, y: int) -> Any:
            return sum(m.fuel_frac[v, y, f] for f in m.F) == 1.0

        model.fuel_sum = pyo.Constraint(model.V, model.Y, rule=fuel_fraction_sum_rule)

        def retrofit_once_rule(m: Any, v: str, r: str) -> Any:
            return sum(m.retrofit[v, y, r] for y in m.Y) <= 1

        model.retrofit_once = pyo.Constraint(model.V, model.R, rule=retrofit_once_rule)

        if self._budget_per_year is not None:
            def budget_rule(m: Any, y: int) -> Any:
                return sum(
                    m.retrofit[v, y, r] * _retrofit_cost.get(r, 0.0)
                    for v in m.V for r in m.R
                ) <= self._budget_per_year

            model.budget = pyo.Constraint(model.Y, rule=budget_rule)

        def objective_rule(m: Any) -> Any:
            total = 0.0
            for v in m.V:
                base_fuel = _base_fuel_tonnes.get(v, 100.0)
                for y in m.Y:
                    year_idx = y - self._base_year
                    eua = eua_base * (1 + eua_growth) ** year_idx
                    discount = 1.0 / (1.08 ** year_idx)

                    retrofit_saving = sum(
                        m.retrofit[v, yy, r] * _retrofit_saving.get(r, 0.0)
                        for yy in m.Y for r in m.R
                        if yy <= y
                    )

                    for f in m.F:
                        frac = m.fuel_frac[v, y, f]
                        fuel_cost = frac * base_fuel * _fuel_prices.get(f, 600.0) * discount
                        co2 = frac * base_fuel * _fuel_co2_factor.get(f, 3.114)
                        ets_cost = co2 * 0.5 * eua * discount
                        total += fuel_cost + ets_cost

                    for r in m.R:
                        total += m.retrofit[v, y, r] * _retrofit_cost.get(r, 0.0) * discount

            return total

        model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        self._model = model
        return model

    def solve(
        self,
        solver: str = "glpk",
        time_limit: int = 300,
    ) -> dict[str, Any]:
        """Solve the fleet transition model."""
        if self._model is None:
            self.build_model()

        from carbon_twin.optimization.solvers import solve_milp

        result = solve_milp(self._model, solver=solver, time_limit=time_limit)
        self._solved = True
        self._results = result
        return result

    def pareto_frontier(
        self,
        n_points: int = 10,
        emission_weights: list[float] | None = None,
    ) -> pd.DataFrame:
        """Cost vs emissions tradeoff frontier.

        Solves the model with varying emission cost multipliers to trace
        the Pareto frontier.
        """
        if emission_weights is None:
            emission_weights = np.linspace(0.0, 5.0, n_points).tolist()

        rows: list[dict[str, Any]] = []

        for w in emission_weights:
            try:
                model = self.build_model()

                import pyomo.environ as pyo

                original_obj = model.obj
                model.del_component(model.obj)

                def weighted_obj(m: Any) -> Any:
                    cost_term = original_obj.expr
                    emission_term = sum(
                        m.fuel_frac[v, y, f] * 100.0
                        for v in m.V for y in m.Y for f in m.F
                    )
                    return cost_term + w * emission_term * 1e6

                model.obj = pyo.Objective(rule=weighted_obj, sense=pyo.minimize)

                from carbon_twin.optimization.solvers import solve_milp

                result = solve_milp(model, solver="glpk", time_limit=60)

                rows.append({
                    "emission_weight": w,
                    "total_cost": result.get("objective_value", float("inf")),
                    "status": result.get("status", "unknown"),
                })
            except Exception as e:
                rows.append({
                    "emission_weight": w,
                    "total_cost": float("inf"),
                    "status": f"error: {e}",
                })

        return pd.DataFrame(rows)

    def evaluate_strategy(self, strategy: dict[str, Any]) -> UnifiedComplianceResult:
        """Given fixed decisions, compute all regulation outcomes.

        Strategy format:
            {
                'vessel_imo': str,
                'year': int,
                'speed_knots': float,
                'fuel_mix': {FuelType: fraction},
                'retrofits': [str],
            }
        """
        vessel_imo = strategy.get("vessel_imo", "unknown")
        year = strategy.get("year", self._base_year)
        speed = strategy.get("speed_knots", 14.0)
        fuel_mix: dict[FuelType, float] = strategy.get("fuel_mix", {FuelType.VLSFO: 1.0})
        retrofits: list[str] = strategy.get("retrofits", [])

        # Validate fuel_mix sums to 1 (normalise if slightly off, raise if badly wrong)
        fuel_mix_sum = sum(fuel_mix.values())
        if fuel_mix_sum <= 0:
            raise ValueError("fuel_mix fractions must sum to 1; got non-positive sum.")
        if abs(fuel_mix_sum - 1.0) > 0.01:
            raise ValueError(
                f"fuel_mix fractions must sum to 1; got {fuel_mix_sum:.4f}. "
                "Normalise the fractions before calling evaluate_strategy()."
            )
        # Normalise minor floating-point deviations
        fuel_mix = {ft: frac / fuel_mix_sum for ft, frac in fuel_mix.items()}

        retrofit_saving = 0.0
        for r in self._retrofit_options:
            if r.name in retrofits:
                retrofit_saving += r.fuel_saving_pct / 100.0
        retrofit_saving = min(retrofit_saving, 0.5)

        vessel = None
        for v in self._fleet.vessels:
            if v.imo_number == vessel_imo:
                vessel = v
                break

        base_fuel = 1000.0
        if vessel:
            base_fuel = vessel.gross_tonnage * 0.05

        speed_factor = (speed / 14.0) ** 3
        effective_fuel = base_fuel * speed_factor * (1.0 - retrofit_saving)

        total_co2e_ttw = 0.0
        total_co2e_wtw = 0.0
        total_cost = 0.0

        # --- Surrogate path: use fitted EmissionSurrogate for fuel/emission estimates ---
        if self._surrogate is not None and self._surrogate._fitted and vessel is not None:
            from carbon_twin.data.schemas import Voyage as _Voyage  # local import to avoid circularity

            # Build a minimal feature row for the surrogate
            dominant_fuel = max(fuel_mix, key=fuel_mix.__getitem__)
            # Build a minimal DataFrame matching EmissionSurrogate._FEATURE_COLS
            import datetime

            feat_row: dict[str, Any] = {
                "ship_type_code": list(ShipType).index(vessel.ship_type),
                "gross_tonnage": vessel.gross_tonnage,
                "deadweight": vessel.deadweight or 0.0,
                "main_engine_power_kw": vessel.main_engine_power_kw or 0.0,
                "design_speed_knots": vessel.design_speed_knots or 0.0,
                "age": year - (vessel.year_built or (year - 15)),
                "distance_nm": 1000.0,   # representative voyage
                "avg_speed_knots": speed,
                "sea_hours": 1000.0 / max(speed, 0.1),
                "weather_factor": 1.0,
                "cargo_tonnes": 0.0,
                "draught_m": vessel.draught_m or 0.0,
                "length_m": vessel.length_m or 0.0,
                "beam_m": vessel.beam_m or 0.0,
                "speed_cubed": speed ** 3,
                "distance_x_gt": 1000.0 * vessel.gross_tonnage,
            }
            feat_df = pd.DataFrame([feat_row])
            surrogate_preds = self._surrogate.predict(feat_df)
            # Scale surrogate predictions (which are per-voyage) to effective_fuel basis
            surrogate_fuel = float(surrogate_preds["fuel_tonnes"][0]) * (1.0 - retrofit_saving)
            total_co2e_ttw = float(surrogate_preds["co2e_ttw"][0]) * (1.0 - retrofit_saving)
            total_co2e_wtw = float(surrogate_preds["co2e_wtw"][0]) * (1.0 - retrofit_saving)
            effective_fuel = surrogate_fuel
        else:
            # --- Simplified fallback fuel-burn model ---
            for ft, frac in fuel_mix.items():
                fuel_tonnes = effective_fuel * frac
                co2_factor = 3.114
                if ft == FuelType.LNG:
                    co2_factor = 2.750
                elif ft in (FuelType.METHANOL_BIO, FuelType.METHANOL_E):
                    co2_factor = 1.375
                elif ft == FuelType.AMMONIA_GREEN:
                    co2_factor = 0.0
                elif ft == FuelType.BIODIESEL_FAME:
                    co2_factor = 2.834

                total_co2e_ttw += fuel_tonnes * co2_factor
                total_co2e_wtw += fuel_tonnes * co2_factor * 1.2

        return UnifiedComplianceResult(
            vessel_imo=vessel_imo,
            year=year,
            total_compliance_cost_eur=total_cost,
            total_co2e_ttw_tonnes=total_co2e_ttw,
            total_co2e_wtw_tonnes=total_co2e_wtw,
            regulation_versions={},
            audit_metadata={
                "strategy": strategy,
                "effective_fuel_tonnes": effective_fuel,
                "retrofit_saving_pct": retrofit_saving * 100,
                "used_surrogate": (
                    self._surrogate is not None and self._surrogate._fitted and vessel is not None
                ),
            },
        )
