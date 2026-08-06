"""Solver utilities for optimization problems."""

from __future__ import annotations

from itertools import product
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_minimize


def solve_milp(
    model: Any,
    solver: str = "glpk",
    time_limit: int = 300,
) -> dict[str, Any]:
    """Solve a Pyomo MILP model.

    Wraps the Pyomo solver interface with standardized output format.
    Returns objective value, solver status, and variable values.
    """
    try:
        import pyomo.environ as pyo
    except ImportError:
        raise ImportError("pyomo is required for MILP solving")

    solver_factory = pyo.SolverFactory(solver)

    if hasattr(solver_factory, "options"):
        if solver == "glpk":
            solver_factory.options["tmlim"] = time_limit
        elif solver == "cbc":
            solver_factory.options["seconds"] = time_limit
        elif solver in ("gurobi", "cplex"):
            solver_factory.options["TimeLimit"] = time_limit

    try:
        result = solver_factory.solve(model, tee=False)
    except Exception as e:
        return {
            "status": "error",
            "termination": str(e),
            "objective_value": float("inf"),
            "variables": {},
        }

    status = str(result.solver.status)
    termination = str(result.solver.termination_condition)

    variables: dict[str, Any] = {}
    try:
        for var in model.component_objects(pyo.Var, active=True):
            var_dict: dict[str, float] = {}
            for idx in var:
                val = pyo.value(var[idx])
                if val is not None:
                    key = str(idx) if idx is not None else "scalar"
                    var_dict[key] = float(val)
            variables[var.name] = var_dict
    except Exception:
        pass

    obj_value = float("inf")
    try:
        for obj in model.component_objects(pyo.Objective, active=True):
            obj_value = float(pyo.value(obj))
            break
    except Exception:
        pass

    return {
        "status": status,
        "termination": termination,
        "objective_value": obj_value,
        "variables": variables,
        "solver": solver,
    }


def solve_scipy_minimize(
    objective: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    constraints: list[dict[str, Any]] | None = None,
    x0: np.ndarray | None = None,
    method: str = "SLSQP",
) -> dict[str, Any]:
    """Solve a continuous optimization problem with scipy.

    Returns optimal parameters, objective value, and convergence info.
    """
    if x0 is None:
        x0 = np.array([(lo + hi) / 2.0 for lo, hi in bounds])

    scipy_constraints = constraints or []

    result = scipy_minimize(
        objective,
        x0,
        method=method,
        bounds=bounds,
        constraints=scipy_constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    return {
        "x": result.x.tolist(),
        "objective_value": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
        "n_iterations": int(result.nit) if hasattr(result, "nit") else 0,
        "n_function_evals": int(result.nfev) if hasattr(result, "nfev") else 0,
    }


def grid_search(
    param_grid: dict[str, list[Any]],
    evaluate_fn: Callable[..., float],
) -> pd.DataFrame:
    """Brute-force parameter search over a grid.

    param_grid: {param_name: [values]}
    evaluate_fn: callable(**params) -> float (objective value to minimize)
    Returns DataFrame with all parameter combinations and their scores.
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    rows: list[dict[str, Any]] = []

    for combo in product(*param_values):
        params = dict(zip(param_names, combo))
        try:
            score = evaluate_fn(**params)
            params["objective_value"] = score
            params["status"] = "ok"
        except Exception as e:
            params["objective_value"] = float("inf")
            params["status"] = f"error: {e}"
        rows.append(params)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("objective_value").reset_index(drop=True)
    return df


def decomposition_solve(
    master_model: Any,
    subproblem_fn: Callable[[dict[str, Any]], dict[str, Any]],
    max_iter: int = 50,
    gap_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Simple Benders-like decomposition.

    1. Solve master problem to get complicating variables.
    2. Fix those variables and solve subproblems via subproblem_fn.
    3. subproblem_fn returns cuts to add to master.
    4. Repeat until convergence or max_iter.

    subproblem_fn(master_solution: dict) -> {
        'objective': float,
        'cuts': [{'coeffs': dict, 'rhs': float}],
        'feasible': bool,
    }
    """
    lower_bound = float("-inf")
    upper_bound = float("inf")
    history: list[dict[str, Any]] = []

    try:
        import pyomo.environ as pyo
    except ImportError:
        raise ImportError("pyomo is required for decomposition solve")

    for iteration in range(max_iter):
        master_result = solve_milp(master_model, solver="glpk", time_limit=60)
        master_obj = master_result.get("objective_value", float("inf"))
        lower_bound = max(lower_bound, master_obj)

        sub_result = subproblem_fn(master_result.get("variables", {}))
        sub_obj = sub_result.get("objective", float("inf"))

        if sub_result.get("feasible", True):
            upper_bound = min(upper_bound, master_obj + sub_obj)
        else:
            upper_bound = float("inf")

        gap = (upper_bound - lower_bound) / max(abs(upper_bound), 1e-10)

        history.append({
            "iteration": iteration,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "gap": gap,
            "master_obj": master_obj,
            "sub_obj": sub_obj,
            "feasible": sub_result.get("feasible", True),
        })

        if gap <= gap_tolerance:
            break

        cuts = sub_result.get("cuts", [])
        if not cuts:
            break

        for cut_idx, cut in enumerate(cuts):
            cut_name = f"benders_cut_{iteration}_{cut_idx}"

            coeffs = cut.get("coeffs", {})
            rhs = cut.get("rhs", 0.0)

            try:
                constraint_expr = sum(
                    coeffs[var_name] * master_model.find_component(var_name)
                    for var_name in coeffs
                    if master_model.find_component(var_name) is not None
                )
                master_model.add_component(
                    cut_name,
                    pyo.Constraint(expr=constraint_expr >= rhs),
                )
            except Exception:
                continue

    return {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "gap": (upper_bound - lower_bound) / max(abs(upper_bound), 1e-10),
        "iterations": len(history),
        "converged": (upper_bound - lower_bound) / max(abs(upper_bound), 1e-10) <= gap_tolerance,
        "history": pd.DataFrame(history),
    }
