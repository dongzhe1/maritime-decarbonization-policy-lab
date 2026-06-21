"""Monte Carlo simulation engine for carbon compliance uncertainty analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from carbon_twin.scenarios.generator import Scenario


@dataclass
class UncertaintySpec:
    param_name: str
    distribution: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def sample(self, rng: np.random.Generator) -> float:
        if self.distribution == "normal":
            return float(rng.normal(self.kwargs["mean"], self.kwargs["std"]))
        elif self.distribution == "lognormal":
            mean = self.kwargs["mean"]
            std = self.kwargs["std"]
            mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
            sigma = np.sqrt(np.log(1 + std**2 / mean**2))
            return float(rng.lognormal(mu, sigma))
        elif self.distribution == "uniform":
            return float(rng.uniform(self.kwargs["low"], self.kwargs["high"]))
        elif self.distribution == "triangular":
            return float(
                rng.triangular(self.kwargs["left"], self.kwargs["mode"], self.kwargs["right"])
            )
        elif self.distribution == "beta":
            return float(rng.beta(self.kwargs.get("a", 2.0), self.kwargs.get("b", 5.0)))
        elif self.distribution == "discrete":
            values = self.kwargs["values"]
            probs = self.kwargs.get("probs", None)
            idx = rng.choice(len(values), p=probs)
            return values[idx]
        raise ValueError(f"Unknown distribution: {self.distribution}")

    def sample_array(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.distribution == "normal":
            return rng.normal(self.kwargs["mean"], self.kwargs["std"], n)
        elif self.distribution == "lognormal":
            mean = self.kwargs["mean"]
            std = self.kwargs["std"]
            mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
            sigma = np.sqrt(np.log(1 + std**2 / mean**2))
            return rng.lognormal(mu, sigma, n)
        elif self.distribution == "uniform":
            return rng.uniform(self.kwargs["low"], self.kwargs["high"], n)
        elif self.distribution == "triangular":
            return rng.triangular(self.kwargs["left"], self.kwargs["mode"], self.kwargs["right"], n)
        elif self.distribution == "beta":
            return rng.beta(self.kwargs.get("a", 2.0), self.kwargs.get("b", 5.0), n)
        elif self.distribution == "discrete":
            values = np.array(self.kwargs["values"])
            probs = self.kwargs.get("probs", None)
            indices = rng.choice(len(values), size=n, p=probs)
            return values[indices]
        raise ValueError(f"Unknown distribution: {self.distribution}")


class MonteCarloEngine:
    """Monte Carlo simulation engine.

    Supports two modes of operation:
    1. Scenario-based: takes a base Scenario, samples uncertainties,
       and calls evaluate_fn(Scenario) -> dict[str, float].
    2. Dict-based (legacy): takes base_params dict, samples,
       and calls evaluate_fn(dict) -> dict.
    """

    def __init__(
        self,
        base_scenario: Scenario | dict[str, Any] | None = None,
        n_samples: int = 1000,
        seed: int = 42,
        *,
        base_params: dict[str, Any] | None = None,
    ) -> None:
        if base_scenario is None and base_params is not None:
            self._base_scenario = None
            self._base_params = dict(base_params)
        elif isinstance(base_scenario, dict):
            self._base_scenario = None
            self._base_params = dict(base_scenario)
        elif isinstance(base_scenario, Scenario):
            self._base_scenario = base_scenario
            self._base_params = base_scenario.to_dict()
        else:
            self._base_scenario = None
            self._base_params = {}

        self.n_samples = n_samples
        self.seed = seed
        self.uncertainties: list[UncertaintySpec] = []
        self._results: pd.DataFrame | None = None

    @property
    def base_params(self) -> dict[str, Any]:
        return self._base_params

    def add_uncertainty(self, param_name: str, distribution: str, **kwargs: Any) -> None:
        self.uncertainties.append(UncertaintySpec(param_name, distribution, kwargs))

    def _build_scenario(self, param_values: dict[str, float]) -> Scenario:
        if self._base_scenario is not None:
            scenario = Scenario(
                name=f"{self._base_scenario.name}_mc",
                year=self._base_scenario.year,
                eua_price=self._base_scenario.eua_price,
                fuel_prices=dict(self._base_scenario.fuel_prices),
                regulation_versions=dict(self._base_scenario.regulation_versions),
                imo_nzf_params=dict(self._base_scenario.imo_nzf_params),
                uncertainty=dict(self._base_scenario.uncertainty),
            )
        else:
            scenario = Scenario(
                name="mc_scenario",
                year=int(self._base_params.get("year", 2026)),
                eua_price=float(self._base_params.get("eua_price", 80.0)),
            )

        for param_name, value in param_values.items():
            if param_name == "eua_price":
                scenario.eua_price = max(float(value), 0.0)
            elif param_name.startswith("fuel_prices."):
                fuel_key = param_name.split(".", 1)[1]
                scenario.fuel_prices[fuel_key] = max(float(value), 0.0)
            elif param_name.startswith("imo_nzf_params."):
                nzf_key = param_name.split(".", 1)[1]
                scenario.imo_nzf_params[nzf_key] = float(value)
            else:
                scenario.uncertainty[param_name] = float(value)

        return scenario

    def run(
        self,
        evaluate_fn: Callable[..., dict[str, Any]],
    ) -> pd.DataFrame:
        """Run evaluate_fn for each sampled scenario.

        The evaluate_fn is called with either a Scenario object (if a
        base Scenario was provided) or a dict of parameters (legacy mode).
        """
        rng = np.random.default_rng(self.seed)
        rows: list[dict[str, Any]] = []

        sampled_arrays: dict[str, np.ndarray] = {}
        for spec in self.uncertainties:
            sampled_arrays[spec.param_name] = spec.sample_array(rng, self.n_samples)

        for i in range(self.n_samples):
            param_values = {name: float(arr[i]) for name, arr in sampled_arrays.items()}

            try:
                if self._base_scenario is not None:
                    scenario = self._build_scenario(param_values)
                    result = evaluate_fn(scenario)
                else:
                    params = dict(self._base_params)
                    params.update(param_values)
                    result = evaluate_fn(params)

                row: dict[str, Any] = {"sample_id": i}
                for name, val in param_values.items():
                    row[f"input_{name}"] = val
                row.update(result)
            except Exception as e:
                row = {"sample_id": i, "_error": str(e)}
                for name, val in param_values.items():
                    row[f"input_{name}"] = val

            rows.append(row)

        self._results = pd.DataFrame(rows)
        return self._results

    def summary_statistics(self, metric: str | None = None) -> dict[str, Any]:
        """Summary statistics for output metric(s).

        If metric is given, returns stats for that column.
        If metric is None, returns stats for all numeric output columns.
        """
        if self._results is None:
            raise RuntimeError("Must call run() before summary_statistics()")

        input_cols = {"sample_id", "_error"}
        for spec in self.uncertainties:
            input_cols.add(f"input_{spec.param_name}")

        if metric is not None:
            if metric not in self._results.columns:
                return {}
            data = self._results[metric].dropna()
            return {
                "mean": float(data.mean()),
                "std": float(data.std()),
                "p5": float(np.percentile(data, 5)),
                "p25": float(np.percentile(data, 25)),
                "p50": float(np.percentile(data, 50)),
                "p75": float(np.percentile(data, 75)),
                "p95": float(np.percentile(data, 95)),
                "min": float(data.min()),
                "max": float(data.max()),
                "n_samples": len(data),
            }

        output_cols = [
            c
            for c in self._results.columns
            if c not in input_cols and self._results[c].dtype in (np.float64, np.int64, float, int)
        ]

        stats: dict[str, dict[str, float]] = {}
        for col in output_cols:
            data = self._results[col].dropna()
            if len(data) == 0:
                continue
            stats[col] = {
                "mean": float(data.mean()),
                "std": float(data.std()),
                "p5": float(np.percentile(data, 5)),
                "p25": float(np.percentile(data, 25)),
                "p50": float(np.percentile(data, 50)),
                "p75": float(np.percentile(data, 75)),
                "p95": float(np.percentile(data, 95)),
                "min": float(data.min()),
                "max": float(data.max()),
            }

        return stats

    def cvar(self, alpha: float = 0.05, metric: str | None = None) -> float:
        """Conditional Value at Risk (Expected Shortfall).

        Returns the mean of the worst alpha-fraction of outcomes.
        """
        if self._results is None:
            raise RuntimeError("Must call run() before cvar()")

        if metric is None:
            candidates = [
                "total_cost_eur",
                "total_cost",
                "compliance_cost",
                "total_compliance_cost_eur",
            ]
            for c in candidates:
                if c in self._results.columns:
                    metric = c
                    break
            if metric is None:
                input_cols = {"sample_id", "_error"}
                for spec in self.uncertainties:
                    input_cols.add(f"input_{spec.param_name}")
                numeric_cols = [
                    c
                    for c in self._results.columns
                    if c not in input_cols
                    and self._results[c].dtype in (np.float64, np.int64, float, int)
                ]
                if not numeric_cols:
                    return float("nan")
                metric = numeric_cols[0]

        data = self._results[metric].dropna().values
        if len(data) == 0:
            return float("nan")
        threshold = np.percentile(data, 100 * (1 - alpha))
        tail = data[data >= threshold]
        return float(tail.mean()) if len(tail) > 0 else float("nan")

    def tornado_diagram_data(self, metric: str | None = None) -> pd.DataFrame:
        """Sensitivity of output to each input parameter.

        For each uncertain input, computes the output when the input is in
        its bottom/top decile. Returns a DataFrame sorted by impact magnitude.
        """
        if self._results is None:
            raise RuntimeError("Must call run() before tornado_diagram_data()")

        if metric is None:
            candidates = [
                "total_cost_eur",
                "total_cost",
                "compliance_cost",
                "total_compliance_cost_eur",
            ]
            for c in candidates:
                if c in self._results.columns:
                    metric = c
                    break
            if metric is None:
                return pd.DataFrame()

        rows: list[dict[str, Any]] = []

        for spec in self.uncertainties:
            input_col = f"input_{spec.param_name}"
            if input_col not in self._results.columns or metric not in self._results.columns:
                continue

            valid = self._results[[input_col, metric]].dropna()
            if len(valid) < 10:
                continue

            corr = valid[[input_col, metric]].corr().iloc[0, 1]
            p10 = valid[input_col].quantile(0.1)
            p90 = valid[input_col].quantile(0.9)

            low_mask = valid[input_col] <= p10
            high_mask = valid[input_col] >= p90

            low_mean = float(valid.loc[low_mask, metric].mean()) if low_mask.any() else float("nan")
            high_mean = (
                float(valid.loc[high_mask, metric].mean()) if high_mask.any() else float("nan")
            )

            swing = (
                abs(high_mean - low_mean)
                if not (np.isnan(high_mean) or np.isnan(low_mean))
                else 0.0
            )

            rows.append(
                {
                    "parameter": spec.param_name,
                    "correlation": float(corr) if not np.isnan(corr) else 0.0,
                    "low_input": float(p10),
                    "high_input": float(p90),
                    "low_10pct_mean": low_mean,
                    "high_90pct_mean": high_mean,
                    "swing": swing,
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("swing", ascending=False).reset_index(drop=True)
        return df
