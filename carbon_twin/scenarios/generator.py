"""Scenario generation for carbon compliance analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class Scenario:
    name: str
    year: int
    eua_price: float = 80.0
    fuel_prices: dict[str, float] = field(default_factory=dict)
    regulation_versions: dict[str, str] = field(default_factory=dict)
    imo_nzf_params: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "year": self.year,
            "eua_price": self.eua_price,
            "fuel_prices": dict(self.fuel_prices),
            "regulation_versions": dict(self.regulation_versions),
            "imo_nzf_params": dict(self.imo_nzf_params),
            "uncertainty": dict(self.uncertainty),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scenario:
        return cls(
            name=data.get("name", "unnamed"),
            year=data.get("year", 2026),
            eua_price=data.get("eua_price", 80.0),
            fuel_prices=data.get("fuel_prices", {}),
            regulation_versions=data.get("regulation_versions", {}),
            imo_nzf_params=data.get("imo_nzf_params", {}),
            uncertainty=data.get("uncertainty", {}),
            metadata=data.get("metadata", {}),
        )


def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario from a YAML file."""
    path = Path(path)
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    scenario_data = raw.get("scenario", raw)

    fuel_prices: dict[str, float] = {}
    fp_ref = scenario_data.get("fuel_prices", {})
    if isinstance(fp_ref, dict):
        fuel_prices = {str(k): float(v) for k, v in fp_ref.items()}

    reg_versions: dict[str, str] = {}
    regs = scenario_data.get("regulations", {})
    for reg_name, reg_info in regs.items():
        if isinstance(reg_info, dict) and "version" in reg_info:
            reg_versions[reg_name] = reg_info["version"]
        elif isinstance(reg_info, str):
            reg_versions[reg_name] = reg_info

    imo_nzf = scenario_data.get("imo_nzf_params", {})
    uncertainty = scenario_data.get("uncertainty", {})

    eua_price = scenario_data.get("eua_price", 80.0)
    if isinstance(eua_price, str):
        eua_price = 80.0

    return Scenario(
        name=scenario_data.get("name", path.stem),
        year=scenario_data.get("year", 2026),
        eua_price=float(eua_price),
        fuel_prices=fuel_prices,
        regulation_versions=reg_versions,
        imo_nzf_params=imo_nzf if isinstance(imo_nzf, dict) else {},
        uncertainty={str(k): float(v) for k, v in uncertainty.items()} if isinstance(uncertainty, dict) else {},
        metadata={"source": str(path)},
    )


def generate_scenario_grid(
    base_scenario: Scenario,
    param_ranges: dict[str, list[Any]],
) -> list[Scenario]:
    """Generate scenarios as a cartesian product of parameter ranges.

    param_ranges: {
        'eua_price': [50, 80, 120],
        'fuel_prices.VLSFO': [500, 600, 700],
        'year': [2026, 2028, 2030],
    }
    """
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())

    scenarios: list[Scenario] = []

    for idx, combo in enumerate(product(*param_values)):
        params = dict(zip(param_names, combo))
        scenario = Scenario(
            name=f"{base_scenario.name}_grid_{idx}",
            year=base_scenario.year,
            eua_price=base_scenario.eua_price,
            fuel_prices=dict(base_scenario.fuel_prices),
            regulation_versions=dict(base_scenario.regulation_versions),
            imo_nzf_params=dict(base_scenario.imo_nzf_params),
            uncertainty=dict(base_scenario.uncertainty),
            metadata={"grid_index": idx, "grid_params": params},
        )

        for param_name, value in params.items():
            if param_name == "eua_price":
                scenario.eua_price = float(value)
            elif param_name == "year":
                scenario.year = int(value)
            elif param_name.startswith("fuel_prices."):
                fuel_key = param_name.split(".", 1)[1]
                scenario.fuel_prices[fuel_key] = float(value)
            elif param_name.startswith("imo_nzf_params."):
                nzf_key = param_name.split(".", 1)[1]
                scenario.imo_nzf_params[nzf_key] = value
            elif param_name.startswith("uncertainty."):
                unc_key = param_name.split(".", 1)[1]
                scenario.uncertainty[unc_key] = float(value)
            elif param_name.startswith("regulation_versions."):
                reg_key = param_name.split(".", 1)[1]
                scenario.regulation_versions[reg_key] = str(value)

        scenarios.append(scenario)

    return scenarios


def generate_random_scenarios(
    base_scenario: Scenario,
    distributions: dict[str, dict[str, Any]],
    n: int = 1000,
    seed: int = 42,
) -> list[Scenario]:
    """Sample scenarios from specified distributions.

    distributions: {
        'eua_price': {'distribution': 'lognormal', 'mean': 80, 'std': 20},
        'fuel_prices.VLSFO': {'distribution': 'normal', 'mean': 600, 'std': 50},
    }

    Supported distributions: normal, lognormal, uniform, triangular.
    """
    rng = np.random.default_rng(seed)
    scenarios: list[Scenario] = []

    sampled_values: dict[str, np.ndarray] = {}
    for param_name, dist_spec in distributions.items():
        dist = dist_spec.get("distribution", "normal")
        if dist == "normal":
            mean = dist_spec.get("mean", 0.0)
            std = dist_spec.get("std", 1.0)
            sampled_values[param_name] = rng.normal(mean, std, n)
        elif dist == "lognormal":
            mean = dist_spec.get("mean", 1.0)
            std = dist_spec.get("std", 0.5)
            mu = np.log(mean ** 2 / np.sqrt(mean ** 2 + std ** 2))
            sigma = np.sqrt(np.log(1 + (std / mean) ** 2))
            sampled_values[param_name] = rng.lognormal(mu, sigma, n)
        elif dist == "uniform":
            low = dist_spec.get("low", 0.0)
            high = dist_spec.get("high", 1.0)
            sampled_values[param_name] = rng.uniform(low, high, n)
        elif dist == "triangular":
            left = dist_spec.get("left", 0.0)
            mode = dist_spec.get("mode", 0.5)
            right = dist_spec.get("right", 1.0)
            sampled_values[param_name] = rng.triangular(left, mode, right, n)
        else:
            raise ValueError(f"Unsupported distribution: {dist}")

    for i in range(n):
        scenario = Scenario(
            name=f"{base_scenario.name}_mc_{i}",
            year=base_scenario.year,
            eua_price=base_scenario.eua_price,
            fuel_prices=dict(base_scenario.fuel_prices),
            regulation_versions=dict(base_scenario.regulation_versions),
            imo_nzf_params=dict(base_scenario.imo_nzf_params),
            uncertainty=dict(base_scenario.uncertainty),
            metadata={"sample_index": i, "seed": seed},
        )

        for param_name, values in sampled_values.items():
            value = float(values[i])
            if param_name == "eua_price":
                scenario.eua_price = max(value, 0.0)
            elif param_name == "year":
                scenario.year = int(round(value))
            elif param_name.startswith("fuel_prices."):
                fuel_key = param_name.split(".", 1)[1]
                scenario.fuel_prices[fuel_key] = max(value, 0.0)
            elif param_name.startswith("imo_nzf_params."):
                nzf_key = param_name.split(".", 1)[1]
                scenario.imo_nzf_params[nzf_key] = value
            elif param_name.startswith("uncertainty."):
                unc_key = param_name.split(".", 1)[1]
                scenario.uncertainty[unc_key] = value

        scenarios.append(scenario)

    return scenarios
