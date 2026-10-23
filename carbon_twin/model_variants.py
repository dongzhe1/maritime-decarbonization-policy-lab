from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from carbon_twin.data.schemas import FuelType
from carbon_twin.emissions.factors import GWP100

_DEFAULT_CONFIG = Path("configs/model_variants.yaml")
_SCOPE_MODES = ("full", "mrv", "mrv_routed", "mrv_priority", "mrv_routed_voyage",
                "mrv_priority_bd", "mrv_bd", "mrv_routed_voyage_bd")

_LNG_CF_CO2 = 2.750
_LNG_CF_CH4 = 0.0
_LNG_CF_N2O = 0.00011
_LNG_LCV_MJ_PER_G = 0.0491


@dataclass(frozen=True)
class ModelVariant:
    name: str = "v23"
    description: str = "Original specification (no-op)"
    lng_cslip_pct: float | None = None
    lng_wtt_g_per_mj: float = 18.5
    fueleu_gwp: dict[str, float] = field(default_factory=lambda: {"CH4": 25.0, "N2O": 298.0})
    scope_mode: str = "full"
    fueleu_version: str | None = None
    fueleu_factors_from_config: bool = False
    report_annual: bool = False
    base_fuels: tuple[str, ...] = ("VLSFO",)
    wtw_overrides: dict[str, float] = field(default_factory=dict)
    exact_blend: bool = False
    clean_objective: bool = False

    def __post_init__(self) -> None:
        if self.scope_mode not in _SCOPE_MODES:
            raise ValueError(f"scope_mode must be one of {_SCOPE_MODES}, got {self.scope_mode!r}")
        if self.lng_cslip_pct is not None and not 0.0 <= self.lng_cslip_pct < 100.0:
            raise ValueError(f"lng_cslip_pct out of range: {self.lng_cslip_pct!r}")
        for b in self.base_fuels:
            FuelType(b)
        for f, v in self.wtw_overrides.items():
            FuelType(f)
            if not 0.0 <= float(v) < 200.0:
                raise ValueError(f"wtw_overrides[{f!r}] out of range: {v!r}")


    def lng_factors(self) -> dict[str, float] | None:
        if self.lng_cslip_pct is None:
            return None
        return lng_annex2_factors(self.lng_cslip_pct, self.lng_wtt_g_per_mj,
                                  self.fueleu_gwp)

    def fuel_factors(self, factors: dict | None) -> dict | None:
        if factors is None:
            return None
        out = dict(factors)
        lng = self.lng_factors()
        if lng is not None and FuelType.LNG in factors:
            out[FuelType.LNG] = dataclasses.replace(
                factors[FuelType.LNG],
                ttw_co2_g_per_g_fuel=lng["ttw_co2_g_per_g_fuel"],
                ttw_ch4_g_per_g_fuel=lng["ttw_ch4_g_per_g_fuel"],
                ttw_n2o_g_per_g_fuel=lng["ttw_n2o_g_per_g_fuel"],
                wtw_co2e_g_per_mj=lng["wtw_co2e_g_per_mj"],
                fueleu_default_emission_factor_gCO2e_per_MJ=lng["wtw_co2e_g_per_mj"],
            )
        for f, wtw in self.wtw_overrides.items():
            ft = FuelType(f)
            if ft in out:
                out[ft] = dataclasses.replace(
                    out[ft], wtw_co2e_g_per_mj=float(wtw),
                    fueleu_default_emission_factor_gCO2e_per_MJ=float(wtw))
        return out


    def fueleu_version_or(self, default: str) -> str:
        return self.fueleu_version or default

    def fuel_optimizer_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "scope_mode": self.scope_mode,
            "report_annual": self.report_annual,
            "fueleu_factors_from_config": self.fueleu_factors_from_config,
            "base_fuels": list(self.base_fuels),
        }
        if self.exact_blend:
            kw["exact_blend"] = True
        return kw

    def cii_kwargs(self) -> dict[str, Any]:
        return {"include_cii_in_objective": False} if self.clean_objective else {}

    def speed_optimizer_kwargs(self) -> dict[str, Any]:
        return {"scope_mode": self.scope_mode, "report_annual": self.report_annual,
                "fueleu_factors_from_config": self.fueleu_factors_from_config}

    def compliance_kwargs(self, factors: dict | None = None) -> dict[str, Any]:
        kw: dict[str, Any] = {"scope_mode": self.scope_mode}
        if self.fueleu_factors_from_config:
            from carbon_twin.emissions.proxy import load_fuel_factors_safe
            from carbon_twin.optimization.fuel_choice import fueleu_factor_table

            base = factors if factors is not None else load_fuel_factors_safe()
            kw["fueleu_emission_factors"] = fueleu_factor_table(self.fuel_factors(base))
        return kw

    def manifest(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["base_fuels"] = list(self.base_fuels)
        d["lng_factors_derived"] = self.lng_factors()
        return d


def lng_annex2_factors(cslip_pct: float, wtt_g_per_mj: float = 18.5,
                       fueleu_gwp: dict[str, float] | None = None) -> dict[str, float]:
    s = cslip_pct / 100.0
    gwp = fueleu_gwp or {"CH4": 25.0, "N2O": 298.0}
    ttw_co2 = (1.0 - s) * _LNG_CF_CO2
    ttw_ch4 = (1.0 - s) * _LNG_CF_CH4 + s
    ttw_n2o = (1.0 - s) * _LNG_CF_N2O
    fueleu_ttw_g_per_g = ttw_co2 + ttw_ch4 * gwp["CH4"] + ttw_n2o * gwp["N2O"]
    ets_ttw_g_per_g = ttw_co2 + ttw_ch4 * GWP100["CH4"] + ttw_n2o * GWP100["N2O"]
    return {
        "cslip_pct": cslip_pct,
        "ttw_co2_g_per_g_fuel": ttw_co2,
        "ttw_ch4_g_per_g_fuel": ttw_ch4,
        "ttw_n2o_g_per_g_fuel": ttw_n2o,
        "ets_ttw_co2e_g_per_mj": ets_ttw_g_per_g / _LNG_LCV_MJ_PER_G,
        "fueleu_ttw_co2e_g_per_mj": fueleu_ttw_g_per_g / _LNG_LCV_MJ_PER_G,
        "wtw_co2e_g_per_mj": wtt_g_per_mj + fueleu_ttw_g_per_g / _LNG_LCV_MJ_PER_G,
    }


def available_variants(config_path: str | Path = _DEFAULT_CONFIG) -> list[str]:
    with open(config_path) as f:
        return list((yaml.safe_load(f) or {}).get("variants", {}))


def load_variant(name: str | None,
                 config_path: str | Path = _DEFAULT_CONFIG) -> ModelVariant:
    if name is None:
        return ModelVariant()
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    variants = cfg.get("variants", {})
    if name not in variants:
        raise KeyError(f"unknown model variant {name!r}; available: {sorted(variants)}")
    raw = dict(variants[name])
    known = {f.name for f in dataclasses.fields(ModelVariant)}
    unknown = set(raw) - known
    if unknown:
        raise KeyError(f"variant {name!r} has unknown keys: {sorted(unknown)}")
    if "base_fuels" in raw:
        raw["base_fuels"] = tuple(raw["base_fuels"])
    return ModelVariant(name=name, **raw)


def add_variant_argument(ap: Any) -> None:
    ap.add_argument(
        "--variant", default=None,
        help="model variant from configs/model_variants.yaml "
             "(default: v23, the original specification)")


def write_variant_manifest(out_path: str | Path, variant: ModelVariant) -> Path:
    p = Path(str(out_path) + ".variant.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(variant.manifest(), indent=2, sort_keys=True))
    return p
