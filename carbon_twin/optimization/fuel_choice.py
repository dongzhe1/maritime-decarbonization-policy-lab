from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from carbon_twin.data.schemas import FuelRecord, FuelType, Vessel, Voyage
from carbon_twin.emissions.factors import (
    GWP100,
    _DEFAULT_TTW_CO2_G_PER_G,
    FuelEmissionFactors,
    get_co2e_ttw,
    get_co2e_wtw,
)
from carbon_twin.emissions.physics import (
    voyage_fuel_consumption,
)


@dataclass
class FuelOption:
    fuel_type: FuelType
    price_per_tonne: float
    availability: float = 1.0
    max_blend_pct: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.fuel_type, FuelType):
            self.fuel_type = FuelType(self.fuel_type)


def _voyage_energy_mj(
    vessel: Vessel,
    voyage: Voyage,
    fuel_factors: dict[FuelType, FuelEmissionFactors],
    base_fuel: FuelType = FuelType.VLSFO,
) -> float:
    fc = voyage_fuel_consumption(vessel, voyage)
    total_tonnes = fc["total_tonnes"]
    if base_fuel in fuel_factors:
        lcv = fuel_factors[base_fuel].lcv_mj_per_kg
    else:
        lcv = 41.0
    return total_tonnes * 1e3 * lcv


def _fuel_mass_for_energy(
    energy_mj: float,
    fuel_type: FuelType,
    fuel_factors: dict[FuelType, FuelEmissionFactors],
) -> float:
    if fuel_type not in fuel_factors:
        return energy_mj / (41.0 * 1e3)
    lcv = fuel_factors[fuel_type].lcv_mj_per_kg
    if lcv <= 0:
        return 0.0
    return energy_mj / (lcv * 1e3)


def optimal_fuel_mix(
    vessel: Vessel,
    voyage: Voyage,
    fuel_options: list[FuelOption],
    fuel_factors: dict[FuelType, FuelEmissionFactors],
    regulations: list[str] | None = None,
    year: int = 2026,
    blend_resolution: int = 20,
) -> dict[str, Any]:
    total_energy = _voyage_energy_mj(vessel, voyage, fuel_factors)

    if len(fuel_options) == 0:
        return {
            "fuel_mix": {},
            "total_cost": 0.0,
            "ghg_intensity": 0.0,
            "co2e_ttw": 0.0,
            "co2e_wtw": 0.0,
        }

    if len(fuel_options) == 1:
        fo = fuel_options[0]
        fraction = min(fo.max_blend_pct, 1.0)
        energy = total_energy * fraction
        mass = _fuel_mass_for_energy(energy, fo.fuel_type, fuel_factors)
        cost = mass * fo.price_per_tonne
        co2e_ttw = get_co2e_ttw(fo.fuel_type, mass, fuel_factors)
        co2e_wtw = get_co2e_wtw(fo.fuel_type, energy, fuel_factors)
        ghg_i = fuel_factors[fo.fuel_type].wtw_co2e_g_per_mj if fo.fuel_type in fuel_factors else 0.0
        return {
            "fuel_mix": {fo.fuel_type: fraction},
            "total_cost": cost,
            "ghg_intensity": ghg_i,
            "co2e_ttw": co2e_ttw,
            "co2e_wtw": co2e_wtw,
        }

    steps = np.arange(0, blend_resolution + 1) / blend_resolution
    n_fuels = len(fuel_options)

    best_cost = float("inf")
    best_result: dict[str, Any] = {}

    feasible_steps_per_fuel = []
    for fo in fuel_options:
        max_steps = int(fo.max_blend_pct * blend_resolution)
        feasible_steps_per_fuel.append(range(0, max_steps + 1))

    for combo in product(*feasible_steps_per_fuel):
        total_steps = sum(combo)
        if total_steps == 0 or total_steps != blend_resolution:
            continue

        fractions = [c / blend_resolution for c in combo]

        total_cost = 0.0
        total_co2e_ttw = 0.0
        total_co2e_wtw = 0.0
        weighted_ghg = 0.0
        fuel_mix: dict[FuelType, float] = {}

        for i, fo in enumerate(fuel_options):
            frac = fractions[i]
            if frac <= 0:
                continue
            energy = total_energy * frac
            mass = _fuel_mass_for_energy(energy, fo.fuel_type, fuel_factors)
            cost = mass * fo.price_per_tonne
            co2e_ttw = get_co2e_ttw(fo.fuel_type, mass, fuel_factors)
            co2e_wtw = get_co2e_wtw(fo.fuel_type, energy, fuel_factors)

            total_cost += cost
            total_co2e_ttw += co2e_ttw
            total_co2e_wtw += co2e_wtw
            fuel_mix[fo.fuel_type] = frac

            if fo.fuel_type in fuel_factors:
                weighted_ghg += fuel_factors[fo.fuel_type].wtw_co2e_g_per_mj * frac

        if total_cost < best_cost:
            best_cost = total_cost
            best_result = {
                "fuel_mix": fuel_mix,
                "total_cost": total_cost,
                "ghg_intensity": weighted_ghg,
                "co2e_ttw": total_co2e_ttw,
                "co2e_wtw": total_co2e_wtw,
            }

    if not best_result:
        return {
            "fuel_mix": {},
            "total_cost": float("inf"),
            "ghg_intensity": 0.0,
            "co2e_ttw": 0.0,
            "co2e_wtw": 0.0,
        }

    return best_result


def fuel_switching_threshold(
    vessel: Vessel,
    voyage: Voyage,
    from_fuel: FuelOption,
    to_fuel: FuelOption,
    fuel_factors: dict[FuelType, FuelEmissionFactors],
    eua_prices: list[float] | None = None,
) -> float:
    total_energy = _voyage_energy_mj(vessel, voyage, fuel_factors)
    ets_scope = voyage.ets_scope_factor

    mass_from = _fuel_mass_for_energy(total_energy, from_fuel.fuel_type, fuel_factors)
    mass_to = _fuel_mass_for_energy(total_energy, to_fuel.fuel_type, fuel_factors)

    fuel_cost_from = mass_from * from_fuel.price_per_tonne
    fuel_cost_to = mass_to * to_fuel.price_per_tonne

    co2e_from = get_co2e_ttw(from_fuel.fuel_type, mass_from, fuel_factors)
    co2e_to = get_co2e_ttw(to_fuel.fuel_type, mass_to, fuel_factors)

    delta_co2e = (co2e_from - co2e_to) * ets_scope
    delta_fuel_cost = fuel_cost_to - fuel_cost_from

    if delta_co2e <= 0:
        return float("inf")

    threshold = delta_fuel_cost / delta_co2e
    return max(threshold, 0.0)


def marginal_abatement_cost(
    vessel: Vessel,
    voyage: Voyage,
    fuel_options: list[FuelOption],
    fuel_factors: dict[FuelType, FuelEmissionFactors],
    baseline_fuel: FuelType = FuelType.VLSFO,
) -> pd.DataFrame:
    total_energy = _voyage_energy_mj(vessel, voyage, fuel_factors)

    baseline_option = None
    for fo in fuel_options:
        if fo.fuel_type == baseline_fuel:
            baseline_option = fo
            break

    if baseline_option is None:
        baseline_option = FuelOption(fuel_type=baseline_fuel, price_per_tonne=600.0)

    mass_baseline = _fuel_mass_for_energy(total_energy, baseline_fuel, fuel_factors)
    cost_baseline = mass_baseline * baseline_option.price_per_tonne
    co2e_baseline = get_co2e_ttw(baseline_fuel, mass_baseline, fuel_factors)

    rows: list[dict[str, Any]] = []
    for fo in fuel_options:
        if fo.fuel_type == baseline_fuel:
            continue
        mass = _fuel_mass_for_energy(total_energy, fo.fuel_type, fuel_factors)
        cost = mass * fo.price_per_tonne
        co2e = get_co2e_ttw(fo.fuel_type, mass, fuel_factors)

        delta_cost = cost - cost_baseline
        delta_co2e = co2e_baseline - co2e

        mac = delta_cost / delta_co2e if delta_co2e > 0 else float("inf")

        rows.append({
            "fuel_type": fo.fuel_type.value,
            "price_per_tonne": fo.price_per_tonne,
            "fuel_mass_tonnes": mass,
            "fuel_cost": cost,
            "co2e_tonnes": co2e,
            "delta_cost": delta_cost,
            "delta_co2e_tonnes": delta_co2e,
            "mac_eur_per_tco2e": mac,
            "abatement_pct": (delta_co2e / co2e_baseline * 100) if co2e_baseline > 0 else 0.0,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("mac_eur_per_tco2e").reset_index(drop=True)
    return df


from carbon_twin.optimization.speed import (
    _DEFAULT_CII_PRESSURE,
    _cii_annual_penalty,
    _fuel_at_speed,
    _fueleu_penalty,
)

_DROPIN_FUELS: list[FuelType] = [
    FuelType.VLSFO,
    FuelType.MGO,
    FuelType.BIODIESEL_FAME,
]
_GAS_CAPABLE_TYPES: set[str] = {"gas_carrier", "lng_carrier"}
_EXTENDED_FUELS: list[FuelType] = [
    FuelType.LNG,
    FuelType.METHANOL_BIO,
    FuelType.METHANOL_E,
]
_BIOGENIC_CATEGORIES: set[str] = {"biofuel", "rfnbo"}
_BIOFUEL_CATEGORIES: set[str] = {"biofuel"}


def _is_biofuel(ef: FuelEmissionFactors | None) -> bool:
    return ef is not None and ef.category in _BIOFUEL_CATEGORIES


def fueleu_factor_table(
    fuel_factors: dict[FuelType, FuelEmissionFactors] | None,
) -> dict[str, float]:
    from carbon_twin.regulations.fueleu import FuelEUEngine

    table = dict(FuelEUEngine._DEFAULT_FACTORS)
    for ft, ef in (fuel_factors or {}).items():
        key = ft.value if isinstance(ft, FuelType) else str(ft)
        table[key] = float(ef.wtw_co2e_g_per_mj)
    return table


def _is_low_carbon(ef: FuelEmissionFactors | None) -> bool:
    return ef is not None and ef.category in _BIOGENIC_CATEGORIES


_EXTENSIVE_PROFILE_COLUMNS: list[str] = [
    "fuel_cost", "ets_cost", "fueleu_penalty", "fueleu_penalty_full", "cii_penalty",
    "delay_cost", "total_cost", "carbon_fuel_cost", "carbon_fuel_cost_wtw",
    "co2e_wtw_t", "co2e_ttw_t", "fueleu_balance", "fueleu_energy_mj",
]


def fuel_menu(vessel: Vessel, extended: bool = False) -> list[FuelType]:
    menu = list(_DROPIN_FUELS)
    if vessel.ship_type.value in _GAS_CAPABLE_TYPES and FuelType.LNG not in menu:
        menu.append(FuelType.LNG)
    if extended:
        for f in _EXTENDED_FUELS:
            if f not in menu:
                menu.append(f)
    return menu


def _ets_covered_tonnes(
    fuel: FuelType, ef: FuelEmissionFactors | None, mass_tonnes: float, year: int
) -> float:
    if ef is None:
        return mass_tonnes * _DEFAULT_TTW_CO2_G_PER_G
    mass_g = mass_tonnes * 1e6
    biogenic = ef.category in _BIOGENIC_CATEGORIES
    co2e_g = 0.0 if biogenic else ef.ttw_co2_g_per_g_fuel * mass_g
    if year >= 2026:
        co2e_g += ef.ttw_ch4_g_per_g_fuel * mass_g * GWP100["CH4"]
        co2e_g += ef.ttw_n2o_g_per_g_fuel * mass_g * GWP100["N2O"]
    return co2e_g / 1e6


def _cii_co2_tonnes(ef: FuelEmissionFactors | None, mass_tonnes: float) -> float:
    if ef is None:
        return mass_tonnes * _DEFAULT_TTW_CO2_G_PER_G
    return ef.ttw_co2_g_per_g_fuel * mass_tonnes


def _evaluate_point(
    vessel: Vessel,
    voyage: Voyage,
    fuel: FuelType,
    speed: float,
    fuel_price: float,
    eua_price: float,
    fuel_factors: dict[FuelType, FuelEmissionFactors],
    cii_engine: Any,
    fueleu_engine: Any,
    ets_phase_in: float,
    year: int,
    annual_dist: float,
    annual_fuel_scale: float,
    cii_pressure: float,
    charter_per_day: float,
    delay_cost_per_hour: float,
    scheduled_hours: float | None,
) -> dict[str, Any]:
    fc = _fuel_at_speed(vessel, voyage, speed, fuel, fuel_factors)
    mass = fc["total_tonnes"]
    ef = fuel_factors.get(fuel)

    fuel_cost = mass * fuel_price

    ets_co2e = _ets_covered_tonnes(fuel, ef, mass, year)
    ets_cost = ets_co2e * voyage.ets_scope_factor * ets_phase_in * eua_price

    annual_cii_co2 = _cii_co2_tonnes(ef, mass) * annual_fuel_scale
    cii_annual, attained = _cii_annual_penalty(
        cii_engine, vessel, annual_cii_co2, annual_dist, year,
        cii_pressure, charter_per_day,
    )
    cii_penalty = cii_annual / max(annual_fuel_scale, 1.0)

    fueleu_penalty = _fueleu_penalty(
        fueleu_engine, vessel, fuel, fc["energy_mj"], year, fuel_factors
    )

    delay_cost = 0.0
    if scheduled_hours is not None and delay_cost_per_hour > 0:
        delay_cost = max(fc["sea_hours"] - scheduled_hours, 0.0) * delay_cost_per_hour

    total = fuel_cost + ets_cost + fueleu_penalty + cii_penalty + delay_cost
    return {
        "fuel": fuel.value,
        "speed_knots": float(speed),
        "fuel_cost": fuel_cost,
        "ets_cost": ets_cost,
        "fueleu_penalty": fueleu_penalty,
        "cii_penalty": cii_penalty,
        "delay_cost": delay_cost,
        "total_cost": total,
        "carbon_fuel_cost": fuel_cost + ets_cost + delay_cost,
        "attained_cii": attained,
    }


def optimal_fuel_speed_multi_regulation(
    vessel: Vessel,
    voyage: Voyage,
    fuel_prices: dict[str, float],
    eua_price: float,
    fuel_factors: dict[FuelType, FuelEmissionFactors],
    cii_engine: Any,
    fueleu_engine: Any,
    ets_engine: Any,
    year: int,
    fuel_options: list[FuelType] | None = None,
    speed_range: tuple[float, float] = (8, 22),
    n_points: int = 30,
    annual_distance_nm: float | None = None,
    delay_cost_per_hour: float = 0.0,
    scheduled_hours: float | None = None,
    cii_pressure: float = _DEFAULT_CII_PRESSURE,
    extended_fuels: bool = False,
) -> dict[str, Any]:
    menu = fuel_options or fuel_menu(vessel, extended=extended_fuels)
    speeds = np.linspace(speed_range[0], speed_range[1], n_points)
    charter_per_day = delay_cost_per_hour * 24.0

    ets_phase_in = 1.0
    if ets_engine is not None and hasattr(ets_engine, "_phase_in_factor"):
        try:
            ets_phase_in = float(ets_engine._phase_in_factor(year))
        except Exception:
            ets_phase_in = 1.0

    annual_dist = annual_distance_nm or voyage.distance_nm * 12
    annual_fuel_scale = annual_dist / max(voyage.distance_nm, 1.0)

    rows: list[dict[str, Any]] = []
    for fuel in menu:
        price = fuel_prices.get(fuel.value, fuel_prices.get(fuel))
        if price is None:
            continue
        for spd in speeds:
            rows.append(_evaluate_point(
                vessel, voyage, fuel, float(spd), float(price), eua_price,
                fuel_factors, cii_engine, fueleu_engine, ets_phase_in, year,
                annual_dist, annual_fuel_scale, cii_pressure, charter_per_day,
                delay_cost_per_hour, scheduled_hours,
            ))

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    joint = df.loc[df["total_cost"].idxmin()]
    siloed = df.loc[df["carbon_fuel_cost"].idxmin()]
    cost_joint = float(joint["total_cost"])
    cost_siloed = float(siloed["total_cost"])
    rci = (cost_siloed - cost_joint) / cost_siloed if cost_siloed > 0 else 0.0

    return {
        "joint_fuel": str(joint["fuel"]),
        "joint_speed": float(joint["speed_knots"]),
        "siloed_fuel": str(siloed["fuel"]),
        "siloed_speed": float(siloed["speed_knots"]),
        "cost_joint_eur": cost_joint,
        "cost_siloed_eur": cost_siloed,
        "rci": max(rci, 0.0),
        "fuel_switch": str(joint["fuel"]) != str(siloed["fuel"]),
        "ets_cost_eur": float(joint["ets_cost"]),
        "fueleu_penalty_eur": float(joint["fueleu_penalty"]),
        "cii_penalty_eur": float(joint["cii_penalty"]),
        "n_options": len(menu),
        "profile": df,
    }


def _fueleu_blend_eval(
    fueleu_engine: Any,
    vessel: Vessel,
    fuel_energy: list[tuple[FuelType, float]],
    year: int,
    emission_factors: dict[str, float] | None = None,
) -> tuple[float, float, float, float]:
    if fueleu_engine is None:
        return 0.0, 0.0, 0.0, 0.0
    records: list[FuelRecord] = []
    total = 0.0
    for ft, e in fuel_energy:
        if e <= 0:
            continue
        records.append(FuelRecord(
            vessel_imo=vessel.imo_number, fuel_type=ft, mass_tonnes=0.0, energy_mj=e))
        total += e
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0
    try:
        r = fueleu_engine.evaluate(vessel, records, total, year,
                                   emission_factors=emission_factors)
        return r.penalty_eur, r.compliance_balance, r.actual_ghg_intensity, total
    except Exception:
        return 0.0, 0.0, 0.0, total


def _evaluate_blend_point(
    vessel: Vessel,
    voyage: Voyage,
    base_fuel: FuelType,
    alt_fuel: FuelType,
    phi: float,
    speed: float,
    base_price: float,
    alt_price: float,
    eua_price: float,
    fuel_factors: dict[FuelType, FuelEmissionFactors],
    cii_engine: Any,
    fueleu_engine: Any,
    ets_phase_in: float,
    year: int,
    annual_dist: float,
    annual_fuel_scale: float,
    cii_pressure: float,
    charter_per_day: float,
    delay_cost_per_hour: float,
    scheduled_hours: float | None,
    energy_scale: float = 1.0,
    fueleu_price_fraction: float = 1.0,
    fueleu_scope_factor: float = 1.0,
    fueleu_emission_factors: dict[str, float] | None = None,
    energy_reference_fuel: FuelType | None = None,
    scope_allocation: str = "uniform",
    fixed_annual_cost: dict[FuelType, float] | None = None,
) -> dict[str, Any]:
    fc = _fuel_at_speed(vessel, voyage, speed, energy_reference_fuel or base_fuel,
                        fuel_factors)
    energy = fc["energy_mj"] * energy_scale
    bef = fuel_factors.get(base_fuel)
    aef = fuel_factors.get(alt_fuel)
    lcv_b = bef.lcv_mj_per_kg if bef else 41.0
    lcv_a = aef.lcv_mj_per_kg if aef else 41.0

    if scope_allocation in _POOL_ALLOCATIONS:
        return _evaluate_pooled_allocation(
            vessel, voyage, base_fuel, alt_fuel, phi, speed, base_price, alt_price,
            eua_price, fuel_factors, cii_engine, fueleu_engine, ets_phase_in, year,
            annual_dist, annual_fuel_scale, cii_pressure, charter_per_day,
            delay_cost_per_hour, scheduled_hours, fc, energy, bef, aef, lcv_b, lcv_a,
            fueleu_price_fraction, fueleu_scope_factor, fueleu_emission_factors,
            scope_allocation, fixed_annual_cost)

    routed = scope_allocation == "routed"
    e_in = energy * (fueleu_scope_factor if routed else 1.0)
    e_out = energy - e_in
    e_alt = phi * e_in
    e_base = (1.0 - phi) * e_in
    m_base = e_base / (1000.0 * lcv_b) if lcv_b > 0 else 0.0
    m_alt = e_alt / (1000.0 * lcv_a) if lcv_a > 0 else 0.0
    m_out = e_out / (1000.0 * lcv_b) if lcv_b > 0 else 0.0

    fuel_cost = (m_base + m_out) * base_price + m_alt * alt_price
    capex_cost = _leg_capex(fixed_annual_cost, base_fuel, alt_fuel, phi, annual_fuel_scale)
    if capex_cost:
        fuel_cost += capex_cost

    ets_co2e = (_ets_covered_tonnes(base_fuel, bef, m_base, year)
                + _ets_covered_tonnes(alt_fuel, aef, m_alt, year))
    ets_cost = ets_co2e * voyage.ets_scope_factor * ets_phase_in * eua_price

    cii_co2_leg = _cii_co2_tonnes(bef, m_base + m_out) + _cii_co2_tonnes(aef, m_alt)
    annual_cii_co2 = cii_co2_leg * annual_fuel_scale
    cii_annual, attained = _cii_annual_penalty(
        cii_engine, vessel, annual_cii_co2, annual_dist, year,
        cii_pressure, charter_per_day)
    cii_penalty = cii_annual / max(annual_fuel_scale, 1.0)

    fe_scope = 1.0 if routed else fueleu_scope_factor
    fueleu_penalty, fueleu_balance, fueleu_actual_int, fueleu_energy = _fueleu_blend_eval(
        fueleu_engine, vessel,
        [(base_fuel, e_base * fe_scope), (alt_fuel, e_alt * fe_scope)],
        year, emission_factors=fueleu_emission_factors)

    delay_cost = 0.0
    if scheduled_hours is not None and delay_cost_per_hour > 0:
        delay_cost = max(fc["sea_hours"] - scheduled_hours, 0.0) * delay_cost_per_hour

    wtw_b = bef.wtw_co2e_g_per_mj if bef else _DEFAULT_WTW_G_PER_MJ
    wtw_a = aef.wtw_co2e_g_per_mj if aef else _DEFAULT_WTW_G_PER_MJ
    co2e_wtw_in_t = (e_base * wtw_b + e_alt * wtw_a) / 1e6
    co2e_wtw_t = co2e_wtw_in_t + e_out * wtw_b / 1e6
    co2e_ttw_t = (_ttw_co2e_g(bef, m_base + m_out) + _ttw_co2e_g(aef, m_alt)) / 1e6

    ets_wtw_proxy = co2e_wtw_in_t * voyage.ets_scope_factor * ets_phase_in * eua_price

    total = (fuel_cost + ets_cost
             + fueleu_penalty * fueleu_price_fraction
             + cii_penalty + delay_cost)
    return {
        "base_fuel": base_fuel.value,
        "alt_fuel": alt_fuel.value,
        "blend_pct": float(phi),
        "speed_knots": float(speed),
        "fuel_cost": fuel_cost,
        "ets_cost": ets_cost,
        "fueleu_penalty": fueleu_penalty,
        "fueleu_penalty_full": fueleu_penalty,
        "cii_penalty": cii_penalty,
        "delay_cost": delay_cost,
        "total_cost": total,
        "carbon_fuel_cost": fuel_cost + ets_cost + delay_cost,
        "carbon_fuel_cost_wtw": fuel_cost + ets_wtw_proxy + delay_cost,
        "attained_cii": attained,
        "co2e_wtw_t": co2e_wtw_t,
        "co2e_ttw_t": co2e_ttw_t,
        "fueleu_balance": fueleu_balance,
        "fueleu_actual_intensity": fueleu_actual_int,
        "fueleu_energy_mj": fueleu_energy,
        "is_biofuel_blend": _is_biofuel(aef) and phi > 0,
    }


_DEFAULT_WTW_G_PER_MJ: float = 91.16

_POOL_ALLOCATIONS: frozenset[str] = frozenset({"fueleu_priority", "routed_voyage"})


def scope_split(scope_weight: float) -> tuple[float, float, float]:
    s = min(max(float(scope_weight), 0.0), 1.0)
    a = max(0.0, 2.0 * s - 1.0)
    b = 1.0 - a
    f = (s - a) / b if b > 0 else 0.0
    return a, b, f


def scope_energy_allocation(phi: float, energy: float, scope_weight: float,
                            scope_allocation: str, alt_first: bool = True) -> dict[str, float]:
    a, b, f = scope_split(scope_weight)
    s = a + f * b
    e_alt = phi * energy
    e_base = energy - e_alt
    if scope_allocation == "routed_voyage":
        if alt_first:
            alt_full = min(e_alt, a * energy)
            base_full = a * energy - alt_full
        else:
            base_full = min(e_base, a * energy)
            alt_full = a * energy - base_full
        alt_part = e_alt - alt_full
        base_part = e_base - base_full
        ets_alt_e = alt_full + f * alt_part
        ets_base_e = base_full + f * base_part
    elif scope_allocation == "fueleu_priority":
        ets_alt_e = s * e_alt
        ets_base_e = s * e_base
    else:
        raise ValueError(f"not a pooled allocation: {scope_allocation!r}")
    pool = s * energy
    fe_alt_e = min(e_alt, pool)
    return {"alt_energy": e_alt, "base_energy": e_base,
            "ets_alt_energy": ets_alt_e, "ets_base_energy": ets_base_e,
            "fueleu_alt_energy": fe_alt_e, "fueleu_base_energy": pool - fe_alt_e}


def _leg_capex(fixed_annual_cost: dict | None, base_fuel: FuelType, alt_fuel: FuelType,
               phi: float, annual_fuel_scale: float) -> float:
    if not fixed_annual_cost:
        return 0.0
    annual = fixed_annual_cost.get(base_fuel, 0.0)
    if phi > 0 and alt_fuel != base_fuel:
        annual += fixed_annual_cost.get(alt_fuel, 0.0)
    return annual / max(annual_fuel_scale, 1.0)


def _evaluate_pooled_allocation(
    vessel, voyage, base_fuel, alt_fuel, phi, speed, base_price, alt_price,
    eua_price, fuel_factors, cii_engine, fueleu_engine, ets_phase_in, year,
    annual_dist, annual_fuel_scale, cii_pressure, charter_per_day,
    delay_cost_per_hour, scheduled_hours, fc, energy, bef, aef, lcv_b, lcv_a,
    fueleu_price_fraction, fueleu_scope_factor, fueleu_emission_factors,
    scope_allocation, fixed_annual_cost,
) -> dict[str, Any]:
    def _tonnes(e, lcv):
        return e / (1000.0 * lcv) if lcv > 0 else 0.0

    alt_first = True
    if scope_allocation == "routed_voyage":
        alt_first = (_ets_covered_tonnes(alt_fuel, aef, _tonnes(1.0, lcv_a), year)
                     <= _ets_covered_tonnes(base_fuel, bef, _tonnes(1.0, lcv_b), year))
    alloc = scope_energy_allocation(phi, energy, fueleu_scope_factor, scope_allocation,
                                    alt_first)
    e_alt, e_base = alloc["alt_energy"], alloc["base_energy"]
    ets_alt_e, ets_base_e = alloc["ets_alt_energy"], alloc["ets_base_energy"]
    fe_alt_e, fe_base_e = alloc["fueleu_alt_energy"], alloc["fueleu_base_energy"]

    m_base, m_alt = _tonnes(e_base, lcv_b), _tonnes(e_alt, lcv_a)
    fuel_cost = m_base * base_price + m_alt * alt_price
    capex_cost = _leg_capex(fixed_annual_cost, base_fuel, alt_fuel, phi, annual_fuel_scale)
    fuel_cost += capex_cost

    ets_co2e = (_ets_covered_tonnes(base_fuel, bef, _tonnes(ets_base_e, lcv_b), year)
                + _ets_covered_tonnes(alt_fuel, aef, _tonnes(ets_alt_e, lcv_a), year))
    ets_cost = ets_co2e * voyage.ets_scope_factor * ets_phase_in * eua_price

    cii_co2_leg = _cii_co2_tonnes(bef, m_base) + _cii_co2_tonnes(aef, m_alt)
    cii_annual, attained = _cii_annual_penalty(
        cii_engine, vessel, cii_co2_leg * annual_fuel_scale, annual_dist, year,
        cii_pressure, charter_per_day)
    cii_penalty = cii_annual / max(annual_fuel_scale, 1.0)

    fueleu_penalty, fueleu_balance, fueleu_actual_int, fueleu_energy = _fueleu_blend_eval(
        fueleu_engine, vessel, [(base_fuel, fe_base_e), (alt_fuel, fe_alt_e)],
        year, emission_factors=fueleu_emission_factors)

    delay_cost = 0.0
    if scheduled_hours is not None and delay_cost_per_hour > 0:
        delay_cost = max(fc["sea_hours"] - scheduled_hours, 0.0) * delay_cost_per_hour

    wtw_b = bef.wtw_co2e_g_per_mj if bef else _DEFAULT_WTW_G_PER_MJ
    wtw_a = aef.wtw_co2e_g_per_mj if aef else _DEFAULT_WTW_G_PER_MJ
    co2e_wtw_t = (e_base * wtw_b + e_alt * wtw_a) / 1e6
    co2e_ttw_t = (_ttw_co2e_g(bef, m_base) + _ttw_co2e_g(aef, m_alt)) / 1e6
    ets_wtw_proxy = ((ets_base_e * wtw_b + ets_alt_e * wtw_a) / 1e6
                     * voyage.ets_scope_factor * ets_phase_in * eua_price)

    total = (fuel_cost + ets_cost + fueleu_penalty * fueleu_price_fraction
             + cii_penalty + delay_cost)
    return {
        "base_fuel": base_fuel.value,
        "alt_fuel": alt_fuel.value,
        "blend_pct": float(phi),
        "speed_knots": float(speed),
        "fuel_cost": fuel_cost,
        "ets_cost": ets_cost,
        "fueleu_penalty": fueleu_penalty,
        "fueleu_penalty_full": fueleu_penalty,
        "cii_penalty": cii_penalty,
        "delay_cost": delay_cost,
        "total_cost": total,
        "carbon_fuel_cost": fuel_cost + ets_cost + delay_cost,
        "carbon_fuel_cost_wtw": fuel_cost + ets_wtw_proxy + delay_cost,
        "attained_cii": attained,
        "co2e_wtw_t": co2e_wtw_t,
        "co2e_ttw_t": co2e_ttw_t,
        "fueleu_balance": fueleu_balance,
        "fueleu_actual_intensity": fueleu_actual_int,
        "fueleu_energy_mj": fueleu_energy,
        "is_biofuel_blend": _is_biofuel(aef) and phi > 0,
    }


def _exact_blend_context(fueleu_engine: Any, table: dict[str, float] | None,
                         year: int) -> dict[str, Any] | None:
    if fueleu_engine is None:
        return None
    from carbon_twin.regulations.fueleu import FuelEUEngine

    params = fueleu_engine._params
    until = params.get("rfnbo_multiplier_until_year")
    active = until is None or year <= int(until)
    return {
        "target": float(fueleu_engine._target_intensity(year)),
        "wtw": table if table is not None else dict(FuelEUEngine._DEFAULT_FACTORS),
        "rfnbo_mult": float(params.get("rfnbo_multiplier", 2.0)) if active else 1.0,
    }


def _pair_exact_points(ctx: dict[str, Any], base: FuelType, alt: FuelType,
                       scope_allocation: str, scope_weight: float) -> list[float]:
    from carbon_twin.regulations.fueleu import _RFNBO_FUEL_TYPES

    wb, wa = ctx["wtw"].get(base.value), ctx["wtw"].get(alt.value)
    if wb is None or wa is None:
        return []
    mult = ctx["rfnbo_mult"] if alt in _RFNBO_FUEL_TYPES else 1.0
    return exact_blend_points(wb, wa, ctx["target"], scope_allocation, scope_weight, mult)


def exact_blend_points(
    base_wtw: float, alt_wtw: float, target: float, scope_allocation: str,
    scope_weight: float, rfnbo_multiplier: float = 1.0,
) -> list[float]:
    pts: list[float] = []
    m1 = rfnbo_multiplier - 1.0
    denom = base_wtw - alt_wtw + target * m1
    pooled = scope_allocation in _POOL_ALLOCATIONS
    if denom > 0 and base_wtw > target:
        share = (base_wtw - target) / denom
        if pooled:
            share *= scope_weight
        pts.append(share)
    if pooled:
        pts.append(scope_weight)
    if scope_allocation == "routed_voyage":
        a = scope_split(scope_weight)[0]
        pts += [a, 1.0 - a]
    return sorted({round(p, 12) for p in pts if 1e-9 < p < 1.0 - 1e-9})


def _ttw_co2e_g(ef: FuelEmissionFactors | None, mass_tonnes: float) -> float:
    if ef is None:
        return mass_tonnes * 1e6 * _DEFAULT_TTW_CO2_G_PER_G
    mass_g = mass_tonnes * 1e6
    return mass_g * (ef.ttw_co2_g_per_g_fuel
                     + ef.ttw_ch4_g_per_g_fuel * GWP100["CH4"]
                     + ef.ttw_n2o_g_per_g_fuel * GWP100["N2O"])


def validate_planner_fueleu_internalization(value: Any) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"planner_fueleu_internalization must be finite, got {value!r}")
    if not 0.0 <= v <= 1.0:
        raise ValueError(
            f"planner_fueleu_internalization must be in [0, 1], got {value!r}")
    return v


def _fuel_pathway_label(base_fuel_label: str, alt_fuel_label: str,
                        blend_pct: float) -> str:
    if blend_pct <= 1e-9:
        return f"{base_fuel_label}_only"
    if blend_pct >= 1.0 - 1e-9:
        return f"{alt_fuel_label}_only"
    return f"{base_fuel_label}+{alt_fuel_label}"


def _energy_dominant_fuel(base_fuel_label: str, alt_fuel_label: str,
                          blend_pct: float) -> str:
    return alt_fuel_label if blend_pct >= 0.5 else base_fuel_label


def partial_planner_decision(
    profile: pd.DataFrame,
    lambda_fueleu: float,
    include_cii_in_objective: bool,
    base_fuel: FuelType = FuelType.VLSFO,
) -> dict[str, Any]:
    lam = validate_planner_fueleu_internalization(lambda_fueleu)
    if profile is None or len(profile) == 0:
        raise ValueError("partial_planner_decision requires a non-empty profile")

    def _full_cost(row: pd.Series) -> float:
        c = float(row["fuel_cost"] + row["ets_cost"] + row["delay_cost"]
                  + row["fueleu_penalty_full"])
        if include_cii_in_objective:
            c += float(row["cii_penalty"])
        return c

    cii_w = profile["cii_penalty"] if include_cii_in_objective else 0.0
    benchmark_score = (profile["fuel_cost"] + profile["ets_cost"]
                       + profile["delay_cost"] + profile["fueleu_penalty_full"] + cii_w)
    partial_score = (profile["fuel_cost"] + profile["ets_cost"]
                     + profile["delay_cost"] + lam * profile["fueleu_penalty_full"] + cii_w)

    bench = profile.loc[benchmark_score.idxmin()]
    part = profile.loc[partial_score.idxmin()]

    full_bench = _full_cost(bench)
    full_part = _full_cost(part)
    delta = full_part - full_bench
    rci_proxy = delta / full_part if full_part > 0 else 0.0
    rci_joint_norm = delta / full_bench if full_bench > 0 else 0.0

    part_alt = str(part["alt_fuel"])
    bench_alt = str(bench["alt_fuel"])
    part_blend = float(part["blend_pct"])
    bench_blend = float(bench["blend_pct"])
    has_base_col = "base_fuel" in profile.columns
    part_base = str(part["base_fuel"]) if has_base_col else base_fuel.value
    bench_base = str(bench["base_fuel"]) if has_base_col else base_fuel.value
    path_part = _fuel_pathway_label(part_base, part_alt, part_blend)
    path_bench = _fuel_pathway_label(bench_base, bench_alt, bench_blend)

    if has_base_col and profile["base_fuel"].nunique() > 1:
        fuel_match = path_part == path_bench
    else:
        fuel_match = part_alt == bench_alt
    speed_diff = abs(float(part["speed_knots"]) - float(bench["speed_knots"]))
    blend_diff = abs(part_blend - bench_blend)
    full_match = fuel_match and speed_diff <= 1e-9 and blend_diff <= 1e-9
    fuel_plan_1pp = fuel_match and blend_diff <= 0.01

    return {
        "partial_alt_fuel": part_alt,
        "partial_base_fuel": part_base,
        "partial_blend_pct": part_blend,
        "partial_speed": float(part["speed_knots"]),
        "partial_low_carbon_share": part_blend,
        "partial_biofuel_share":
            part_blend if bool(part["is_biofuel_blend"]) else 0.0,
        "partial_fuel_cost_full_eur": float(part["fuel_cost"]),
        "partial_ets_cost_full_eur": float(part["ets_cost"]),
        "partial_fueleu_cost_full_eur": float(part["fueleu_penalty_full"]),
        "partial_delay_cost_full_eur": float(part["delay_cost"]),
        "partial_total_cost_full_eur": full_part,
        "partial_fueleu_balance": float(part["fueleu_balance"]),
        "partial_fueleu_actual_intensity": float(part["fueleu_actual_intensity"]),
        "partial_fueleu_energy_mj": float(part["fueleu_energy_mj"]),
        "benchmark_alt_fuel": bench_alt,
        "benchmark_base_fuel": bench_base,
        "benchmark_blend_pct": bench_blend,
        "benchmark_speed": float(bench["speed_knots"]),
        "benchmark_low_carbon_share": bench_blend,
        "benchmark_is_biofuel": bool(bench["is_biofuel_blend"]),
        "benchmark_fuel_cost_full_eur": float(bench["fuel_cost"]),
        "benchmark_ets_cost_full_eur": float(bench["ets_cost"]),
        "benchmark_fueleu_cost_full_eur": float(bench["fueleu_penalty_full"]),
        "benchmark_delay_cost_full_eur": float(bench["delay_cost"]),
        "benchmark_total_cost_full_eur": full_bench,
        "benchmark_fueleu_balance": float(bench["fueleu_balance"]),
        "benchmark_fueleu_actual_intensity": float(bench["fueleu_actual_intensity"]),
        "delta_cost_eur": delta,
        "rci_proxy": max(rci_proxy, 0.0),
        "rci_joint_norm": max(rci_joint_norm, 0.0),
        "fuel_pathway_label_match": fuel_match,
        "blend_share_match_1pp": blend_diff <= 0.01,
        "fuel_plan_match_1pp": fuel_plan_1pp,
        "speed_match_0p01kn": speed_diff <= 0.01,
        "full_decision_match": full_match,
        "speed_abs_diff_kn": speed_diff,
        "blend_share_abs_diff": blend_diff,
        "numerical_regret_nonnegative": delta >= -1e-6,
        "fuel_pathway_partial": path_part,
        "fuel_pathway_benchmark": path_bench,
        "alternative_component_share_partial": part_blend,
        "alternative_component_share_benchmark": bench_blend,
        "energy_dominant_fuel_partial":
            _energy_dominant_fuel(part_base, part_alt, part_blend),
        "energy_dominant_fuel_benchmark":
            _energy_dominant_fuel(bench_base, bench_alt, bench_blend),
        "full_decision_exact": full_match,
        "full_decision_1pp_0p01kn": fuel_plan_1pp and speed_diff <= 0.01,
    }


def optimal_fuel_blend_speed_multi_regulation(
    vessel: Vessel,
    voyage: Voyage,
    fuel_prices: dict[str, float],
    eua_price: float,
    fuel_factors: dict[FuelType, FuelEmissionFactors],
    cii_engine: Any,
    fueleu_engine: Any,
    ets_engine: Any,
    year: int,
    fuel_options: list[FuelType] | None = None,
    base_fuel: FuelType = FuelType.VLSFO,
    blend_steps: int = 11,
    speed_range: tuple[float, float] = (8, 22),
    n_points: int = 15,
    annual_distance_nm: float | None = None,
    delay_cost_per_hour: float = 0.0,
    scheduled_hours: float | None = None,
    cii_pressure: float = _DEFAULT_CII_PRESSURE,
    extended_fuels: bool = False,
    silo_carbon_basis: str = "ttw_ets",
    include_cii_in_objective: bool = True,
    max_biofuel_share: float | None = None,
    energy_scale: float = 1.0,
    fueleu_price_fraction: float = 1.0,
    planner_fueleu_internalization: float = 0.0,
    fueleu_scope_factor: float = 1.0,
    fueleu_emission_factors: dict[str, float] | None = None,
    report_annual: bool = False,
    base_fuels: list[FuelType] | None = None,
    scope_allocation: str = "uniform",
    exact_blend: bool = False,
    fixed_annual_cost: dict[FuelType, float] | None = None,
) -> dict[str, Any]:
    validate_planner_fueleu_internalization(planner_fueleu_internalization)
    menu = fuel_options or fuel_menu(vessel, extended=extended_fuels)
    alts = [f for f in menu if f != base_fuel]
    if not alts:
        alts = [base_fuel]
    base_price = fuel_prices.get(base_fuel.value, fuel_prices.get(base_fuel))
    if base_price is None:
        return {}
    plans: list[tuple[FuelType, float, list[FuelType], bool]] = [
        (base_fuel, float(base_price), alts, False)]
    for extra in base_fuels or []:
        if extra == base_fuel or extra not in menu:
            continue
        extra_price = fuel_prices.get(extra.value, fuel_prices.get(extra))
        if extra_price is None:
            continue
        extra_alts = [f for f in menu if f != extra and _is_low_carbon(fuel_factors.get(f))]
        if extra_alts:
            plans.append((extra, float(extra_price), extra_alts, True))

    speeds = np.linspace(speed_range[0], speed_range[1], n_points)
    phis = np.linspace(0.0, 1.0, blend_steps)
    charter_per_day = delay_cost_per_hour * 24.0

    ets_phase_in = 1.0
    if ets_engine is not None and hasattr(ets_engine, "_phase_in_factor"):
        try:
            ets_phase_in = float(ets_engine._phase_in_factor(year))
        except Exception:
            ets_phase_in = 1.0

    annual_dist = annual_distance_nm or voyage.distance_nm * 12
    annual_fuel_scale = annual_dist / max(voyage.distance_nm, 1.0)

    exact_ctx = _exact_blend_context(fueleu_engine, fueleu_emission_factors, year) \
        if exact_blend else None

    rows: list[dict[str, Any]] = []
    for plan_base, plan_base_price, plan_alts, interior_only in plans:
        for alt in plan_alts:
            alt_price = fuel_prices.get(alt.value, fuel_prices.get(alt))
            if alt_price is None:
                continue
            alt_is_biofuel = _is_biofuel(fuel_factors.get(alt))
            pair_phis = list(phis)
            if exact_ctx is not None:
                grid = {round(float(x), 12) for x in phis}
                extra_pts = _pair_exact_points(
                    exact_ctx, plan_base, alt, scope_allocation, fueleu_scope_factor)
                if max_biofuel_share is not None and alt_is_biofuel \
                        and 0.0 < max_biofuel_share < 1.0:
                    extra_pts = [*extra_pts, float(max_biofuel_share)]
                pair_phis += [x for x in sorted(set(extra_pts)) if round(x, 12) not in grid]
            for phi in pair_phis:
                if interior_only and not 0.0 < float(phi) < 1.0:
                    continue
                if (max_biofuel_share is not None and alt_is_biofuel
                        and float(phi) > max_biofuel_share + 1e-9):
                    continue
                for spd in speeds:
                    row = _evaluate_blend_point(
                        vessel, voyage, plan_base, alt, float(phi), float(spd),
                        plan_base_price, float(alt_price), eua_price, fuel_factors,
                        cii_engine, fueleu_engine, ets_phase_in, year, annual_dist,
                        annual_fuel_scale, cii_pressure, charter_per_day,
                        delay_cost_per_hour, scheduled_hours, energy_scale=energy_scale,
                        fueleu_price_fraction=fueleu_price_fraction,
                        fueleu_scope_factor=fueleu_scope_factor,
                        fueleu_emission_factors=fueleu_emission_factors,
                        energy_reference_fuel=base_fuel,
                        scope_allocation=scope_allocation,
                        fixed_annual_cost=fixed_annual_cost)
                    rows.append(row)

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    activity_scale = annual_fuel_scale if report_annual else 1.0
    if activity_scale != 1.0:
        df[_EXTENSIVE_PROFILE_COLUMNS] = df[_EXTENSIVE_PROFILE_COLUMNS] * activity_scale
    objective = (df["total_cost"] - df["cii_penalty"]
                 if not include_cii_in_objective else df["total_cost"])
    silo_col = "carbon_fuel_cost_wtw" if silo_carbon_basis == "wtw_proxy" else "carbon_fuel_cost"
    joint = df.loc[objective.idxmin()]
    siloed = df.loc[df[silo_col].idxmin()]
    cost_joint = float(objective.loc[objective.idxmin()])
    cost_siloed = float(siloed["total_cost"] - (
        siloed["cii_penalty"] if not include_cii_in_objective else 0.0))
    rci = (cost_siloed - cost_joint) / cost_siloed if cost_siloed > 0 else 0.0

    delta_c = cost_siloed - cost_joint
    rci_joint = delta_c / cost_joint if cost_joint > 0 else 0.0

    ppd = partial_planner_decision(
        df, planner_fueleu_internalization, include_cii_in_objective, base_fuel)

    return {
        "joint_base_fuel": str(joint["base_fuel"]),
        "siloed_base_fuel": str(siloed["base_fuel"]),
        "annual_scale": float(annual_fuel_scale),
        "cost_basis": "annual" if report_annual else "leg",
        "joint_fuel_pathway": _fuel_pathway_label(
            str(joint["base_fuel"]), str(joint["alt_fuel"]), float(joint["blend_pct"])),
        "siloed_fuel_pathway": _fuel_pathway_label(
            str(siloed["base_fuel"]), str(siloed["alt_fuel"]), float(siloed["blend_pct"])),
        "joint_alt_fuel": str(joint["alt_fuel"]),
        "joint_blend_pct": float(joint["blend_pct"]),
        "joint_speed": float(joint["speed_knots"]),
        "siloed_alt_fuel": str(siloed["alt_fuel"]),
        "siloed_blend_pct": float(siloed["blend_pct"]),
        "siloed_speed": float(siloed["speed_knots"]),
        "cost_joint_eur": cost_joint,
        "cost_siloed_eur": cost_siloed,
        "delta_cost_eur": delta_c,
        "rci": max(rci, 0.0),
        "rci_joint": max(rci_joint, 0.0),
        "blends": float(joint["blend_pct"]) > 0.01,
        "fuel_cost_joint_eur": float(joint["fuel_cost"]),
        "ets_cost_joint_eur": float(joint["ets_cost"]),
        "fueleu_penalty_joint_eur": float(joint["fueleu_penalty"]),
        "cii_consequence_joint_eur": float(joint["cii_penalty"]),
        "delay_cost_joint_eur": float(joint["delay_cost"]),
        "fuel_cost_siloed_eur": float(siloed["fuel_cost"]),
        "ets_cost_siloed_eur": float(siloed["ets_cost"]),
        "fueleu_penalty_siloed_eur": float(siloed["fueleu_penalty"]),
        "cii_consequence_siloed_eur": float(siloed["cii_penalty"]),
        "delay_cost_siloed_eur": float(siloed["delay_cost"]),
        "ets_cost_eur": float(joint["ets_cost"]),
        "fueleu_penalty_eur": float(joint["fueleu_penalty"]),
        "cii_penalty_eur": float(joint["cii_penalty"]),
        "fueleu_balance_joint": float(joint["fueleu_balance"]),
        "fueleu_actual_intensity_joint": float(joint["fueleu_actual_intensity"]),
        "fueleu_energy_mj_joint": float(joint["fueleu_energy_mj"]),
        "fueleu_balance_siloed": float(siloed["fueleu_balance"]),
        "fueleu_actual_intensity_siloed": float(siloed["fueleu_actual_intensity"]),
        "fueleu_energy_mj_siloed": float(siloed["fueleu_energy_mj"]),
        "low_carbon_share_joint": float(joint["blend_pct"]),
        "low_carbon_share_siloed": float(siloed["blend_pct"]),
        "biofuel_share_joint":
            float(joint["blend_pct"]) if bool(joint["is_biofuel_blend"]) else 0.0,
        "biofuel_share_siloed":
            float(siloed["blend_pct"]) if bool(siloed["is_biofuel_blend"]) else 0.0,
        "partial_alt_fuel": ppd["partial_alt_fuel"],
        "partial_base_fuel": ppd["partial_base_fuel"],
        "partial_blend_pct": ppd["partial_blend_pct"],
        "partial_speed": ppd["partial_speed"],
        "partial_low_carbon_share": ppd["partial_low_carbon_share"],
        "partial_biofuel_share": ppd["partial_biofuel_share"],
        "partial_fuel_cost_full_eur": ppd["partial_fuel_cost_full_eur"],
        "partial_ets_cost_full_eur": ppd["partial_ets_cost_full_eur"],
        "partial_fueleu_cost_full_eur": ppd["partial_fueleu_cost_full_eur"],
        "partial_delay_cost_full_eur": ppd["partial_delay_cost_full_eur"],
        "partial_total_cost_full_eur": ppd["partial_total_cost_full_eur"],
        "partial_fueleu_balance": ppd["partial_fueleu_balance"],
        "partial_fueleu_actual_intensity": ppd["partial_fueleu_actual_intensity"],
        "partial_fueleu_energy_mj": ppd["partial_fueleu_energy_mj"],
        "delta_cost_partial_eur": ppd["delta_cost_eur"],
        "rci_partial": ppd["rci_proxy"],
        "rci_partial_joint_norm": ppd["rci_joint_norm"],
        "fuel_pathway_label_match": ppd["fuel_pathway_label_match"],
        "blend_share_match_1pp": ppd["blend_share_match_1pp"],
        "fuel_plan_match_1pp": ppd["fuel_plan_match_1pp"],
        "speed_match_0p01kn": ppd["speed_match_0p01kn"],
        "full_decision_match": ppd["full_decision_match"],
        "speed_abs_diff_kn": ppd["speed_abs_diff_kn"],
        "blend_share_abs_diff": ppd["blend_share_abs_diff"],
        "numerical_regret_nonnegative": ppd["numerical_regret_nonnegative"],
        "benchmark_base_fuel": ppd["benchmark_base_fuel"],
        "benchmark_low_carbon_share": ppd["benchmark_low_carbon_share"],
        "fuel_pathway_partial": ppd["fuel_pathway_partial"],
        "fuel_pathway_benchmark": ppd["fuel_pathway_benchmark"],
        "alternative_component_share_partial": ppd["alternative_component_share_partial"],
        "alternative_component_share_benchmark": ppd["alternative_component_share_benchmark"],
        "energy_dominant_fuel_partial": ppd["energy_dominant_fuel_partial"],
        "energy_dominant_fuel_benchmark": ppd["energy_dominant_fuel_benchmark"],
        "full_decision_exact": ppd["full_decision_exact"],
        "full_decision_1pp_0p01kn": ppd["full_decision_1pp_0p01kn"],
        "profile": df,
    }
