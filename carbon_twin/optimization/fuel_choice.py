"""Fuel selection optimization."""

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
    """Find cost-minimizing fuel blend satisfying regulatory constraints.

    Discretizes blend fractions at 1/blend_resolution increments and evaluates
    all feasible combinations, choosing the minimum-cost blend.
    """
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
    """EUA price at which switching from_fuel to to_fuel becomes cost-effective.

    Finds the breakeven EUA price where total_cost(from) == total_cost(to),
    accounting for fuel price differential and emission reduction.
    Returns the threshold EUA price (EUR/tCO2e). Returns inf if switching
    never becomes cheaper (e.g., to_fuel emits more).
    """
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
    """MAC curve: for each fuel switch from baseline, cost per tonne CO2e reduced.

    Returns a DataFrame sorted by MAC ascending (cheapest abatement first).
    """
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


# ===========================================================================
# Regulation-aware fuel × speed optimization (fuel-dimension RCI)
# ---------------------------------------------------------------------------
# The speed-only optimizer shows near-zero coupling because slow steaming is
# co-monotonic across CII / ETS / FuelEU. Genuine coupling lives in the *fuel*
# decision, where the regulations price different things and conflict:
#   EU ETS  prices Tank-to-Wake CO2 (quantity); biogenic carbon is zero-rated.
#   FuelEU  prices Well-to-Wake GHG *intensity*; rewards low-WtW & RFNBO fuels.
#   CII     prices Tank-to-Wake CO2 / transport work; biofuel carbon still counts.
# So a drop-in biofuel cuts the FuelEU penalty and (being biogenic) the ETS bill
# but does nothing for CII, while LNG cuts CII and ETS yet barely helps FuelEU.
# Optimising one regulation's fuel choice in isolation diverges from the joint
# optimum; the gap is the fuel-dimension RCI.
# ===========================================================================

from carbon_twin.optimization.speed import (  # noqa: E402  (after utility defs)
    _DEFAULT_CII_PRESSURE,
    _cii_annual_penalty,
    _fuel_at_speed,
    _fueleu_penalty,
)

# Drop-in fuels usable by any vessel without a new engine (near-term FuelEU
# compliance levers). VLSFO is the conventional baseline.
_DROPIN_FUELS: list[FuelType] = [
    FuelType.VLSFO,
    FuelType.MGO,
    FuelType.BIODIESEL_FAME,
]
# Ship types with gas-capable (dual-fuel) machinery → LNG is a real option.
_GAS_CAPABLE_TYPES: set[str] = {"gas_carrier", "lng_carrier"}
# Forward-looking newbuild/retrofit options (behind the ``extended`` flag).
_EXTENDED_FUELS: list[FuelType] = [
    FuelType.LNG,
    FuelType.METHANOL_BIO,
    FuelType.METHANOL_E,
]
# Categories whose CO2 is biogenic / renewable → zero-rated under EU ETS.
_BIOGENIC_CATEGORIES: set[str] = {"biofuel", "rfnbo"}
# Bio-derived (crop/waste) fuels, distinct from synthetic e-fuels/RFNBO. A
# "no biofuel" feasibility scenario removes these but keeps RFNBO e-fuels; the
# biofuel share cap targets these too.
_BIOFUEL_CATEGORIES: set[str] = {"biofuel"}


def _is_biofuel(ef: FuelEmissionFactors | None) -> bool:
    """True if the fuel is a bio-derived (crop/waste) fuel, e.g. FAME or
    bio-methanol — not a synthetic e-fuel/RFNBO."""
    return ef is not None and ef.category in _BIOFUEL_CATEGORIES


def fuel_menu(vessel: Vessel, extended: bool = False) -> list[FuelType]:
    """Switchable fuel set for a vessel: drop-in fuels for all, LNG for
    gas-capable types; ``extended`` adds newbuild/retrofit options to every
    vessel for forward-looking scenarios."""
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
    """EU-ETS-covered emissions (tCO2e). Biogenic CO2 (biofuel/RFNBO) is
    zero-rated; CH4 and N2O enter the scope from 2026 (GWP-100 weighted)."""
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
    """Operational-CII CO2 (tonnes): Tank-to-Wake CO2 only on the fuel's actual
    carbon (biofuel carbon is not credited — current MEPC operational CII). This
    is what makes biofuels help FuelEU/ETS but not CII, and lets LNG's lower
    carbon help CII without its methane slip (which CII ignores)."""
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
    """All cost components (EUR, representative leg) at one (fuel, speed)."""
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
        # siloed objective: an operator who prices carbon only through ETS and
        # is blind to FuelEU and CII, but still pays real fuel and delay costs.
        # (Excluding delay here would make the siloed over-slow to the floor and
        # inflate RCI — a speed artifact, not a fuel-choice effect.)
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
    """Joint (fuel × speed) optimisation vs a carbon-only siloed baseline.

    Returns the joint optimum (minimise total cost over all four regulations),
    the siloed optimum (minimise fuel + ETS only — an operator blind to FuelEU
    and CII), and the fuel-dimension RCI = fractional cost the siloed operator
    overpays once the full regulatory burden is applied.
    """
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
    cost_siloed = float(siloed["total_cost"])  # full burden at the siloed optimum
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


# ===========================================================================
# v2: continuous fuel BLENDING × speed (rigorous fuel-dimension RCI)
# ---------------------------------------------------------------------------
# v1 forces a single fuel (all-or-nothing), so RCI is a conservative lower bound
# and the switch is abrupt. Operators in reality blend just enough low-carbon
# fuel (e.g. FAME, bio-methanol) into a fossil base to hit the FuelEU target —
# cheaper than 100% switching, which *lowers* the joint cost and *raises* RCI.
# This evaluates blends on an ENERGY basis (the fuel-independent invariant), which
# also corrects v1's mass-basis understatement of low-LCV fuels (methanol, LNG).
# ===========================================================================


def _fueleu_blend_eval(
    fueleu_engine: Any,
    vessel: Vessel,
    fuel_energy: list[tuple[FuelType, float]],
    year: int,
) -> tuple[float, float, float, float]:
    """FuelEU (penalty_eur, signed compliance_balance, actual_intensity, energy_mj)
    for an energy-weighted blend via the engine (handles energy-weighted WtW
    intensity + RFNBO bonus). The signed balance (deficit < 0 < surplus), actual
    intensity and total energy are all needed for fleet-level pooling."""
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
        r = fueleu_engine.evaluate(vessel, records, total, year)
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
) -> dict[str, Any]:
    """Cost components for a blend of ``phi`` energy-fraction ``alt_fuel`` into
    ``base_fuel`` at ``speed``. Energy demand is fuel-independent (taken from the
    base-fuel physics) and split across components by energy.

    ``energy_scale`` (MRV calibration) multiplies the propulsion energy so the
    vessel's baseline energy/emissions are anchored to its MRV total while the
    speed elasticity (cubic law) is preserved. Delay cost is time-based and is
    NOT scaled."""
    fc = _fuel_at_speed(vessel, voyage, speed, base_fuel, fuel_factors)
    energy = fc["energy_mj"] * energy_scale   # total propulsion energy (MJ), fuel-independent
    bef = fuel_factors.get(base_fuel)
    aef = fuel_factors.get(alt_fuel)
    lcv_b = bef.lcv_mj_per_kg if bef else 41.0
    lcv_a = aef.lcv_mj_per_kg if aef else 41.0

    e_alt = phi * energy
    e_base = (1.0 - phi) * energy
    m_base = e_base / (1000.0 * lcv_b) if lcv_b > 0 else 0.0   # tonnes
    m_alt = e_alt / (1000.0 * lcv_a) if lcv_a > 0 else 0.0

    fuel_cost = m_base * base_price + m_alt * alt_price

    ets_co2e = (_ets_covered_tonnes(base_fuel, bef, m_base, year)
                + _ets_covered_tonnes(alt_fuel, aef, m_alt, year))
    ets_cost = ets_co2e * voyage.ets_scope_factor * ets_phase_in * eua_price

    cii_co2_leg = _cii_co2_tonnes(bef, m_base) + _cii_co2_tonnes(aef, m_alt)
    annual_cii_co2 = cii_co2_leg * annual_fuel_scale
    cii_annual, attained = _cii_annual_penalty(
        cii_engine, vessel, annual_cii_co2, annual_dist, year,
        cii_pressure, charter_per_day)
    cii_penalty = cii_annual / max(annual_fuel_scale, 1.0)

    fueleu_penalty, fueleu_balance, fueleu_actual_int, fueleu_energy = _fueleu_blend_eval(
        fueleu_engine, vessel, [(base_fuel, e_base), (alt_fuel, e_alt)], year)

    delay_cost = 0.0
    if scheduled_hours is not None and delay_cost_per_hour > 0:
        delay_cost = max(fc["sea_hours"] - scheduled_hours, 0.0) * delay_cost_per_hour

    # --- physical emissions for abatement accounting (energy basis) ---
    # WtW CO2e (lifecycle, gCO2e/MJ): the climate-relevant abatement metric and
    # what FuelEU regulates. TtW CO2e (GWP-100) is the combustion-only quantity;
    # unlike the ETS-covered figure it does NOT zero-rate biogenic carbon, so it
    # is the true tank-to-wake mass. Both are per representative leg (tonnes).
    wtw_b = bef.wtw_co2e_g_per_mj if bef else _DEFAULT_WTW_G_PER_MJ
    wtw_a = aef.wtw_co2e_g_per_mj if aef else _DEFAULT_WTW_G_PER_MJ
    co2e_wtw_t = (e_base * wtw_b + e_alt * wtw_a) / 1e6
    co2e_ttw_t = (_ttw_co2e_g(bef, m_base) + _ttw_co2e_g(aef, m_alt)) / 1e6

    # WtW-proxy carbon cost: the same carbon price applied to the fuel's
    # *well-to-wake* CO2e instead of the ETS-covered (tank-to-wake, biogenic-
    # zeroed) figure — i.e. an ETS-aware operator whose fuel ranking already
    # reflects lifecycle GHG. Diagnostic counterfactual only.
    ets_wtw_proxy = co2e_wtw_t * voyage.ets_scope_factor * ets_phase_in * eua_price

    total = (fuel_cost + ets_cost
             + fueleu_penalty * fueleu_price_fraction
             + cii_penalty + delay_cost)
    return {
        "alt_fuel": alt_fuel.value,
        "blend_pct": float(phi),
        "speed_knots": float(speed),
        "fuel_cost": fuel_cost,
        "ets_cost": ets_cost,
        "fueleu_penalty": fueleu_penalty,
        # Unscaled statutory FuelEU term (weight 1.0). ``fueleu_penalty`` above is
        # the credit-price-scaled value used in ``total_cost``; the full term is
        # needed by the FuelEU planning-internalization experiment (λ_fueleu),
        # whose full evaluation must never be scaled by λ or the credit fraction.
        "fueleu_penalty_full": fueleu_penalty,
        "cii_penalty": cii_penalty,
        "delay_cost": delay_cost,
        "total_cost": total,
        "carbon_fuel_cost": fuel_cost + ets_cost + delay_cost,
        "carbon_fuel_cost_wtw": fuel_cost + ets_wtw_proxy + delay_cost,
        "attained_cii": attained,
        "co2e_wtw_t": co2e_wtw_t,
        "co2e_ttw_t": co2e_ttw_t,
        "fueleu_balance": fueleu_balance,   # signed: deficit < 0 < surplus (for pooling)
        "fueleu_actual_intensity": fueleu_actual_int,
        "fueleu_energy_mj": fueleu_energy,
        "is_biofuel_blend": _is_biofuel(aef) and phi > 0,
    }


_DEFAULT_WTW_G_PER_MJ: float = 91.16  # VLSFO/HFO-like WtW fallback (gCO2e/MJ)


def _ttw_co2e_g(ef: FuelEmissionFactors | None, mass_tonnes: float) -> float:
    """Tank-to-Wake CO2e (grams, GWP-100), physical — biogenic carbon NOT
    zero-rated (this is the actual combustion mass, not the ETS-counted one)."""
    if ef is None:
        return mass_tonnes * 1e6 * _DEFAULT_TTW_CO2_G_PER_G
    mass_g = mass_tonnes * 1e6
    return mass_g * (ef.ttw_co2_g_per_g_fuel
                     + ef.ttw_ch4_g_per_g_fuel * GWP100["CH4"]
                     + ef.ttw_n2o_g_per_g_fuel * GWP100["N2O"])


def validate_planner_fueleu_internalization(value: Any) -> float:
    """Range-validate the planner-only FuelEU internalization weight (λ_fueleu).

    ``planner_fueleu_internalization`` ∈ [0, 1] is the weight a *partial planner*
    puts on the FuelEU term at the decision stage only. It is semantically
    separate from ``fueleu_price_fraction`` (the effective-credit-price
    experiment): λ scales the selection objective, never the stored or reported
    full-evaluation FuelEU penalty. Rejects negative, >1, NaN and infinite
    values; accepts ints and floats in range."""
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"planner_fueleu_internalization must be finite, got {value!r}")
    if not 0.0 <= v <= 1.0:
        raise ValueError(
            f"planner_fueleu_internalization must be in [0, 1], got {value!r}")
    return v


def _fuel_pathway_label(base_fuel_label: str, alt_fuel_label: str,
                        blend_pct: float) -> str:
    """Deterministic fuel-pathway string from the selected (base, alt, blend)
    triple: blend_pct≈0 is pure base fuel, ≈1 is pure alternative, otherwise a
    blend. Uses the actual selected base/alt fuels, never the energy-dominant
    label -- a 5% biodiesel blend and a 100% biodiesel switch must not collapse
    to the same label."""
    if blend_pct <= 1e-9:
        return f"{base_fuel_label}_only"
    if blend_pct >= 1.0 - 1e-9:
        return f"{alt_fuel_label}_only"
    return f"{base_fuel_label}+{alt_fuel_label}"


def _energy_dominant_fuel(base_fuel_label: str, alt_fuel_label: str,
                          blend_pct: float) -> str:
    """Descriptive-only energy-dominant fuel label -- the component holding the
    larger energy share. Never used to define fuel pathway."""
    return alt_fuel_label if blend_pct >= 0.5 else base_fuel_label


def partial_planner_decision(
    profile: pd.DataFrame,
    lambda_fueleu: float,
    include_cii_in_objective: bool,
    base_fuel: FuelType = FuelType.VLSFO,
) -> dict[str, Any]:
    """Partial-planner decision and planning regret vs the full FuelEU benchmark.

    Models a planner who only partially internalizes the FuelEU cost signal at
    decision time: it minimises ``base + λ·F`` (base = fuel + ETS + delay; F =
    full statutory FuelEU term, weight 1.0) over the SAME candidate set the joint
    optimizer evaluates, and both the partial decision and the full benchmark
    (λ = 1) are then evaluated under the full FuelEU term. ``profile`` is the
    candidate component table built by
    :func:`optimal_fuel_blend_speed_multi_regulation` (or a frame with the same
    columns), so candidate evaluations are never repeated across λ.

    ``include_cii_in_objective`` mirrors the joint optimizer's switch: the
    primary case keeps the CII-proxy consequence out of both objectives AND out
    of the full evaluation (a clean zero-regret endpoint at λ = 1); an optional
    variant holds it at full weight everywhere instead.

    Determinism: ties on the partial score follow the repository's existing
    rule — ``Series.idxmin()`` returns the first candidate in the enumeration
    order (menu × blend × speed), so the same candidate set and objective values
    produce the same decision across reruns.

    Never multiplies the stored or reported full-evaluation FuelEU penalty by λ
    and is independent of ``fueleu_price_fraction``."""
    lam = validate_planner_fueleu_internalization(lambda_fueleu)
    if profile is None or len(profile) == 0:
        raise ValueError("partial_planner_decision requires a non-empty profile")

    def _full_cost(row: pd.Series) -> float:
        """Full-policy evaluation of one candidate: base + FuelEU (weight 1.0),
        plus the CII consequence only when the case includes it."""
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

    fuel_match = str(part["alt_fuel"]) == str(bench["alt_fuel"])
    speed_diff = abs(float(part["speed_knots"]) - float(bench["speed_knots"]))
    blend_diff = abs(float(part["blend_pct"]) - float(bench["blend_pct"]))
    full_match = fuel_match and speed_diff <= 1e-9 and blend_diff <= 1e-9
    fuel_plan_1pp = fuel_match and blend_diff <= 0.01

    part_alt = str(part["alt_fuel"])
    bench_alt = str(bench["alt_fuel"])
    part_blend = float(part["blend_pct"])
    bench_blend = float(bench["blend_pct"])

    return {
        # partial-planner decision (selected under base + λ·F)
        "partial_alt_fuel": part_alt,
        "partial_base_fuel": base_fuel.value,
        "partial_blend_pct": part_blend,
        "partial_speed": float(part["speed_knots"]),
        "partial_low_carbon_share": part_blend,
        "partial_biofuel_share":
            part_blend if bool(part["is_biofuel_blend"]) else 0.0,
        # partial decision evaluated under the FULL FuelEU term (weight 1.0)
        "partial_fuel_cost_full_eur": float(part["fuel_cost"]),
        "partial_ets_cost_full_eur": float(part["ets_cost"]),
        "partial_fueleu_cost_full_eur": float(part["fueleu_penalty_full"]),
        "partial_delay_cost_full_eur": float(part["delay_cost"]),
        "partial_total_cost_full_eur": full_part,
        "partial_fueleu_balance": float(part["fueleu_balance"]),
        "partial_fueleu_actual_intensity": float(part["fueleu_actual_intensity"]),
        "partial_fueleu_energy_mj": float(part["fueleu_energy_mj"]),
        # full benchmark decision (x_star: FuelEU weight 1.0 in selection and eval)
        "benchmark_alt_fuel": bench_alt,
        "benchmark_base_fuel": base_fuel.value,
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
        # planning regret: full-policy cost of the partial decision minus the
        # full benchmark, normalized two ways (vs. the partial decision's own
        # cost, and vs. the benchmark's cost)
        "delta_cost_eur": delta,
        "rci_proxy": max(rci_proxy, 0.0),
        "rci_joint_norm": max(rci_joint_norm, 0.0),
        # decision-agreement metrics: the label match compares the
        # alternative-fuel LABEL/pathway only (two 100%-share plans with
        # different labels count as different; equal labels at different
        # shares count as equal). ``fuel_plan_match_1pp`` is the COMPOSITE plan
        # agreement: same label AND share within 1 pp — a plan metric, unlike
        # the label or share match alone. Speed adds the full-decision
        # tolerance on top.
        "fuel_pathway_label_match": fuel_match,
        "blend_share_match_1pp": blend_diff <= 0.01,
        "fuel_plan_match_1pp": fuel_plan_1pp,
        "speed_match_0p01kn": speed_diff <= 0.01,
        "full_decision_match": full_match,
        "speed_abs_diff_kn": speed_diff,
        "blend_share_abs_diff": blend_diff,
        "numerical_regret_nonnegative": delta >= -1e-6,
        # Unambiguous fuel-identity fields, purely descriptive/derived from the
        # existing (base_fuel, alt_fuel, blend_pct) selection above -- base_fuel
        # is a single constant per call, so a "fuel pathway" is fully expressed
        # by base+alt+blend_pct without adding a new candidate dimension.
        "fuel_pathway_partial": _fuel_pathway_label(base_fuel.value, part_alt, part_blend),
        "fuel_pathway_benchmark": _fuel_pathway_label(base_fuel.value, bench_alt, bench_blend),
        "alternative_component_share_partial": part_blend,
        "alternative_component_share_benchmark": bench_blend,
        "energy_dominant_fuel_partial":
            _energy_dominant_fuel(base_fuel.value, part_alt, part_blend),
        "energy_dominant_fuel_benchmark":
            _energy_dominant_fuel(base_fuel.value, bench_alt, bench_blend),
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
) -> dict[str, Any]:
    """Joint (blend × speed) optimisation vs a carbon-only siloed baseline.

    Decision space: a continuous energy-fraction ``phi`` of one low-carbon
    ``alt_fuel`` blended into ``base_fuel`` (VLSFO), plus speed. Joint minimises
    total cost over all four regulations; siloed minimises fuel+ETS+delay (carbon
    priced only via ETS, blind to FuelEU/CII). RCI = fractional overpayment of the
    siloed operator. Reports the joint low-carbon blend fraction.

    Revision switches (all default to the headline behaviour):

    * ``silo_carbon_basis`` — ``"ttw_ets"`` (default) ranks the siloed operator's
      fuels by ETS-covered tank-to-wake CO2; ``"wtw_proxy"`` ranks them by
      well-to-wake CO2e at the same carbon price.
    * ``include_cii_in_objective`` — drop the modelled CII consequence from the
      joint objective (clean ETS+FuelEU planning loss only).
    * ``max_biofuel_share`` — cap the energy fraction of bio-derived fuels.
    * ``energy_scale`` — per-vessel MRV-anchoring multiplier.
    * ``planner_fueleu_internalization`` (λ_fueleu) — FuelEU planning-integration
      weight used ONLY to select the partial-planner decision ``argmin(base +
      λ·F)``; the partial decision and the full benchmark are always evaluated at
      the full statutory FuelEU term (weight 1.0, independent of
      ``fueleu_price_fraction``). Default 0.0 makes the partial decision coincide
      with the siloed decision and leaves every pre-existing output unchanged.
    """
    validate_planner_fueleu_internalization(planner_fueleu_internalization)
    menu = fuel_options or fuel_menu(vessel, extended=extended_fuels)
    alts = [f for f in menu if f != base_fuel]
    if not alts:
        alts = [base_fuel]
    base_price = fuel_prices.get(base_fuel.value, fuel_prices.get(base_fuel))
    if base_price is None:
        return {}

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

    rows: list[dict[str, Any]] = []
    for alt in alts:
        alt_price = fuel_prices.get(alt.value, fuel_prices.get(alt))
        if alt_price is None:
            continue
        alt_is_biofuel = _is_biofuel(fuel_factors.get(alt))
        for phi in phis:
            # cap the energy share of bio-derived fuels
            if (max_biofuel_share is not None and alt_is_biofuel
                    and float(phi) > max_biofuel_share + 1e-9):
                continue
            for spd in speeds:
                row = _evaluate_blend_point(
                    vessel, voyage, base_fuel, alt, float(phi), float(spd),
                    float(base_price), float(alt_price), eua_price, fuel_factors,
                    cii_engine, fueleu_engine, ets_phase_in, year, annual_dist,
                    annual_fuel_scale, cii_pressure, charter_per_day,
                    delay_cost_per_hour, scheduled_hours, energy_scale=energy_scale,
                    fueleu_price_fraction=fueleu_price_fraction)
                rows.append(row)

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    # Joint objective: full burden, optionally excluding the modelled CII
    # consequence. Siloed objective: carbon ranked by ETS (TtW) or by a WtW
    # proxy; always evaluated on the full burden afterwards.
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

    # FuelEU planning-internalization block: the partial-planner decision at
    # λ_fueleu over the SAME candidate set. Purely additive — joint/siloed
    # selection, costs and RCI above are untouched, so existing experiments are
    # numerically unchanged when the new parameter is omitted (default λ = 0,
    # which reproduces the siloed decision).
    ppd = partial_planner_decision(
        df, planner_fueleu_internalization, include_cii_in_objective, base_fuel)

    return {
        "joint_alt_fuel": str(joint["alt_fuel"]),
        "joint_blend_pct": float(joint["blend_pct"]),
        "joint_speed": float(joint["speed_knots"]),
        "siloed_alt_fuel": str(siloed["alt_fuel"]),
        "siloed_blend_pct": float(siloed["blend_pct"]),
        "siloed_speed": float(siloed["speed_knots"]),
        "cost_joint_eur": cost_joint,
        "cost_siloed_eur": cost_siloed,
        "delta_cost_eur": delta_c,
        "rci": max(rci, 0.0),          # = delta_C / C_joint(x_silo)
        "rci_joint": max(rci_joint, 0.0),  # = delta_C / C_joint(x_joint)
        "blends": float(joint["blend_pct"]) > 0.01,
        # joint-point cost components
        "fuel_cost_joint_eur": float(joint["fuel_cost"]),
        "ets_cost_joint_eur": float(joint["ets_cost"]),
        "fueleu_penalty_joint_eur": float(joint["fueleu_penalty"]),
        "cii_consequence_joint_eur": float(joint["cii_penalty"]),
        "delay_cost_joint_eur": float(joint["delay_cost"]),
        # siloed-point cost components (evaluated under all instruments)
        "fuel_cost_siloed_eur": float(siloed["fuel_cost"]),
        "ets_cost_siloed_eur": float(siloed["ets_cost"]),
        "fueleu_penalty_siloed_eur": float(siloed["fueleu_penalty"]),
        "cii_consequence_siloed_eur": float(siloed["cii_penalty"]),
        "delay_cost_siloed_eur": float(siloed["delay_cost"]),
        # backward-compatible aliases (joint-point)
        "ets_cost_eur": float(joint["ets_cost"]),
        "fueleu_penalty_eur": float(joint["fueleu_penalty"]),
        "cii_penalty_eur": float(joint["cii_penalty"]),
        # FuelEU pooling inputs: signed balance, actual WtW intensity and energy
        # at each decision point, so a fleet/type-level pool penalty can be
        # recomputed in post-processing without re-optimising.
        "fueleu_balance_joint": float(joint["fueleu_balance"]),
        "fueleu_actual_intensity_joint": float(joint["fueleu_actual_intensity"]),
        "fueleu_energy_mj_joint": float(joint["fueleu_energy_mj"]),
        "fueleu_balance_siloed": float(siloed["fueleu_balance"]),
        "fueleu_actual_intensity_siloed": float(siloed["fueleu_actual_intensity"]),
        "fueleu_energy_mj_siloed": float(siloed["fueleu_energy_mj"]),
        # low-carbon / biofuel energy shares at each decision point
        "low_carbon_share_joint": float(joint["blend_pct"]),
        "low_carbon_share_siloed": float(siloed["blend_pct"]),
        "biofuel_share_joint":
            float(joint["blend_pct"]) if bool(joint["is_biofuel_blend"]) else 0.0,
        "biofuel_share_siloed":
            float(siloed["blend_pct"]) if bool(siloed["is_biofuel_blend"]) else 0.0,
        # FuelEU planning-internalization block (additive; see above)
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
        # unambiguous fuel-identity fields (see partial_planner_decision above)
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
