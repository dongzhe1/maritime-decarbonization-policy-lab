"""Fuel selection optimization."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from carbon_twin.data.schemas import FuelRecord, FuelType, Vessel, Voyage
from carbon_twin.emissions.factors import (
    _DEFAULT_TTW_CO2_G_PER_G,
    GWP100,
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
        ghg_i = (
            fuel_factors[fo.fuel_type].wtw_co2e_g_per_mj if fo.fuel_type in fuel_factors else 0.0
        )
        return {
            "fuel_mix": {fo.fuel_type: fraction},
            "total_cost": cost,
            "ghg_intensity": ghg_i,
            "co2e_ttw": co2e_ttw,
            "co2e_wtw": co2e_wtw,
        }

    np.arange(0, blend_resolution + 1) / blend_resolution
    len(fuel_options)

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

        rows.append(
            {
                "fuel_type": fo.fuel_type.value,
                "price_per_tonne": fo.price_per_tonne,
                "fuel_mass_tonnes": mass,
                "fuel_cost": cost,
                "co2e_tonnes": co2e,
                "delta_cost": delta_cost,
                "delta_co2e_tonnes": delta_co2e,
                "mac_eur_per_tco2e": mac,
                "abatement_pct": (delta_co2e / co2e_baseline * 100) if co2e_baseline > 0 else 0.0,
            }
        )

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
        cii_engine,
        vessel,
        annual_cii_co2,
        annual_dist,
        year,
        cii_pressure,
        charter_per_day,
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
            rows.append(
                _evaluate_point(
                    vessel,
                    voyage,
                    fuel,
                    float(spd),
                    float(price),
                    eua_price,
                    fuel_factors,
                    cii_engine,
                    fueleu_engine,
                    ets_phase_in,
                    year,
                    annual_dist,
                    annual_fuel_scale,
                    cii_pressure,
                    charter_per_day,
                    delay_cost_per_hour,
                    scheduled_hours,
                )
            )

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
) -> tuple[float, float]:
    """FuelEU (penalty_eur, signed compliance_balance) for an energy-weighted blend
    via the engine (handles energy-weighted WtW intensity + RFNBO bonus). The
    signed balance (deficit < 0 < surplus) is needed for fleet pooling."""
    if fueleu_engine is None:
        return 0.0, 0.0
    records: list[FuelRecord] = []
    total = 0.0
    for ft, e in fuel_energy:
        if e <= 0:
            continue
        records.append(
            FuelRecord(vessel_imo=vessel.imo_number, fuel_type=ft, mass_tonnes=0.0, energy_mj=e)
        )
        total += e
    if total <= 0:
        return 0.0, 0.0
    try:
        r = fueleu_engine.evaluate(vessel, records, total, year)
        return r.penalty_eur, r.compliance_balance
    except Exception:
        return 0.0, 0.0


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
) -> dict[str, Any]:
    """Cost components for a blend of ``phi`` energy-fraction ``alt_fuel`` into
    ``base_fuel`` at ``speed``. Energy demand is fuel-independent (taken from the
    base-fuel physics) and split across components by energy."""
    fc = _fuel_at_speed(vessel, voyage, speed, base_fuel, fuel_factors)
    energy = fc["energy_mj"]  # total propulsion energy (MJ), fuel-independent
    bef = fuel_factors.get(base_fuel)
    aef = fuel_factors.get(alt_fuel)
    lcv_b = bef.lcv_mj_per_kg if bef else 41.0
    lcv_a = aef.lcv_mj_per_kg if aef else 41.0

    e_alt = phi * energy
    e_base = (1.0 - phi) * energy
    m_base = e_base / (1000.0 * lcv_b) if lcv_b > 0 else 0.0  # tonnes
    m_alt = e_alt / (1000.0 * lcv_a) if lcv_a > 0 else 0.0

    fuel_cost = m_base * base_price + m_alt * alt_price

    ets_co2e = _ets_covered_tonnes(base_fuel, bef, m_base, year) + _ets_covered_tonnes(
        alt_fuel, aef, m_alt, year
    )
    ets_cost = ets_co2e * voyage.ets_scope_factor * ets_phase_in * eua_price

    cii_co2_leg = _cii_co2_tonnes(bef, m_base) + _cii_co2_tonnes(aef, m_alt)
    annual_cii_co2 = cii_co2_leg * annual_fuel_scale
    cii_annual, attained = _cii_annual_penalty(
        cii_engine, vessel, annual_cii_co2, annual_dist, year, cii_pressure, charter_per_day
    )
    cii_penalty = cii_annual / max(annual_fuel_scale, 1.0)

    fueleu_penalty, fueleu_balance = _fueleu_blend_eval(
        fueleu_engine, vessel, [(base_fuel, e_base), (alt_fuel, e_alt)], year
    )

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

    total = fuel_cost + ets_cost + fueleu_penalty + cii_penalty + delay_cost
    return {
        "alt_fuel": alt_fuel.value,
        "blend_pct": float(phi),
        "speed_knots": float(speed),
        "fuel_cost": fuel_cost,
        "ets_cost": ets_cost,
        "fueleu_penalty": fueleu_penalty,
        "cii_penalty": cii_penalty,
        "delay_cost": delay_cost,
        "total_cost": total,
        "carbon_fuel_cost": fuel_cost + ets_cost + delay_cost,
        "attained_cii": attained,
        "co2e_wtw_t": co2e_wtw_t,
        "co2e_ttw_t": co2e_ttw_t,
        "fueleu_balance": fueleu_balance,  # signed: deficit < 0 < surplus (for pooling)
    }


_DEFAULT_WTW_G_PER_MJ: float = 91.16  # VLSFO/HFO-like WtW fallback (gCO2e/MJ)


def _ttw_co2e_g(ef: FuelEmissionFactors | None, mass_tonnes: float) -> float:
    """Tank-to-Wake CO2e (grams, GWP-100), physical — biogenic carbon NOT
    zero-rated (this is the actual combustion mass, not the ETS-counted one)."""
    if ef is None:
        return mass_tonnes * 1e6 * _DEFAULT_TTW_CO2_G_PER_G
    mass_g = mass_tonnes * 1e6
    return mass_g * (
        ef.ttw_co2_g_per_g_fuel
        + ef.ttw_ch4_g_per_g_fuel * GWP100["CH4"]
        + ef.ttw_n2o_g_per_g_fuel * GWP100["N2O"]
    )


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
) -> dict[str, Any]:
    """Joint (blend × speed) optimisation vs a carbon-only siloed baseline.

    Decision space: a continuous energy-fraction ``phi`` of one low-carbon
    ``alt_fuel`` blended into ``base_fuel`` (VLSFO), plus speed. Joint minimises
    total cost over all four regulations; siloed minimises fuel+ETS+delay (carbon
    priced only via ETS, blind to FuelEU/CII). RCI = fractional overpayment of the
    siloed operator. Reports the joint low-carbon blend fraction.
    """
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
        for phi in phis:
            for spd in speeds:
                row = _evaluate_blend_point(
                    vessel,
                    voyage,
                    base_fuel,
                    alt,
                    float(phi),
                    float(spd),
                    float(base_price),
                    float(alt_price),
                    eua_price,
                    fuel_factors,
                    cii_engine,
                    fueleu_engine,
                    ets_phase_in,
                    year,
                    annual_dist,
                    annual_fuel_scale,
                    cii_pressure,
                    charter_per_day,
                    delay_cost_per_hour,
                    scheduled_hours,
                )
                rows.append(row)

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    joint = df.loc[df["total_cost"].idxmin()]
    siloed = df.loc[df["carbon_fuel_cost"].idxmin()]
    cost_joint = float(joint["total_cost"])
    cost_siloed = float(siloed["total_cost"])
    rci = (cost_siloed - cost_joint) / cost_siloed if cost_siloed > 0 else 0.0

    return {
        "joint_alt_fuel": str(joint["alt_fuel"]),
        "joint_blend_pct": float(joint["blend_pct"]),
        "joint_speed": float(joint["speed_knots"]),
        "siloed_blend_pct": float(siloed["blend_pct"]),
        "siloed_speed": float(siloed["speed_knots"]),
        "cost_joint_eur": cost_joint,
        "cost_siloed_eur": cost_siloed,
        "rci": max(rci, 0.0),
        "blends": float(joint["blend_pct"]) > 0.01,
        "ets_cost_eur": float(joint["ets_cost"]),
        "fueleu_penalty_eur": float(joint["fueleu_penalty"]),
        "cii_penalty_eur": float(joint["cii_penalty"]),
        "profile": df,
    }
