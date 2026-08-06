"""Speed optimization under regulatory constraints."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from carbon_twin.data.schemas import CIIRating, FuelRecord, FuelType, Vessel, Voyage
from carbon_twin.emissions.factors import (
    FuelEmissionFactors,
    get_co2e_ttw,
)

# Energy content of one tonne of VLSFO-equivalent (MJ); FuelEU Annex IV constant.
_VLSFO_ENERGY_MJ_PER_TONNE: float = 41000.0
from carbon_twin.emissions.physics import (
    _estimate_displacement,
    estimate_admiralty_coeff,
    main_engine_fuel_rate,
    aux_engine_fuel_rate,
    VesselActivity,
)


# Slow-steaming floor as a fraction of design speed. Since power ∝ speed³, a
# 0.5×design floor corresponds to ~12% MCR — the practical super-slow-steaming
# limit. An absolute 6 kn floor preserves steerage. Commercial operation does
# not exceed design speed, so that is the upper bound.
_MIN_SPEED_FRACTION: float = 0.5
_ABS_MIN_SPEED: float = 6.0


def speed_bounds(
    vessel: Vessel, operating_speed: float | None = None
) -> tuple[float, float]:
    """Realistic per-vessel optimization band [v_min, v_max] in knots.

    Replaces the flat (8, 22) band that let every vessel rail to an unrealistic
    8 kn floor. Anchored on the vessel's design speed (now real, from the IHS
    register). The actual operating speed, if known, is always kept inside the
    band so the optimizer can reproduce it.
    """
    from carbon_twin.emissions.physics import _DESIGN_SPEEDS

    design = vessel.design_speed_knots or _DESIGN_SPEEDS.get(vessel.ship_type, 14.0)
    v_min = max(design * _MIN_SPEED_FRACTION, _ABS_MIN_SPEED)
    v_max = float(design)
    if operating_speed and operating_speed > 0:
        v_max = max(v_max, operating_speed)
        v_min = min(v_min, operating_speed)
    if v_max <= v_min:
        v_max = v_min + 1.0
    return round(v_min, 2), round(v_max, 2)


def _sea_hours_at_speed(voyage: Voyage, speed_knots: float) -> float:
    if speed_knots <= 0:
        return float("inf")
    return voyage.distance_nm / speed_knots


def _me_power_at_speed(vessel: Vessel, speed_knots: float) -> float:
    displacement = _estimate_displacement(vessel)
    ac = estimate_admiralty_coeff(vessel)
    if speed_knots <= 0:
        return 0.0
    power = (displacement ** (2 / 3) * speed_knots ** 3) / ac
    age = 0
    if vessel.year_built:
        age = 2026 - vessel.year_built
    fouling_margin = 1.0 + min(age * 0.005, 0.15)
    power *= fouling_margin
    me_max = vessel.main_engine_power_kw
    if me_max and power > me_max:
        power = me_max
    return power


def _fuel_at_speed(
    vessel: Vessel,
    voyage: Voyage,
    speed_knots: float,
    fuel_type: FuelType = FuelType.VLSFO,
    fuel_factors: dict[FuelType, FuelEmissionFactors] | None = None,
) -> dict[str, float]:
    sea_hours = _sea_hours_at_speed(voyage, speed_knots)
    me_power = _me_power_at_speed(vessel, speed_knots)
    me_tonnes = main_engine_fuel_rate(me_power) * sea_hours
    aux_tonnes = aux_engine_fuel_rate(vessel, VesselActivity.UNDERWAY) * sea_hours
    total_tonnes = me_tonnes + aux_tonnes
    if fuel_factors and fuel_type in fuel_factors:
        lcv = fuel_factors[fuel_type].lcv_mj_per_kg
    else:
        lcv = 41.0
    energy_mj = total_tonnes * 1e3 * lcv
    return {
        "me_tonnes": me_tonnes,
        "aux_tonnes": aux_tonnes,
        "total_tonnes": total_tonnes,
        "energy_mj": energy_mj,
        "sea_hours": sea_hours,
    }


def optimal_speed_single_voyage(
    vessel: Vessel,
    voyage: Voyage,
    fuel_price: float,
    eua_price: float,
    fuel_factors: dict[FuelType, FuelEmissionFactors],
    regulations: list[str] | None = None,
    speed_range: tuple[float, float] = (8, 22),
    n_points: int = 50,
    fuel_type: FuelType = FuelType.VLSFO,
    delay_cost_per_hour: float = 0.0,
    scheduled_hours: float | None = None,
    energy_scale: float = 1.0,
) -> dict[str, Any]:
    """Brute-force speed optimization over a grid.

    Total cost = fuel_cost + ets_cost + fueleu_penalty_share + delay_cost.
    Returns optimal speed and associated costs.
    """
    speeds = np.linspace(speed_range[0], speed_range[1], n_points)
    best: dict[str, Any] = {}
    best_total = float("inf")
    profile: list[dict[str, Any]] = []

    ets_scope = voyage.ets_scope_factor

    for spd in speeds:
        fc = _fuel_at_speed(vessel, voyage, float(spd), fuel_type, fuel_factors)
        scaled_t = fc["total_tonnes"] * energy_scale
        fuel_cost = scaled_t * fuel_price
        co2e = get_co2e_ttw(fuel_type, scaled_t, fuel_factors)
        ets_cost = co2e * ets_scope * eua_price

        delay_cost = 0.0
        if scheduled_hours is not None and delay_cost_per_hour > 0:
            extra_hours = max(fc["sea_hours"] - scheduled_hours, 0.0)
            delay_cost = extra_hours * delay_cost_per_hour

        total_cost = fuel_cost + ets_cost + delay_cost

        row = {
            "speed_knots": float(spd),
            "fuel_tonnes": scaled_t,
            "fuel_cost": fuel_cost,
            "ets_cost": ets_cost,
            "delay_cost": delay_cost,
            "total_cost": total_cost,
            "co2e_tonnes": co2e,
            "sea_hours": fc["sea_hours"],
        }
        profile.append(row)

        if total_cost < best_total:
            best_total = total_cost
            best = row.copy()

    return {
        "optimal_speed": best.get("speed_knots", 0.0),
        "fuel_cost": best.get("fuel_cost", 0.0),
        "ets_cost": best.get("ets_cost", 0.0),
        "total_cost": best.get("total_cost", 0.0),
        "co2e": best.get("co2e_tonnes", 0.0),
        "fuel_tonnes": best.get("fuel_tonnes", 0.0),
        "sea_hours": best.get("sea_hours", 0.0),
        "speed_profile": pd.DataFrame(profile),
    }


def speed_cost_curve(
    vessel: Vessel,
    voyage: Voyage,
    fuel_price: float,
    eua_price: float,
    fuel_factors: dict[FuelType, FuelEmissionFactors],
    speeds: list[float] | np.ndarray | None = None,
    fuel_type: FuelType = FuelType.VLSFO,
) -> pd.DataFrame:
    """Compute cost at each speed point."""
    if speeds is None:
        speeds = np.linspace(8, 22, 50)
    speeds = np.asarray(speeds, dtype=float)

    ets_scope = voyage.ets_scope_factor
    rows: list[dict[str, Any]] = []

    for spd in speeds:
        fc = _fuel_at_speed(vessel, voyage, float(spd), fuel_type, fuel_factors)
        fuel_cost = fc["total_tonnes"] * fuel_price
        co2e = get_co2e_ttw(fuel_type, fc["total_tonnes"], fuel_factors)
        ets_cost = co2e * ets_scope * eua_price
        rows.append({
            "speed_knots": float(spd),
            "fuel_tonnes": fc["total_tonnes"],
            "fuel_cost": fuel_cost,
            "ets_cost": ets_cost,
            "total_cost": fuel_cost + ets_cost,
            "co2e_tonnes": co2e,
            "sea_hours": fc["sea_hours"],
            "energy_mj": fc["energy_mj"],
        })

    return pd.DataFrame(rows)


def _cii_capacity(vessel: Vessel) -> float:
    """MEPC capacity for this vessel — mirrors the CII engine's basis (GT vs DWT
    per ship type / DWT threshold). Used only for the engine-absent fallback;
    when an engine is present the optimizer defers to it so the attained-CII
    denominator and the rating always agree (see _cii_annual_penalty)."""
    try:
        from carbon_twin.regulations.cii import CIIEngine
        return CIIEngine.capacity_for(vessel)
    except Exception:
        if vessel.deadweight and vessel.deadweight > 0:
            return vessel.deadweight
        return vessel.gross_tonnage


# Default CII consequence model. The penalty is a charter-rate discount that
# grows with how far a vessel's carbon-intensity ratio sits beyond the C/D
# compliance boundary. This replaces an arbitrary 50k/100k step (which created a
# discontinuity and an undefendable magnitude) with a continuous, interpretable
# cost. `cii_pressure` (k) is the discount sensitivity and the single knob for
# the RCI sensitivity sweep; default 0.2 ≈ a ~2–3% charter discount for a solidly
# D/E vessel — consistent with emerging charter-market signals. See
# docs/coupling_model_design.md.
_DEFAULT_CII_PRESSURE: float = 0.2
_CII_OPERATING_DAYS: float = 300.0


def _cii_annual_penalty(
    cii_engine: Any,
    vessel: Vessel,
    annual_co2_tonnes: float,
    annual_dist_nm: float,
    year: int,
    cii_pressure: float,
    charter_per_day: float,
    cii_rho_multiplier: float = 1.0,
) -> tuple[float, float]:
    """Return (annual_penalty_eur, attained_cii).

    Continuous: penalty = charter_per_day · operating_days · k · max(0, ratio −
    ratio_CD), where ratio_CD is the C/D boundary. Zero for A/B/C, ramps into
    D/E. The caller scales this annual figure to the representative leg so it sits
    on the same basis as fuel/ETS. Falls back to attained-only when no engine or
    no pressure/charter context is available.
    """
    capacity = _cii_capacity(vessel)
    attained = (annual_co2_tonnes * 1e6) / (capacity * annual_dist_nm)

    if cii_engine is None or cii_pressure <= 0 or charter_per_day <= 0:
        return 0.0, attained

    try:
        # Pass capacity=None so the engine selects the correct MEPC basis
        # (GT vs DWT, incl. DWT thresholds) itself — overriding it here is what
        # previously inflated RCI for GT-based types (vehicle carrier, ro-pax).
        result = cii_engine.evaluate(
            vessel=vessel,
            annual_co2_tonnes=annual_co2_tonnes,
            distance_nm=annual_dist_nm,
            year=year,
            capacity=None,
        )
        attained = result.attained_cii
        ratio_cd = cii_engine.cd_boundary_ratio(vessel.ship_type.value, vessel.deadweight)
        # cii_rho_multiplier scales the attained/required ratio to probe the
        # EU-MRV-vs-official-CII scope bias (1.0 = neutral; see configs/
        # mrv_cii_proxy_bias_scenarios.csv).
        excess = max(0.0, result.ratio * cii_rho_multiplier - ratio_cd)
    except Exception:
        return 0.0, attained

    penalty = charter_per_day * _CII_OPERATING_DAYS * cii_pressure * excess
    return penalty, attained


def _fueleu_penalty(
    fueleu_engine: Any,
    vessel: Vessel,
    fuel_type: FuelType,
    energy_mj: float,
    year: int,
    fuel_factors: dict[FuelType, FuelEmissionFactors] | None,
) -> float:
    """FuelEU penalty (EUR) for burning ``energy_mj`` of ``fuel_type``.

    Delegates to the engine when available; otherwise applies the same
    Annex IV formula directly so the magnitude stays correct.
    """
    if energy_mj <= 0:
        return 0.0

    if fueleu_engine is not None:
        try:
            record = FuelRecord(
                vessel_imo=vessel.imo_number,
                fuel_type=fuel_type,
                mass_tonnes=0.0,
                energy_mj=energy_mj,
            )
            result = fueleu_engine.evaluate(
                vessel=vessel,
                fuel_records=[record],
                energy_mj=energy_mj,
                year=year,
            )
            return result.penalty_eur
        except Exception:
            pass

    # Fallback: Annex IV formula |balance| / (actual * 41000) * 2400.
    if not fuel_factors or fuel_type not in fuel_factors:
        return 0.0
    actual = fuel_factors[fuel_type].wtw_co2e_g_per_mj
    target = 91.16 * (1.0 - 0.02)
    if actual <= target or actual <= 0:
        return 0.0
    balance_gco2e = (actual - target) * energy_mj
    vlsfoe_tonnes = balance_gco2e / (actual * _VLSFO_ENERGY_MJ_PER_TONNE)
    return vlsfoe_tonnes * 2400.0


def optimal_speed_multi_regulation(
    vessel: Vessel,
    voyage: Voyage,
    fuel_price: float,
    eua_price: float,
    fuel_factors: dict[FuelType, FuelEmissionFactors],
    cii_engine: Any,
    fueleu_engine: Any,
    ets_engine: Any,
    year: int,
    fuel_type: FuelType = FuelType.VLSFO,
    speed_range: tuple[float, float] = (8, 22),
    n_points: int = 50,
    annual_distance_nm: float | None = None,
    delay_cost_per_hour: float = 0.0,
    scheduled_hours: float | None = None,
    cii_pressure: float = _DEFAULT_CII_PRESSURE,
    cii_rho_multiplier: float = 1.0,
    energy_scale: float = 1.0,
) -> dict[str, Any]:
    """Speed optimization including CII impact in annual context.

    Evaluates each speed point for fuel cost, ETS cost, FuelEU penalty share,
    and CII rating implications. The CII consequence is a continuous,
    charter-anchored cost (see ``_cii_annual_penalty``); ``cii_pressure`` scales
    it and is the knob for the RCI sensitivity sweep.

    The passed ``cii_engine``, ``fueleu_engine`` and ``ets_engine`` are used
    directly (regulation-as-code), so the active regulation *version* drives the
    optimisation. When an engine is ``None`` a dimensionally-correct fallback is
    used instead.
    """
    speeds = np.linspace(speed_range[0], speed_range[1], n_points)
    best: dict[str, Any] = {}
    best_total = float("inf")
    profile: list[dict[str, Any]] = []

    ets_scope = voyage.ets_scope_factor
    annual_dist = annual_distance_nm or voyage.distance_nm * 12
    annual_fuel_scale = annual_dist / max(voyage.distance_nm, 1.0)

    # ETS phase-in is version-specific; honour the engine if available.
    ets_phase_in = 1.0
    if ets_engine is not None and hasattr(ets_engine, "_phase_in_factor"):
        try:
            ets_phase_in = float(ets_engine._phase_in_factor(year))
        except Exception:
            ets_phase_in = 1.0

    # Charter day-rate underlying both the delay cost and the CII consequence.
    charter_per_day = delay_cost_per_hour * 24.0

    for spd in speeds:
        fc = _fuel_at_speed(vessel, voyage, float(spd), fuel_type, fuel_factors)
        # energy_scale: MRV-calibration multiplier; scales fuel mass/energy so the
        # vessel's baseline emissions are anchored to its MRV-reported CO2 while
        # preserving the cubic-law speed elasticity. Delay cost is time-based, unscaled.
        scaled_tonnes = fc["total_tonnes"] * energy_scale
        scaled_energy = fc["energy_mj"] * energy_scale
        fuel_cost = scaled_tonnes * fuel_price
        co2e = get_co2e_ttw(fuel_type, scaled_tonnes, fuel_factors)
        ets_cost = co2e * ets_scope * ets_phase_in * eua_price

        annual_co2 = co2e * annual_fuel_scale

        # ---- CII: continuous charter-anchored consequence, computed in annual
        # context then scaled to the representative leg so it sits on the same
        # basis as fuel/ETS (which are per-leg).
        cii_annual, attained_cii = _cii_annual_penalty(
            cii_engine, vessel, annual_co2, annual_dist, year,
            cii_pressure, charter_per_day, cii_rho_multiplier,
        )
        cii_penalty = cii_annual / max(annual_fuel_scale, 1.0)

        # ---- FuelEU: penalty from the engine (Annex IV formula).
        fueleu_penalty = _fueleu_penalty(
            fueleu_engine, vessel, fuel_type, scaled_energy, year, fuel_factors
        )

        delay_cost = 0.0
        if scheduled_hours is not None and delay_cost_per_hour > 0:
            extra_hours = max(fc["sea_hours"] - scheduled_hours, 0.0)
            delay_cost = extra_hours * delay_cost_per_hour

        total_cost = fuel_cost + ets_cost + fueleu_penalty + cii_penalty + delay_cost

        row = {
            "speed_knots": float(spd),
            "fuel_tonnes": scaled_tonnes,
            "fuel_cost": fuel_cost,
            "ets_cost": ets_cost,
            "fueleu_penalty": fueleu_penalty,
            "cii_penalty": cii_penalty,
            "delay_cost": delay_cost,
            "total_cost": total_cost,
            "co2e_tonnes": co2e,
            "attained_cii": attained_cii,
            "sea_hours": fc["sea_hours"],
        }
        profile.append(row)

        if total_cost < best_total:
            best_total = total_cost
            best = row.copy()

    return {
        "optimal_speed": best.get("speed_knots", 0.0),
        "fuel_cost": best.get("fuel_cost", 0.0),
        "ets_cost": best.get("ets_cost", 0.0),
        "fueleu_penalty": best.get("fueleu_penalty", 0.0),
        "cii_penalty": best.get("cii_penalty", 0.0),
        "total_cost": best.get("total_cost", 0.0),
        "co2e": best.get("co2e_tonnes", 0.0),
        "attained_cii": best.get("attained_cii", 0.0),
        "speed_profile": pd.DataFrame(profile),
    }
