"""Physics-based ship energy and fuel consumption models."""

from __future__ import annotations

import numpy as np

from carbon_twin.data.schemas import (
    PortCall,
    ShipType,
    Vessel,
    VesselActivity,
    Voyage,
)

# ---- Typical design speeds (knots) by ship type, used as fallback ----
_DESIGN_SPEEDS: dict[ShipType, float] = {
    ShipType.BULK_CARRIER: 14.0,
    ShipType.TANKER: 14.5,
    ShipType.CONTAINER_SHIP: 20.0,
    ShipType.GENERAL_CARGO: 14.0,
    ShipType.GAS_CARRIER: 16.5,
    ShipType.LNG_CARRIER: 19.5,
    ShipType.RO_RO_CARGO: 17.0,
    ShipType.REFRIGERATED_CARGO: 18.0,
    ShipType.CRUISE_PASSENGER: 21.0,
    ShipType.RO_PAX: 18.0,
    ShipType.VEHICLE_CARRIER: 18.5,
    ShipType.OFFSHORE: 12.0,
    ShipType.OTHER: 14.0,
}

# ---- Aux engine load fraction of installed ME power, by activity ----
_AUX_FRACTION: dict[ShipType, dict[VesselActivity, float]] = {
    ShipType.CONTAINER_SHIP: {
        VesselActivity.UNDERWAY: 0.06,
        VesselActivity.ANCHORED: 0.04,
        VesselActivity.BERTHED: 0.05,
        VesselActivity.DRIFTING: 0.03,
        VesselActivity.MANEUVERING: 0.05,
    },
    ShipType.BULK_CARRIER: {
        VesselActivity.UNDERWAY: 0.04,
        VesselActivity.ANCHORED: 0.025,
        VesselActivity.BERTHED: 0.035,
        VesselActivity.DRIFTING: 0.02,
        VesselActivity.MANEUVERING: 0.04,
    },
    ShipType.TANKER: {
        VesselActivity.UNDERWAY: 0.05,
        VesselActivity.ANCHORED: 0.03,
        VesselActivity.BERTHED: 0.04,
        VesselActivity.DRIFTING: 0.025,
        VesselActivity.MANEUVERING: 0.045,
    },
    ShipType.CRUISE_PASSENGER: {
        VesselActivity.UNDERWAY: 0.12,
        VesselActivity.ANCHORED: 0.09,
        VesselActivity.BERTHED: 0.10,
        VesselActivity.DRIFTING: 0.08,
        VesselActivity.MANEUVERING: 0.10,
    },
}

_DEFAULT_AUX_FRACTION: dict[VesselActivity, float] = {
    VesselActivity.UNDERWAY: 0.05,
    VesselActivity.ANCHORED: 0.03,
    VesselActivity.BERTHED: 0.04,
    VesselActivity.DRIFTING: 0.025,
    VesselActivity.MANEUVERING: 0.045,
}


def admiralty_power(
    displacement_tonnes: float,
    speed_knots: float,
    admiralty_coeff: float,
) -> float:
    """Estimate shaft power (kW) via the Admiralty coefficient formula.

    P = (displacement^(2/3) * speed^3) / C
    """
    if displacement_tonnes <= 0 or speed_knots <= 0 or admiralty_coeff <= 0:
        return 0.0
    return (displacement_tonnes ** (2 / 3) * speed_knots**3) / admiralty_coeff


def estimate_admiralty_coeff(vessel: Vessel) -> float:
    """Heuristic Admiralty coefficient derived from ship type, DWT, and design speed.

    Reverse-engineers C from known ME power: C = disp^(2/3) * Vd^3 / P_me.
    Falls back to ship-type lookup tables when ME power is unavailable.
    """
    displacement = _estimate_displacement(vessel)
    design_speed = vessel.design_speed_knots or _DESIGN_SPEEDS.get(vessel.ship_type, 14.0)

    if vessel.main_engine_power_kw and vessel.main_engine_power_kw > 0:
        return (displacement ** (2 / 3) * design_speed**3) / vessel.main_engine_power_kw

    # Fallback admiralty coefficients by ship type (typical values)
    _FALLBACK_C: dict[ShipType, float] = {
        ShipType.BULK_CARRIER: 550.0,
        ShipType.TANKER: 530.0,
        ShipType.CONTAINER_SHIP: 350.0,
        ShipType.GENERAL_CARGO: 500.0,
        ShipType.GAS_CARRIER: 480.0,
        ShipType.LNG_CARRIER: 400.0,
        ShipType.RO_RO_CARGO: 420.0,
        ShipType.REFRIGERATED_CARGO: 400.0,
        ShipType.CRUISE_PASSENGER: 320.0,
        ShipType.RO_PAX: 350.0,
        ShipType.VEHICLE_CARRIER: 430.0,
        ShipType.OFFSHORE: 600.0,
        ShipType.OTHER: 500.0,
    }
    return _FALLBACK_C.get(vessel.ship_type, 500.0)


def speed_power_curve(
    vessel: Vessel,
    speeds_knots: np.ndarray,
) -> np.ndarray:
    """Power (kW) at each speed, using cubic law with corrections for hull form.

    Applies a shallow-water / fouling margin of 15 % above the Admiralty baseline
    for vessels older than 10 years.
    """
    displacement = _estimate_displacement(vessel)
    ac = estimate_admiralty_coeff(vessel)

    powers = np.where(
        speeds_knots > 0,
        (displacement ** (2 / 3) * speeds_knots**3) / ac,
        0.0,
    )

    age = 0
    if vessel.year_built:
        age = 2026 - vessel.year_built
    fouling_margin = 1.0 + min(age * 0.005, 0.15)

    me_max = vessel.main_engine_power_kw or np.inf
    powers = np.clip(powers * fouling_margin, 0.0, me_max)
    return powers


def main_engine_fuel_rate(power_kw: float, sfoc_g_per_kwh: float = 180.0) -> float:
    """Main engine fuel consumption in tonnes/hour."""
    if power_kw <= 0:
        return 0.0
    return power_kw * sfoc_g_per_kwh / 1e6


def aux_engine_fuel_rate(
    vessel: Vessel,
    activity: VesselActivity,
    sfoc_g_per_kwh: float = 220.0,
) -> float:
    """Auxiliary engine fuel consumption in tonnes/hour.

    Estimates aux power as a fraction of installed main engine power, varying
    by ship type and activity mode.
    """
    me_power = vessel.main_engine_power_kw or _estimate_me_power(vessel)

    type_fractions = _AUX_FRACTION.get(vessel.ship_type, _DEFAULT_AUX_FRACTION)
    frac = type_fractions.get(activity, _DEFAULT_AUX_FRACTION.get(activity, 0.04))

    aux_power = me_power * frac
    return aux_power * sfoc_g_per_kwh / 1e6


def voyage_fuel_consumption(
    vessel: Vessel,
    voyage: Voyage,
    speed_profile: np.ndarray | None = None,
    weather_factor: float = 1.0,
) -> dict[str, float]:
    """Estimate fuel consumption for a single voyage.

    Returns dict with keys: main_engine_tonnes, aux_engine_tonnes,
    total_tonnes, energy_mj.
    """
    displacement = _estimate_displacement(vessel)
    ac = estimate_admiralty_coeff(vessel)

    if speed_profile is not None and len(speed_profile) > 0:
        avg_power = float(np.mean(np.where(
            speed_profile > 0,
            (displacement ** (2 / 3) * speed_profile**3) / ac,
            0.0,
        )))
    else:
        avg_speed = voyage.avg_speed_knots
        if avg_speed is None or avg_speed <= 0:
            sea_hours = voyage.sea_hours
            if sea_hours and sea_hours > 0:
                avg_speed = voyage.distance_nm / sea_hours
            else:
                avg_speed = vessel.design_speed_knots or _DESIGN_SPEEDS.get(
                    vessel.ship_type, 14.0
                )
        avg_power = admiralty_power(displacement, avg_speed, ac)

    age = 0
    if vessel.year_built:
        age = 2026 - vessel.year_built
    fouling_margin = 1.0 + min(age * 0.005, 0.15)
    avg_power *= fouling_margin

    me_max = vessel.main_engine_power_kw
    if me_max and avg_power > me_max:
        avg_power = me_max

    avg_power *= weather_factor

    sea_hours = voyage.sea_hours
    if sea_hours is None or sea_hours <= 0:
        avg_speed = voyage.avg_speed_knots
        if avg_speed is None or avg_speed <= 0:
            avg_speed = vessel.design_speed_knots or _DESIGN_SPEEDS.get(
                vessel.ship_type, 14.0
            )
        sea_hours = voyage.distance_nm / max(avg_speed, 0.1)

    me_tonnes = main_engine_fuel_rate(avg_power) * sea_hours
    aux_tonnes = aux_engine_fuel_rate(vessel, VesselActivity.UNDERWAY) * sea_hours
    total = me_tonnes + aux_tonnes

    lcv_mj_per_kg = 40.2  # HFO default
    energy_mj = total * 1e3 * lcv_mj_per_kg

    return {
        "main_engine_tonnes": round(me_tonnes, 4),
        "aux_engine_tonnes": round(aux_tonnes, 4),
        "total_tonnes": round(total, 4),
        "energy_mj": round(energy_mj, 2),
    }


def port_fuel_consumption(
    vessel: Vessel,
    port_call: PortCall,
    shore_power: bool = False,
) -> dict[str, float]:
    """Estimate fuel consumption during a port call."""
    if shore_power or port_call.shore_power_used:
        return {
            "main_engine_tonnes": 0.0,
            "aux_engine_tonnes": 0.0,
            "total_tonnes": 0.0,
            "energy_mj": 0.0,
            "shore_power_kwh": port_call.shore_power_kwh,
        }

    anchored_rate = aux_engine_fuel_rate(vessel, VesselActivity.ANCHORED)
    berthed_rate = aux_engine_fuel_rate(vessel, VesselActivity.BERTHED)

    anch_tonnes = anchored_rate * port_call.anchorage_hours
    berth_tonnes = berthed_rate * port_call.berth_hours
    total = anch_tonnes + berth_tonnes

    lcv_mj_per_kg = 42.7  # MGO default for port
    energy_mj = total * 1e3 * lcv_mj_per_kg

    return {
        "main_engine_tonnes": 0.0,
        "aux_engine_tonnes": round(total, 4),
        "total_tonnes": round(total, 4),
        "energy_mj": round(energy_mj, 2),
    }


def annual_fuel_consumption(
    vessel: Vessel,
    voyages: list[Voyage],
    port_calls: list[PortCall],
    weather_factors: list[float] | None = None,
) -> dict[str, float]:
    """Aggregate fuel consumption over a year of voyages and port calls."""
    total_me = 0.0
    total_aux = 0.0
    total_energy = 0.0

    for i, voy in enumerate(voyages):
        wf = weather_factors[i] if weather_factors and i < len(weather_factors) else 1.0
        result = voyage_fuel_consumption(vessel, voy, weather_factor=wf)
        total_me += result["main_engine_tonnes"]
        total_aux += result["aux_engine_tonnes"]
        total_energy += result["energy_mj"]

    for pc in port_calls:
        result = port_fuel_consumption(vessel, pc)
        total_aux += result["aux_engine_tonnes"]
        total_energy += result["energy_mj"]

    total = total_me + total_aux
    return {
        "main_engine_tonnes": round(total_me, 4),
        "aux_engine_tonnes": round(total_aux, 4),
        "total_tonnes": round(total, 4),
        "energy_mj": round(total_energy, 2),
        "num_voyages": len(voyages),
        "num_port_calls": len(port_calls),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _estimate_displacement(vessel: Vessel) -> float:
    """Estimate displacement from DWT and GT (lightweight ~ 0.3*GT for cargo)."""
    dwt = vessel.deadweight
    if dwt and dwt > 0:
        lightweight_ratio: dict[ShipType, float] = {
            ShipType.BULK_CARRIER: 0.20,
            ShipType.TANKER: 0.22,
            ShipType.CONTAINER_SHIP: 0.35,
            ShipType.GENERAL_CARGO: 0.30,
            ShipType.GAS_CARRIER: 0.40,
            ShipType.LNG_CARRIER: 0.50,
            ShipType.RO_RO_CARGO: 0.45,
            ShipType.REFRIGERATED_CARGO: 0.35,
            ShipType.CRUISE_PASSENGER: 0.65,
            ShipType.RO_PAX: 0.50,
            ShipType.VEHICLE_CARRIER: 0.45,
            ShipType.OFFSHORE: 0.55,
            ShipType.OTHER: 0.30,
        }
        ratio = lightweight_ratio.get(vessel.ship_type, 0.30)
        return dwt / (1 - ratio)
    return vessel.gross_tonnage * 1.5


def _estimate_me_power(vessel: Vessel) -> float:
    """Rough ME power estimate when not available on the vessel record."""
    displacement = _estimate_displacement(vessel)
    design_speed = vessel.design_speed_knots or _DESIGN_SPEEDS.get(vessel.ship_type, 14.0)
    ac = 500.0  # generic
    return (displacement ** (2 / 3) * design_speed**3) / ac
