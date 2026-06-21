"""Just-In-Time arrival optimization."""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

from carbon_twin.data.schemas import FuelType, PortCall, Vessel, Voyage
from carbon_twin.emissions.factors import (
    FuelEmissionFactors,
    get_co2e_ttw,
)
from carbon_twin.emissions.physics import (
    VesselActivity,
    _estimate_displacement,
    aux_engine_fuel_rate,
    estimate_admiralty_coeff,
    main_engine_fuel_rate,
)


def jit_speed_adjustment(
    vessel: Vessel,
    voyage: Voyage,
    port_call: PortCall,
    berth_window_start: dt.datetime,
    berth_window_end: dt.datetime,
    fuel_type: FuelType = FuelType.VLSFO,
    fuel_factors: dict[FuelType, FuelEmissionFactors] | None = None,
    fuel_price: float = 600.0,
    eua_price: float = 80.0,
) -> dict[str, Any]:
    """Compute reduced speed to arrive at berth_window_start instead of anchoring early.

    Returns fuel saved, CO2 saved, and ETS cost saving from JIT arrival.
    """
    distance = voyage.distance_nm
    departure = voyage.departure_time
    original_arrival = voyage.arrival_time

    original_sea_seconds = (original_arrival - departure).total_seconds()
    original_sea_hours = original_sea_seconds / 3600.0
    original_speed = distance / max(original_sea_hours, 0.1)

    wait_hours = port_call.anchorage_hours

    jit_sea_seconds = (berth_window_start - departure).total_seconds()
    jit_sea_hours = max(jit_sea_seconds / 3600.0, 1.0)
    jit_speed = distance / jit_sea_hours

    min_speed = 4.0
    if jit_speed < min_speed:
        jit_speed = min_speed
        jit_sea_hours = distance / jit_speed

    displacement = _estimate_displacement(vessel)
    ac = estimate_admiralty_coeff(vessel)

    age = 0
    if vessel.year_built:
        age = 2026 - vessel.year_built
    fouling_margin = 1.0 + min(age * 0.005, 0.15)

    me_max = vessel.main_engine_power_kw or float("inf")

    def _fuel_for_speed(speed: float, hours: float) -> float:
        power = (displacement ** (2 / 3) * speed**3) / ac * fouling_margin
        power = min(power, me_max)
        me_rate = main_engine_fuel_rate(power)
        aux_rate = aux_engine_fuel_rate(vessel, VesselActivity.UNDERWAY)
        return (me_rate + aux_rate) * hours

    original_sea_fuel = _fuel_for_speed(original_speed, original_sea_hours)
    anchor_fuel = aux_engine_fuel_rate(vessel, VesselActivity.ANCHORED) * wait_hours
    original_total_fuel = original_sea_fuel + anchor_fuel

    jit_fuel = _fuel_for_speed(jit_speed, jit_sea_hours)

    fuel_saved = original_total_fuel - jit_fuel

    if fuel_factors is None:
        fuel_factors_local: dict[FuelType, FuelEmissionFactors] = {}
    else:
        fuel_factors_local = fuel_factors

    if fuel_type in fuel_factors_local:
        co2_original = get_co2e_ttw(fuel_type, original_total_fuel, fuel_factors_local)
        co2_jit = get_co2e_ttw(fuel_type, jit_fuel, fuel_factors_local)
    else:
        co2_factor = 3.114
        co2_original = original_total_fuel * co2_factor
        co2_jit = jit_fuel * co2_factor

    co2_saved = co2_original - co2_jit
    ets_scope = voyage.ets_scope_factor
    ets_saving = co2_saved * ets_scope * eua_price

    fuel_cost_saving = fuel_saved * fuel_price

    return {
        "jit_speed": round(jit_speed, 2),
        "original_speed": round(original_speed, 2),
        "fuel_saved_tonnes": round(max(fuel_saved, 0.0), 4),
        "co2_saved_tonnes": round(max(co2_saved, 0.0), 4),
        "ets_saving_eur": round(max(ets_saving, 0.0), 2),
        "fuel_cost_saving_eur": round(max(fuel_cost_saving, 0.0), 2),
        "wait_hours_eliminated": round(wait_hours, 2),
        "original_sea_hours": round(original_sea_hours, 2),
        "jit_sea_hours": round(jit_sea_hours, 2),
        "original_fuel_tonnes": round(original_total_fuel, 4),
        "jit_fuel_tonnes": round(jit_fuel, 4),
    }


def fleet_jit_potential(
    vessels: list[Vessel],
    voyages: list[Voyage],
    port_calls: list[PortCall],
    avg_wait_hours_by_port: dict[str, float],
    fuel_type: FuelType = FuelType.VLSFO,
    fuel_factors: dict[FuelType, FuelEmissionFactors] | None = None,
    fuel_price: float = 600.0,
    eua_price: float = 80.0,
) -> pd.DataFrame:
    """Estimate JIT savings across the fleet.

    Uses average wait hours per port to approximate potential savings
    if all vessels adopted JIT arrival.
    """
    vessel_map = {v.imo_number: v for v in vessels}
    voyage_map: dict[str, list[Voyage]] = {}
    for voy in voyages:
        voyage_map.setdefault(voy.vessel_imo, []).append(voy)

    pc_map: dict[str, list[PortCall]] = {}
    for pc in port_calls:
        pc_map.setdefault(pc.vessel_imo, []).append(pc)

    rows: list[dict[str, Any]] = []

    for imo, vessel in vessel_map.items():
        vessel_voyages = voyage_map.get(imo, [])
        vessel_pcs = pc_map.get(imo, [])

        total_fuel_saved = 0.0
        total_co2_saved = 0.0
        total_ets_saving = 0.0
        total_wait_eliminated = 0.0
        n_jit_voyages = 0

        for voy in vessel_voyages:
            port_id = voy.arrival_port_id
            avg_wait = avg_wait_hours_by_port.get(port_id, 0.0)
            if avg_wait <= 0:
                continue

            matching_pcs = [
                pc for pc in vessel_pcs if pc.port_id == port_id and pc.vessel_imo == imo
            ]
            if not matching_pcs:
                pc = PortCall(
                    port_call_id=f"synthetic_{voy.voyage_id}",
                    vessel_imo=imo,
                    port_id=port_id,
                    arrival_time=voy.arrival_time,
                    anchorage_hours=avg_wait,
                )
            else:
                pc = matching_pcs[0]
                if pc.anchorage_hours <= 0:
                    pc = pc.model_copy(update={"anchorage_hours": avg_wait})

            berth_start = voy.arrival_time + dt.timedelta(hours=avg_wait)
            berth_end = berth_start + dt.timedelta(hours=1)

            result = jit_speed_adjustment(
                vessel=vessel,
                voyage=voy,
                port_call=pc,
                berth_window_start=berth_start,
                berth_window_end=berth_end,
                fuel_type=fuel_type,
                fuel_factors=fuel_factors,
                fuel_price=fuel_price,
                eua_price=eua_price,
            )

            total_fuel_saved += result["fuel_saved_tonnes"]
            total_co2_saved += result["co2_saved_tonnes"]
            total_ets_saving += result["ets_saving_eur"]
            total_wait_eliminated += result["wait_hours_eliminated"]
            n_jit_voyages += 1

        rows.append(
            {
                "vessel_imo": imo,
                "vessel_name": vessel.name or imo,
                "ship_type": vessel.ship_type.value,
                "n_voyages": len(vessel_voyages),
                "n_jit_applicable": n_jit_voyages,
                "total_fuel_saved_tonnes": round(total_fuel_saved, 4),
                "total_co2_saved_tonnes": round(total_co2_saved, 4),
                "total_ets_saving_eur": round(total_ets_saving, 2),
                "total_wait_hours_eliminated": round(total_wait_eliminated, 2),
            }
        )

    return pd.DataFrame(rows)


def jit_sensitivity(
    vessel: Vessel,
    voyage: Voyage,
    wait_hours_range: list[float] | np.ndarray,
    fuel_price: float = 600.0,
    eua_price: float = 80.0,
    fuel_type: FuelType = FuelType.VLSFO,
    fuel_factors: dict[FuelType, FuelEmissionFactors] | None = None,
) -> pd.DataFrame:
    """How JIT savings vary with wait time."""
    rows: list[dict[str, Any]] = []

    for wait_hours in wait_hours_range:
        wait_hours = float(wait_hours)
        if wait_hours <= 0:
            rows.append(
                {
                    "wait_hours": wait_hours,
                    "fuel_saved_tonnes": 0.0,
                    "co2_saved_tonnes": 0.0,
                    "ets_saving_eur": 0.0,
                    "fuel_cost_saving_eur": 0.0,
                    "jit_speed": voyage.avg_speed_knots or 14.0,
                }
            )
            continue

        pc = PortCall(
            port_call_id="sensitivity",
            vessel_imo=vessel.imo_number,
            port_id=voyage.arrival_port_id,
            arrival_time=voyage.arrival_time,
            anchorage_hours=wait_hours,
        )

        berth_start = voyage.arrival_time + dt.timedelta(hours=wait_hours)
        berth_end = berth_start + dt.timedelta(hours=1)

        result = jit_speed_adjustment(
            vessel=vessel,
            voyage=voyage,
            port_call=pc,
            berth_window_start=berth_start,
            berth_window_end=berth_end,
            fuel_type=fuel_type,
            fuel_factors=fuel_factors,
            fuel_price=fuel_price,
            eua_price=eua_price,
        )

        rows.append(
            {
                "wait_hours": wait_hours,
                "fuel_saved_tonnes": result["fuel_saved_tonnes"],
                "co2_saved_tonnes": result["co2_saved_tonnes"],
                "ets_saving_eur": result["ets_saving_eur"],
                "fuel_cost_saving_eur": result["fuel_cost_saving_eur"],
                "jit_speed": result["jit_speed"],
            }
        )

    return pd.DataFrame(rows)
