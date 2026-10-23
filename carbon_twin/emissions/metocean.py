from __future__ import annotations

import math

import numpy as np

from carbon_twin.data.schemas import ShipType, Vessel, Voyage

_BEAUFORT_SPEED_MS: list[float] = [
    0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7, 36.0,
]

_TYPE_AVERAGE_PENALTY: dict[ShipType, float] = {
    ShipType.BULK_CARRIER: 0.10,
    ShipType.TANKER: 0.09,
    ShipType.CONTAINER_SHIP: 0.12,
    ShipType.GENERAL_CARGO: 0.11,
    ShipType.GAS_CARRIER: 0.10,
    ShipType.LNG_CARRIER: 0.09,
    ShipType.RO_RO_CARGO: 0.13,
    ShipType.REFRIGERATED_CARGO: 0.12,
    ShipType.CRUISE_PASSENGER: 0.14,
    ShipType.RO_PAX: 0.15,
    ShipType.VEHICLE_CARRIER: 0.14,
    ShipType.OFFSHORE: 0.08,
    ShipType.OTHER: 0.11,
}


def beaufort_to_wind_speed(beaufort: int) -> float:
    beaufort = max(0, min(beaufort, 12))
    if beaufort == 0:
        return 0.0
    lower = _BEAUFORT_SPEED_MS[beaufort - 1]
    upper = _BEAUFORT_SPEED_MS[beaufort]
    return (lower + upper) / 2.0


def compute_weather_factor(
    wind_speed_ms: float,
    wave_height_m: float,
    ship_type: ShipType,
    vessel_length_m: float | None = None,
) -> float:
    wind_frac = _added_resistance_wind_simple(wind_speed_ms, ship_type)
    wave_frac = _added_resistance_waves_simple(wave_height_m, vessel_length_m or 200.0)
    return 1.0 + wind_frac + wave_frac


def added_resistance_wind(
    wind_speed: float,
    wind_angle: float,
    vessel: Vessel,
) -> float:
    if wind_speed <= 0:
        return 0.0

    beam = vessel.beam_m or _default_beam(vessel)
    draught = vessel.draught_m or _default_draught(vessel)
    frontal_area = beam * (draught * 0.5 + 8.0)

    rho_air = 1.225
    angle_rad = math.radians(wind_angle)
    cx = 0.8 * math.cos(angle_rad) + 0.2 * abs(math.cos(angle_rad))

    force_n = 0.5 * rho_air * cx * frontal_area * wind_speed**2

    displacement = vessel.deadweight / 0.75 if vessel.deadweight else vessel.gross_tonnage * 1.5
    reference_resistance = displacement * 9.81 * 0.002

    if reference_resistance <= 0:
        return 0.0
    return max(force_n / reference_resistance, 0.0)


def added_resistance_waves(
    wave_height: float,
    wave_period: float,
    vessel: Vessel,
) -> float:
    if wave_height <= 0 or wave_period <= 0:
        return 0.0

    lpp = vessel.length_m or _default_length(vessel)
    beam = vessel.beam_m or _default_beam(vessel)

    wavelength = 1.56 * wave_period**2
    ratio_wl = wavelength / lpp if lpp > 0 else 1.0

    response_peak = math.exp(-2.0 * (ratio_wl - 1.0) ** 2)

    rho_water = 1025.0
    raw_coefficient = 0.5 * rho_water * 9.81 * beam * (wave_height**2) * response_peak

    displacement = vessel.deadweight / 0.75 if vessel.deadweight else vessel.gross_tonnage * 1.5
    reference_resistance = displacement * 9.81 * 0.002

    if reference_resistance <= 0:
        return 0.0
    return max(raw_coefficient / reference_resistance, 0.0)


def voyage_weather_penalty(
    voyage: Voyage,
    era5_data: dict | None = None,
) -> float:
    if era5_data is not None:
        wind_speeds: np.ndarray = np.asarray(era5_data.get("wind_speed_ms", []))
        wave_heights: np.ndarray = np.asarray(era5_data.get("wave_height_m", []))
        ship_type = ShipType(era5_data.get("ship_type", "bulk_carrier"))
        vessel_length = era5_data.get("vessel_length_m", 200.0)

        if len(wind_speeds) == 0 and len(wave_heights) == 0:
            return 1.0 + _TYPE_AVERAGE_PENALTY.get(ship_type, 0.10)

        n = max(len(wind_speeds), len(wave_heights))
        if len(wind_speeds) == 0:
            wind_speeds = np.zeros(n)
        if len(wave_heights) == 0:
            wave_heights = np.zeros(n)

        min_len = min(len(wind_speeds), len(wave_heights))
        wind_speeds = wind_speeds[:min_len]
        wave_heights = wave_heights[:min_len]

        factors = np.array([
            compute_weather_factor(float(ws), float(wh), ship_type, vessel_length)
            for ws, wh in zip(wind_speeds, wave_heights)
        ])
        return float(np.mean(factors))

    if voyage.weather_factor != 1.0:
        return voyage.weather_factor

    return 1.0 + _TYPE_AVERAGE_PENALTY.get(ShipType.BULK_CARRIER, 0.10)


def _added_resistance_wind_simple(wind_speed_ms: float, ship_type: ShipType) -> float:
    if wind_speed_ms <= 0:
        return 0.0
    exposure: dict[ShipType, float] = {
        ShipType.CONTAINER_SHIP: 1.2,
        ShipType.VEHICLE_CARRIER: 1.4,
        ShipType.CRUISE_PASSENGER: 1.3,
        ShipType.RO_PAX: 1.2,
        ShipType.RO_RO_CARGO: 1.1,
    }
    k = exposure.get(ship_type, 1.0)
    return k * 0.0003 * wind_speed_ms**2


def _added_resistance_waves_simple(wave_height_m: float, vessel_length_m: float) -> float:
    if wave_height_m <= 0 or vessel_length_m <= 0:
        return 0.0
    ratio = wave_height_m / vessel_length_m
    return 150.0 * ratio**2


def _default_beam(vessel: Vessel) -> float:
    dwt = vessel.deadweight or vessel.gross_tonnage
    return 10.0 + 0.00015 * dwt


def _default_draught(vessel: Vessel) -> float:
    dwt = vessel.deadweight or vessel.gross_tonnage
    return 5.0 + 0.00006 * dwt


def _default_length(vessel: Vessel) -> float:
    dwt = vessel.deadweight or vessel.gross_tonnage
    return 50.0 + 0.001 * dwt
