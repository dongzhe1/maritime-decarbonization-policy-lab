from __future__ import annotations

import datetime as dt

import pytest

from carbon_twin.data.schemas import (
    CIIRating,
    EEAScope,
    EmissionEstimate,
    Fleet,
    FuelRecord,
    FuelType,
    Port,
    PortCall,
    ShipType,
    Vessel,
    Voyage,
)


@pytest.fixture
def sample_bulk_carrier() -> Vessel:
    return Vessel(
        imo_number="IMO9876543",
        mmsi="211000001",
        name="MV Test Bulk",
        ship_type=ShipType.BULK_CARRIER,
        flag="DE",
        gross_tonnage=45000,
        deadweight=82000,
        design_speed_knots=14.5,
        main_engine_power_kw=12000,
        aux_engine_power_kw=2500,
        year_built=2015,
        length_m=229,
        beam_m=32,
        draught_m=14.5,
    )


@pytest.fixture
def sample_container_ship() -> Vessel:
    return Vessel(
        imo_number="IMO1234567",
        mmsi="636012345",
        name="MV Test Container",
        ship_type=ShipType.CONTAINER_SHIP,
        flag="LR",
        gross_tonnage=95000,
        deadweight=105000,
        design_speed_knots=22.0,
        main_engine_power_kw=55000,
        aux_engine_power_kw=8000,
        year_built=2018,
        length_m=335,
        beam_m=48,
        draught_m=15.5,
    )


@pytest.fixture
def sample_tanker() -> Vessel:
    return Vessel(
        imo_number="IMO5555555",
        mmsi="538001234",
        name="MT Test Tanker",
        ship_type=ShipType.TANKER,
        flag="MH",
        gross_tonnage=60000,
        deadweight=110000,
        design_speed_knots=15.0,
        main_engine_power_kw=16000,
        aux_engine_power_kw=3000,
        year_built=2012,
        length_m=250,
        beam_m=44,
        draught_m=16.0,
    )


@pytest.fixture
def rotterdam_port() -> Port:
    return Port(
        port_id="NLRTM",
        name="Rotterdam",
        country="NL",
        unlocode="NLRTM",
        lat=51.9225,
        lon=4.4792,
        is_eea=True,
        is_eu=True,
        has_ops=True,
    )


@pytest.fixture
def shanghai_port() -> Port:
    return Port(
        port_id="CNSHA",
        name="Shanghai",
        country="CN",
        unlocode="CNSHA",
        lat=31.2304,
        lon=121.4737,
        is_eea=False,
        is_eu=False,
    )


@pytest.fixture
def hamburg_port() -> Port:
    return Port(
        port_id="DEHAM",
        name="Hamburg",
        country="DE",
        unlocode="DEHAM",
        lat=53.5511,
        lon=9.9937,
        is_eea=True,
        is_eu=True,
        has_ops=True,
    )


@pytest.fixture
def sample_intra_eea_voyage(sample_bulk_carrier, rotterdam_port, hamburg_port) -> Voyage:
    return Voyage(
        voyage_id="VOY001",
        vessel_imo=sample_bulk_carrier.imo_number,
        departure_port_id=rotterdam_port.port_id,
        arrival_port_id=hamburg_port.port_id,
        departure_time=dt.datetime(2026, 3, 1, 8, 0),
        arrival_time=dt.datetime(2026, 3, 2, 14, 0),
        distance_nm=260,
        avg_speed_knots=12.0,
        sea_hours=22,
        eea_scope=EEAScope.INTRA_EEA,
        ets_scope_factor=1.0,
    )


@pytest.fixture
def sample_into_eea_voyage(sample_container_ship, shanghai_port, rotterdam_port) -> Voyage:
    return Voyage(
        voyage_id="VOY002",
        vessel_imo=sample_container_ship.imo_number,
        departure_port_id=shanghai_port.port_id,
        arrival_port_id=rotterdam_port.port_id,
        departure_time=dt.datetime(2026, 4, 1, 0, 0),
        arrival_time=dt.datetime(2026, 4, 28, 12, 0),
        distance_nm=10500,
        avg_speed_knots=16.0,
        sea_hours=656,
        eea_scope=EEAScope.INTO_EEA,
        ets_scope_factor=0.5,
    )


@pytest.fixture
def sample_fuel_records_hfo(sample_bulk_carrier) -> list[FuelRecord]:
    return [
        FuelRecord(
            voyage_id="VOY001",
            vessel_imo=sample_bulk_carrier.imo_number,
            fuel_type=FuelType.HFO,
            mass_tonnes=45.0,
            energy_mj=45.0 * 40.2 * 1000,
            purpose="main_engine",
        ),
        FuelRecord(
            voyage_id="VOY001",
            vessel_imo=sample_bulk_carrier.imo_number,
            fuel_type=FuelType.MGO,
            mass_tonnes=3.5,
            energy_mj=3.5 * 42.7 * 1000,
            purpose="aux_engine",
        ),
    ]


@pytest.fixture
def sample_emission_estimate(sample_bulk_carrier) -> EmissionEstimate:
    return EmissionEstimate(
        vessel_imo=sample_bulk_carrier.imo_number,
        voyage_id="VOY001",
        year=2026,
        co2_tonnes=150.0,
        ch4_tonnes=0.003,
        n2o_tonnes=0.008,
        co2e_ttw_tonnes=150.0 + 0.003 * 28 + 0.008 * 265,
        co2e_wtw_tonnes=165.0,
        energy_mj=1_959_450,
        estimation_method="physics_proxy",
    )


@pytest.fixture
def sample_port_call(sample_bulk_carrier, rotterdam_port) -> PortCall:
    return PortCall(
        port_call_id="PC001",
        vessel_imo=sample_bulk_carrier.imo_number,
        port_id=rotterdam_port.port_id,
        arrival_time=dt.datetime(2026, 3, 2, 14, 0),
        departure_time=dt.datetime(2026, 3, 5, 6, 0),
        anchorage_hours=8.0,
        berth_hours=56.0,
        is_eea=True,
        shore_power_used=False,
    )
