from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ShipType(str, Enum):
    BULK_CARRIER = "bulk_carrier"
    TANKER = "tanker"
    CONTAINER_SHIP = "container_ship"
    GENERAL_CARGO = "general_cargo"
    GAS_CARRIER = "gas_carrier"
    LNG_CARRIER = "lng_carrier"
    RO_RO_CARGO = "ro_ro_cargo"
    REFRIGERATED_CARGO = "refrigerated_cargo"
    CRUISE_PASSENGER = "cruise_passenger"
    RO_PAX = "ro_pax"
    VEHICLE_CARRIER = "vehicle_carrier"
    OFFSHORE = "offshore"
    OTHER = "other"


class FuelType(str, Enum):
    HFO = "HFO"
    VLSFO = "VLSFO"
    MGO = "MGO"
    LNG = "LNG"
    LPG = "LPG"
    METHANOL_FOSSIL = "methanol_fossil"
    METHANOL_BIO = "methanol_bio"
    METHANOL_E = "methanol_e"
    AMMONIA_FOSSIL = "ammonia_fossil"
    AMMONIA_GREEN = "ammonia_green"
    BIODIESEL_FAME = "biodiesel_fame"
    HYDROGEN_GREEN = "hydrogen_green"
    SHORE_POWER = "shore_power"


class CIIRating(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


class VesselActivity(str, Enum):
    UNDERWAY = "underway"
    ANCHORED = "anchored"
    BERTHED = "berthed"
    DRIFTING = "drifting"
    MANEUVERING = "maneuvering"


class EEAScope(str, Enum):
    INTRA_EEA = "intra_eea"
    INTO_EEA = "into_eea"
    OUT_OF_EEA = "out_of_eea"
    NON_EEA = "non_eea"
    PORT = "port"


class Vessel(BaseModel):
    imo_number: str
    mmsi: str | None = None
    name: str | None = None
    ship_type: ShipType
    flag: str | None = None
    gross_tonnage: float
    deadweight: float | None = None
    net_tonnage: float | None = None
    design_speed_knots: float | None = None
    main_engine_power_kw: float | None = None
    aux_engine_power_kw: float | None = None
    year_built: int | None = None
    ice_class: str | None = None
    length_m: float | None = None
    beam_m: float | None = None
    draught_m: float | None = None


class Port(BaseModel):
    port_id: str
    name: str
    country: str
    unlocode: str | None = None
    lat: float
    lon: float
    is_eea: bool = False
    is_eu: bool = False
    has_ops: bool = False
    is_transhipment_excluded: bool = False
    region: str | None = None


class AISPoint(BaseModel):
    mmsi: str
    imo_number: str | None = None
    timestamp: dt.datetime
    lat: float
    lon: float
    sog: float | None = None
    cog: float | None = None
    heading: float | None = None
    nav_status: int | None = None
    draught: float | None = None
    destination: str | None = None


class PortCall(BaseModel):
    port_call_id: str
    vessel_imo: str
    port_id: str
    arrival_time: dt.datetime
    departure_time: dt.datetime | None = None
    anchorage_hours: float = 0.0
    berth_hours: float = 0.0
    is_eea: bool = False
    shore_power_used: bool = False
    shore_power_kwh: float = 0.0


class Voyage(BaseModel):
    voyage_id: str
    vessel_imo: str
    departure_port_id: str
    arrival_port_id: str
    departure_time: dt.datetime
    arrival_time: dt.datetime
    distance_nm: float
    distance_gc_nm: float | None = None
    avg_speed_knots: float | None = None
    sea_hours: float | None = None
    port_wait_hours: float = 0.0
    eea_scope: EEAScope = EEAScope.NON_EEA
    ets_scope_factor: float = 0.0
    weather_factor: float = 1.0
    cargo_tonnes: float | None = None
    draught_m: float | None = None


class FuelRecord(BaseModel):
    voyage_id: str | None = None
    port_call_id: str | None = None
    vessel_imo: str
    fuel_type: FuelType
    mass_tonnes: float
    energy_mj: float | None = None
    purpose: str = "main_engine"


class EmissionEstimate(BaseModel):
    vessel_imo: str
    voyage_id: str | None = None
    port_call_id: str | None = None
    year: int
    co2_tonnes: float = 0.0
    ch4_tonnes: float = 0.0
    n2o_tonnes: float = 0.0
    co2e_ttw_tonnes: float = 0.0
    co2e_wtw_tonnes: float = 0.0
    energy_mj: float = 0.0
    fuel_records: list[FuelRecord] = Field(default_factory=list)
    estimation_method: str = "proxy"
    confidence_lower: float | None = None
    confidence_upper: float | None = None


class CIIResult(BaseModel):
    vessel_imo: str
    year: int
    version: str
    attained_cii: float
    required_cii: float
    rating: CIIRating
    ratio: float
    distance_to_next_rating: float | None = None
    consecutive_d_years: int = 0
    corrective_action_required: bool = False


class FuelEUResult(BaseModel):
    vessel_imo: str
    year: int
    version: str
    actual_ghg_intensity: float
    target_ghg_intensity: float
    compliance_balance: float
    surplus: bool
    penalty_eur: float = 0.0
    rfnbo_bonus_applied: bool = False
    banking_balance: float = 0.0
    borrowing_balance: float = 0.0
    pooling_group: str | None = None
    pooling_contribution: float = 0.0
    ops_compliant: bool | None = None


class ETSResult(BaseModel):
    vessel_imo: str
    year: int
    version: str
    total_emissions_tco2e: float
    covered_emissions_tco2e: float
    scope_breakdown: dict[str, float] = Field(default_factory=dict)
    phase_in_factor: float = 1.0
    allowances_required: float
    eua_price_eur: float
    ets_cost_eur: float
    cost_per_tonne_cargo: float | None = None


class IMONZFResult(BaseModel):
    vessel_imo: str
    year: int
    scenario: str
    gfi_attained: float | None = None
    gfi_target: float | None = None
    deficit_or_surplus: float | None = None
    ghg_price_per_tco2e: float | None = None
    penalty_or_reward_eur: float = 0.0
    reward_units: float = 0.0


class UnifiedComplianceResult(BaseModel):
    vessel_imo: str
    year: int
    cii: CIIResult | None = None
    fueleu: FuelEUResult | None = None
    ets: ETSResult | None = None
    imo_nzf: IMONZFResult | None = None
    total_compliance_cost_eur: float = 0.0
    total_co2e_ttw_tonnes: float = 0.0
    total_co2e_wtw_tonnes: float = 0.0
    regulation_versions: dict[str, str] = Field(default_factory=dict)
    audit_metadata: dict[str, Any] = Field(default_factory=dict)


class Fleet(BaseModel):
    fleet_id: str
    name: str | None = None
    vessels: list[Vessel] = Field(default_factory=list)
    voyages: list[Voyage] = Field(default_factory=list)
    port_calls: list[PortCall] = Field(default_factory=list)
    fuel_records: list[FuelRecord] = Field(default_factory=list)
    emissions: list[EmissionEstimate] = Field(default_factory=list)
