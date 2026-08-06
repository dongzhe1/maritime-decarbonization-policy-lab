"""End-to-end orchestration tying voyages → emissions → regulations → optimization.

This is the glue layer the CLI commands call. The heavy lifting lives in the
engines (``regulations/``, ``optimization/``, ``scenarios/``); here we only
assemble inputs, fan out per vessel, and collect the unified compliance state.

The unified compliance state vector per vessel-year (see README "Unified
compliance state vector") is the central output of :func:`assemble_compliance`:

    (attained_CII, CII_rating, FuelEU_balance, ETS_covered_CO2e,
     total_compliance_cost)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from carbon_twin.data.loaders import load_usd_to_eur
from carbon_twin.data.schemas import (
    EEAScope,
    EmissionEstimate,
    FuelRecord,
    FuelType,
    Vessel,
    Voyage,
)
from carbon_twin.emissions.proxy import _row_to_voyage
from carbon_twin.regulations.cii import CIIEngine
from carbon_twin.regulations.eu_ets import ETSEngine
from carbon_twin.regulations.fueleu import FuelEUEngine

# VLSFO lower calorific value (MJ/kg) → energy from fuel mass for FuelEU
_LCV_MJ_PER_KG = 40.2

# Plausibility guards for CII. Real attained CII is ~1–30 gCO2/(capacity·nm) and
# a sailing cargo vessel covers thousands of n.miles/yr. A handful of MRV rows
# carry corrupt efficiency ratios that back-calculate to near-zero distance,
# producing absurd attained values (1e3–1e6). Reject those so they don't pollute
# ratings, costs and RCI. (Systematic ship-type misrating from tier-1 fleet
# capacity estimates is a separate, documented limitation — not guarded here.)
_MIN_CII_DISTANCE_NM = 500.0
_MAX_PLAUSIBLE_ATTAINED_CII = 100.0

# Indicative time-charter rates (USD/day) by ship type → hourly delay cost.
# Without a time cost, speed optimisation degenerates to "go as slow as possible"
# (fuel/ETS/penalties all fall monotonically with speed). The charter rate is the
# opportunity cost of slow steaming and creates the real fuel-vs-time trade-off.
_CHARTER_USD_PER_DAY: dict[str, float] = {
    "bulk_carrier": 14000, "tanker": 26000, "container_ship": 42000,
    "general_cargo": 8000, "gas_carrier": 45000, "lng_carrier": 70000,
    "ro_ro_cargo": 20000, "refrigerated_cargo": 16000, "cruise_passenger": 60000,
    "ro_pax": 28000, "vehicle_carrier": 24000, "offshore": 18000, "other": 15000,
}


# Charter day-rates (USD) are converted to EUR so delay cost and the CII commercial
# consequence share the numeraire of fuel/ETS/FuelEU (all EUR). See configs/currency.yaml.
_USD_TO_EUR: float = load_usd_to_eur()


def _charter_rate_per_hour(ship_type: str) -> float:
    return _CHARTER_USD_PER_DAY.get(ship_type, 15000) / 24.0 * _USD_TO_EUR


# ---------------------------------------------------------------------------
# MRV-direct compliance  (no AIS needed — Path C)
# ---------------------------------------------------------------------------

def assemble_compliance_from_mrv(
    mrv_df: pd.DataFrame,
    fleet: dict[str, Vessel],
    year: int,
    eua_price: float,
    cii_version: str,
    fueleu_version: str,
    ets_version: str,
    fuel_type: FuelType = FuelType.VLSFO,
) -> pd.DataFrame:
    """Compute CII + FuelEU + EU ETS straight from THETIS-MRV annual totals.

    This is the AIS-free path: MRV already reports each vessel's annual EU-related
    CO2 (``total_co2_tonnes``), the ETS-regulated subset (``co2_ets_tonnes``),
    fuel and distance. These ARE the regulated quantities, so the engines run on
    them directly — no voyage reconstruction or proxy calibration required.
    """
    cii = CIIEngine(version=cii_version)
    fueleu = FuelEUEngine(version=fueleu_version)
    ets = ETSEngine(version=ets_version)
    phase_in = 1.0
    try:
        phase_in = float(ets._phase_in_factor(year))
    except Exception:
        pass

    rows: list[dict[str, Any]] = []
    for r in mrv_df.to_dict(orient="records"):
        imo = str(r.get("imo_number", "")).strip()
        vessel = fleet.get(imo)
        if vessel is None:
            continue

        annual_co2 = _num(r.get("total_co2_tonnes"))
        distance = _mrv_distance(r)
        fuel_t = _num(r.get("total_fuel_tonnes"))
        # ETS-regulated emissions: prefer the MRV ETS column, else fall back to total
        co2_ets = r.get("co2_ets_tonnes")
        co2_ets = _num(co2_ets) if co2_ets is not None and pd.notna(co2_ets) else annual_co2

        # CII (global regulation), with a plausibility guard against corrupt MRV
        cii_rating = cii_attained = cii_required = None
        try:
            if (distance and distance >= _MIN_CII_DISTANCE_NM
                    and annual_co2 and annual_co2 > 0):
                cr = cii.evaluate(vessel, annual_co2, distance, year)
                if cr.attained_cii <= _MAX_PLAUSIBLE_ATTAINED_CII:
                    cii_rating = cr.rating.value
                    cii_attained = cr.attained_cii
                    cii_required = cr.required_cii
        except Exception:
            pass

        # ETS cost = regulated CO2 × phase-in × EUA price
        ets_cost = (co2_ets or 0.0) * phase_in * eua_price

        # FuelEU (energy from MRV fuel mass)
        fueleu_balance = fueleu_penalty = None
        if fuel_t and fuel_t > 0:
            energy_mj = fuel_t * 1e3 * _LCV_MJ_PER_KG
            try:
                fr = FuelRecord(vessel_imo=imo, fuel_type=fuel_type,
                                mass_tonnes=fuel_t, energy_mj=energy_mj)
                fe = fueleu.evaluate(vessel, [fr], energy_mj, year)
                fueleu_balance, fueleu_penalty = fe.compliance_balance, fe.penalty_eur
            except Exception:
                pass

        total_cost = ets_cost + (fueleu_penalty or 0.0)
        rows.append({
            "vessel_imo": imo,
            "ship_type": vessel.ship_type.value,
            "year": year,
            "annual_co2_tonnes": round(annual_co2 or 0.0, 2),
            "distance_nm": round(distance or 0.0, 1),
            "cii_attained": cii_attained,
            "cii_required": cii_required,
            "cii_rating": cii_rating,
            "fueleu_balance": fueleu_balance,
            "fueleu_penalty_eur": fueleu_penalty,
            "ets_covered_tco2e": round(co2_ets or 0.0, 2),
            "ets_cost_eur": round(ets_cost, 2),
            "total_compliance_cost_eur": round(total_cost, 2),
        })

    return pd.DataFrame(rows)


def run_optimization_from_mrv(
    compliance_df: pd.DataFrame,
    mrv_df: pd.DataFrame,
    fleet: dict[str, Vessel],
    year: int,
    eua_price: float,
    fuel_price: float,
    cii_version: str,
    fueleu_version: str,
    ets_version: str,
    fuel_factors: dict | None,
    fuel_type: FuelType = FuelType.VLSFO,
    cii_pressure: float | None = None,
    cii_rho_multiplier: dict[str, float] | float = 1.0,
    calibrate_to_mrv: bool = False,
) -> pd.DataFrame:
    """Speed optimisation + RCI using a representative voyage synthesised per
    vessel from MRV (distance + hours → operating speed). No AIS required.

    ``cii_pressure`` scales the continuous CII consequence (charter-discount
    sensitivity); None uses the optimizer default. Sweeping it produces the RCI
    sensitivity curve (see scripts/rci_sensitivity.py).

    ``cii_rho_multiplier`` scales the attained/required CII ratio to probe the
    EU-MRV-vs-official-CII scope bias; a float applies fleet-wide, a dict keyed by
    ship_type applies per type (see scripts/cii_proxy_sensitivity.py)."""
    from carbon_twin.optimization.speed import (
        _DEFAULT_CII_PRESSURE,
        optimal_speed_multi_regulation,
        optimal_speed_single_voyage,
        speed_bounds,
    )

    if cii_pressure is None:
        cii_pressure = _DEFAULT_CII_PRESSURE
    import datetime as dt

    cii = CIIEngine(version=cii_version)
    fueleu = FuelEUEngine(version=fueleu_version)
    ets = ETSEngine(version=ets_version)

    rows: list[dict[str, Any]] = []
    for r in mrv_df.to_dict(orient="records"):
        imo = str(r.get("imo_number", "")).strip()
        vessel = fleet.get(imo)
        if vessel is None:
            continue
        annual_dist = _mrv_distance(r)
        hours = _num(r.get("hours_at_sea"))
        # Same plausibility floor as CII: skip corrupt near-zero-distance rows
        # so they don't produce garbage optima / inflated RCI.
        if not annual_dist or annual_dist < _MIN_CII_DISTANCE_NM:
            continue
        avg_speed = (annual_dist / hours) if hours and hours > 0 else (
            vessel.design_speed_knots or 14.0)
        avg_speed = float(min(max(avg_speed, 6.0), 25.0))

        # Representative leg: a typical voyage length; annual context via annual_distance_nm
        leg_nm = float(min(annual_dist, 2000.0))
        leg_hours = leg_nm / max(avg_speed, 0.1)
        voy = Voyage(
            voyage_id=f"{imo}_rep", vessel_imo=imo,
            departure_port_id="REP_A", arrival_port_id="REP_B",
            departure_time=dt.datetime(year, 1, 1),
            arrival_time=dt.datetime(year, 1, 1) + dt.timedelta(hours=leg_hours),
            distance_nm=leg_nm, avg_speed_knots=avg_speed, sea_hours=leg_hours,
            eea_scope=EEAScope.INTRA_EEA, ets_scope_factor=1.0,
        )

        # Time cost of slow steaming: schedule the leg at the operating speed,
        # charge the charter rate for hours beyond schedule.
        charter_per_hour = _charter_rate_per_hour(vessel.ship_type.value)

        # Realistic per-vessel speed band (design-speed anchored) instead of the
        # flat 8–22 kn that let every vessel rail to an unrealistic 8 kn floor.
        v_range = speed_bounds(vessel, avg_speed)

        rho_mult = (cii_rho_multiplier.get(vessel.ship_type.value, 1.0)
                    if isinstance(cii_rho_multiplier, dict) else float(cii_rho_multiplier))

        # MRV-calibrated speed RCI: anchor propulsion energy to MRV-reported
        # CO2 (same logic as the fuel-optimizer calibrate_to_mrv path).
        spd_energy_scale = 1.0
        if calibrate_to_mrv:
            mrv_co2 = _num(r.get("total_co2_tonnes"))
            proxy_co2 = _proxy_baseline_annual_co2(
                vessel, voy, avg_speed, annual_dist, leg_nm, fuel_factors or {})
            if mrv_co2 and proxy_co2 and proxy_co2 > 0:
                spd_energy_scale = float(min(max(mrv_co2 / proxy_co2, 0.2), 5.0))

        joint = optimal_speed_multi_regulation(
            vessel, voy, fuel_price, eua_price, fuel_factors or {},
            cii_engine=cii, fueleu_engine=fueleu, ets_engine=ets,
            year=year, fuel_type=fuel_type, annual_distance_nm=annual_dist,
            speed_range=v_range, cii_pressure=cii_pressure,
            cii_rho_multiplier=rho_mult,
            delay_cost_per_hour=charter_per_hour, scheduled_hours=leg_hours,
            energy_scale=spd_energy_scale)
        single = optimal_speed_single_voyage(
            vessel, voy, fuel_price, eua_price, fuel_factors or {}, fuel_type=fuel_type,
            speed_range=v_range,
            delay_cost_per_hour=charter_per_hour, scheduled_hours=leg_hours,
            energy_scale=spd_energy_scale)

        jp = joint["speed_profile"]
        s_speed = single["optimal_speed"]
        j_speed = float(joint["optimal_speed"])
        silo_row = jp.iloc[(jp["speed_knots"] - s_speed).abs().argsort().iloc[0]]
        joint_row = jp.iloc[(jp["speed_knots"] - j_speed).abs().argsort().iloc[0]]
        cost_siloed = float(silo_row["total_cost"])
        cost_joint = float(joint["total_cost"])
        delta_c = cost_siloed - cost_joint
        rci = delta_c / cost_siloed if cost_siloed > 0 else 0.0
        rci_joint = delta_c / cost_joint if cost_joint > 0 else 0.0

        rows.append({
            "vessel_imo": imo,
            "ship_type": vessel.ship_type.value,
            "operating_speed_knots": round(avg_speed, 2),
            "joint_optimal_speed": round(j_speed, 2),
            "siloed_optimal_speed": round(float(s_speed), 2),
            "cost_joint_eur": round(cost_joint, 2),
            "cost_siloed_eur": round(cost_siloed, 2),
            # absolute opportunity loss and both normalisations (RCI economic interpretation)
            "delta_cost_eur": round(delta_c, 2),
            "rci": round(rci, 4),          # = delta_C / C_joint(x_silo)
            "rci_joint": round(rci_joint, 4),  # = delta_C / C_joint(x_joint)
            # joint-point cost components
            "fuel_cost_joint_eur": round(float(joint_row["fuel_cost"]), 2),
            "ets_cost_joint_eur": round(float(joint_row["ets_cost"]), 2),
            "fueleu_penalty_joint_eur": round(float(joint_row["fueleu_penalty"]), 2),
            "cii_consequence_joint_eur": round(float(joint_row["cii_penalty"]), 2),
            "delay_cost_joint_eur": round(float(joint_row["delay_cost"]), 2),
            # siloed-point cost components (evaluated under all instruments)
            "fuel_cost_siloed_eur": round(float(silo_row["fuel_cost"]), 2),
            "ets_cost_siloed_eur": round(float(silo_row["ets_cost"]), 2),
            "fueleu_penalty_siloed_eur": round(float(silo_row["fueleu_penalty"]), 2),
            "cii_consequence_siloed_eur": round(float(silo_row["cii_penalty"]), 2),
            "delay_cost_siloed_eur": round(float(silo_row["delay_cost"]), 2),
            # backward-compatible aliases (joint-point), kept so existing figures work
            "ets_cost_eur": round(float(joint_row["ets_cost"]), 2),
            "fueleu_penalty_eur": round(float(joint_row["fueleu_penalty"]), 2),
            "cii_penalty_eur": round(float(joint_row["cii_penalty"]), 2),
        })

    return pd.DataFrame(rows)


def build_fuel_optimization_context(
    r: dict[str, Any],
    fleet: dict[str, Vessel],
    year: int,
    fuel_factors: dict,
    calibrate_to_mrv: bool = False,
    charter_multiplier: float = 1.0,
) -> dict[str, Any] | None:
    """Representative-voyage + optimizer context for one MRV row (fuel path).

    Shared by :func:`run_fuel_optimization_from_mrv` and the FuelEU
    planning-internalization sweep runner so both optimise the exact same
    synthetic vessel-year under the same sample filters. Returns None for
    unmatched vessels and corrupt near-zero-distance rows (the same plausibility
    floor as the CII path)."""
    import datetime as dt

    from carbon_twin.optimization.speed import speed_bounds

    imo = str(r.get("imo_number", "")).strip()
    vessel = fleet.get(imo)
    if vessel is None:
        return None
    annual_dist = _mrv_distance(r)
    hours = _num(r.get("hours_at_sea"))
    if not annual_dist or annual_dist < _MIN_CII_DISTANCE_NM:
        return None
    avg_speed = (annual_dist / hours) if hours and hours > 0 else (
        vessel.design_speed_knots or 14.0)
    avg_speed = float(min(max(avg_speed, 6.0), 25.0))

    # Representative leg: a typical voyage length; annual context via annual_distance_nm
    leg_nm = float(min(annual_dist, 2000.0))
    leg_hours = leg_nm / max(avg_speed, 0.1)
    voy = Voyage(
        voyage_id=f"{imo}_rep", vessel_imo=imo,
        departure_port_id="REP_A", arrival_port_id="REP_B",
        departure_time=dt.datetime(year, 1, 1),
        arrival_time=dt.datetime(year, 1, 1) + dt.timedelta(hours=leg_hours),
        distance_nm=leg_nm, avg_speed_knots=avg_speed, sea_hours=leg_hours,
        eea_scope=EEAScope.INTRA_EEA, ets_scope_factor=1.0,
    )

    # Time cost of slow steaming: schedule the leg at the operating speed,
    # charge the charter rate for hours beyond schedule.
    charter_per_hour = _charter_rate_per_hour(vessel.ship_type.value) * charter_multiplier

    # Realistic per-vessel speed band (design-speed anchored) instead of the
    # flat 8–22 kn that let every vessel rail to an unrealistic 8 kn floor.
    v_range = speed_bounds(vessel, avg_speed)

    # MRV calibration: anchor the vessel's baseline energy/emissions to its
    # MRV-reported CO2 while keeping the cubic-law speed elasticity. The scale
    # is (MRV annual CO2) / (proxy annual CO2 at the base fuel and operating
    # speed); clamped to guard against corrupt near-zero MRV rows.
    energy_scale = 1.0
    if calibrate_to_mrv:
        mrv_co2 = _num(r.get("total_co2_tonnes"))
        proxy_co2 = _proxy_baseline_annual_co2(
            vessel, voy, avg_speed, annual_dist, leg_nm, fuel_factors or {})
        if mrv_co2 and proxy_co2 and proxy_co2 > 0:
            energy_scale = float(min(max(mrv_co2 / proxy_co2, 0.2), 5.0))

    return {
        "imo": imo,
        "vessel": vessel,
        "voyage": voy,
        "annual_dist": annual_dist,
        "charter_per_hour": charter_per_hour,
        "speed_range": v_range,
        "energy_scale": energy_scale,
    }


_FUEL_DROPIN: set[FuelType] = {FuelType.VLSFO, FuelType.MGO, FuelType.BIODIESEL_FAME}
_FUEL_GAS_CAPABLE: set[str] = {"gas_carrier", "lng_carrier"}


def feasible_fuel_options(
    vessel: Vessel,
    fuel_feasibility: str,
    extended_fuels: bool,
    exclude_fuels: set | None,
) -> list[FuelType] | None:
    """Feasibility-restricted alternative-fuel set for one vessel.

    ``continuous_base`` → the optimiser's full ``fuel_menu`` (None sentinel);
    ``dropin_only`` → drop-in fuels (VLSFO/MGO/biodiesel); ``compatible_switch``
    → drop-ins plus LNG for gas-capable types. ``exclude_fuels`` drops named
    fuels from whatever menu remains (a biofuel-availability sensitivity, e.g.
    excluding FAME to test a no-biofuel scenario)."""
    from carbon_twin.optimization.fuel_choice import fuel_menu

    if fuel_feasibility == "continuous_base":
        feas: list[FuelType] | None = None  # optimiser uses the full fuel_menu
    else:
        menu = fuel_menu(vessel, extended=extended_fuels)
        if fuel_feasibility == "dropin_only":
            feas = [f for f in menu if f in _FUEL_DROPIN]
        elif fuel_feasibility == "compatible_switch":
            allowed = set(_FUEL_DROPIN)
            if vessel.ship_type.value in _FUEL_GAS_CAPABLE:
                allowed.add(FuelType.LNG)
            feas = [f for f in menu if f in allowed]
        else:
            raise ValueError(f"unknown fuel_feasibility: {fuel_feasibility}")
    if exclude_fuels:
        base = feas if feas is not None else fuel_menu(vessel, extended=extended_fuels)
        feas = [f for f in base if f not in exclude_fuels]
    return feas


def run_fuel_optimization_from_mrv(
    mrv_df: pd.DataFrame,
    fleet: dict[str, Vessel],
    year: int,
    eua_price: float,
    fuel_prices: dict[str, float],
    cii_version: str,
    fueleu_version: str,
    ets_version: str,
    fuel_factors: dict | None,
    cii_pressure: float | None = None,
    extended_fuels: bool = False,
    blend: bool = False,
    charter_multiplier: float = 1.0,
    fuel_feasibility: str = "continuous_base",
    silo_carbon_basis: str = "ttw_ets",
    include_cii_in_objective: bool = True,
    max_biofuel_share: float | None = None,
    calibrate_to_mrv: bool = False,
    exclude_fuels: set | None = None,
    fueleu_price_fraction: float = 1.0,
    planner_fueleu_internalization: float = 0.0,
) -> pd.DataFrame:
    """Fuel × speed optimisation + fuel-dimension RCI per vessel.

    ``fuel_feasibility`` restricts the alternative-fuel menu (robustness to the
    continuous-blending assumption): ``continuous_base`` = full extended menu;
    ``dropin_only`` = drop-in fuels only (MGO, biodiesel); ``compatible_switch`` =
    drop-ins plus LNG for gas-capable ships. See configs/fuel_feasibility_scenarios.csv.

    Joint = minimise cost over all four regulations choosing both speed AND fuel;
    siloed = an operator pricing carbon only via ETS (blind to FuelEU/CII). RCI is
    the fractional cost the siloed operator overpays. Unlike the speed-only RCI,
    this captures the fuel decision where the regulations genuinely conflict
    (see docs/coupling_model_design.md).

    ``blend=True`` uses the v2 continuous-blending optimiser (a vessel blends just
    enough low-carbon fuel into a fossil base to meet FuelEU) — smoother and a
    *higher* RCI than the discrete v1 (which is a conservative lower bound).

    ``planner_fueleu_internalization`` (λ_fueleu) adds the FuelEU
    planning-internalization partial-planner decision (blend path only): the
    partial decision and its full-evaluation components are extra columns and
    never change the joint/siloed selection or costs, so existing runs are
    numerically unchanged when the parameter is omitted."""
    from carbon_twin.optimization.fuel_choice import (
        optimal_fuel_blend_speed_multi_regulation,
        optimal_fuel_speed_multi_regulation,
    )
    from carbon_twin.optimization.speed import _DEFAULT_CII_PRESSURE

    if cii_pressure is None:
        cii_pressure = _DEFAULT_CII_PRESSURE

    cii = CIIEngine(version=cii_version)
    fueleu = FuelEUEngine(version=fueleu_version)
    ets = ETSEngine(version=ets_version)

    rows: list[dict[str, Any]] = []
    for r in mrv_df.to_dict(orient="records"):
        ctx = build_fuel_optimization_context(
            r, fleet, year, fuel_factors or {}, calibrate_to_mrv, charter_multiplier)
        if ctx is None:
            continue
        imo = ctx["imo"]
        vessel = ctx["vessel"]
        voy = ctx["voyage"]
        annual_dist = ctx["annual_dist"]
        charter_per_hour = ctx["charter_per_hour"]
        v_range = ctx["speed_range"]
        energy_scale = ctx["energy_scale"]

        common = dict(
            cii_engine=cii, fueleu_engine=fueleu, ets_engine=ets,
            year=year, annual_distance_nm=annual_dist, speed_range=v_range,
            delay_cost_per_hour=charter_per_hour,
            scheduled_hours=voy.sea_hours,  # leg hours (see build_fuel_optimization_context)
            cii_pressure=cii_pressure, extended_fuels=extended_fuels,
        )
        if blend:
            res = optimal_fuel_blend_speed_multi_regulation(
                vessel, voy, fuel_prices, eua_price, fuel_factors or {},
                fuel_options=feasible_fuel_options(
                    vessel, fuel_feasibility, extended_fuels, exclude_fuels),
                silo_carbon_basis=silo_carbon_basis,
                include_cii_in_objective=include_cii_in_objective,
                max_biofuel_share=max_biofuel_share,
                energy_scale=energy_scale,
                fueleu_price_fraction=fueleu_price_fraction,
                planner_fueleu_internalization=planner_fueleu_internalization,
                **common)
            if not res:
                continue
            row = {
                "joint_fuel": res["joint_alt_fuel"],
                "joint_blend_pct": round(res["joint_blend_pct"], 3),
                "siloed_blend_pct": round(res["siloed_blend_pct"], 3),
                "blends": res["blends"],
                "energy_scale": round(energy_scale, 4),
                # FuelEU planning-internalization block (additive — existing
                # consumers select columns explicitly, so shared values and the
                # joint/siloed outputs are numerically unchanged).
                "partial_fuel": res["partial_alt_fuel"],
                "partial_base_fuel": res["partial_base_fuel"],
                "partial_blend_pct": round(res["partial_blend_pct"], 3),
                "partial_speed": round(res["partial_speed"], 2),
                "partial_low_carbon_share": round(res["partial_low_carbon_share"], 3),
                "partial_biofuel_share": round(res["partial_biofuel_share"], 3),
                "partial_fuel_cost_full_eur": round(res["partial_fuel_cost_full_eur"], 2),
                "partial_ets_cost_full_eur": round(res["partial_ets_cost_full_eur"], 2),
                "partial_fueleu_cost_full_eur": round(res["partial_fueleu_cost_full_eur"], 2),
                "partial_delay_cost_full_eur": round(res["partial_delay_cost_full_eur"], 2),
                "partial_total_cost_full_eur": round(res["partial_total_cost_full_eur"], 2),
                "partial_fueleu_balance": round(res["partial_fueleu_balance"], 2),
                "partial_fueleu_actual_intensity":
                    round(res["partial_fueleu_actual_intensity"], 4),
                "partial_fueleu_energy_mj": round(res["partial_fueleu_energy_mj"], 1),
                "delta_cost_partial_eur": round(res["delta_cost_partial_eur"], 2),
                "rci_partial": round(res["rci_partial"], 4),
                "rci_partial_joint_norm": round(res["rci_partial_joint_norm"], 4),
                "fuel_pathway_label_match": res["fuel_pathway_label_match"],
                "blend_share_match_1pp": res["blend_share_match_1pp"],
                "fuel_plan_match_1pp": res["fuel_plan_match_1pp"],
                "speed_match_0p01kn": res["speed_match_0p01kn"],
                "full_decision_match": res["full_decision_match"],
                "speed_abs_diff_kn": round(res["speed_abs_diff_kn"], 4),
                "blend_share_abs_diff": round(res["blend_share_abs_diff"], 4),
                "numerical_regret_nonnegative": res["numerical_regret_nonnegative"],
                # unambiguous fuel-identity fields (see fuel_choice.py)
                "benchmark_base_fuel": res["benchmark_base_fuel"],
                "benchmark_low_carbon_share":
                    round(res["benchmark_low_carbon_share"], 3),
                "fuel_pathway_partial": res["fuel_pathway_partial"],
                "fuel_pathway_benchmark": res["fuel_pathway_benchmark"],
                "alternative_component_share_partial":
                    round(res["alternative_component_share_partial"], 3),
                "alternative_component_share_benchmark":
                    round(res["alternative_component_share_benchmark"], 3),
                "energy_dominant_fuel_partial": res["energy_dominant_fuel_partial"],
                "energy_dominant_fuel_benchmark": res["energy_dominant_fuel_benchmark"],
                "full_decision_exact": res["full_decision_exact"],
                "full_decision_1pp_0p01kn": res["full_decision_1pp_0p01kn"],
            }
        else:
            res = optimal_fuel_speed_multi_regulation(
                vessel, voy, fuel_prices, eua_price, fuel_factors or {}, **common)
            if not res:
                continue
            row = {
                "siloed_fuel": res["siloed_fuel"],
                "joint_fuel": res["joint_fuel"],
                "fuel_switch": res["fuel_switch"],
            }
        row.update({
            "vessel_imo": imo,
            "ship_type": vessel.ship_type.value,
            "year": year,
            "siloed_speed": round(res["siloed_speed"], 2),
            "joint_speed": round(res["joint_speed"], 2),
            "cost_siloed_eur": round(res["cost_siloed_eur"], 2),
            "cost_joint_eur": round(res["cost_joint_eur"], 2),
            "rci": round(res["rci"], 4),
            "ets_cost_eur": round(res["ets_cost_eur"], 2),
            "fueleu_penalty_eur": round(res["fueleu_penalty_eur"], 2),
            "cii_penalty_eur": round(res["cii_penalty_eur"], 2),
        })
        # absolute opportunity loss + component decomposition (blend optimiser only)
        if "delta_cost_eur" in res:
            row.update({
                "delta_cost_eur": round(res["delta_cost_eur"], 2),
                "rci_joint": round(res["rci_joint"], 4),
                "siloed_fuel": res.get("siloed_alt_fuel"),
                "fuel_cost_joint_eur": round(res["fuel_cost_joint_eur"], 2),
                "ets_cost_joint_eur": round(res["ets_cost_joint_eur"], 2),
                "fueleu_penalty_joint_eur": round(res["fueleu_penalty_joint_eur"], 2),
                "cii_consequence_joint_eur": round(res["cii_consequence_joint_eur"], 2),
                "delay_cost_joint_eur": round(res["delay_cost_joint_eur"], 2),
                "fuel_cost_siloed_eur": round(res["fuel_cost_siloed_eur"], 2),
                "ets_cost_siloed_eur": round(res["ets_cost_siloed_eur"], 2),
                "fueleu_penalty_siloed_eur": round(res["fueleu_penalty_siloed_eur"], 2),
                "cii_consequence_siloed_eur": round(res["cii_consequence_siloed_eur"], 2),
                "delay_cost_siloed_eur": round(res["delay_cost_siloed_eur"], 2),
                # FuelEU pooling inputs + low-carbon/biofuel shares
                "fueleu_balance_joint": round(res["fueleu_balance_joint"], 2),
                "fueleu_actual_intensity_joint": round(res["fueleu_actual_intensity_joint"], 4),
                "fueleu_energy_mj_joint": round(res["fueleu_energy_mj_joint"], 1),
                "fueleu_balance_siloed": round(res["fueleu_balance_siloed"], 2),
                "fueleu_actual_intensity_siloed": round(res["fueleu_actual_intensity_siloed"], 4),
                "fueleu_energy_mj_siloed": round(res["fueleu_energy_mj_siloed"], 1),
                "low_carbon_share_joint": round(res["low_carbon_share_joint"], 3),
                "low_carbon_share_siloed": round(res["low_carbon_share_siloed"], 3),
                "biofuel_share_joint": round(res["biofuel_share_joint"], 3),
                "biofuel_share_siloed": round(res["biofuel_share_siloed"], 3),
            })
        rows.append(row)

    return pd.DataFrame(rows)


def run_ae_metrics_from_mrv(
    mrv_df: pd.DataFrame,
    fleet: dict[str, Vessel],
    year: int,
    fuel_prices: dict[str, float],
    cii_version: str,
    fueleu_version: str,
    ets_version: str,
    fuel_factors: dict | None,
    target_euas: list[float],
    breakeven_grid: list[float],
    eua_ref: float = 100.0,
    cii_pressure: float | None = None,
    extended_fuels: bool = True,
    rci_eps: float = 0.01,
    price_het_cv: float = 0.0,
    price_het_seed: int = 0,
) -> pd.DataFrame:
    """Applied-Energy metrics per vessel: abatement & MAC (AE-1), FuelEU-redundancy
    breakeven EUA (AE-2), and instrument attribution of the RCI (AE-3).

    ``price_het_cv`` > 0 draws a per-vessel low-carbon-fuel price multiplier from a
    mean-1 lognormal with that coefficient of variation (seed ``price_het_seed``),
    so operators face heterogeneous alternative-fuel prices and the redundancy
    breakeven becomes a distribution rather than a single fleet-wide value.

    Computes the blend profile once per vessel at ``eua_ref`` and derives all
    metrics analytically across ``target_euas`` (the EU ETS cost is linear in EUA;
    every other term is EUA-independent — see ``ae_metrics.ae_vessel_metrics``).
    Returns one row per vessel × target EUA. Uses the same representative-voyage
    build as ``run_fuel_optimization_from_mrv`` so results are consistent."""
    import datetime as dt
    import math

    import numpy as np

    from carbon_twin.optimization.ae_metrics import ae_vessel_metrics
    from carbon_twin.optimization.fuel_choice import (
        optimal_fuel_blend_speed_multi_regulation,
    )
    from carbon_twin.optimization.speed import _DEFAULT_CII_PRESSURE, speed_bounds

    if cii_pressure is None:
        cii_pressure = _DEFAULT_CII_PRESSURE

    cii = CIIEngine(version=cii_version)
    fueleu = FuelEUEngine(version=fueleu_version)
    ets = ETSEngine(version=ets_version)

    # per-vessel low-carbon-fuel price multiplier (heterogeneity, AE robustness)
    _rng = np.random.default_rng(price_het_seed)
    _sigma = math.sqrt(math.log(1.0 + price_het_cv ** 2)) if price_het_cv > 0 else 0.0
    _FOSSIL = {"VLSFO", "HFO", "MGO", "LPG"}

    rows: list[dict[str, Any]] = []
    for r in mrv_df.to_dict(orient="records"):
        imo = str(r.get("imo_number", "")).strip()
        vessel = fleet.get(imo)
        if vessel is None:
            continue
        annual_dist = _mrv_distance(r)
        if not annual_dist or annual_dist < _MIN_CII_DISTANCE_NM:
            continue
        hours = _num(r.get("hours_at_sea"))
        avg_speed = (annual_dist / hours) if hours and hours > 0 else (
            vessel.design_speed_knots or 14.0)
        avg_speed = float(min(max(avg_speed, 6.0), 25.0))

        leg_nm = float(min(annual_dist, 2000.0))
        leg_hours = leg_nm / max(avg_speed, 0.1)
        voy = Voyage(
            voyage_id=f"{imo}_rep", vessel_imo=imo,
            departure_port_id="REP_A", arrival_port_id="REP_B",
            departure_time=dt.datetime(year, 1, 1),
            arrival_time=dt.datetime(year, 1, 1) + dt.timedelta(hours=leg_hours),
            distance_nm=leg_nm, avg_speed_knots=avg_speed, sea_hours=leg_hours,
            eea_scope=EEAScope.INTRA_EEA, ets_scope_factor=1.0,
        )
        charter_per_hour = _charter_rate_per_hour(vessel.ship_type.value)
        v_range = speed_bounds(vessel, avg_speed)

        # heterogeneous low-carbon-fuel price per operator (alt fuels only)
        vessel_prices = fuel_prices
        if _sigma > 0:
            mult = float(_rng.lognormal(mean=-0.5 * _sigma ** 2, sigma=_sigma))
            vessel_prices = {k: (v * mult if k not in _FOSSIL else v)
                             for k, v in fuel_prices.items()}

        res = optimal_fuel_blend_speed_multi_regulation(
            vessel, voy, vessel_prices, eua_ref, fuel_factors or {},
            cii_engine=cii, fueleu_engine=fueleu, ets_engine=ets,
            year=year, annual_distance_nm=annual_dist, speed_range=v_range,
            delay_cost_per_hour=charter_per_hour, scheduled_hours=leg_hours,
            cii_pressure=cii_pressure, extended_fuels=extended_fuels,
        )
        prof = res.get("profile") if res else None
        if prof is None or len(prof) == 0:
            continue
        annual_scale = annual_dist / max(leg_nm, 1.0)   # leg → annual (≥1)
        for m in ae_vessel_metrics(prof, eua_ref, target_euas, breakeven_grid, rci_eps):
            m.update({"vessel_imo": imo, "ship_type": vessel.ship_type.value,
                      "year": year, "annual_scale": annual_scale})
            rows.append(m)

    return pd.DataFrame(rows)


def _num(v: Any) -> float | None:
    """Coerce to float, returning None on failure/NaN."""
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _mrv_distance(r: dict[str, Any]) -> float | None:
    """Distance in n.miles, back-calculated when MRV omits the direct column.

    THETIS-MRV 2022+ exports drop 'Distance travelled' and only give efficiency
    ratios. Units: fuel [m tonnes], efficiency [kg / n mile], so
        distance_nm = fuel_tonnes × 1000 (kg/t) ÷ fuel_per_nm (kg/nm),
    falling back to the CO2-per-distance ratio.
    """
    d = _num(r.get("distance_nm"))
    if d and d > 0:
        return d
    fuel = _num(r.get("total_fuel_tonnes"))
    eff = _num(r.get("fuel_per_nm_kg"))
    if fuel and fuel > 0 and eff and eff > 0:
        return fuel * 1e3 / eff
    co2 = _num(r.get("total_co2_tonnes"))
    co2_eff = _num(r.get("co2_per_nm_kg"))
    if co2 and co2 > 0 and co2_eff and co2_eff > 0:
        return co2 * 1e3 / co2_eff
    return None


def _proxy_baseline_annual_co2(
    vessel: Any, voy: Any, avg_speed: float, annual_dist: float, leg_nm: float,
    fuel_factors: dict,
) -> float | None:
    """Propulsion-model annual TtW CO2 (tonnes) at the base fuel and operating
    speed — the denominator for the MRV-calibration scale."""
    from carbon_twin.optimization.speed import _fuel_at_speed
    try:
        fc = _fuel_at_speed(vessel, voy, avg_speed, FuelType.VLSFO, fuel_factors)
    except Exception:
        return None
    ef = fuel_factors.get(FuelType.VLSFO)
    co2_per_t = ef.ttw_co2_g_per_g_fuel if ef else 3.114
    annual_scale = annual_dist / max(leg_nm, 1.0)
    return fc["total_tonnes"] * co2_per_t * annual_scale


# ---------------------------------------------------------------------------
# Compliance assembly  (run-regulations, AIS path)
# ---------------------------------------------------------------------------

def assemble_compliance(
    voyages_df: pd.DataFrame,
    proxy_df: pd.DataFrame,
    calibrator: Any,
    fleet: dict[str, Vessel],
    year: int,
    eua_price: float,
    cii_version: str,
    fueleu_version: str,
    ets_version: str,
    fuel_type: FuelType = FuelType.VLSFO,
) -> pd.DataFrame:
    """Run CII + FuelEU + EU ETS for every vessel and return the state vectors.

    Parameters
    ----------
    voyages_df : per-voyage rows (for ETS scope, matched by voyage_id)
    proxy_df   : per-voyage proxy emissions from ``build_proxy_emissions``
    calibrator : fitted MRVCalibrator (job 02). If None, proxy is used uncalibrated.
    fleet      : IMO → Vessel
    eua_price  : EUR / tCO2e
    """
    # 1. Calibrate per-voyage proxy emissions to MRV-consistent values
    if calibrator is not None:
        adjusted = calibrator.transform(proxy_df)
        emis_col = "adjusted_emission_tonnes"
        if emis_col not in adjusted.columns:
            adjusted = proxy_df.copy()
            adjusted[emis_col] = adjusted["proxy_emission_tonnes"]
    else:
        adjusted = proxy_df.copy()
        adjusted["adjusted_emission_tonnes"] = adjusted["proxy_emission_tonnes"]
        emis_col = "adjusted_emission_tonnes"

    # Scale factor per voyage: how much calibration moved the proxy
    adjusted["_scale"] = adjusted[emis_col] / adjusted["proxy_emission_tonnes"].replace(0, 1.0)

    # 2. Engines (regulation-as-code; version drives behaviour)
    cii = CIIEngine(version=cii_version)
    fueleu = FuelEUEngine(version=fueleu_version)
    ets = ETSEngine(version=ets_version)

    # 3. Build Voyage objects + per-voyage EmissionEstimate (for ETS)
    voyage_objs: list[Voyage] = []
    emissions: list[EmissionEstimate] = []
    scale_by_voyage = dict(zip(adjusted["voyage_id"], adjusted["_scale"]))
    energy_by_voyage = dict(zip(adjusted["voyage_id"], adjusted["energy_mj"]))
    adj_by_voyage = dict(zip(adjusted["voyage_id"], adjusted[emis_col]))

    for record in voyages_df.to_dict(orient="records"):
        voy = _row_to_voyage(record)
        if voy is None or voy.voyage_id not in adj_by_voyage:
            continue
        voyage_objs.append(voy)
        co2 = float(adj_by_voyage[voy.voyage_id])
        emissions.append(EmissionEstimate(
            vessel_imo=voy.vessel_imo,
            voyage_id=voy.voyage_id,
            year=year,
            co2_tonnes=co2,
            co2e_ttw_tonnes=co2,
            energy_mj=float(energy_by_voyage.get(voy.voyage_id, 0.0)),
            estimation_method="mrv_calibrated" if calibrator is not None else "proxy",
        ))

    # 4. ETS — per vessel, covered emissions across voyages
    ets_results = ets.evaluate(voyage_objs, emissions, eua_price, year)
    ets_by_imo = {r.vessel_imo: r for r in ets_results}

    # 5. Annual aggregates per vessel
    annual = (
        adjusted.groupby("vessel_imo")
        .agg(
            annual_co2_tonnes=(emis_col, "sum"),
            distance_nm=("distance_nm", "sum"),
            energy_mj=("energy_mj", "sum"),
            num_voyages=("voyage_id", "nunique"),
        )
        .reset_index()
    )

    # 6. CII + FuelEU per vessel → assemble state vector
    rows: list[dict[str, Any]] = []
    for _, a in annual.iterrows():
        imo = str(a["vessel_imo"])
        vessel = fleet.get(imo)
        if vessel is None:
            continue

        annual_co2 = float(a["annual_co2_tonnes"])
        distance = float(a["distance_nm"])
        energy = float(a["energy_mj"])

        # CII (needs positive capacity + distance)
        cii_rating = cii_attained = cii_required = None
        try:
            if distance > 0:
                cr = cii.evaluate(vessel, annual_co2, distance, year)
                cii_rating = cr.rating.value
                cii_attained = cr.attained_cii
                cii_required = cr.required_cii
        except Exception:
            pass

        # FuelEU (single-fuel record for the vessel's annual energy)
        fueleu_balance = fueleu_penalty = None
        try:
            fr = FuelRecord(vessel_imo=imo, fuel_type=fuel_type,
                            mass_tonnes=0.0, energy_mj=energy)
            fe = fueleu.evaluate(vessel, [fr], energy, year)
            fueleu_balance = fe.compliance_balance
            fueleu_penalty = fe.penalty_eur
        except Exception:
            pass

        # ETS
        ets_r = ets_by_imo.get(imo)
        ets_covered = ets_r.covered_emissions_tco2e if ets_r else 0.0
        ets_cost = ets_r.ets_cost_eur if ets_r else 0.0

        total_cost = (ets_cost or 0.0) + (fueleu_penalty or 0.0)

        rows.append({
            "vessel_imo": imo,
            "ship_type": vessel.ship_type.value,
            "year": year,
            "annual_co2_tonnes": round(annual_co2, 2),
            "distance_nm": round(distance, 1),
            "num_voyages": int(a["num_voyages"]),
            "cii_attained": cii_attained,
            "cii_required": cii_required,
            "cii_rating": cii_rating,
            "fueleu_balance": fueleu_balance,
            "fueleu_penalty_eur": fueleu_penalty,
            "ets_covered_tco2e": round(ets_covered, 2),
            "ets_cost_eur": round(ets_cost, 2),
            "total_compliance_cost_eur": round(total_cost, 2),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Optimization  (optimize)  — single-regulation vs joint → RCI
# ---------------------------------------------------------------------------

def run_optimization(
    compliance_df: pd.DataFrame,
    voyages_df: pd.DataFrame,
    fleet: dict[str, Vessel],
    year: int,
    eua_price: float,
    fuel_price: float,
    cii_version: str,
    fueleu_version: str,
    ets_version: str,
    fuel_factors: dict | None,
    fuel_type: FuelType = FuelType.VLSFO,
) -> pd.DataFrame:
    """For each vessel, optimise speed jointly vs per-regulation; compute RCI.

    The Regulation Coupling Index (RCI) is the fractional cost gap between
    optimising each regulation alone and optimising all jointly. A positive RCI
    means joint optimisation recovers material savings the siloed view misses.
    """
    from carbon_twin.optimization.speed import (
        optimal_speed_multi_regulation,
        optimal_speed_single_voyage,
        speed_bounds,
    )

    cii = CIIEngine(version=cii_version)
    fueleu = FuelEUEngine(version=fueleu_version)
    ets = ETSEngine(version=ets_version)

    # Representative voyage per vessel: the longest (most cost leverage)
    voyages_by_imo: dict[str, Voyage] = {}
    annual_dist: dict[str, float] = {}
    for record in voyages_df.to_dict(orient="records"):
        voy = _row_to_voyage(record)
        if voy is None:
            continue
        annual_dist[voy.vessel_imo] = annual_dist.get(voy.vessel_imo, 0.0) + voy.distance_nm
        cur = voyages_by_imo.get(voy.vessel_imo)
        if cur is None or voy.distance_nm > cur.distance_nm:
            voyages_by_imo[voy.vessel_imo] = voy

    rows: list[dict[str, Any]] = []
    for imo, voy in voyages_by_imo.items():
        vessel = fleet.get(imo)
        if vessel is None or voy.distance_nm <= 0:
            continue

        v_range = speed_bounds(vessel, voy.avg_speed_knots)

        # Joint (all regulations active)
        joint = optimal_speed_multi_regulation(
            vessel, voy, fuel_price, eua_price, fuel_factors or {},
            cii_engine=cii, fueleu_engine=fueleu, ets_engine=ets,
            year=year, fuel_type=fuel_type, speed_range=v_range,
            annual_distance_nm=annual_dist.get(imo),
        )

        # Single-regulation baseline: optimise for fuel+ETS only (ignore CII/FuelEU)
        single = optimal_speed_single_voyage(
            vessel, voy, fuel_price, eua_price, fuel_factors or {},
            fuel_type=fuel_type, speed_range=v_range,
        )

        # Cost of the single-regulation speed under the *full* regulatory burden:
        # re-evaluate joint cost at the single optimum by locating nearest grid row
        joint_profile = joint["speed_profile"]
        single_speed = single["optimal_speed"]
        nearest = joint_profile.iloc[
            (joint_profile["speed_knots"] - single_speed).abs().argsort().iloc[0]
        ]
        cost_siloed = float(nearest["total_cost"])
        cost_joint = float(joint["total_cost"])
        rci = (cost_siloed - cost_joint) / cost_siloed if cost_siloed > 0 else 0.0

        rows.append({
            "vessel_imo": imo,
            "ship_type": vessel.ship_type.value,
            "joint_optimal_speed": round(float(joint["optimal_speed"]), 2),
            "siloed_optimal_speed": round(float(single_speed), 2),
            "cost_joint_eur": round(cost_joint, 2),
            "cost_siloed_eur": round(cost_siloed, 2),
            "rci": round(rci, 4),
            "ets_cost_eur": round(joint["ets_cost"], 2),
            "fueleu_penalty_eur": round(joint["fueleu_penalty"], 2),
            "cii_penalty_eur": round(joint["cii_penalty"], 2),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Monte Carlo  (run-scenarios)
# ---------------------------------------------------------------------------

def run_monte_carlo(
    compliance_df: pd.DataFrame,
    scenario_cfg: dict,
    n_samples: int,
    eua_price: float,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Propagate EUA / fuel-price uncertainty through the fleet compliance cost.

    Uses the fleet's deterministic compliance cost as the base, then perturbs
    EUA price (and a fuel-price proxy) per the scenario's uncertainty block.
    Returns (raw_samples_df, summary_dict_with_cvar).
    """
    from carbon_twin.scenarios.monte_carlo import MonteCarloEngine

    unc = scenario_cfg.get("uncertainty", {})
    eua_std_pct = float(unc.get("eua_price_std_pct", 0.15))

    base_ets = float(compliance_df["ets_cost_eur"].sum())
    base_fueleu = float(compliance_df["fueleu_penalty_eur"].fillna(0).sum())
    base_eua = max(eua_price, 1e-6)

    def evaluate(params: dict) -> dict:
        # ETS cost scales linearly with EUA price; FuelEU penalty independent of EUA
        eua = float(params.get("eua_price", base_eua))
        ets_cost = base_ets * (eua / base_eua)
        total = ets_cost + base_fueleu
        return {
            "total_compliance_cost_eur": total,
            "ets_cost_eur": ets_cost,
            "fueleu_penalty_eur": base_fueleu,
        }

    mc = MonteCarloEngine(base_params={"eua_price": base_eua}, n_samples=n_samples, seed=seed)
    mc.add_uncertainty("eua_price", "normal", mean=base_eua, std=base_eua * eua_std_pct)
    raw = mc.run(evaluate)

    summary = mc.summary_statistics("total_compliance_cost_eur")
    try:
        summary["cvar_95"] = mc.cvar(alpha=0.05, metric="total_compliance_cost_eur")
    except Exception:
        summary["cvar_95"] = None
    summary["base_total_cost_eur"] = base_ets + base_fueleu
    return raw, summary
