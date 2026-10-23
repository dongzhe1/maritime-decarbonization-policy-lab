from __future__ import annotations

from pathlib import Path
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

_LCV_MJ_PER_KG = 40.2

_MIN_CII_DISTANCE_NM = 500.0
_MAX_PLAUSIBLE_ATTAINED_CII = 100.0

_CHARTER_USD_PER_DAY: dict[str, float] = {
    "bulk_carrier": 14000, "tanker": 26000, "container_ship": 42000,
    "general_cargo": 8000, "gas_carrier": 45000, "lng_carrier": 70000,
    "ro_ro_cargo": 20000, "refrigerated_cargo": 16000, "cruise_passenger": 60000,
    "ro_pax": 28000, "vehicle_carrier": 24000, "offshore": 18000, "other": 15000,
}


_USD_TO_EUR: float = load_usd_to_eur()


def _charter_rate_per_hour(ship_type: str) -> float:
    return _CHARTER_USD_PER_DAY.get(ship_type, 15000) / 24.0 * _USD_TO_EUR


def assemble_compliance_from_mrv(
    mrv_df: pd.DataFrame,
    fleet: dict[str, Vessel],
    year: int,
    eua_price: float,
    cii_version: str,
    fueleu_version: str,
    ets_version: str,
    fuel_type: FuelType = FuelType.VLSFO,
    scope_mode: str = "full",
    fueleu_emission_factors: dict[str, float] | None = None,
) -> pd.DataFrame:
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
        co2_ets = r.get("co2_ets_tonnes")
        co2_ets = _num(co2_ets) if co2_ets is not None and pd.notna(co2_ets) else annual_co2

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

        ets_cost = (co2_ets or 0.0) * phase_in * eua_price

        fueleu_balance = fueleu_penalty = None
        scope_w, _ = _scope_weight(r, scope_mode)
        if fuel_t and fuel_t > 0:
            energy_mj = fuel_t * 1e3 * _LCV_MJ_PER_KG * scope_w
            try:
                fr = FuelRecord(vessel_imo=imo, fuel_type=fuel_type,
                                mass_tonnes=fuel_t, energy_mj=energy_mj)
                fe = fueleu.evaluate(vessel, [fr], energy_mj, year,
                                     emission_factors=fueleu_emission_factors)
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
        if scope_mode != "full":
            rows[-1]["scope_weight"] = round(scope_w, 6)

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
    scope_mode: str = "full",
    report_annual: bool = False,
    fueleu_factors_from_config: bool = False,
    return_rows: bool = False,
) -> pd.DataFrame | list[dict[str, Any]]:
    from carbon_twin.optimization.speed import (
        _DEFAULT_CII_PRESSURE,
        optimal_speed_multi_regulation,
        optimal_speed_single_voyage,
        speed_bounds,
    )

    if cii_pressure is None:
        cii_pressure = _DEFAULT_CII_PRESSURE
    import datetime as dt

    from carbon_twin.optimization.fuel_choice import fueleu_factor_table

    fueleu_table = fueleu_factor_table(fuel_factors) if fueleu_factors_from_config else None

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
        if not annual_dist or annual_dist < _MIN_CII_DISTANCE_NM:
            continue
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
        scope_w, _ = _scope_weight(r, scope_mode)
        if scope_mode != "full":
            voy = voy.model_copy(update={"ets_scope_factor": scope_w})
        annual_scale = annual_dist / max(leg_nm, 1.0)

        charter_per_hour = _charter_rate_per_hour(vessel.ship_type.value)

        v_range = speed_bounds(vessel, avg_speed)

        rho_mult = (cii_rho_multiplier.get(vessel.ship_type.value, 1.0)
                    if isinstance(cii_rho_multiplier, dict) else float(cii_rho_multiplier))

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
            energy_scale=spd_energy_scale, fueleu_scope_factor=scope_w,
            fueleu_emission_factors=fueleu_table)
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
            "delta_cost_eur": round(delta_c, 2),
            "rci": round(rci, 4),
            "rci_joint": round(rci_joint, 4),
            "fuel_cost_joint_eur": round(float(joint_row["fuel_cost"]), 2),
            "ets_cost_joint_eur": round(float(joint_row["ets_cost"]), 2),
            "fueleu_penalty_joint_eur": round(float(joint_row["fueleu_penalty"]), 2),
            "cii_consequence_joint_eur": round(float(joint_row["cii_penalty"]), 2),
            "delay_cost_joint_eur": round(float(joint_row["delay_cost"]), 2),
            "fuel_cost_siloed_eur": round(float(silo_row["fuel_cost"]), 2),
            "ets_cost_siloed_eur": round(float(silo_row["ets_cost"]), 2),
            "fueleu_penalty_siloed_eur": round(float(silo_row["fueleu_penalty"]), 2),
            "cii_consequence_siloed_eur": round(float(silo_row["cii_penalty"]), 2),
            "delay_cost_siloed_eur": round(float(silo_row["delay_cost"]), 2),
            "ets_cost_eur": round(float(joint_row["ets_cost"]), 2),
            "fueleu_penalty_eur": round(float(joint_row["fueleu_penalty"]), 2),
            "cii_penalty_eur": round(float(joint_row["cii_penalty"]), 2),
        })
        if scope_mode != "full" or report_annual:
            row = rows[-1]
            row["scope_weight"] = round(scope_w, 6)
            row["annual_scale"] = round(annual_scale, 6)
            row["cost_basis"] = "annual" if report_annual else "leg"
            if report_annual:
                for k in _SPEED_EUR_COLUMNS:
                    row[k] = round(row[k] * annual_scale, 2)

    return rows if return_rows else pd.DataFrame(rows)


def build_fuel_optimization_context(
    r: dict[str, Any],
    fleet: dict[str, Vessel],
    year: int,
    fuel_factors: dict,
    calibrate_to_mrv: bool = False,
    charter_multiplier: float = 1.0,
    scope_mode: str = "full",
) -> dict[str, Any] | None:
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
    scope_w, scope_src = _scope_weight(r, scope_mode)
    if scope_mode in ("mrv", "mrv_bd"):
        voy = voy.model_copy(update={"ets_scope_factor": scope_w})

    charter_per_hour = _charter_rate_per_hour(vessel.ship_type.value) * charter_multiplier

    v_range = speed_bounds(vessel, avg_speed)

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
        "scope_weight": scope_w,
        "scope_weight_source": scope_src,
        "scope_allocation": _SCOPE_ALLOCATION[scope_mode],
        "annual_scale": annual_dist / max(leg_nm, 1.0),
    }


_FUEL_DROPIN: set[FuelType] = {FuelType.VLSFO, FuelType.MGO, FuelType.BIODIESEL_FAME}
_METHANOL: set[FuelType] = {FuelType.METHANOL_BIO, FuelType.METHANOL_E}


def retrofit_annual_cost(
    vessel: Vessel,
    capex_eur_per_kw: dict[str, float] | None,
    crf: float,
    fuel_capability: dict[str, dict[str, bool]] | None = None,
) -> dict[FuelType, float] | None:
    if not capex_eur_per_kw:
        return None
    kw = vessel.main_engine_power_kw or 0.0
    cap = (fuel_capability or {}).get(str(vessel.imo_number).strip())
    out: dict[FuelType, float] = {}
    for fuel, per_kw in capex_eur_per_kw.items():
        ft = FuelType(fuel)
        if cap is not None:
            capable = bool(cap.get("lng_capable")) if ft == FuelType.LNG else \
                bool(cap.get("methanol_capable")) if ft in _METHANOL else False
        else:
            capable = ft == FuelType.LNG and vessel.ship_type.value in _FUEL_GAS_CAPABLE
        if not capable and per_kw > 0 and kw > 0:
            out[ft] = crf * per_kw * kw
    return out or None


def load_fuel_capability(path: str | Path | None) -> dict[str, dict[str, bool]] | None:
    if not path:
        return None
    df = pd.read_csv(path, dtype={"imo_number": str})
    return {str(r.imo_number).strip(): {"lng_capable": bool(r.lng_capable),
                                        "methanol_capable": bool(r.methanol_capable)}
            for r in df.itertuples(index=False)}
_FUEL_GAS_CAPABLE: set[str] = {"gas_carrier", "lng_carrier"}


def feasible_fuel_options(
    vessel: Vessel,
    fuel_feasibility: str,
    extended_fuels: bool,
    exclude_fuels: set | None,
    fuel_capability: dict[str, dict[str, bool]] | None = None,
) -> list[FuelType] | None:
    from carbon_twin.optimization.fuel_choice import fuel_menu

    if fuel_feasibility == "continuous_base":
        feas: list[FuelType] | None = None
    else:
        menu = fuel_menu(vessel, extended=extended_fuels)
        if fuel_feasibility == "dropin_only":
            feas = [f for f in menu if f in _FUEL_DROPIN]
        elif fuel_feasibility == "compatible_switch":
            allowed = set(_FUEL_DROPIN)
            if vessel.ship_type.value in _FUEL_GAS_CAPABLE:
                allowed.add(FuelType.LNG)
            feas = [f for f in menu if f in allowed]
        elif fuel_feasibility == "engine_capability":
            if fuel_capability is None:
                raise ValueError("engine_capability needs a fuel_capability map")
            cap = fuel_capability.get(str(vessel.imo_number).strip(), {})
            allowed = set(_FUEL_DROPIN)
            if cap.get("lng_capable"):
                allowed.add(FuelType.LNG)
            if cap.get("methanol_capable"):
                allowed |= {FuelType.METHANOL_BIO, FuelType.METHANOL_E}
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
    scope_mode: str = "full",
    report_annual: bool = False,
    fueleu_factors_from_config: bool = False,
    base_fuels: list | None = None,
    exact_blend: bool = False,
    fuel_capability: dict[str, dict[str, bool]] | None = None,
    retrofit_capex_eur_per_kw: dict[str, float] | None = None,
    capex_crf: float = 0.15,
    return_rows: bool = False,
) -> pd.DataFrame | list[dict[str, Any]]:
    from carbon_twin.optimization.fuel_choice import (
        fueleu_factor_table,
        optimal_fuel_blend_speed_multi_regulation,
        optimal_fuel_speed_multi_regulation,
    )
    from carbon_twin.optimization.speed import _DEFAULT_CII_PRESSURE

    if cii_pressure is None:
        cii_pressure = _DEFAULT_CII_PRESSURE
    fueleu_table = fueleu_factor_table(fuel_factors) if fueleu_factors_from_config else None
    base_fuel_types = [FuelType(b) for b in (base_fuels or [])]

    cii = CIIEngine(version=cii_version)
    fueleu = FuelEUEngine(version=fueleu_version)
    ets = ETSEngine(version=ets_version)

    rows: list[dict[str, Any]] = []
    for r in mrv_df.to_dict(orient="records"):
        ctx = build_fuel_optimization_context(
            r, fleet, year, fuel_factors or {}, calibrate_to_mrv, charter_multiplier,
            scope_mode=scope_mode)
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
            scheduled_hours=voy.sea_hours,
            cii_pressure=cii_pressure, extended_fuels=extended_fuels,
        )
        if blend:
            res = optimal_fuel_blend_speed_multi_regulation(
                vessel, voy, fuel_prices, eua_price, fuel_factors or {},
                fuel_options=feasible_fuel_options(
                    vessel, fuel_feasibility, extended_fuels, exclude_fuels,
                    fuel_capability),
                silo_carbon_basis=silo_carbon_basis,
                include_cii_in_objective=include_cii_in_objective,
                max_biofuel_share=max_biofuel_share,
                energy_scale=energy_scale,
                fueleu_price_fraction=fueleu_price_fraction,
                planner_fueleu_internalization=planner_fueleu_internalization,
                fueleu_scope_factor=ctx["scope_weight"],
                fueleu_emission_factors=fueleu_table,
                report_annual=report_annual,
                base_fuels=base_fuel_types,
                scope_allocation=ctx["scope_allocation"],
                exact_blend=exact_blend,
                fixed_annual_cost=retrofit_annual_cost(
                    vessel, retrofit_capex_eur_per_kw, capex_crf, fuel_capability),
                **common)
            if not res:
                continue
            row = {
                "joint_fuel": res["joint_alt_fuel"],
                "joint_blend_pct": round(res["joint_blend_pct"], 3),
                "siloed_blend_pct": round(res["siloed_blend_pct"], 3),
                "blends": res["blends"],
                "energy_scale": round(energy_scale, 4),
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
        if blend and (scope_mode != "full" or report_annual or fueleu_factors_from_config
                      or any(b != FuelType.VLSFO for b in base_fuel_types)):
            row.update({
                "joint_base_fuel": res["joint_base_fuel"],
                "siloed_base_fuel": res["siloed_base_fuel"],
                "joint_fuel_pathway": res["joint_fuel_pathway"],
                "siloed_fuel_pathway": res["siloed_fuel_pathway"],
                "scope_weight": round(ctx["scope_weight"], 6),
                "scope_weight_source": ctx["scope_weight_source"],
                "annual_scale": round(res["annual_scale"], 6),
                "cost_basis": res["cost_basis"],
            })
        rows.append(row)

    return rows if return_rows else pd.DataFrame(rows)


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
        annual_scale = annual_dist / max(leg_nm, 1.0)
        for m in ae_vessel_metrics(prof, eua_ref, target_euas, breakeven_grid, rci_eps):
            m.update({"vessel_imo": imo, "ship_type": vessel.ship_type.value,
                      "year": year, "annual_scale": annual_scale})
            rows.append(m)

    return pd.DataFrame(rows)


def _num(v: Any) -> float | None:
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


_SPEED_EUR_COLUMNS: tuple[str, ...] = (
    "cost_joint_eur", "cost_siloed_eur", "delta_cost_eur",
    "fuel_cost_joint_eur", "ets_cost_joint_eur", "fueleu_penalty_joint_eur",
    "cii_consequence_joint_eur", "delay_cost_joint_eur",
    "fuel_cost_siloed_eur", "ets_cost_siloed_eur", "fueleu_penalty_siloed_eur",
    "cii_consequence_siloed_eur", "delay_cost_siloed_eur",
    "ets_cost_eur", "fueleu_penalty_eur", "cii_penalty_eur",
)


_SCOPE_ALLOCATION: dict[str, str] = {
    "full": "uniform", "mrv": "uniform", "mrv_routed": "routed",
    "mrv_priority": "fueleu_priority", "mrv_routed_voyage": "routed_voyage",
    "mrv_priority_bd": "fueleu_priority",
    "mrv_bd": "uniform", "mrv_routed_voyage_bd": "routed_voyage",
}

_BREAKDOWN_KEYS: dict[str, tuple[str, ...]] = {
    "between": ("between ports",),
    "departed": ("departed from ports",),
    "arrived": ("to ports",),
}


def _norm_col(c: Any) -> str:
    return " ".join(str(c).replace("₂", "2").lower().split())


def mrv_breakdown_columns(columns) -> dict[str, str] | None:
    co2 = {c: n for c in columns if (n := _norm_col(c)).startswith("co2 emissions")}
    found: dict[str, str] = {}
    for key, needles in _BREAKDOWN_KEYS.items():
        hits = [c for c, n in co2.items() if any(k in n for k in needles)
                and not (key == "arrived" and ("between" in n or "departed" in n))]
        if len(hits) != 1:
            return None
        found[key] = hits[0]
    within = [c for c, n in co2.items() if "within ports" in n and "at berth" not in n]
    berth = [c for c, n in co2.items() if "at berth" in n]
    if len(within) > 1 or len(berth) > 1 or not (within or berth):
        return None
    if within:
        found["port"] = within[0]
    if berth:
        found["berth"] = berth[0]
    return found


def mrv_port_term(v: dict[str, float | None]) -> float | None:
    ports = [v[k] for k in ("port", "berth") if v.get(k) is not None]
    if not ports or min(ports) < 0:
        return None
    return max(ports)


def mrv_breakdown_scope(r: dict[str, Any], cols: dict[str, str] | None = None) -> float | None:
    cols = cols or mrv_breakdown_columns(r.keys())
    if cols is None:
        return None
    v = {k: _num(r.get(c)) for k, c in cols.items()}
    voyages = [v["between"], v["departed"], v["arrived"]]
    port = mrv_port_term(v)
    if port is None or any(x is None or x < 0 for x in voyages):
        return None
    total = sum(voyages) + port
    if total <= 0:
        return None
    return float((v["between"] + port + 0.5 * (v["departed"] + v["arrived"])) / total)


def _scope_weight(r: dict[str, Any], scope_mode: str) -> tuple[float, str]:
    if scope_mode == "full":
        return 1.0, "full"
    if scope_mode not in _SCOPE_ALLOCATION or scope_mode == "full":
        raise ValueError(f"unknown scope_mode: {scope_mode!r}")
    if scope_mode.endswith("_bd"):
        s_bd = mrv_breakdown_scope(r)
        if s_bd is not None:
            return min(max(s_bd, 0.0), 1.0), "mrv_voyage_breakdown"
        w, src = _scope_weight(r, scope_mode[:-3])
        return w, f"{src}_bd_fallback"
    total = _num(r.get("total_co2_tonnes"))
    ets = _num(r.get("co2_ets_tonnes"))
    if not total or total <= 0 or ets is None or ets < 0:
        return 1.0, "fallback_full"
    return float(min(max(ets / total, 0.0), 1.0)), "mrv_ets_share"


def _mrv_distance(r: dict[str, Any]) -> float | None:
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
    from carbon_twin.optimization.speed import _fuel_at_speed
    try:
        fc = _fuel_at_speed(vessel, voy, avg_speed, FuelType.VLSFO, fuel_factors)
    except Exception:
        return None
    ef = fuel_factors.get(FuelType.VLSFO)
    co2_per_t = ef.ttw_co2_g_per_g_fuel if ef else 3.114
    annual_scale = annual_dist / max(leg_nm, 1.0)
    return fc["total_tonnes"] * co2_per_t * annual_scale


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

    adjusted["_scale"] = adjusted[emis_col] / adjusted["proxy_emission_tonnes"].replace(0, 1.0)

    cii = CIIEngine(version=cii_version)
    fueleu = FuelEUEngine(version=fueleu_version)
    ets = ETSEngine(version=ets_version)

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

    ets_results = ets.evaluate(voyage_objs, emissions, eua_price, year)
    ets_by_imo = {r.vessel_imo: r for r in ets_results}

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

    rows: list[dict[str, Any]] = []
    for _, a in annual.iterrows():
        imo = str(a["vessel_imo"])
        vessel = fleet.get(imo)
        if vessel is None:
            continue

        annual_co2 = float(a["annual_co2_tonnes"])
        distance = float(a["distance_nm"])
        energy = float(a["energy_mj"])

        cii_rating = cii_attained = cii_required = None
        try:
            if distance > 0:
                cr = cii.evaluate(vessel, annual_co2, distance, year)
                cii_rating = cr.rating.value
                cii_attained = cr.attained_cii
                cii_required = cr.required_cii
        except Exception:
            pass

        fueleu_balance = fueleu_penalty = None
        try:
            fr = FuelRecord(vessel_imo=imo, fuel_type=fuel_type,
                            mass_tonnes=0.0, energy_mj=energy)
            fe = fueleu.evaluate(vessel, [fr], energy, year)
            fueleu_balance = fe.compliance_balance
            fueleu_penalty = fe.penalty_eur
        except Exception:
            pass

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
    from carbon_twin.optimization.speed import (
        optimal_speed_multi_regulation,
        optimal_speed_single_voyage,
        speed_bounds,
    )

    cii = CIIEngine(version=cii_version)
    fueleu = FuelEUEngine(version=fueleu_version)
    ets = ETSEngine(version=ets_version)

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

        joint = optimal_speed_multi_regulation(
            vessel, voy, fuel_price, eua_price, fuel_factors or {},
            cii_engine=cii, fueleu_engine=fueleu, ets_engine=ets,
            year=year, fuel_type=fuel_type, speed_range=v_range,
            annual_distance_nm=annual_dist.get(imo),
        )

        single = optimal_speed_single_voyage(
            vessel, voy, fuel_price, eua_price, fuel_factors or {},
            fuel_type=fuel_type, speed_range=v_range,
        )

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


def run_monte_carlo(
    compliance_df: pd.DataFrame,
    scenario_cfg: dict,
    n_samples: int,
    eua_price: float,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from carbon_twin.scenarios.monte_carlo import MonteCarloEngine

    unc = scenario_cfg.get("uncertainty", {})
    eua_std_pct = float(unc.get("eua_price_std_pct", 0.15))

    base_ets = float(compliance_df["ets_cost_eur"].sum())
    base_fueleu = float(compliance_df["fueleu_penalty_eur"].fillna(0).sum())
    base_eua = max(eua_price, 1e-6)

    def evaluate(params: dict) -> dict:
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
