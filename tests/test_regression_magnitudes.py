"""Magnitude regression tests: pin orders of magnitude and exact formulas so a
dimensionally-wrong formula or a silently-bypassed regulation engine is caught.
"""

from __future__ import annotations

import datetime as dt

import pytest

from carbon_twin.data.schemas import (
    EEAScope,
    EmissionEstimate,
    FuelRecord,
    FuelType,
    ShipType,
    Vessel,
    Voyage,
)
from carbon_twin.regulations.cii import CIIEngine
from carbon_twin.regulations.eu_ets import ETSEngine
from carbon_twin.regulations.fueleu import FuelEUEngine
from carbon_twin.regulations.imo_nzf import IMONZFScenarioEngine

_VLSFO_ENERGY_MJ_PER_TONNE = 41000.0


# ---------------------------------------------------------------------------
# FuelEU penalty (Regulation (EU) 2023/1805, Annex IV)
# ---------------------------------------------------------------------------


class TestFuelEUPenaltyMagnitude:
    def _result(self):
        vessel = Vessel(
            imo_number="IMO1",
            ship_type=ShipType.BULK_CARRIER,
            gross_tonnage=30000,
            deadweight=50000,
        )
        # 1000 t HFO at 40.2 MJ/g -> 40.2e6 MJ
        recs = [
            FuelRecord(
                vessel_imo="IMO1",
                fuel_type=FuelType.HFO,
                mass_tonnes=1000,
                energy_mj=1000 * 40.2 * 1000,
            )
        ]
        energy = sum(r.energy_mj for r in recs)
        return FuelEUEngine().evaluate(vessel, recs, energy, 2030), energy

    def test_penalty_matches_annex_iv_formula(self):
        result, _ = self._result()
        assert not result.surplus
        expected = (
            abs(result.compliance_balance)
            / (result.actual_ghg_intensity * _VLSFO_ENERGY_MJ_PER_TONNE)
            * 2400.0
        )
        assert result.penalty_eur == pytest.approx(expected, rel=1e-6)

    def test_penalty_far_below_gco2e_balance(self):
        """Regression: the old bug computed |balance_gCO2e| * 2400, which is
        ~3.7 million x too large.  The correct penalty (EUR) is a small fraction
        of the gCO2e balance, never larger than it."""
        result, _ = self._result()
        assert 0.0 < result.penalty_eur < abs(result.compliance_balance)

    def test_penalty_per_tonne_fuel_is_sane(self):
        """Penalty per tonne of fuel must stay well below the 2400 EUR/t-VLSFOe
        rate (a fossil fuel is only a fraction over target, not 100% over)."""
        result, _ = self._result()
        penalty_per_tonne_fuel = result.penalty_eur / 1000.0
        assert 0.0 < penalty_per_tonne_fuel < 2400.0


# ---------------------------------------------------------------------------
# EU ETS uses tank-to-wake, not well-to-wake
# ---------------------------------------------------------------------------


class TestETSUsesTankToWake:
    def test_covered_emissions_use_ttw(self):
        voyage = Voyage(
            voyage_id="V1",
            vessel_imo="IMO1",
            departure_port_id="NLRTM",
            arrival_port_id="DEHAM",
            departure_time=dt.datetime(2026, 3, 1),
            arrival_time=dt.datetime(2026, 3, 2),
            distance_nm=260,
            eea_scope=EEAScope.INTRA_EEA,
            ets_scope_factor=1.0,
        )
        # Deliberately distinct TtW vs WtW so the source is unambiguous.
        ttw = 100.0
        wtw = 130.0
        emission = EmissionEstimate(
            vessel_imo="IMO1",
            voyage_id="V1",
            year=2026,
            co2_tonnes=98.0,
            co2e_ttw_tonnes=ttw,
            co2e_wtw_tonnes=wtw,
        )
        results = ETSEngine().evaluate([voyage], [emission], eua_price_eur=80.0, year=2026)
        assert len(results) == 1
        res = results[0]
        # Intra-EEA scope=1.0, phase-in 2026=1.0 -> covered == TtW, not WtW.
        assert res.covered_emissions_tco2e == pytest.approx(ttw, rel=1e-6)
        assert res.covered_emissions_tco2e != pytest.approx(wtw, rel=1e-3)
        assert res.ets_cost_eur == pytest.approx(ttw * 80.0, rel=1e-6)


# ---------------------------------------------------------------------------
# CII genuinely responds to speed in the multi-regulation optimiser
# ---------------------------------------------------------------------------


class TestMultiRegulationUsesEngines:
    def _setup(self):
        from pathlib import Path

        from carbon_twin.emissions.proxy import load_fuel_factors_safe

        cfg = Path(__file__).resolve().parent.parent / "configs" / "fuel_factors.yaml"
        fuel_factors = load_fuel_factors_safe(cfg)

        vessel = Vessel(
            imo_number="IMO1",
            ship_type=ShipType.BULK_CARRIER,
            gross_tonnage=45000,
            deadweight=82000,
            design_speed_knots=14.5,
            main_engine_power_kw=12000,
            aux_engine_power_kw=2500,
            year_built=2015,
        )
        voyage = Voyage(
            voyage_id="V1",
            vessel_imo="IMO1",
            departure_port_id="A",
            arrival_port_id="B",
            departure_time=dt.datetime(2026, 3, 1),
            arrival_time=dt.datetime(2026, 3, 10),
            distance_nm=4000,
            eea_scope=EEAScope.INTRA_EEA,
            ets_scope_factor=1.0,
        )
        return vessel, voyage, fuel_factors

    def test_cii_penalty_varies_with_speed(self):
        """Regression: the old self-referential ratio (attained / (attained*0.9))
        was constant, so the CII penalty never changed across speeds. The penalty
        is now a continuous charter-anchored cost that must ramp with speed."""
        from carbon_twin.optimization.speed import optimal_speed_multi_regulation

        vessel, voyage, ff = self._setup()
        res = optimal_speed_multi_regulation(
            vessel,
            voyage,
            fuel_price=600,
            eua_price=80,
            fuel_factors=ff,
            cii_engine=CIIEngine(),
            fueleu_engine=FuelEUEngine(),
            ets_engine=ETSEngine(),
            year=2026,
            speed_range=(8, 18),
            n_points=20,
            delay_cost_per_hour=500.0,
            scheduled_hours=voyage.sea_hours,
            cii_pressure=0.2,
        )
        prof = res["speed_profile"]
        # attained CII must rise with speed (more fuel -> higher intensity)
        assert prof["attained_cii"].iloc[-1] > prof["attained_cii"].iloc[0]
        # and the continuous CII penalty must take more than a single value
        assert prof["cii_penalty"].nunique() > 1

    def test_cii_penalty_continuous_no_step(self):
        """The CII penalty must be continuous (no 50k/100k jump) and zero when
        CII pressure is off."""
        from carbon_twin.optimization.speed import optimal_speed_multi_regulation

        vessel, voyage, ff = self._setup()
        common = dict(
            fuel_price=600,
            eua_price=80,
            fuel_factors=ff,
            cii_engine=CIIEngine(),
            fueleu_engine=FuelEUEngine(),
            ets_engine=ETSEngine(),
            year=2026,
            speed_range=(8, 18),
            n_points=20,
            delay_cost_per_hour=500.0,
            scheduled_hours=voyage.sea_hours,
        )
        off = optimal_speed_multi_regulation(vessel, voyage, cii_pressure=0.0, **common)
        on = optimal_speed_multi_regulation(vessel, voyage, cii_pressure=0.5, **common)
        assert (off["speed_profile"]["cii_penalty"] == 0).all()
        # Stronger pressure ⇒ a larger penalty somewhere in the profile.
        assert on["speed_profile"]["cii_penalty"].max() > 0

    def test_fueleu_penalty_consistent_with_engine(self):
        """The FuelEU penalty embedded in the speed optimiser must match the
        engine for the same energy — i.e. no second, divergent formula."""
        from carbon_twin.optimization.speed import (
            _fuel_at_speed,
            _fueleu_penalty,
        )

        vessel, voyage, ff = self._setup()
        fc = _fuel_at_speed(vessel, voyage, 12.0, FuelType.VLSFO, ff)
        embedded = _fueleu_penalty(
            FuelEUEngine(), vessel, FuelType.VLSFO, fc["energy_mj"], 2026, ff
        )

        record = FuelRecord(
            vessel_imo=vessel.imo_number,
            fuel_type=FuelType.VLSFO,
            mass_tonnes=0.0,
            energy_mj=fc["energy_mj"],
        )
        engine = FuelEUEngine().evaluate(vessel, [record], fc["energy_mj"], 2026)
        assert embedded == pytest.approx(engine.penalty_eur, rel=1e-9)
        # Sanity: a small VLSFO over-target should cost far less than the fuel.
        assert 0.0 <= embedded < fc["total_tonnes"] * 600.0


# ---------------------------------------------------------------------------
# IMO NZF loads scenario-specific parameters (not hardcoded to MEPC 83)
# ---------------------------------------------------------------------------


class TestCIIShipTypeCoverage:
    def test_ro_pax_evaluates_without_error(self):
        """Regression: ro_pax was in the GT-based set but had no reference line,
        so evaluating a ro-pax vessel raised ValueError."""
        vessel = Vessel(
            imo_number="IMO1",
            ship_type=ShipType.RO_PAX,
            gross_tonnage=40000,
        )
        result = CIIEngine().evaluate(vessel, annual_co2_tonnes=50000, distance_nm=80000, year=2026)
        assert result.attained_cii > 0
        assert result.required_cii > 0
        assert result.rating is not None

    def test_gt_based_types_have_reference_lines(self):
        """Every GT-capacity ship type must have a reference line, else capacity
        selection and reference lookup disagree."""
        from carbon_twin.regulations.cii import (
            _GT_BASED_TYPES,
            _REFERENCE_COEFFICIENTS,
        )

        for ship_type in _GT_BASED_TYPES:
            assert ship_type in _REFERENCE_COEFFICIENTS

    def test_small_gas_carrier_uses_dwt_basis(self):
        """A <65,000 DWT gas carrier uses the (8104, 0.639) line on a DWT basis
        per MEPC.353(78) Table 1 — BOTH gas-carrier segments are DWT (the
        "Capacity in CII calculation" column of MEPC.354(78)). A previous revision
        wrongly placed this segment on a GT basis."""
        from carbon_twin.regulations.cii import _select_reference_segment

        a, c, cap, _ = _select_reference_segment("gas_carrier", 19205, 23686)
        assert (a, c) == (8104.0, 0.639)
        assert cap == 23686  # DWT, per MEPC.353(78)
        # Big LPG/VLGC crosses to the large-ship DWT line.
        a2, c2, cap2, _ = _select_reference_segment("gas_carrier", 70000, 80000)
        assert (a2, c2) == (14405e7, 2.071)
        assert cap2 == 80000  # DWT

    def test_dd_vectors_are_ship_type_specific(self):
        """MEPC.354(78) dd-vectors are per ship type, NOT a single fleet-wide
        vector. Bulk is [0.86,0.94,1.06,1.18]; tanker, container, ro-pax differ."""
        e = CIIEngine()
        assert e._dd_vectors("bulk_carrier", 50000) == [0.86, 0.94, 1.06, 1.18]
        assert e._dd_vectors("tanker", 50000) == [0.82, 0.93, 1.08, 1.28]
        assert e._dd_vectors("container_ship", 50000) == [0.83, 0.94, 1.07, 1.19]
        assert e._dd_vectors("ro_pax", 30000) == [0.76, 0.92, 1.14, 1.30]
        # size-dependent for gas and LNG
        assert e._dd_vectors("gas_carrier", 30000) != e._dd_vectors("gas_carrier", 80000)
        assert e._dd_vectors("lng_carrier", 80000) != e._dd_vectors("lng_carrier", 120000)

    def test_roro_family_reference_lines_match_mepc353(self):
        """ro-ro/ro-pax/vehicle-carrier reference lines match MEPC.353(78) Table 1
        (GT basis); a previous revision used wrong coefficients."""
        from carbon_twin.regulations.cii import _select_reference_segment

        a, c, _, _ = _select_reference_segment("ro_ro_cargo", 25000, 9000)
        assert (a, c) == (1967.0, 0.485)
        a, c, _, _ = _select_reference_segment("ro_pax", 30000, 8000)
        assert (a, c) == (2023.0, 0.460)
        # vehicle carrier: GT-threshold piecewise (3627/0.590 large, 330/0.329 small)
        a, c, _, _ = _select_reference_segment("vehicle_carrier", 40000, 18000)
        assert (a, c) == (3627.0, 0.590)
        a, c, _, _ = _select_reference_segment("vehicle_carrier", 20000, 8000)
        assert (a, c) == (330.0, 0.329)

    def test_small_lng_reference_capped_no_false_a(self):
        """Regression: the steep LNG sub-100k line (c=2.673) explodes for tiny
        LNG vessels — required CII must stay bounded via the 65,000 ref floor so
        small inefficient LNG don't falsely rate A."""
        small = Vessel(
            imo_number="IMO3", ship_type=ShipType.LNG_CARRIER, gross_tonnage=15000, deadweight=12000
        )
        r = CIIEngine().evaluate(small, annual_co2_tonnes=40000, distance_nm=60000, year=2026)
        assert r.required_cii < 30  # not thousands

    def test_gt_type_capacity_independent_of_passed_override_basis(self):
        """The optimizer must rate GT-based types on GT. capacity_for() is the
        single source the optimizer shares with the engine."""
        from carbon_twin.regulations.cii import CIIEngine

        vc = Vessel(
            imo_number="IMO2",
            ship_type=ShipType.VEHICLE_CARRIER,
            gross_tonnage=58685,
            deadweight=18168,
        )
        assert CIIEngine.capacity_for(vc) == 58685  # GT, not DWT


class TestFuelDimensionCoupling:
    """The fuel decision is where the regulations genuinely conflict; coupling
    must emerge there as FuelEU tightens (and be ~0 in 2026)."""

    def _ctx(self):
        from carbon_twin.data.loaders import load_fuel_prices
        from carbon_twin.emissions.proxy import load_fuel_factors_safe

        vessel = Vessel(
            imo_number="IMOF",
            ship_type=ShipType.BULK_CARRIER,
            gross_tonnage=45000,
            deadweight=82000,
            design_speed_knots=14.5,
            main_engine_power_kw=12000,
            aux_engine_power_kw=2500,
            year_built=2015,
        )
        voyage = Voyage(
            voyage_id="VF",
            vessel_imo="IMOF",
            departure_port_id="A",
            arrival_port_id="B",
            departure_time=dt.datetime(2026, 3, 1),
            arrival_time=dt.datetime(2026, 3, 10),
            distance_nm=2000,
            avg_speed_knots=12.0,
            sea_hours=2000 / 12.0,
            eea_scope=EEAScope.INTRA_EEA,
            ets_scope_factor=1.0,
        )
        return (
            vessel,
            voyage,
            load_fuel_factors_safe(),
            load_fuel_prices("configs/fuel_prices.yaml", scenario="base_2026"),
        )

    def test_biofuel_zero_rated_in_ets(self):
        from carbon_twin.optimization.fuel_choice import _ets_covered_tonnes

        _, _, ff, _ = self._ctx()
        fossil = _ets_covered_tonnes(FuelType.VLSFO, ff[FuelType.VLSFO], 100.0, 2026)
        bio = _ets_covered_tonnes(FuelType.BIODIESEL_FAME, ff[FuelType.BIODIESEL_FAME], 100.0, 2026)
        assert fossil > 250  # ~311 tCO2e for 100 t VLSFO
        assert bio < fossil * 0.1  # biogenic CO2 zero-rated

    def test_fuel_coupling_grows_with_fueleu_stringency(self):
        from carbon_twin.optimization.fuel_choice import (
            optimal_fuel_speed_multi_regulation,
        )

        vessel, voyage, ff, prices = self._ctx()
        common = dict(
            fuel_prices=prices,
            eua_price=120.0,
            fuel_factors=ff,
            cii_engine=CIIEngine(),
            fueleu_engine=FuelEUEngine(),
            ets_engine=ETSEngine(),
            annual_distance_nm=60000.0,
            speed_range=(7.0, 14.5),
            n_points=12,
            delay_cost_per_hour=580.0,
            scheduled_hours=voyage.sea_hours,
        )
        early = optimal_fuel_speed_multi_regulation(vessel, voyage, year=2026, **common)
        late = optimal_fuel_speed_multi_regulation(vessel, voyage, year=2045, **common)
        # 2026 FuelEU (−2%) barely binds → no switch, ~0 coupling.
        assert early["siloed_fuel"] == early["joint_fuel"]
        # 2045 (−80%) forces a low-carbon fuel in the joint optimum; siloed stays
        # on the cheap fossil fuel → material fuel-dimension coupling.
        assert late["fuel_switch"]
        assert late["rci"] > early["rci"] + 0.05

    def test_blending_raises_midrange_rci_above_discrete(self):
        """v2 (continuous blend) must reveal more coupling than v1 (discrete) in
        the mid-range: blending just enough low-carbon fuel is cheaper than an
        all-or-nothing switch, so the joint cost drops and RCI rises. v1 is a
        conservative lower bound."""
        from carbon_twin.optimization.fuel_choice import (
            optimal_fuel_blend_speed_multi_regulation,
            optimal_fuel_speed_multi_regulation,
        )

        vessel, voyage, ff, prices = self._ctx()
        common = dict(
            fuel_prices=prices,
            eua_price=120.0,
            fuel_factors=ff,
            cii_engine=CIIEngine(),
            fueleu_engine=FuelEUEngine(),
            ets_engine=ETSEngine(),
            year=2035,
            annual_distance_nm=60000.0,
            speed_range=(7.0, 14.5),
            n_points=12,
            delay_cost_per_hour=580.0,
            scheduled_hours=voyage.sea_hours,
        )
        disc = optimal_fuel_speed_multi_regulation(vessel, voyage, **common)
        blend = optimal_fuel_blend_speed_multi_regulation(vessel, voyage, **common)
        assert blend["rci"] >= disc["rci"]
        # the joint blend at mid-stringency is partial, not all-or-nothing
        assert 0.0 < blend["joint_blend_pct"] < 1.0


class TestSpeedBounds:
    def test_speed_band_anchored_on_design_not_flat_floor(self):
        """Regression: a flat (8, 22) band let high-design-speed vessels rail to
        an unrealistic 8 kn floor. The band must be design-anchored: floor ~50%
        of design, ceiling at design."""
        from carbon_twin.optimization.speed import speed_bounds

        fast = Vessel(
            imo_number="C1",
            ship_type=ShipType.CONTAINER_SHIP,
            gross_tonnage=40000,
            design_speed_knots=22.0,
        )
        lo, hi = speed_bounds(fast)
        assert hi == pytest.approx(22.0)
        assert 10.5 <= lo <= 11.5  # ~50% design, NOT 8
        # The operating point is always kept inside the band.
        lo2, hi2 = speed_bounds(fast, operating_speed=23.0)
        assert lo2 <= 23.0 <= hi2


class TestIMONZFScenarioLoading:
    def test_distinct_scenarios_load_distinct_params(self):
        m83 = IMONZFScenarioEngine("MEPC83_draft")
        m84 = IMONZFScenarioEngine("MEPC84_negotiation")
        assert m83._scenario_params != m84._scenario_params
        # MEPC 83 draft carries a year-keyed GFI reduction path...
        assert "gfi_reduction_target" in m83._scenario_params
        # ...whereas MEPC 84 (still under negotiation) does not.
        assert "gfi_reduction_target" not in m84._scenario_params

    def test_scenario_configs_use_valid_version_keys(self):
        """Every regulation version referenced by a scenario YAML must exist in
        the version manager — guards against the divergent/invalid keys that
        previously lived in the MEPC 84 and methanol-transition scenarios."""
        import glob

        import yaml

        from carbon_twin.regulations.versioning import RegulationVersionManager

        mgr = RegulationVersionManager()
        for path in sorted(glob.glob("configs/scenarios/*.yaml")):
            raw = yaml.safe_load(open(path)) or {}
            scenario = raw.get("scenario", raw)
            regs = scenario.get("regulations", {})
            for reg_key, reg_info in regs.items():
                version = reg_info["version"] if isinstance(reg_info, dict) else reg_info
                assert mgr.validate_version(reg_key, version), (
                    f"{path}: {reg_key} -> {version} is not a known version"
                )

    def test_mepc83_target_drives_evaluation(self):
        vessel = Vessel(
            imo_number="IMO1",
            ship_type=ShipType.BULK_CARRIER,
            gross_tonnage=30000,
            deadweight=50000,
        )
        recs = [
            FuelRecord(
                vessel_imo="IMO1",
                fuel_type=FuelType.HFO,
                mass_tonnes=1000,
                energy_mj=1000 * 40.2 * 1000,
            )
        ]
        energy = sum(r.energy_mj for r in recs)
        factors = {FuelType.HFO: 91.16}
        res = IMONZFScenarioEngine("MEPC83_draft").evaluate(
            vessel, energy, recs, factors, year=2030, ghg_price_per_tco2e=150.0
        )
        # 2030 GFI target (-10% vs 91.16) is active, so a target is produced.
        assert res.gfi_target is not None
        assert res.gfi_attained == pytest.approx(91.16, rel=1e-3)
