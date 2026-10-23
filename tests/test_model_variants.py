from __future__ import annotations

import math

import pandas as pd
import pytest

from carbon_twin.data.loaders import load_fuel_prices, load_mrv_csv, load_vessels_csv
from carbon_twin.data.schemas import FuelRecord, FuelType
from carbon_twin.emissions.proxy import load_fuel_factors_safe, vessels_to_index
from carbon_twin.model_variants import (
    ModelVariant,
    available_variants,
    lng_annex2_factors,
    load_variant,
)
from carbon_twin.optimization.fuel_choice import (
    _leg_capex,
    exact_blend_points,
    fueleu_factor_table,
    scope_energy_allocation,
    scope_split,
)
from carbon_twin.pipeline import (
    _scope_weight,
    feasible_fuel_options,
    load_fuel_capability,
    retrofit_annual_cost,
    run_fuel_optimization_from_mrv,
    run_optimization_from_mrv,
)
from carbon_twin.regulations.fueleu import FuelEUEngine

_REGS = dict(cii_version="IMO_CII_2023_phase1", fueleu_version="EU_2023_1805",
             ets_version="EU_ETS_Maritime_2024")


@pytest.fixture
def fleet_inputs(tmp_path):
    mrv = tmp_path / "mrv.csv"
    pd.DataFrame([
        {"imo_number": "9000001", "total_co2_tonnes": 30000.0, "co2_ets_tonnes": 18000.0,
         "hours_at_sea": 2500.0, "distance_nm": 32000.0, "total_fuel_tonnes": 9600.0},
        {"imo_number": "9000002", "total_co2_tonnes": 50000.0, "co2_ets_tonnes": 50000.0,
         "hours_at_sea": 2800.0, "distance_nm": 38000.0, "total_fuel_tonnes": 16000.0},
        {"imo_number": "9000003", "total_co2_tonnes": 3000.0, "co2_ets_tonnes": None,
         "hours_at_sea": 200.0, "distance_nm": 1500.0, "total_fuel_tonnes": 960.0},
    ]).to_csv(mrv, index=False)
    fleet = tmp_path / "fleet.csv"
    pd.DataFrame([
        {"imo_number": "9000001", "ship_type": "bulk_carrier", "gross_tonnage": 45000,
         "deadweight": 82000, "design_speed_knots": 14.5, "main_engine_power_kw": 12000,
         "aux_engine_power_kw": 2500, "length_m": 229, "beam_m": 32, "draught_m": 14.5,
         "year_built": 2015},
        {"imo_number": "9000002", "ship_type": "container_ship", "gross_tonnage": 95000,
         "deadweight": 105000, "design_speed_knots": 22.0, "main_engine_power_kw": 55000,
         "aux_engine_power_kw": 8000, "length_m": 335, "beam_m": 48, "draught_m": 15.5,
         "year_built": 2018},
        {"imo_number": "9000003", "ship_type": "general_cargo", "gross_tonnage": 8000,
         "deadweight": 11000, "design_speed_knots": 13.0, "main_engine_power_kw": 4000,
         "aux_engine_power_kw": 800, "length_m": 120, "beam_m": 20, "draught_m": 8,
         "year_built": 2005},
    ]).to_csv(fleet, index=False)
    return (load_mrv_csv(mrv), vessels_to_index(load_vessels_csv(fleet)),
            load_fuel_prices("configs/fuel_prices.yaml", scenario="base_2026"))


def _fuel_run(inputs, year, variant: ModelVariant | None = None, **kw):
    mrv_df, fleet_idx, prices = inputs
    v = variant or ModelVariant()
    factors = v.fuel_factors(load_fuel_factors_safe())
    regs = dict(_REGS, fueleu_version=v.fueleu_version_or(_REGS["fueleu_version"]))
    return run_fuel_optimization_from_mrv(
        mrv_df, fleet_idx, year, 80.0, prices, fuel_factors=factors,
        extended_fuels=True, blend=True, **regs, **kw)


class TestVariantLoading:
    def test_all_configured_variants_load(self):
        names = available_variants()
        assert "v23" in names and "r1_lng_otto_ms" in names
        for n in names:
            assert load_variant(n).name == n

    def test_default_is_noop(self):
        v = load_variant(None)
        factors = load_fuel_factors_safe()
        assert v.fuel_factors(factors) == factors
        assert v.fuel_optimizer_kwargs() == {
            "scope_mode": "full", "report_annual": False,
            "fueleu_factors_from_config": False, "base_fuels": ["VLSFO"]}
        assert v.fueleu_version_or("EU_2023_1805") == "EU_2023_1805"

    def test_v23_matches_default(self):
        a, b = load_variant("v23"), ModelVariant()
        assert a.fuel_optimizer_kwargs() == b.fuel_optimizer_kwargs()
        assert a.lng_cslip_pct is None and a.fueleu_version is None

    def test_unknown_variant_rejected(self):
        with pytest.raises(KeyError):
            load_variant("does_not_exist")

    def test_invalid_scope_mode_rejected(self):
        with pytest.raises(ValueError):
            ModelVariant(scope_mode="half")


class TestLngFactors:
    @pytest.mark.parametrize("cslip,ets_ttw,fueleu_wtw", [
        (3.1, 72.53, 89.20),
        (1.7, 65.33, 82.87),
        (0.2, 57.63, 76.08),
    ])
    def test_annex2_values(self, cslip, ets_ttw, fueleu_wtw):
        f = lng_annex2_factors(cslip)
        assert f["ets_ttw_co2e_g_per_mj"] == pytest.approx(ets_ttw, abs=0.01)
        assert f["wtw_co2e_g_per_mj"] == pytest.approx(fueleu_wtw, abs=0.01)
        assert f["ttw_ch4_g_per_g_fuel"] == pytest.approx(cslip / 100)

    def test_variant_replaces_only_lng_and_fossil_wtw(self):
        base = load_fuel_factors_safe()
        adj = load_variant("r1_lng_otto_ms").fuel_factors(base)
        for ft in base:
            if ft not in (FuelType.LNG, FuelType.VLSFO, FuelType.MGO):
                assert adj[ft] == base[ft]
        assert adj[FuelType.MGO].ttw_co2_g_per_g_fuel == base[FuelType.MGO].ttw_co2_g_per_g_fuel
        assert adj[FuelType.LNG].wtw_co2e_g_per_mj == pytest.approx(89.20, abs=0.01)
        assert adj[FuelType.LNG].lcv_mj_per_kg == base[FuelType.LNG].lcv_mj_per_kg
        assert adj[FuelType.VLSFO].wtw_co2e_g_per_mj == pytest.approx(91.55)
        assert adj[FuelType.VLSFO].ttw_co2_g_per_g_fuel == base[FuelType.VLSFO].ttw_co2_g_per_g_fuel


class TestWtwOverride:
    @pytest.mark.parametrize("name,expected", [
        ("v23", 90.0), ("v23_lng_otto_ms", 90.0), ("r1_lng_v23", 91.55),
        ("r1_lng_otto_ms_routed", 91.55)])
    def test_vlsfo_wtw_by_variant(self, name, expected):
        f = load_variant(name).fuel_factors(load_fuel_factors_safe())
        assert f[FuelType.VLSFO].wtw_co2e_g_per_mj == pytest.approx(expected)

    @pytest.mark.parametrize("name,expected", [
        ("v23", 88.0), ("r1_lng_otto_ms_prio_fb_bd", 90.77)])
    def test_mgo_wtw_annex2(self, name, expected):
        f = load_variant(name).fuel_factors(load_fuel_factors_safe())
        assert f[FuelType.MGO].wtw_co2e_g_per_mj == pytest.approx(expected)
        assert 14.4 + (3.206 + 0.00005 * 25 + 0.00018 * 298) / 0.0427 == pytest.approx(
            90.77, abs=0.005)

    def test_invalid_override_rejected(self):
        with pytest.raises(ValueError):
            ModelVariant(wtw_overrides={"VLSFO": -1.0})
        with pytest.raises(ValueError):
            ModelVariant(wtw_overrides={"not_a_fuel": 90.0})

    def test_override_reaches_fuel_path_fueleu(self, fleet_inputs):
        v = load_variant("r1_lng_v23")
        kw = dict(include_cii_in_objective=False, fueleu_factors_from_config=True)
        plain = _fuel_run(fleet_inputs, 2035, **kw).set_index("vessel_imo")
        over = _fuel_run(fleet_inputs, 2035, v, **kw).set_index("vessel_imo")
        assert list(over["fueleu_actual_intensity_siloed"]) == pytest.approx(
            [91.55] * len(over))
        assert (over["fueleu_penalty_siloed_eur"] > plain["fueleu_penalty_siloed_eur"]).all()

    def test_override_reaches_speed_path(self, fleet_inputs):
        mrv_df, fleet_idx, prices = fleet_inputs
        v = load_variant("r1_lng_v23")

        def run(factors, **kw):
            return run_optimization_from_mrv(
                pd.DataFrame(), mrv_df, fleet_idx, 2026, 80.0, prices["VLSFO"],
                fuel_factors=factors, cii_pressure=0.0, **_REGS, **kw).set_index("vessel_imo")
        base = run(load_fuel_factors_safe())
        over = run(v.fuel_factors(load_fuel_factors_safe()), fueleu_factors_from_config=True)
        assert (over["fueleu_penalty_siloed_eur"] > base["fueleu_penalty_siloed_eur"]).all()

    def test_override_reaches_compliance_path(self, fleet_inputs):
        from carbon_twin.pipeline import assemble_compliance_from_mrv

        mrv_df, fleet_idx, _ = fleet_inputs
        kw = load_variant("r1_lng_v23").compliance_kwargs()
        assert kw["fueleu_emission_factors"]["VLSFO"] == pytest.approx(91.55)
        base = assemble_compliance_from_mrv(mrv_df, fleet_idx, 2026, 80.0, **_REGS)
        over = assemble_compliance_from_mrv(
            mrv_df, fleet_idx, 2026, 80.0, **_REGS,
            fueleu_emission_factors=kw["fueleu_emission_factors"])
        assert over["fueleu_penalty_eur"].sum() > base["fueleu_penalty_eur"].sum()

    def test_v23_lng_factors_unchanged(self):
        base = load_fuel_factors_safe()
        assert load_variant("r1_lng_v23").fuel_factors(base)[FuelType.LNG] == base[FuelType.LNG]


class TestFuelEU:
    def test_config_factor_table_equals_engine_defaults(self):
        table = fueleu_factor_table(load_fuel_factors_safe())
        for k, v in FuelEUEngine._DEFAULT_FACTORS.items():
            assert table[k] == pytest.approx(v)

    def _rfnbo_intensity(self, version, year, sample_bulk_carrier):
        eng = FuelEUEngine(version=version)
        recs = [FuelRecord(vessel_imo=sample_bulk_carrier.imo_number,
                           fuel_type=FuelType.METHANOL_E, mass_tonnes=0.0, energy_mj=1e6)]
        return eng.evaluate(sample_bulk_carrier, recs, 1e6, year).actual_ghg_intensity

    def test_rfnbo_reward_ends_after_2033_in_r1_version(self, sample_bulk_carrier):
        assert self._rfnbo_intensity("EU_2023_1805_r1", 2030, sample_bulk_carrier) \
            == pytest.approx(2.5)
        assert self._rfnbo_intensity("EU_2023_1805_r1", 2045, sample_bulk_carrier) \
            == pytest.approx(5.0)

    def test_legacy_version_keeps_reward_every_year(self, sample_bulk_carrier):
        assert self._rfnbo_intensity("EU_2023_1805", 2045, sample_bulk_carrier) \
            == pytest.approx(2.5)


class TestScopeWeight:
    def test_full(self):
        assert _scope_weight({"total_co2_tonnes": 10, "co2_ets_tonnes": 6}, "full") == (1.0, "full")

    def test_mrv_share(self):
        w, src = _scope_weight({"total_co2_tonnes": 10, "co2_ets_tonnes": 6}, "mrv")
        assert w == pytest.approx(0.6) and src == "mrv_ets_share"

    def test_mrv_clipped_and_fallback(self):
        assert _scope_weight({"total_co2_tonnes": 10, "co2_ets_tonnes": 12}, "mrv")[0] == 1.0
        assert _scope_weight({"total_co2_tonnes": 10}, "mrv") == (1.0, "fallback_full")
        assert _scope_weight({"total_co2_tonnes": 10, "co2_ets_tonnes": float("nan")},
                             "mrv") == (1.0, "fallback_full")


class TestFuelPath:
    def test_explicit_legacy_kwargs_identical(self, fleet_inputs):
        a = _fuel_run(fleet_inputs, 2035, include_cii_in_objective=False)
        b = _fuel_run(fleet_inputs, 2035, include_cii_in_objective=False,
                      **ModelVariant().fuel_optimizer_kwargs())
        pd.testing.assert_frame_equal(a, b)

    def test_annual_reporting_scales_costs_not_decisions(self, fleet_inputs):
        leg = _fuel_run(fleet_inputs, 2045, include_cii_in_objective=False,
                        scope_mode="mrv", fueleu_factors_from_config=True)
        ann = _fuel_run(fleet_inputs, 2045, include_cii_in_objective=False,
                        scope_mode="mrv", fueleu_factors_from_config=True,
                        report_annual=True)
        for col in ("joint_fuel_pathway", "siloed_fuel_pathway", "joint_speed",
                    "siloed_speed", "partial_speed"):
            assert list(leg[col]) == list(ann[col]), col
        assert list(leg["rci"]) == pytest.approx(list(ann["rci"]), abs=1e-4)
        assert (ann["cost_basis"] == "annual").all() and (leg["cost_basis"] == "leg").all()
        ratio = ann["delta_cost_eur"] / leg["delta_cost_eur"]
        assert list(ratio) == pytest.approx(list(ann["annual_scale"]), rel=1e-3)
        assert (ann["annual_scale"] >= 1.0).all()

    def test_mrv_scope_scales_fueleu_energy(self, fleet_inputs):
        full = _fuel_run(fleet_inputs, 2035, include_cii_in_objective=False,
                         fueleu_factors_from_config=True, scope_mode="full")
        mrv = _fuel_run(fleet_inputs, 2035, include_cii_in_objective=False,
                        fueleu_factors_from_config=True, scope_mode="mrv")
        by_imo = mrv.set_index("vessel_imo")
        assert by_imo.loc["9000001", "scope_weight"] == pytest.approx(0.6)
        assert by_imo.loc["9000002", "scope_weight"] == pytest.approx(1.0)
        assert by_imo.loc["9000003", "scope_weight_source"] == "fallback_full"
        f = full.set_index("vessel_imo").loc["9000001"]
        m = by_imo.loc["9000001"]
        assert m["fueleu_energy_mj_siloed"] == pytest.approx(
            0.6 * f["fueleu_energy_mj_siloed"], rel=1e-6)

    def test_lng_correction_reverses_2035_divergence(self, fleet_inputs):
        v = load_variant("v23_lng_otto_ms")
        df = _fuel_run(fleet_inputs, 2035, v, include_cii_in_objective=False,
                       **v.fuel_optimizer_kwargs())
        assert set(df["siloed_fuel_pathway"]) == {"LNG_only"}
        assert set(df["joint_fuel_pathway"]) == {"VLSFO+biodiesel_fame"}

    def test_extra_lng_base_adds_only_interior_low_carbon_blends(self, fleet_inputs):
        v = load_variant("r1_lng_otto_ms_freebase")
        df = _fuel_run(fleet_inputs, 2045, v, include_cii_in_objective=False,
                       **v.fuel_optimizer_kwargs())
        lng_base = df[df["joint_base_fuel"] == "LNG"]
        assert len(lng_base) > 0
        assert lng_base["joint_fuel_pathway"].str.startswith("LNG+").all()

    def test_single_base_list_is_noop(self, fleet_inputs):
        a = _fuel_run(fleet_inputs, 2045, include_cii_in_objective=False,
                      scope_mode="mrv", report_annual=True)
        b = _fuel_run(fleet_inputs, 2045, include_cii_in_objective=False,
                      scope_mode="mrv", report_annual=True, base_fuels=["VLSFO"])
        pd.testing.assert_frame_equal(a, b)


class TestSpeedPath:
    def _run(self, inputs, **kw):
        mrv_df, fleet_idx, prices = inputs
        return run_optimization_from_mrv(
            pd.DataFrame(), mrv_df, fleet_idx, 2026, 80.0, prices["VLSFO"],
            fuel_factors=load_fuel_factors_safe(), cii_pressure=2.0, **_REGS, **kw)

    def test_annual_reporting_speed(self, fleet_inputs):
        leg = self._run(fleet_inputs, scope_mode="mrv")
        ann = self._run(fleet_inputs, scope_mode="mrv", report_annual=True)
        assert list(leg["joint_optimal_speed"]) == list(ann["joint_optimal_speed"])
        assert list(leg["rci"]) == list(ann["rci"])
        for _, r in ann.iterrows():
            lr = leg[leg["vessel_imo"] == r["vessel_imo"]].iloc[0]
            if lr["delta_cost_eur"] > 1:
                assert r["delta_cost_eur"] / lr["delta_cost_eur"] == pytest.approx(
                    r["annual_scale"], rel=1e-3)

    def test_default_has_no_new_columns(self, fleet_inputs):
        assert "scope_weight" not in self._run(fleet_inputs).columns

    def test_scope_lowers_policy_cost(self, fleet_inputs):
        full = self._run(fleet_inputs).set_index("vessel_imo")
        mrv = self._run(fleet_inputs, scope_mode="mrv").set_index("vessel_imo")
        assert (mrv.loc["9000001", "ets_cost_siloed_eur"]
                < full.loc["9000001", "ets_cost_siloed_eur"])
        assert math.isclose(mrv.loc["9000002", "ets_cost_siloed_eur"],
                            full.loc["9000002", "ets_cost_siloed_eur"])


class TestScopeAllocation:
    def _pair(self, inputs, year=2045):
        kw = dict(include_cii_in_objective=False, fueleu_factors_from_config=True)
        uni = _fuel_run(inputs, year, scope_mode="mrv", **kw).set_index("vessel_imo")
        rou = _fuel_run(inputs, year, scope_mode="mrv_routed", **kw).set_index("vessel_imo")
        return uni, rou

    def test_identical_when_fully_in_scope(self, fleet_inputs):
        uni, rou = self._pair(fleet_inputs)
        for col in ("cost_joint_eur", "cost_siloed_eur", "joint_fuel_pathway", "rci"):
            assert uni.at["9000002", col] == rou.at["9000002", col], col

    def test_routing_never_raises_joint_cost(self, fleet_inputs):
        uni, rou = self._pair(fleet_inputs)
        assert rou.at["9000001", "cost_joint_eur"] <= uni.at["9000001", "cost_joint_eur"] + 1e-6
        assert rou.at["9000001", "scope_weight"] == pytest.approx(0.6)

    def test_routed_fueleu_energy_is_scoped_share(self, fleet_inputs):
        full = _fuel_run(fleet_inputs, 2045, include_cii_in_objective=False,
                         fueleu_factors_from_config=True).set_index("vessel_imo")
        _, rou = self._pair(fleet_inputs)
        assert rou.at["9000001", "fueleu_energy_mj_siloed"] == pytest.approx(
            0.6 * full.at["9000001", "fueleu_energy_mj_siloed"], rel=1e-6)

    def test_routed_variant_loads(self):
        assert load_variant("r1_lng_otto_ms_routed").scope_mode == "mrv_routed"


_PHIS = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0]
_WEIGHTS = [0.55, 0.6, 0.8, 1.0]


class TestScopeEnergyAllocation:
    def test_scope_split(self):
        assert scope_split(0.6) == pytest.approx((0.2, 0.8, 0.5))
        assert scope_split(1.0) == pytest.approx((1.0, 0.0, 0.0))
        assert scope_split(0.3) == pytest.approx((0.0, 1.0, 0.3))

    @pytest.mark.parametrize("mode", ["fueleu_priority", "routed_voyage"])
    @pytest.mark.parametrize("s", _WEIGHTS)
    @pytest.mark.parametrize("phi", _PHIS)
    def test_conservation(self, mode, s, phi):
        a = scope_energy_allocation(phi, 100.0, s, mode)
        assert a["alt_energy"] + a["base_energy"] == pytest.approx(100.0)
        assert a["ets_alt_energy"] + a["ets_base_energy"] == pytest.approx(100.0 * s)
        assert a["fueleu_alt_energy"] + a["fueleu_base_energy"] == pytest.approx(100.0 * s)
        assert a["fueleu_alt_energy"] == pytest.approx(min(100.0 * phi, 100.0 * s))
        assert 0.0 <= a["ets_alt_energy"] <= a["alt_energy"] + 1e-12

    @pytest.mark.parametrize("alt_first", [True, False])
    @pytest.mark.parametrize("s", _WEIGHTS)
    @pytest.mark.parametrize("phi", _PHIS)
    def test_routed_conservation(self, alt_first, s, phi):
        a = scope_energy_allocation(phi, 100.0, s, "routed_voyage", alt_first)
        assert a["ets_alt_energy"] + a["ets_base_energy"] == pytest.approx(100.0 * s)
        assert 0.0 <= a["ets_alt_energy"] <= a["alt_energy"] + 1e-12
        assert 0.0 <= a["ets_base_energy"] <= a["base_energy"] + 1e-12

    @pytest.mark.parametrize("phi", _PHIS)
    def test_routing_covers_first_fuel_fully(self, phi):
        s = 0.6
        alt = scope_energy_allocation(phi, 100.0, s, "routed_voyage", True)
        base = scope_energy_allocation(phi, 100.0, s, "routed_voyage", False)
        pri = scope_energy_allocation(phi, 100.0, s, "fueleu_priority")
        assert pri["ets_alt_energy"] == pytest.approx(s * 100.0 * phi)
        assert base["ets_alt_energy"] - 1e-12 <= pri["ets_alt_energy"] \
            <= alt["ets_alt_energy"] + 1e-12
        if phi <= 0.2:
            assert alt["ets_alt_energy"] == pytest.approx(100.0 * phi)
        if phi >= 0.8:
            assert base["ets_base_energy"] == pytest.approx(100.0 * (1 - phi))
        assert alt["fueleu_alt_energy"] == base["fueleu_alt_energy"] == \
            pri["fueleu_alt_energy"]

    def test_unknown_allocation_rejected(self):
        with pytest.raises(ValueError):
            scope_energy_allocation(0.5, 1.0, 0.6, "uniform")


class TestExactBlendPoints:
    def test_uniform_share_meets_target(self):
        (x,) = exact_blend_points(91.55, 30.0, 77.94, "uniform", 1.0)
        assert (1 - x) * 91.55 + x * 30.0 == pytest.approx(77.94)

    def test_pooled_points(self):
        x_uni = exact_blend_points(91.55, 30.0, 77.94, "uniform", 1.0)[0]
        pts = exact_blend_points(91.55, 30.0, 77.94, "fueleu_priority", 0.6)
        assert pts == pytest.approx(sorted([0.6 * x_uni, 0.6]))
        pts = exact_blend_points(91.55, 30.0, 77.94, "routed_voyage", 0.6)
        assert pts == pytest.approx(sorted([0.2, 0.6 * x_uni, 0.6, 0.8]))

    def test_no_compliance_point_when_base_complies(self):
        assert exact_blend_points(70.0, 30.0, 77.94, "uniform", 1.0) == []
        assert exact_blend_points(70.0, 30.0, 77.94, "fueleu_priority", 0.6) == [0.6]

    def test_rfnbo_share_meets_target_in_engine(self, sample_bulk_carrier):
        table = fueleu_factor_table(
            load_variant("r1_lng_v23").fuel_factors(load_fuel_factors_safe()))
        eng = FuelEUEngine(version="EU_2023_1805_r1")
        target = eng._target_intensity(2030)
        (x,) = exact_blend_points(table["VLSFO"], table["methanol_e"], target,
                                  "uniform", 1.0, rfnbo_multiplier=2.0)
        recs = [FuelRecord(vessel_imo=sample_bulk_carrier.imo_number, fuel_type=ft,
                           mass_tonnes=0.0, energy_mj=e)
                for ft, e in ((FuelType.VLSFO, (1 - x) * 1e6), (FuelType.METHANOL_E, x * 1e6))]
        r = eng.evaluate(sample_bulk_carrier, recs, 1e6, 2030, emission_factors=table)
        assert r.actual_ghg_intensity == pytest.approx(target, rel=1e-9)


class TestPooledAllocation:
    _KW = dict(include_cii_in_objective=False, fueleu_factors_from_config=True)

    def _run(self, inputs, scope_mode, year=2045, **kw):
        return _fuel_run(inputs, year, scope_mode=scope_mode, **self._KW,
                         **kw).set_index("vessel_imo")

    def test_identical_when_fully_in_scope(self, fleet_inputs):
        prop = self._run(fleet_inputs, "mrv")
        for mode in ("mrv_priority", "mrv_routed_voyage"):
            pool = self._run(fleet_inputs, mode)
            for col in ("cost_joint_eur", "cost_siloed_eur", "joint_fuel_pathway"):
                assert pool.at["9000002", col] == pytest.approx(prop.at["9000002", col],
                                                                rel=1e-9) \
                    if isinstance(prop.at["9000002", col], float) else \
                    pool.at["9000002", col] == prop.at["9000002", col], (mode, col)

    @pytest.mark.parametrize("year", [2035, 2045])
    def test_cost_ordering(self, fleet_inputs, year):
        prop = self._run(fleet_inputs, "mrv", year)
        prio = self._run(fleet_inputs, "mrv_priority", year)
        rout = self._run(fleet_inputs, "mrv_routed_voyage", year)
        v = "9000001"
        assert prio.at[v, "cost_joint_eur"] <= prop.at[v, "cost_joint_eur"] * (1 + 1e-9)
        assert rout.at[v, "cost_joint_eur"] <= prio.at[v, "cost_joint_eur"] * (1 + 1e-9)

    def test_fueleu_energy_is_pool(self, fleet_inputs):
        full = _fuel_run(fleet_inputs, 2045, **self._KW).set_index("vessel_imo")
        prio = self._run(fleet_inputs, "mrv_priority")
        assert prio.at["9000001", "scope_weight"] == pytest.approx(0.6)
        assert prio.at["9000001", "fueleu_energy_mj_siloed"] == pytest.approx(
            0.6 * full.at["9000001", "fueleu_energy_mj_siloed"], rel=1e-6)

    def test_batch2_variants_load(self):
        v = load_variant("r1_lng_otto_ms_prio_fb")
        assert v.scope_mode == "mrv_priority" and v.exact_blend and v.clean_objective
        assert [str(getattr(b, "value", b)) for b in v.base_fuels] == ["VLSFO", "LNG"]
        assert load_variant("r1_lng_v23_route_fb").scope_mode == "mrv_routed_voyage"
        assert load_variant("r1_lng_v23_prop_fb").scope_mode == "mrv"


class TestExactBlendGrid:
    def _solve(self, inputs, blend_steps, exact, year=2035):
        from carbon_twin.optimization.fuel_choice import (
            optimal_fuel_blend_speed_multi_regulation,
        )
        from carbon_twin.pipeline import build_fuel_optimization_context
        from carbon_twin.regulations.cii import CIIEngine
        from carbon_twin.regulations.eu_ets import ETSEngine

        mrv_df, fleet_idx, prices = inputs
        v = load_variant("r1_lng_otto_ms_prio_fb")
        factors = v.fuel_factors(load_fuel_factors_safe())
        out = {}
        for r in mrv_df.to_dict(orient="records"):
            ctx = build_fuel_optimization_context(r, fleet_idx, year, factors,
                                                  scope_mode=v.scope_mode)
            if ctx is None:
                continue
            res = optimal_fuel_blend_speed_multi_regulation(
                ctx["vessel"], ctx["voyage"], prices, 80.0, factors,
                cii_engine=CIIEngine(version=_REGS["cii_version"]),
                fueleu_engine=FuelEUEngine(version="EU_2023_1805_r1"),
                ets_engine=ETSEngine(version=_REGS["ets_version"]), year=year,
                annual_distance_nm=ctx["annual_dist"], speed_range=ctx["speed_range"],
                delay_cost_per_hour=ctx["charter_per_hour"],
                scheduled_hours=ctx["voyage"].sea_hours, extended_fuels=True,
                include_cii_in_objective=False, blend_steps=blend_steps,
                energy_scale=ctx["energy_scale"], fueleu_scope_factor=ctx["scope_weight"],
                fueleu_emission_factors=fueleu_factor_table(factors),
                base_fuels=[FuelType.VLSFO, FuelType.LNG],
                scope_allocation=ctx["scope_allocation"], exact_blend=exact)
            out[str(r["imo_number"])] = res
        return out

    @pytest.mark.parametrize("year", [2035, 2045])
    def test_exact_shares_make_the_grid_irrelevant(self, fleet_inputs, year):
        coarse = self._solve(fleet_inputs, 11, True, year)
        fine = self._solve(fleet_inputs, 51, True, year)
        assert coarse.keys() == fine.keys()
        for imo in coarse:
            for col in ("cost_joint_eur", "cost_siloed_eur"):
                assert coarse[imo][col] == pytest.approx(fine[imo][col], rel=1e-9), (imo, col)
            assert coarse[imo]["joint_fuel_pathway"] == fine[imo]["joint_fuel_pathway"]

    def test_exact_never_worse_than_grid(self, fleet_inputs):
        grid = self._solve(fleet_inputs, 11, False)
        exact = self._solve(fleet_inputs, 11, True)
        for imo in grid:
            assert exact[imo]["cost_joint_eur"] <= grid[imo]["cost_joint_eur"] * (1 + 1e-12)


class TestEngineCapability:
    _CAP = {"9000001": {"lng_capable": True, "methanol_capable": False},
            "9000002": {"lng_capable": False, "methanol_capable": True}}

    def _menu(self, fleet_inputs, imo, cap=_CAP):
        return feasible_fuel_options(fleet_inputs[1][imo], "engine_capability", True,
                                     None, cap)

    def test_menus(self, fleet_inputs):
        lng = self._menu(fleet_inputs, "9000001")
        meoh = self._menu(fleet_inputs, "9000002")
        none = self._menu(fleet_inputs, "9000003")
        assert FuelType.LNG in lng and FuelType.METHANOL_E not in lng
        assert FuelType.LNG not in meoh and FuelType.METHANOL_BIO in meoh
        assert set(none) <= {FuelType.VLSFO, FuelType.MGO, FuelType.BIODIESEL_FAME}
        assert FuelType.BIODIESEL_FAME in none

    def test_requires_map(self, fleet_inputs):
        with pytest.raises(ValueError):
            self._menu(fleet_inputs, "9000001", cap=None)

    def test_load_round_trip(self, tmp_path):
        p = tmp_path / "cap.csv"
        pd.DataFrame({"imo_number": ["0900001", " 9000002"], "lng_capable": [1, 0],
                      "methanol_capable": [0, 1], "evidence": ["", ""]}).to_csv(p, index=False)
        cap = load_fuel_capability(p)
        assert cap == {"0900001": {"lng_capable": True, "methanol_capable": False},
                       "9000002": {"lng_capable": False, "methanol_capable": True}}
        assert load_fuel_capability(None) is None

    def test_sea_web_rules(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("r1_fuel_capability",
                                                      "scripts/r1_fuel_capability.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        info = pd.DataFrame({
            "lrnoimo_ship_no": ["1", "2", "3", "4", "5", "6"],
            "fuel_type1first": ["Residual Fuel", "Residual Fuel", "Residual Fuel",
                                "LNG", "Residual Fuel", "Methanol"],
            "fuel_type2second": ["Distillate Fuel", "Distillate Fuel", "Distillate Fuel",
                                 None, "Liquefied Biogas", None],
            "main_engine_model": ["6G70ME-C10.5-GI", "W6X72DF", "6S50ME-C8.2-LGIP",
                                  "", "6S60ME-C", "6G50ME-C9.6-LGIM"]})
        t = mod.capability_table(info).set_index("imo_number")
        assert t["lng_capable"].tolist() == [1, 1, 0, 1, 1, 0]
        assert t["methanol_capable"].tolist() == [0, 0, 0, 0, 0, 1]


class TestRetrofitCapex:
    def test_annual_cost(self, fleet_inputs):
        bulk = fleet_inputs[1]["9000001"]
        capex = {"LNG": 500.0, "methanol_bio": 300.0}
        assert retrofit_annual_cost(bulk, capex, 0.15) == pytest.approx(
            {FuelType.LNG: 0.15 * 500 * 12000, FuelType.METHANOL_BIO: 0.15 * 300 * 12000})
        cap = {"9000001": {"lng_capable": True, "methanol_capable": False}}
        assert retrofit_annual_cost(bulk, capex, 0.15, cap) == pytest.approx(
            {FuelType.METHANOL_BIO: 0.15 * 300 * 12000})
        assert retrofit_annual_cost(bulk, None, 0.15) is None
        assert retrofit_annual_cost(bulk, {"LNG": 0.0}, 0.15) is None

    def test_missing_power_uses_estimate(self, fleet_inputs):
        from carbon_twin.emissions.physics import installed_power_kw
        bulk = fleet_inputs[1]["9000001"].model_copy(update={"main_engine_power_kw": None})
        kw = installed_power_kw(bulk)
        assert kw > 0
        assert retrofit_annual_cost(bulk, {"LNG": 500.0}, 0.15) == pytest.approx(
            {FuelType.LNG: 0.15 * 500 * kw})

    def test_leg_capex_charged_only_when_used(self):
        fixed = {FuelType.LNG: 1.4e6}
        assert _leg_capex(fixed, FuelType.VLSFO, FuelType.LNG, 0.0, 14.0) == 0.0
        assert _leg_capex(fixed, FuelType.VLSFO, FuelType.LNG, 0.3, 14.0) == pytest.approx(1e5)
        assert _leg_capex(fixed, FuelType.LNG, FuelType.BIODIESEL_FAME, 0.0, 14.0) == \
            pytest.approx(1e5)
        assert _leg_capex(None, FuelType.LNG, FuelType.LNG, 1.0, 14.0) == 0.0

    def test_prohibitive_capex_removes_lng(self, fleet_inputs):
        v = load_variant("r1_lng_otto_ms_prio_fb")
        kw = dict(scope_mode=v.scope_mode, include_cii_in_objective=False,
                  fueleu_factors_from_config=True, base_fuels=v.base_fuels, exact_blend=True)
        free = _fuel_run(fleet_inputs, 2035, v, **kw).set_index("vessel_imo")
        dear = _fuel_run(fleet_inputs, 2035, v, retrofit_capex_eur_per_kw={"LNG": 1e7},
                         **kw).set_index("vessel_imo")
        assert (dear["cost_joint_eur"] >= free["cost_joint_eur"] - 1e-6).all()
        assert not dear["joint_fuel_pathway"].str.contains("LNG").any()


_BD_COLS = {
    "between": "CO₂ emissions from all voyages between ports under a MS jurisdiction [m tonnes]",
    "departed": ("CO₂ emissions from all voyages which departed from ports under a MS "
                 "jurisdiction [m tonnes]"),
    "arrived": "CO₂ emissions from all voyages to ports under a MS jurisdiction [m tonnes]",
    "berth": ("CO₂ emissions which occurred within ports under a MS jurisdiction at berth "
              "[m tonnes]"),
}
_WITHIN = "CO₂ emissions which occurred within ports under a MS jurisdiction [m tonnes]"


class TestBreakdownScope:
    def test_columns_located(self):
        from carbon_twin.pipeline import mrv_breakdown_columns

        cols = [*_BD_COLS.values(), "Total CO₂ emissions [m tonnes]",
                "CO₂eq emissions from all voyages between ports under a MS jurisdiction [m tonnes]"]
        assert mrv_breakdown_columns(cols) == _BD_COLS
        assert mrv_breakdown_columns(cols + [_WITHIN]) == {**_BD_COLS, "port": _WITHIN}
        no_berth = [c for c in cols if c != _BD_COLS["berth"]] + [_WITHIN]
        assert mrv_breakdown_columns(no_berth) == {
            **{k: v for k, v in _BD_COLS.items() if k != "berth"}, "port": _WITHIN}
        assert mrv_breakdown_columns([c for c in cols if c != _BD_COLS["berth"]]) is None
        assert mrv_breakdown_columns(cols + [_BD_COLS["between"] + " (copy)"]) is None
        assert mrv_breakdown_columns(["Total CO2 emissions [m tonnes]"]) is None

    def test_scope_value_and_fallback(self):
        from carbon_twin.pipeline import mrv_breakdown_scope

        r = {"total_co2_tonnes": 100.0, "co2_ets_tonnes": 0.0,
             _BD_COLS["between"]: 10.0, _BD_COLS["departed"]: 40.0,
             _BD_COLS["arrived"]: 40.0, _BD_COLS["berth"]: 10.0}
        assert mrv_breakdown_scope(r) == pytest.approx(0.6)
        assert _scope_weight(r, "mrv_priority_bd") == (pytest.approx(0.6), "mrv_voyage_breakdown")
        assert _scope_weight(r, "mrv_priority") == (0.0, "mrv_ets_share")
        r_w = {**r, _WITHIN: 14.0}
        assert mrv_breakdown_scope(r_w) == pytest.approx((10 + 14 + 40) / 104)
        assert mrv_breakdown_scope({**r, _WITHIN: 0.0}) == pytest.approx(0.6)
        assert mrv_breakdown_scope({**r, _WITHIN: 6.0}) == pytest.approx(0.6)
        assert mrv_breakdown_scope({**r, _WITHIN: None}) == pytest.approx(0.6)
        assert mrv_breakdown_scope({**r, _WITHIN: -1.0}) is None
        bare = {"total_co2_tonnes": 100.0, "co2_ets_tonnes": 55.0}
        assert _scope_weight(bare, "mrv_priority_bd") == (pytest.approx(0.55),
                                                          "mrv_ets_share_bd_fallback")

    def test_bd_variants(self, fleet_inputs):
        from carbon_twin.pipeline import _SCOPE_ALLOCATION

        v = load_variant("r1_lng_otto_ms_prio_fb_bd")
        assert v.scope_mode == "mrv_priority_bd"
        assert _SCOPE_ALLOCATION[v.scope_mode] == "fueleu_priority"
        a = _fuel_run(fleet_inputs, 2035, v, include_cii_in_objective=False,
                      **v.fuel_optimizer_kwargs())
        c = load_variant("r1_lng_otto_ms_prio_fb")
        b = _fuel_run(fleet_inputs, 2035, c, include_cii_in_objective=False,
                      **c.fuel_optimizer_kwargs())
        assert a["cost_joint_eur"].tolist() == pytest.approx(b["cost_joint_eur"].tolist())


class TestBreakdownTwins:
    _PAIRS = [("r1_lng_otto_ms_prop_fb_bd", "r1_lng_otto_ms_prop_fb"),
              ("r1_lng_otto_ms_route_fb_bd", "r1_lng_otto_ms_route_fb"),
              ("r1_lng_lbsi_prio_fb_bd", "r1_lng_lbsi_prio_fb")]

    @pytest.mark.parametrize("bd,base", _PAIRS)
    def test_fall_back_without_breakdown(self, fleet_inputs, bd, base):
        a, b = load_variant(bd), load_variant(base)
        ra = _fuel_run(fleet_inputs, 2045, a, include_cii_in_objective=False,
                       **a.fuel_optimizer_kwargs())
        rb = _fuel_run(fleet_inputs, 2045, b, include_cii_in_objective=False,
                       **b.fuel_optimizer_kwargs())
        assert ra["cost_joint_eur"].tolist() == pytest.approx(rb["cost_joint_eur"].tolist())
        assert ra["scope_weight"].tolist() == pytest.approx(rb["scope_weight"].tolist())

    @pytest.mark.parametrize("variant", ["r1_lng_otto_ms_prop_fb_bd", "r1_lng_otto_ms_prio_fb_bd",
                                         "r1_lng_otto_ms_route_fb_bd"])
    def test_uses_breakdown(self, fleet_inputs, variant):
        mrv_df, fleet_idx, prices = fleet_inputs
        mrv = mrv_df.copy()
        shares = {"between": 0.5, "departed": 0.2, "arrived": 0.2, "berth": 0.1}
        for k, c in _BD_COLS.items():
            mrv[c] = mrv["total_co2_tonnes"] * shares[k]
        v = load_variant(variant)
        out = _fuel_run((mrv, fleet_idx, prices), 2045, v, include_cii_in_objective=False,
                        **v.fuel_optimizer_kwargs()).set_index("vessel_imo")
        assert out.at["9000001", "scope_weight"] == pytest.approx(0.8)
        assert out.at["9000002", "scope_weight"] == pytest.approx(0.8)


class TestSlimIntervals:
    def test_regret_and_path_intervals(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("slim", "scripts/r1_slim_vessel_outputs.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        lams = [0.0, 0.2, 0.4, 0.6, 1.0]
        rows = []
        for lam, rci, match in zip(lams, [0.1, 0.1, 0.0005, 0.0005, 0.0],
                                   [False, False, True, True, True]):
            rows.append(("A", lam, rci, match))
        for lam in lams:
            rows.append(("B", lam, 0.0, True))
        df = pd.DataFrame(rows, columns=["vessel_id", "lambda_fueleu", "rci_proxy",
                                         "fuel_pathway_label_match"]).assign(
            scenario_id="S", year=2035, blend_steps=11, ship_type="tanker", scope_weight=0.6)
        out = mod.closing_lambda(df).set_index("vessel_id")
        assert (out.at["A", "lambda_open"], out.at["A", "lambda_close"]) == (0.6, 1.0)
        assert (out.at["A", "lambda_open_1pct"], out.at["A", "lambda_close_1pct"]) == (0.2, 0.4)
        assert (out.at["A", "lambda_path_open"], out.at["A", "lambda_path_close"]) == (0.2, 0.4)
        assert not out.at["A", "path_match_lambda0"]
        assert out.at["B", "lambda_close"] == 0.0 and pd.isna(out.at["B", "lambda_open"])
        assert out.at["B", "path_match_lambda0"]


class TestRetrofitPathways:
    def test_pathway_fuels(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "retro", "scripts/r1_retrofit_cost_sensitivity.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod._pathway_fuels("LNG+biodiesel_fame") == {"LNG", "biodiesel_fame"}
        assert mod._pathway_fuels("VLSFO_only") == {"VLSFO"}
        assert mod._pathway_fuels("VLSFO+methanol_bio") == {"VLSFO", "methanol_bio"}


class TestExactCandidateSet:

    def _solve(self, inputs, variant, blend_steps, exact, year, **kw):
        from carbon_twin.optimization.fuel_choice import (
            optimal_fuel_blend_speed_multi_regulation,
        )
        from carbon_twin.pipeline import build_fuel_optimization_context
        from carbon_twin.regulations.cii import CIIEngine
        from carbon_twin.regulations.eu_ets import ETSEngine

        mrv_df, fleet_idx, prices = inputs
        v = load_variant(variant)
        factors = v.fuel_factors(load_fuel_factors_safe())
        out = {}
        for r in mrv_df.to_dict(orient="records")[:2]:
            ctx = build_fuel_optimization_context(r, fleet_idx, year, factors,
                                                  scope_mode=v.scope_mode)
            res = optimal_fuel_blend_speed_multi_regulation(
                ctx["vessel"], ctx["voyage"], prices, 80.0, factors,
                cii_engine=CIIEngine(version=_REGS["cii_version"]),
                fueleu_engine=FuelEUEngine(version="EU_2023_1805_r1"),
                ets_engine=ETSEngine(version=_REGS["ets_version"]), year=year,
                annual_distance_nm=ctx["annual_dist"], speed_range=ctx["speed_range"],
                delay_cost_per_hour=ctx["charter_per_hour"],
                scheduled_hours=ctx["voyage"].sea_hours, extended_fuels=True,
                include_cii_in_objective=False, blend_steps=blend_steps, n_points=5,
                energy_scale=ctx["energy_scale"], fueleu_scope_factor=ctx["scope_weight"],
                fueleu_emission_factors=fueleu_factor_table(factors),
                base_fuels=[FuelType(b) for b in v.base_fuels],
                scope_allocation=ctx["scope_allocation"], exact_blend=exact, **kw)
            out[str(r["imo_number"])] = res["cost_joint_eur"]
        return out

    @pytest.mark.parametrize("variant,year,kw", [
        ("r1_lng_otto_ms_prio_fb", 2035, {}),
        ("r1_lng_otto_ms_prio_fb", 2045, {}),
        ("r1_lng_otto_ms_route_fb", 2035, {}),
        ("r1_lng_otto_ms_prop_fb", 2045, {}),
        ("r1_lng_otto_ms_prio_fb", 2045, {"max_biofuel_share": 0.25}),
        ("r1_lng_otto_ms_prio_fb", 2035, {"fixed_annual_cost": {FuelType.LNG: 2.0e6}}),
    ])
    def test_exact_not_beaten_by_fine_grid(self, fleet_inputs, variant, year, kw):
        exact = self._solve(fleet_inputs, variant, 11, True, year, **kw)
        fine = self._solve(fleet_inputs, variant, 101, False, year, **kw)
        for imo in exact:
            assert exact[imo] <= fine[imo] * (1 + 1e-9), (variant, year, kw, imo)
