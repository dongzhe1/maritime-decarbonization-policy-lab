from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from carbon_twin.emissions.proxy import load_fuel_factors_safe
from carbon_twin.optimization.fuel_choice import (
    optimal_fuel_blend_speed_multi_regulation,
    partial_planner_decision,
    validate_planner_fueleu_internalization,
)
from carbon_twin.regulations.cii import CIIEngine
from carbon_twin.regulations.eu_ets import ETSEngine
from carbon_twin.regulations.fueleu import FuelEUEngine

CII_V = "IMO_CII_2023_phase1"
FUELEU_V = "EU_2023_1805"
ETS_V = "EU_ETS_Maritime_2024"

FUEL_PRICES = {
    "VLSFO": 600.0, "MGO": 700.0, "LNG": 800.0,
    "biodiesel_fame": 1100.0, "methanol_bio": 1200.0, "methanol_e": 1500.0,
}


@pytest.fixture
def factors() -> dict:
    return load_fuel_factors_safe()


@pytest.fixture
def engines():
    return (CIIEngine(version=CII_V), FuelEUEngine(version=FUELEU_V),
            ETSEngine(version=ETS_V))


def _toy_profile() -> pd.DataFrame:

    def row(alt: str, base: float, f: float, idx: int) -> dict:
        return {
            "alt_fuel": alt, "blend_pct": float(idx), "speed_knots": 14.0,
            "fuel_cost": base, "ets_cost": 0.0, "delay_cost": 0.0,
            "fueleu_penalty_full": f, "cii_penalty": 0.0,
            "fueleu_balance": 0.0, "fueleu_actual_intensity": 0.0,
            "fueleu_energy_mj": 0.0, "is_biofuel_blend": False,
        }

    return pd.DataFrame([row("A", 100.0, 100.0, 0), row("B", 140.0, 0.0, 1)])


def _optimizer_kwargs(vessel, voyage, factors, engines, **extra):
    cii, fueleu, ets = engines
    base = dict(
        vessel=vessel, voyage=voyage, fuel_prices=FUEL_PRICES, eua_price=80.0,
        fuel_factors=factors, cii_engine=cii, fueleu_engine=fueleu,
        ets_engine=ets, year=2035, extended_fuels=True,
        include_cii_in_objective=False, speed_range=(10, 18), n_points=8,
        blend_steps=6,
    )
    base.update(extra)
    return base


class TestParameterValidation:
    @pytest.mark.parametrize("bad", [-0.1, -1e-9, 1.0001, 5.0])
    def test_rejects_out_of_range(self, bad):
        with pytest.raises(ValueError, match="[0, 1]"):
            validate_planner_fueleu_internalization(bad)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite(self, bad):
        with pytest.raises(ValueError, match="finite"):
            validate_planner_fueleu_internalization(bad)

    @pytest.mark.parametrize("good", [0, 1, 0.3, 1.0, 0.0, 0.75])
    def test_accepts_in_range(self, good):
        v = validate_planner_fueleu_internalization(good)
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0

    def test_optimizer_validates_too(self, sample_bulk_carrier,
                                     sample_intra_eea_voyage, factors, engines):
        with pytest.raises(ValueError):
            optimal_fuel_blend_speed_multi_regulation(
                **_optimizer_kwargs(sample_bulk_carrier, sample_intra_eea_voyage,
                                    factors, engines,
                                    planner_fueleu_internalization=1.5))


class TestToySwitchingThreshold:
    def test_switch_at_lambda_0p40(self):
        prof = _toy_profile()
        below = partial_planner_decision(prof, 0.39, include_cii_in_objective=False)
        above = partial_planner_decision(prof, 0.41, include_cii_in_objective=False)
        assert below["partial_alt_fuel"] == "A"
        assert below["delta_cost_eur"] == pytest.approx(60.0)
        assert below["rci_proxy"] == pytest.approx(60.0 / 200.0)
        assert below["rci_joint_norm"] == pytest.approx(60.0 / 140.0)
        assert above["partial_alt_fuel"] == "B"
        assert above["delta_cost_eur"] == pytest.approx(0.0)
        assert above["full_decision_match"] is True
        assert above["fuel_pathway_label_match"] is True
        assert above["blend_share_match_1pp"] is True
        assert above["fuel_plan_match_1pp"] is True
        assert above["speed_match_0p01kn"] is True

    def test_exact_tie_follows_deterministic_rule(self):
        d = partial_planner_decision(_toy_profile(), 0.40, include_cii_in_objective=False)
        assert d["partial_alt_fuel"] == "A"
        assert d["delta_cost_eur"] == pytest.approx(60.0)

    def test_benchmark_is_b_action(self):
        d = partial_planner_decision(_toy_profile(), 1.0, include_cii_in_objective=False)
        assert d["benchmark_alt_fuel"] == "B"
        assert d["benchmark_total_cost_full_eur"] == pytest.approx(140.0)


class TestAgreementSemantics:

    def test_label_match_with_blend_mismatch(self):
        prof = pd.DataFrame([
            {"alt_fuel": "A", "blend_pct": 0.0, "speed_knots": 14.0,
             "fuel_cost": 100.0, "ets_cost": 0.0, "delay_cost": 0.0,
             "fueleu_penalty_full": 100.0, "cii_penalty": 0.0,
             "fueleu_balance": 0.0, "fueleu_actual_intensity": 0.0,
             "fueleu_energy_mj": 0.0, "is_biofuel_blend": False},
            {"alt_fuel": "B", "blend_pct": 0.9, "speed_knots": 14.0,
             "fuel_cost": 120.0, "ets_cost": 0.0, "delay_cost": 0.0,
             "fueleu_penalty_full": 30.0, "cii_penalty": 0.0,
             "fueleu_balance": 0.0, "fueleu_actual_intensity": 0.0,
             "fueleu_energy_mj": 0.0, "is_biofuel_blend": False},
            {"alt_fuel": "B", "blend_pct": 1.0, "speed_knots": 14.0,
             "fuel_cost": 130.0, "ets_cost": 0.0, "delay_cost": 0.0,
             "fueleu_penalty_full": 0.0, "cii_penalty": 0.0,
             "fueleu_balance": 0.0, "fueleu_actual_intensity": 0.0,
             "fueleu_energy_mj": 0.0, "is_biofuel_blend": False},
        ])
        d = partial_planner_decision(prof, 0.30, include_cii_in_objective=False)
        assert d["partial_alt_fuel"] == "B"
        assert d["benchmark_alt_fuel"] == "B"
        assert d["fuel_pathway_label_match"] is True
        assert d["blend_share_match_1pp"] is False
        assert d["fuel_plan_match_1pp"] is False
        assert d["full_decision_match"] is False
        assert d["blend_share_abs_diff"] == pytest.approx(0.1)
        assert d["delta_cost_eur"] == pytest.approx(150.0 - 130.0)


class TestFinalCompletionAgreementMetrics:

    def test_share_agreement_but_different_pathway(self):
        prof = pd.DataFrame([
            {"alt_fuel": "LNG", "blend_pct": 0.5, "speed_knots": 14.0,
             "fuel_cost": 100.0, "ets_cost": 0.0, "delay_cost": 0.0,
             "fueleu_penalty_full": 50.0, "cii_penalty": 0.0,
             "fueleu_balance": 0.0, "fueleu_actual_intensity": 0.0,
             "fueleu_energy_mj": 0.0, "is_biofuel_blend": False},
            {"alt_fuel": "biodiesel_fame", "blend_pct": 0.5, "speed_knots": 14.0,
             "fuel_cost": 140.0, "ets_cost": 0.0, "delay_cost": 0.0,
             "fueleu_penalty_full": 0.0, "cii_penalty": 0.0,
             "fueleu_balance": 0.0, "fueleu_actual_intensity": 0.0,
             "fueleu_energy_mj": 0.0, "is_biofuel_blend": True},
        ])
        d = partial_planner_decision(prof, 0.0, include_cii_in_objective=False)
        assert d["partial_alt_fuel"] == "LNG"
        assert d["benchmark_alt_fuel"] == "biodiesel_fame"
        assert d["blend_share_match_1pp"] is True
        assert d["fuel_pathway_label_match"] is False
        assert d["fuel_plan_match_1pp"] is False
        assert d["fuel_pathway_partial"] == "VLSFO+LNG"
        assert d["fuel_pathway_benchmark"] == "VLSFO+biodiesel_fame"

    def test_pathway_and_share_within_1pp_gives_plan_agreement(self):
        prof = pd.DataFrame([
            {"alt_fuel": "LNG", "blend_pct": 0.50, "speed_knots": 14.0,
             "fuel_cost": 100.0, "ets_cost": 0.0, "delay_cost": 0.0,
             "fueleu_penalty_full": 100.0, "cii_penalty": 0.0,
             "fueleu_balance": 0.0, "fueleu_actual_intensity": 0.0,
             "fueleu_energy_mj": 0.0, "is_biofuel_blend": False},
            {"alt_fuel": "LNG", "blend_pct": 0.505, "speed_knots": 14.0,
             "fuel_cost": 100.4, "ets_cost": 0.0, "delay_cost": 0.0,
             "fueleu_penalty_full": 0.0, "cii_penalty": 0.0,
             "fueleu_balance": 0.0, "fueleu_actual_intensity": 0.0,
             "fueleu_energy_mj": 0.0, "is_biofuel_blend": False},
        ])
        d = partial_planner_decision(prof, 0.0, include_cii_in_objective=False)
        assert d["partial_blend_pct"] == pytest.approx(0.50)
        assert d["benchmark_blend_pct"] == pytest.approx(0.505)
        assert d["fuel_pathway_label_match"] is True
        assert d["blend_share_match_1pp"] is True
        assert d["fuel_plan_match_1pp"] is True
        assert d["full_decision_exact"] is False

    def test_same_plan_but_speed_mismatch_fails_tolerant_full_decision(self):
        prof = pd.DataFrame([
            {"alt_fuel": "LNG", "blend_pct": 0.5, "speed_knots": 14.0,
             "fuel_cost": 100.0, "ets_cost": 0.0, "delay_cost": 0.0,
             "fueleu_penalty_full": 100.0, "cii_penalty": 0.0,
             "fueleu_balance": 0.0, "fueleu_actual_intensity": 0.0,
             "fueleu_energy_mj": 0.0, "is_biofuel_blend": False},
            {"alt_fuel": "LNG", "blend_pct": 0.5, "speed_knots": 14.5,
             "fuel_cost": 100.2, "ets_cost": 0.0, "delay_cost": 0.0,
             "fueleu_penalty_full": 0.0, "cii_penalty": 0.0,
             "fueleu_balance": 0.0, "fueleu_actual_intensity": 0.0,
             "fueleu_energy_mj": 0.0, "is_biofuel_blend": False},
        ])
        d = partial_planner_decision(prof, 0.0, include_cii_in_objective=False)
        assert d["partial_speed"] == pytest.approx(14.0)
        assert d["benchmark_speed"] == pytest.approx(14.5)
        assert d["fuel_plan_match_1pp"] is True
        assert d["full_decision_1pp_0p01kn"] is False
        assert d["full_decision_exact"] is False

    def test_energy_dominant_fuel_not_used_for_pathway(self):
        from carbon_twin.optimization.fuel_choice import (
            _energy_dominant_fuel,
            _fuel_pathway_label,
        )
        assert _energy_dominant_fuel("VLSFO", "biodiesel_fame", 0.05) == "VLSFO"
        assert _fuel_pathway_label("VLSFO", "biodiesel_fame", 0.05) == \
            "VLSFO+biodiesel_fame"
        prof = pd.DataFrame([
            {"alt_fuel": "biodiesel_fame", "blend_pct": 0.05, "speed_knots": 14.0,
             "fuel_cost": 100.0, "ets_cost": 0.0, "delay_cost": 0.0,
             "fueleu_penalty_full": 10.0, "cii_penalty": 0.0,
             "fueleu_balance": 0.0, "fueleu_actual_intensity": 0.0,
             "fueleu_energy_mj": 0.0, "is_biofuel_blend": True},
        ])
        d = partial_planner_decision(prof, 1.0, include_cii_in_objective=False)
        assert d["energy_dominant_fuel_partial"] == "VLSFO"
        assert d["fuel_pathway_partial"] == "VLSFO+biodiesel_fame"
        assert d["fuel_pathway_partial"] != f"{d['energy_dominant_fuel_partial']}_only"


class TestEvaluationWeightInvariant:
    def test_full_evaluation_never_scaled_by_lambda(self):
        prof = _toy_profile()
        as_sel = [partial_planner_decision(prof, lam, False) for lam in (0.05, 0.10, 0.20)]
        bs_sel = [partial_planner_decision(prof, lam, False) for lam in (0.50, 0.70, 0.90)]
        for d in as_sel:
            assert d["partial_alt_fuel"] == "A"
            assert d["partial_fueleu_cost_full_eur"] == pytest.approx(100.0)
            assert d["partial_total_cost_full_eur"] == pytest.approx(200.0)
            assert d["partial_fueleu_balance"] == pytest.approx(0.0)
            assert d["partial_fueleu_actual_intensity"] == pytest.approx(0.0)
        for d in bs_sel:
            assert d["partial_alt_fuel"] == "B"
            assert d["partial_fueleu_cost_full_eur"] == pytest.approx(0.0)
            assert d["partial_total_cost_full_eur"] == pytest.approx(140.0)

    def test_lambda_one_reports_full_components(self, sample_bulk_carrier,
                                                sample_intra_eea_voyage,
                                                factors, engines):
        res = optimal_fuel_blend_speed_multi_regulation(
            **_optimizer_kwargs(sample_bulk_carrier, sample_intra_eea_voyage,
                                factors, engines, planner_fueleu_internalization=1.0))
        assert res["partial_alt_fuel"] == res["joint_alt_fuel"]
        assert res["partial_fuel_cost_full_eur"] == res["fuel_cost_joint_eur"]
        assert res["partial_fueleu_cost_full_eur"] == res["fueleu_penalty_joint_eur"]
        assert res["partial_total_cost_full_eur"] == res["cost_joint_eur"]


class TestEndpointEquivalence:
    def test_optimizer_lambda0_equals_siloed(self, sample_bulk_carrier,
                                             sample_intra_eea_voyage,
                                             factors, engines):
        res = optimal_fuel_blend_speed_multi_regulation(
            **_optimizer_kwargs(sample_bulk_carrier, sample_intra_eea_voyage,
                                factors, engines))
        assert res["partial_alt_fuel"] == res["siloed_alt_fuel"]
        assert res["partial_blend_pct"] == pytest.approx(res["siloed_blend_pct"])
        assert res["partial_speed"] == pytest.approx(res["siloed_speed"])
        assert res["partial_total_cost_full_eur"] == pytest.approx(
            res["cost_siloed_eur"])

    def test_optimizer_lambda1_zero_regret(self, sample_bulk_carrier,
                                           sample_intra_eea_voyage,
                                           factors, engines):
        res = optimal_fuel_blend_speed_multi_regulation(
            **_optimizer_kwargs(sample_bulk_carrier, sample_intra_eea_voyage,
                                factors, engines, planner_fueleu_internalization=1.0))
        assert res["partial_alt_fuel"] == res["joint_alt_fuel"]
        assert res["partial_blend_pct"] == pytest.approx(res["joint_blend_pct"])
        assert res["partial_speed"] == pytest.approx(res["joint_speed"])
        assert res["delta_cost_partial_eur"] == pytest.approx(0.0, abs=1e-9)
        assert res["full_decision_match"] is True
        assert res["numerical_regret_nonnegative"] is True

    def test_pipeline_m1_endpoints(self, sample_bulk_carrier, sample_container_ship,
                                   sample_tanker, factors, engines):
        from carbon_twin.pipeline import run_fuel_optimization_from_mrv

        mrv = pd.DataFrame([
            {"imo_number": "IMO9876543", "total_co2_tonnes": 30000.0,
             "hours_at_sea": 2500.0, "distance_nm": 32000.0, "total_fuel_tonnes": 10000.0},
            {"imo_number": "IMO1234567", "total_co2_tonnes": 50000.0,
             "hours_at_sea": 2800.0, "distance_nm": 38000.0, "total_fuel_tonnes": 17000.0},
            {"imo_number": "IMO5555555", "total_co2_tonnes": 40000.0,
             "hours_at_sea": 2600.0, "distance_nm": 35000.0, "total_fuel_tonnes": 13000.0},
        ])
        fleet = {v.imo_number: v for v in (sample_bulk_carrier,
                                           sample_container_ship, sample_tanker)}
        base = dict(cii_version=CII_V, fueleu_version=FUELEU_V, ets_version=ETS_V,
                    fuel_factors=factors, extended_fuels=True, blend=True,
                    include_cii_in_objective=False, eua_price=80.0,
                    fuel_prices=FUEL_PRICES)
        df0 = run_fuel_optimization_from_mrv(mrv, fleet, 2035, **base)
        df1 = run_fuel_optimization_from_mrv(mrv, fleet, 2035, **base,
                                             planner_fueleu_internalization=1.0)
        assert len(df0) == 3
        assert (df0["partial_fuel"] == df0["siloed_fuel"]).all()
        assert (df1["partial_fuel"] == df1["joint_fuel"]).all()
        assert np.allclose(df1["delta_cost_partial_eur"], 0.0, atol=1e-6)
        assert bool(df1["full_decision_match"].all())


class TestRegretProperties:
    @pytest.fixture
    def grid_results(self, sample_bulk_carrier, sample_intra_eea_voyage,
                     factors, engines):
        return [
            optimal_fuel_blend_speed_multi_regulation(
                **_optimizer_kwargs(sample_bulk_carrier, sample_intra_eea_voyage,
                                    factors, engines,
                                    planner_fueleu_internalization=lam))
            for lam in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]

    def test_regret_nonnegative(self, grid_results):
        for res in grid_results:
            assert res["delta_cost_partial_eur"] >= -1e-6
            assert res["numerical_regret_nonnegative"] is True

    def test_regret_monotonic(self, grid_results):
        deltas = [res["delta_cost_partial_eur"] for res in grid_results]
        for a, b in zip(deltas, deltas[1:]):
            assert b <= a + 1e-6

    def test_grid_covers_full_range(self, grid_results):
        assert grid_results[0]["partial_total_cost_full_eur"] >= \
            grid_results[-1]["partial_total_cost_full_eur"] - 1e-6

    def test_fleet_aggregate_regret_monotonic(self, sample_bulk_carrier,
                                              sample_container_ship,
                                              sample_tanker, factors, engines):
        from carbon_twin.pipeline import run_fuel_optimization_from_mrv

        mrv = pd.DataFrame([
            {"imo_number": "IMO9876543", "total_co2_tonnes": 30000.0,
             "hours_at_sea": 2500.0, "distance_nm": 32000.0, "total_fuel_tonnes": 10000.0},
            {"imo_number": "IMO1234567", "total_co2_tonnes": 50000.0,
             "hours_at_sea": 2800.0, "distance_nm": 38000.0, "total_fuel_tonnes": 17000.0},
            {"imo_number": "IMO5555555", "total_co2_tonnes": 40000.0,
             "hours_at_sea": 2600.0, "distance_nm": 35000.0, "total_fuel_tonnes": 13000.0},
        ])
        fleet = {v.imo_number: v for v in (sample_bulk_carrier,
                                           sample_container_ship, sample_tanker)}
        base = dict(cii_version=CII_V, fueleu_version=FUELEU_V, ets_version=ETS_V,
                    fuel_factors=factors, extended_fuels=True, blend=True,
                    include_cii_in_objective=False, eua_price=80.0,
                    fuel_prices=FUEL_PRICES)
        fleet_deltas = []
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            df = run_fuel_optimization_from_mrv(mrv, fleet, 2035, **base,
                                                planner_fueleu_internalization=lam)
            fleet_deltas.append(float(df["delta_cost_partial_eur"].sum()))
        for a, b in zip(fleet_deltas, fleet_deltas[1:]):
            assert b <= a + 1e-6
        assert fleet_deltas[-1] == pytest.approx(0.0, abs=1e-6)


class TestBackwardCompatibility:

    def test_default_param_identical_to_explicit_zero(self, sample_bulk_carrier,
                                                      sample_intra_eea_voyage,
                                                      factors, engines):
        kw = _optimizer_kwargs(sample_bulk_carrier, sample_intra_eea_voyage,
                               factors, engines)
        default = optimal_fuel_blend_speed_multi_regulation(**kw)
        explicit = optimal_fuel_blend_speed_multi_regulation(
            **kw, planner_fueleu_internalization=0.0)
        for k in default:
            if k == "profile":
                pd.testing.assert_frame_equal(default[k], explicit[k])
                continue
            if isinstance(default[k], float):
                assert default[k] == pytest.approx(explicit[k], abs=0.0)
            else:
                assert default[k] == explicit[k]

    def test_joint_siloed_outputs_untouched_by_partial_block(self,
                                                             sample_bulk_carrier,
                                                             sample_intra_eea_voyage,
                                                             factors, engines):
        res0 = optimal_fuel_blend_speed_multi_regulation(
            **_optimizer_kwargs(sample_bulk_carrier, sample_intra_eea_voyage,
                                factors, engines))
        res1 = optimal_fuel_blend_speed_multi_regulation(
            **_optimizer_kwargs(sample_bulk_carrier, sample_intra_eea_voyage,
                                factors, engines, planner_fueleu_internalization=1.0))
        assert res0["joint_alt_fuel"] == res1["joint_alt_fuel"]
        assert res0["joint_blend_pct"] == res1["joint_blend_pct"]
        assert res0["cost_joint_eur"] == res1["cost_joint_eur"]


class TestDeterminism:
    def test_reruns_identical(self, sample_bulk_carrier, sample_intra_eea_voyage,
                              factors, engines):
        kw = _optimizer_kwargs(sample_bulk_carrier, sample_intra_eea_voyage,
                               factors, engines, planner_fueleu_internalization=0.5)
        a = optimal_fuel_blend_speed_multi_regulation(**kw)
        b = optimal_fuel_blend_speed_multi_regulation(**kw)
        assert a["partial_alt_fuel"] == b["partial_alt_fuel"]
        assert a["partial_blend_pct"] == b["partial_blend_pct"]
        assert a["partial_speed"] == b["partial_speed"]
        assert a["delta_cost_partial_eur"] == b["delta_cost_partial_eur"]
        assert a["rci_partial"] == b["rci_partial"]


class TestByTypeClosure:
    def _synthetic_by_type(self) -> pd.DataFrame:
        rows = []
        for st, d0 in (("bulk_carrier", 100.0), ("container_ship", 200.0)):
            for lam, frac in ((0.0, 1.0), (0.5, 0.5), (1.0, 0.0)):
                rows.append({"case": "FI_PRIMARY_NO_CII", "year": 2035,
                             "lambda_fueleu": lam, "ship_type": st,
                             "n_vessels": 100, "fleet_delta_cost_eur": d0 * frac,
                             "mean_rci_proxy": 0.1 * frac,
                             "aggregate_rci_proxy": 0.1 * frac,
                             "dominant_fuel_label_agreement_share": 0.5,
                             "mean_low_carbon_share_partial": 0.5,
                             "mean_low_carbon_share_joint": 0.9,
                             "dominant_fuel_partial": "MGO",
                             "dominant_fuel_joint": "LNG"})
        return pd.DataFrame(rows)

    def test_each_type_normalized_against_own_baseline(self):
        from scripts.fueleu_internalization_sweep import _add_closure_metrics

        bt = _add_closure_metrics(self._synthetic_by_type(),
                                  ("case", "year", "ship_type"))
        for st, d0 in (("bulk_carrier", 100.0), ("container_ship", 200.0)):
            g = bt[(bt["ship_type"] == st) & (bt["case"] == "FI_PRIMARY_NO_CII")]
            g = g.sort_values("lambda_fueleu").reset_index(drop=True)
            assert g.loc[0, "fleet_regret_remaining_fraction"] == pytest.approx(1.0)
            assert g.loc[0, "fleet_regret_closed_fraction"] == pytest.approx(0.0)
            assert g.loc[1, "fleet_regret_closed_fraction"] == pytest.approx(0.5)
            assert g.loc[2, "fleet_regret_remaining_fraction"] == pytest.approx(0.0)
            assert g.loc[2, "fleet_regret_closed_fraction"] == pytest.approx(1.0)

    def test_audit_by_type_endpoints(self):
        from scripts.fueleu_internalization_sweep import (
            _add_closure_metrics,
            _audit_by_type_endpoints,
        )

        bt = _add_closure_metrics(self._synthetic_by_type(),
                                  ("case", "year", "ship_type"))
        assert _audit_by_type_endpoints(bt) == []
        bt.loc[(bt["ship_type"] == "container_ship") & (bt["lambda_fueleu"] == 0.0),
               "fleet_regret_closed_fraction"] = 0.5
        problems = _audit_by_type_endpoints(bt)
        assert len(problems) == 1
        assert "container_ship" in problems[0]

    def test_audit_accepts_zero_baseline(self):
        from scripts.fueleu_internalization_sweep import (
            _add_closure_metrics,
            _audit_by_type_endpoints,
        )

        raw = self._synthetic_by_type()
        raw["fleet_delta_cost_eur"] = 0.0
        bt = _add_closure_metrics(raw, ("case", "year", "ship_type"))
        assert bt["fleet_regret_closed_fraction"].isna().all()
        assert _audit_by_type_endpoints(bt) == []


class TestVesselThresholdSummary:
    def test_first_lambda_columns(self):
        from scripts.fueleu_internalization_sweep import _vessel_threshold_summary

        rows = []
        for vid, match_at, base_delta in (("v1", 0.05, 100.0), ("v2", 0.30, 50.0)):
            for lam in (0.0, 0.05, 0.10, 0.20, 0.30, 1.0):
                rows.append({
                    "case": "FI_PRIMARY_NO_CII", "year": 2035,
                    "lambda_fueleu": lam, "vessel_id": vid, "ship_type": "tanker",
                    "fuel_pathway_label_match": lam >= match_at,
                    "blend_share_match_1pp": lam >= match_at + 0.05,
                    "full_decision_match": lam >= match_at + 0.05,
                    "delta_cost_eur": base_delta * max(0.0, 1.0 - lam / 0.4),
                })
        out = _vessel_threshold_summary(pd.DataFrame(rows))
        assert len(out) == 2
        g = out.set_index("vessel_id")
        assert g.loc["v1", "first_lambda_fuel_pathway_label_match"] == pytest.approx(0.05)
        assert g.loc["v2", "first_lambda_fuel_pathway_label_match"] == pytest.approx(0.30)
        assert g.loc["v1", "first_lambda_90pct_individual_regret_closed"] == pytest.approx(1.0)
        assert g.loc["v2", "first_lambda_90pct_individual_regret_closed"] == pytest.approx(1.0)


class TestRunnerSmoke:
    def _write_inputs(self, tmp_path):
        mrv = tmp_path / "mrv.csv"
        pd.DataFrame([
            {"imo_number": "IMO9876543", "total_co2_tonnes": 30000.0,
             "hours_at_sea": 2500.0, "distance_nm": 32000.0, "total_fuel_tonnes": 10000.0},
            {"imo_number": "IMO1234567", "total_co2_tonnes": 50000.0,
             "hours_at_sea": 2800.0, "distance_nm": 38000.0, "total_fuel_tonnes": 17000.0},
        ]).to_csv(mrv, index=False)
        fleet = tmp_path / "fleet.csv"
        pd.DataFrame([
            {"imo_number": "IMO9876543", "ship_type": "bulk_carrier",
             "gross_tonnage": 45000, "deadweight": 82000, "design_speed_knots": 14.5,
             "main_engine_power_kw": 12000, "aux_engine_power_kw": 2500,
             "length_m": 229, "beam_m": 32, "draught_m": 14.5, "year_built": 2015},
            {"imo_number": "IMO1234567", "ship_type": "container_ship",
             "gross_tonnage": 95000, "deadweight": 105000, "design_speed_knots": 22.0,
             "main_engine_power_kw": 55000, "aux_engine_power_kw": 8000,
             "length_m": 335, "beam_m": 48, "draught_m": 15.5, "year_built": 2018},
        ]).to_csv(fleet, index=False)
        return mrv, fleet

    def _run_once(self, tmp_path, out: pathlib.Path) -> None:
        from scripts.fueleu_internalization_sweep import main

        mrv, fleet = self._write_inputs(tmp_path)
        sys_argv = [
            "fueleu_internalization_sweep.py",
            "--mrv", str(mrv), "--fleet", str(fleet),
            "--scenario", "configs/scenarios/base_2026.yaml",
            "--years", "2035", "--lambdas", "0", "0.25", "0.5", "0.75", "1",
            "--fuel-price-scenario", "base_2026", "--eua", "80",
            "--case", "FI_PRIMARY_NO_CII", "--output-dir", str(out),
            "--workers", "2", "--sample-limit", "5",
        ]
        import sys
        sys.argv = sys_argv
        with pytest.raises(SystemExit) as ei:
            main()
        assert ei.value.code == 0

    def test_runner_smoke_grid(self, tmp_path):
        out = tmp_path / "out"
        self._run_once(tmp_path, out)
        for name in ("fueleu_internalization_curve.csv",
                     "fueleu_internalization_vessel.parquet",
                     "fueleu_internalization_by_type.csv",
                     "fueleu_internalization_thresholds.csv",
                     "vessel_threshold_summary.csv",
                     "run_manifest.json", "validation_report.txt", "run_log.md"):
            assert (out / name).exists(), name
        vessel = pd.read_parquet(out / "fueleu_internalization_vessel.parquet")
        curve = pd.read_csv(out / "fueleu_internalization_curve.csv")
        by_type = pd.read_csv(out / "fueleu_internalization_by_type.csv")
        assert set(vessel["lambda_fueleu"]) == {0.0, 0.25, 0.5, 0.75, 1.0}
        assert len(vessel) == 10
        lam1 = curve[curve["lambda_fueleu"] == 1.0].iloc[0]
        assert abs(float(lam1["fleet_delta_cost_eur"])) <= 1.0
        assert float(lam1["full_decision_agreement_share"]) == 1.0
        assert "dominant_fuel_label_agreement_share" in curve.columns
        assert "fuel_choice_agreement_share" not in curve.columns
        assert "alternative_component_share_agreement_1pp" in curve.columns
        assert "blend_share_agreement_1pp" not in curve.columns
        assert "fuel_plan_agreement_1pp" in curve.columns
        assert "fuel_pathway_label_match" in vessel.columns
        assert "fuel_choice_match" not in vessel.columns
        assert "blend_share_match_1pp" in vessel.columns
        assert "fuel_plan_match_1pp" in vessel.columns
        for (yr, lam), g in vessel.groupby(["year", "lambda_fueleu"]):
            recomputed = float(g["rci_proxy"].mean())
            stored = float(curve[(curve["year"] == yr)
                                 & (curve["lambda_fueleu"] == lam)]
                           ["mean_rci_proxy"].iloc[0])
            assert recomputed == pytest.approx(stored, rel=1e-10)
        for st, g in by_type.groupby("ship_type"):
            g = g.sort_values("lambda_fueleu")
            assert g["fleet_regret_remaining_fraction"].iloc[0] == pytest.approx(1.0)
            assert g["fleet_regret_closed_fraction"].iloc[0] == pytest.approx(0.0)
            assert g["fleet_regret_closed_fraction"].iloc[-1] == pytest.approx(1.0)
        assert "IMO" not in "".join(vessel["vessel_id"].astype(str))

    def test_runner_determinism(self, tmp_path):
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        self._run_once(tmp_path, out1)
        self._run_once(tmp_path, out2)
        a = pd.read_parquet(out1 / "fueleu_internalization_vessel.parquet")
        b = pd.read_parquet(out2 / "fueleu_internalization_vessel.parquet")
        a = a.sort_values(["lambda_fueleu", "vessel_id"]).reset_index(drop=True)
        b = b.sort_values(["lambda_fueleu", "vessel_id"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(
            pd.read_csv(out1 / "fueleu_internalization_curve.csv").sort_values(
                "lambda_fueleu").reset_index(drop=True),
            pd.read_csv(out2 / "fueleu_internalization_curve.csv").sort_values(
                "lambda_fueleu").reset_index(drop=True))


class TestGlobalThresholdCensoring:

    def _curve(self, rows: list[dict]) -> pd.DataFrame:
        base = {"scenario_id": "CENTRAL_EXTENDED", "year": 2035, "blend_steps": 11}
        return pd.DataFrame([{**base, **r} for r in rows])

    def test_bracket_normal_crossing(self):
        from scripts.run_fueleu_internalization_final import _bracket_for_condition

        g = self._curve([
            {"lambda_fueleu": 0.0, "fleet_regret_closed_fraction": 0.0},
            {"lambda_fueleu": 0.5, "fleet_regret_closed_fraction": 0.40},
            {"lambda_fueleu": 1.0, "fleet_regret_closed_fraction": 1.00},
        ])
        b = _bracket_for_condition(g, "fleet_regret_closed_fraction", 0.90, "ge")
        assert b == (0.5, 1.0)

    def test_bracket_left_censored_when_already_true_at_zero(self):
        from scripts.run_fueleu_internalization_final import _bracket_for_condition

        g = self._curve([
            {"lambda_fueleu": 0.0, "fleet_regret_closed_fraction": 0.95},
            {"lambda_fueleu": 1.0, "fleet_regret_closed_fraction": 1.00},
        ])
        assert _bracket_for_condition(g, "fleet_regret_closed_fraction", 0.90, "ge") is None

    def test_bracket_right_censored_when_never_reached(self):
        from scripts.run_fueleu_internalization_final import _bracket_for_condition

        g = self._curve([
            {"lambda_fueleu": 0.0, "fleet_regret_closed_fraction": 0.0},
            {"lambda_fueleu": 1.0, "fleet_regret_closed_fraction": 0.50},
        ])
        assert _bracket_for_condition(g, "fleet_regret_closed_fraction", 0.90, "ge") is None

    def test_merge_windows_combines_overlapping_and_adjacent(self):
        from scripts.run_fueleu_internalization_final import _merge_windows

        merged = _merge_windows([(0.05, 0.10), (0.10, 0.15), (0.40, 0.45), (0.46, 0.50)])
        assert merged == [(0.05, 0.15), (0.40, 0.45), (0.46, 0.50)]

    def test_global_thresholds_union_across_coarse_and_refined(self):
        from scripts.run_fueleu_internalization_final import _global_threshold_rows

        coarse = self._curve([
            {"lambda_fueleu": 0.00, "fleet_regret_closed_fraction": 0.0,
             "fuel_pathway_label_agreement_share": 0.0, "fuel_plan_agreement_1pp_share": 0.0,
             "mean_rci_proxy": 0.20},
            {"lambda_fueleu": 0.05, "fleet_regret_closed_fraction": 1.0,
             "fuel_pathway_label_agreement_share": 1.0, "fuel_plan_agreement_1pp_share": 1.0,
             "mean_rci_proxy": 0.0},
        ])
        refined = self._curve([
            {"lambda_fueleu": 0.02, "fleet_regret_closed_fraction": 0.30,
             "fuel_pathway_label_agreement_share": 0.30, "fuel_plan_agreement_1pp_share": 0.30,
             "mean_rci_proxy": 0.14},
            {"lambda_fueleu": 0.03, "fleet_regret_closed_fraction": 0.95,
             "fuel_pathway_label_agreement_share": 0.95, "fuel_plan_agreement_1pp_share": 0.95,
             "mean_rci_proxy": 0.01},
        ])
        merged = pd.concat([coarse, refined], ignore_index=True)
        thr = _global_threshold_rows(merged)
        row = thr.iloc[0]
        assert row["n_tested_points"] == 4
        assert row["fleet_regret_closed_fraction_ge0p9_lambda_low"] == pytest.approx(0.02)
        assert row["fleet_regret_closed_fraction_ge0p9_lambda_high"] == pytest.approx(0.03)
        assert row["fleet_regret_closed_fraction_ge0p9_interval"] == "(0.0200, 0.0300]"
        assert row["fleet_regret_closed_fraction_ge0p9_left_censored"] == False
        assert row["fleet_regret_closed_fraction_ge0p9_right_censored"] == False

    def test_global_thresholds_no_observation_rows(self):
        from scripts.run_fueleu_internalization_final import _global_threshold_rows

        g = self._curve([
            {"lambda_fueleu": 0.0, "fleet_regret_closed_fraction": 0.0,
             "fuel_pathway_label_agreement_share": 0.0, "fuel_plan_agreement_1pp_share": 0.0,
             "mean_rci_proxy": 0.2},
            {"lambda_fueleu": 1.0, "fleet_regret_closed_fraction": 1.0,
             "fuel_pathway_label_agreement_share": 1.0, "fuel_plan_agreement_1pp_share": 1.0,
             "mean_rci_proxy": 0.0},
        ])
        thr = _global_threshold_rows(g)
        assert "lambda_fueleu" not in thr.columns
        assert len(thr) == 1


class TestGlobalClosureRecomputation:

    def _curve(self, rows: list[dict], **over) -> pd.DataFrame:
        base = {
            "scenario_id": "CENTRAL_EXTENDED", "case": "FI_PRIMARY_NO_CII",
            "year": 2035, "blend_steps": 11, "fuel_price_scenario": "base_2026",
            "eua_price_eur_per_tco2": 80.0, "feasibility_mode": "continuous_base",
            "exclude_fuels": "", "include_cii_in_planner": False,
            "include_cii_in_evaluation": False, "fueleu_price_fraction": 1.0,
            "fueleu_evaluation_multiplier": 1.0,
        }
        base.update(over)
        return pd.DataFrame([{**base, **r} for r in rows])

    def test_denominator_is_lambda_zero_not_first_row(self):
        from scripts.run_fueleu_internalization_final import (
            CURVE_GROUP_KEYS,
            recompute_global_closure_metrics,
        )
        df = self._curve([
            {"lambda_fueleu": 0.8, "fleet_delta_cost_eur": 400.0},
            {"lambda_fueleu": 1.0, "fleet_delta_cost_eur": 0.0},
            {"lambda_fueleu": 0.0, "fleet_delta_cost_eur": 500.0},
        ])
        out = recompute_global_closure_metrics(df, CURVE_GROUP_KEYS)
        out = out.sort_values("lambda_fueleu").reset_index(drop=True)
        assert out.loc[out["lambda_fueleu"] == 0.8,
                       "fleet_regret_closed_fraction"].iloc[0] == pytest.approx(0.20)
        assert out.loc[out["lambda_fueleu"] == 0.0,
                       "fleet_regret_closed_fraction"].iloc[0] == pytest.approx(0.0)

    def test_missing_lambda_zero_raises(self):
        from scripts.run_fueleu_internalization_final import (
            CURVE_GROUP_KEYS,
            recompute_global_closure_metrics,
        )
        df = self._curve([
            {"lambda_fueleu": 0.80, "fleet_delta_cost_eur": 400.0},
            {"lambda_fueleu": 0.85, "fleet_delta_cost_eur": 300.0},
        ])
        with pytest.raises(ValueError, match="no lambda=0 anchor"):
            recompute_global_closure_metrics(df, CURVE_GROUP_KEYS)

    def test_duplicate_lambda_zero_raises(self):
        from scripts.run_fueleu_internalization_final import (
            CURVE_GROUP_KEYS,
            recompute_global_closure_metrics,
        )
        df = self._curve([
            {"lambda_fueleu": 0.0, "fleet_delta_cost_eur": 500.0},
            {"lambda_fueleu": 0.0, "fleet_delta_cost_eur": 500.0},
        ])
        with pytest.raises(ValueError, match="lambda=0 rows"):
            recompute_global_closure_metrics(df, CURVE_GROUP_KEYS)

    def test_merge_of_coarse_and_refined_does_not_reset_closure(self):
        from scripts.run_fueleu_internalization_final import (
            CURVE_GROUP_KEYS,
            _dedupe_cells,
            recompute_global_closure_metrics,
        )
        coarse = self._curve([
            {"lambda_fueleu": 0.00, "fleet_delta_cost_eur": 500.0},
            {"lambda_fueleu": 0.05, "fleet_delta_cost_eur": 498.0},
            {"lambda_fueleu": 0.10, "fleet_delta_cost_eur": 1.0},
            {"lambda_fueleu": 1.00, "fleet_delta_cost_eur": 0.0},
        ])
        refined = self._curve([
            {"lambda_fueleu": 0.080, "fleet_delta_cost_eur": 400.0},
            {"lambda_fueleu": 0.085, "fleet_delta_cost_eur": 300.0},
            {"lambda_fueleu": 0.100, "fleet_delta_cost_eur": 1.0},
        ])
        merged = _dedupe_cells(pd.concat([coarse, refined], ignore_index=True))
        out = recompute_global_closure_metrics(merged, CURVE_GROUP_KEYS)
        out = out.sort_values("lambda_fueleu").reset_index(drop=True)
        closed = out["fleet_regret_closed_fraction"].to_numpy()
        assert np.all(np.diff(closed) >= -1e-12), f"closure reset detected: {closed}"
        assert out.loc[out["lambda_fueleu"] == 0.080,
                       "fleet_regret_closed_fraction"].iloc[0] == pytest.approx(0.20)

    def test_closure_endpoints(self):
        from scripts.run_fueleu_internalization_final import (
            CURVE_GROUP_KEYS,
            recompute_global_closure_metrics,
        )
        df = self._curve([
            {"lambda_fueleu": 0.0, "fleet_delta_cost_eur": 500.0},
            {"lambda_fueleu": 0.5, "fleet_delta_cost_eur": 250.0},
            {"lambda_fueleu": 1.0, "fleet_delta_cost_eur": 0.0},
        ])
        out = recompute_global_closure_metrics(df, CURVE_GROUP_KEYS
                                               ).sort_values("lambda_fueleu")
        assert out["fleet_regret_closed_fraction"].iloc[0] == pytest.approx(0.0)
        assert out["fleet_regret_closed_fraction"].iloc[-1] == pytest.approx(1.0)
        assert out["fleet_regret_remaining_fraction"].iloc[0] == pytest.approx(1.0)
        assert out["fleet_regret_remaining_fraction"].iloc[-1] == pytest.approx(0.0)

    def test_zero_baseline_marks_undefined_not_zero(self):
        from scripts.run_fueleu_internalization_final import (
            CURVE_GROUP_KEYS,
            recompute_global_closure_metrics,
        )
        df = self._curve([
            {"lambda_fueleu": 0.0, "fleet_delta_cost_eur": 0.0},
            {"lambda_fueleu": 1.0, "fleet_delta_cost_eur": 0.0},
        ])
        out = recompute_global_closure_metrics(df, CURVE_GROUP_KEYS)
        assert out["fleet_regret_closed_fraction"].isna().all()
        assert bool(out["baseline_regret_zero"].all())

    def test_resolutions_normalized_independently(self):
        from scripts.run_fueleu_internalization_final import (
            CURVE_GROUP_KEYS,
            recompute_global_closure_metrics,
        )
        df = pd.concat([
            self._curve([{"lambda_fueleu": 0.0, "fleet_delta_cost_eur": 500.0},
                         {"lambda_fueleu": 0.5, "fleet_delta_cost_eur": 250.0}],
                        blend_steps=11),
            self._curve([{"lambda_fueleu": 0.0, "fleet_delta_cost_eur": 800.0},
                         {"lambda_fueleu": 0.5, "fleet_delta_cost_eur": 200.0}],
                        blend_steps=21),
        ], ignore_index=True)
        out = recompute_global_closure_metrics(df, CURVE_GROUP_KEYS)
        b11 = out[(out["blend_steps"] == 11) & (out["lambda_fueleu"] == 0.5)]
        b21 = out[(out["blend_steps"] == 21) & (out["lambda_fueleu"] == 0.5)]
        assert b11["fleet_regret_closed_fraction"].iloc[0] == pytest.approx(0.50)
        assert b21["fleet_regret_closed_fraction"].iloc[0] == pytest.approx(0.75)


class TestThresholdStatusAndCensoring:

    def _g(self, lam_vals, cond_vals) -> pd.DataFrame:
        return pd.DataFrame({"lambda_fueleu": lam_vals,
                             "fleet_regret_closed_fraction": cond_vals})

    def test_bracketed(self):
        from scripts.run_fueleu_internalization_final import _condition_status
        r = _condition_status(self._g([0.0, 0.5, 1.0], [0.0, 0.4, 1.0]),
                              "fleet_regret_closed_fraction", 0.90, "ge")
        assert r["status"] == "bracketed"
        assert r["interval"] == "(0.5000, 1.0000]"
        assert not r["left_censored"] and not r["right_censored"]

    def test_exact_at_zero(self):
        from scripts.run_fueleu_internalization_final import _condition_status
        r = _condition_status(self._g([0.0, 1.0], [0.95, 1.0]),
                              "fleet_regret_closed_fraction", 0.90, "ge")
        assert r["status"] == "exact_at_zero"
        assert r["interval"] == "[0, 0]"

    def test_left_censored_when_window_starts_above_zero(self):
        from scripts.run_fueleu_internalization_final import _condition_status
        r = _condition_status(self._g([0.80, 0.85, 0.90], [0.99, 1.0, 1.0]),
                              "fleet_regret_closed_fraction", 0.90, "ge")
        assert r["status"] == "left_censored"
        assert r["left_censored"] is True
        assert r["right_censored"] is False
        assert r["interval"] == "<= 0.8000"

    def test_right_censored_when_grid_stops_short(self):
        from scripts.run_fueleu_internalization_final import _condition_status
        r = _condition_status(self._g([0.0, 0.4, 0.8], [0.0, 0.2, 0.5]),
                              "fleet_regret_closed_fraction", 0.90, "ge")
        assert r["status"] == "right_censored"
        assert r["right_censored"] is True

    def test_not_reached_on_domain(self):
        from scripts.run_fueleu_internalization_final import _condition_status
        r = _condition_status(self._g([0.0, 0.5, 1.0], [0.0, 0.2, 0.5]),
                              "fleet_regret_closed_fraction", 0.90, "ge")
        assert r["status"] == "not_reached_on_domain"
        assert not r["right_censored"]

    def test_baseline_undefined(self):
        from scripts.run_fueleu_internalization_final import _condition_status
        r = _condition_status(self._g([0.0, 1.0], [np.nan, np.nan]),
                              "fleet_regret_closed_fraction", 0.90, "ge")
        assert r["status"] == "baseline_undefined"

    def test_long_table_has_no_observation_rows(self):
        from scripts.run_fueleu_internalization_final import _threshold_long_table
        base = {"scenario_id": "S", "year": 2035, "blend_steps": 11}
        curve = pd.DataFrame([
            {**base, "lambda_fueleu": lam, "fleet_regret_closed_fraction": c,
             "fuel_pathway_label_agreement_share": c,
             "fuel_plan_agreement_1pp_share": c, "mean_rci_proxy": 0.2 * (1 - c)}
            for lam, c in ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0))
        ])
        out = _threshold_long_table(curve)
        assert "lambda_fueleu" not in out.columns
        assert set(out["condition_name"]) <= {
            "fleet_regret_closed_fraction", "fuel_pathway_label_agreement_share",
            "fuel_plan_agreement_1pp_share", "mean_rci_proxy"}

    def test_closure_threshold_without_anchor_raises(self):
        from scripts.run_fueleu_internalization_final import _threshold_long_table
        base = {"scenario_id": "S", "year": 2035, "blend_steps": 51}
        curve = pd.DataFrame([
            {**base, "lambda_fueleu": lam, "fleet_regret_closed_fraction": c,
             "fuel_pathway_label_agreement_share": c,
             "fuel_plan_agreement_1pp_share": c, "mean_rci_proxy": 0.1}
            for lam, c in ((0.80, 0.5), (0.85, 0.9))
        ])
        with pytest.raises(ValueError, match="no same-resolution lambda=0 anchor"):
            _threshold_long_table(curve)


class TestVesselThresholdResolutionSeparation:

    def _vessel_rows(self, blend_steps: int, match_from: float) -> list[dict]:
        return [{
            "scenario_id": "CENTRAL_EXTENDED", "case": "FI_PRIMARY_NO_CII",
            "year": 2035, "blend_steps": blend_steps, "vessel_id": "v1",
            "ship_type": "tanker", "lambda_fueleu": lam,
            "fuel_pathway_label_match": lam >= match_from,
            "fuel_plan_match_1pp": lam >= match_from,
            "full_decision_exact": lam >= match_from,
            "full_decision_1pp_0p01kn": lam >= match_from,
            "delta_cost_eur": 100.0 * max(0.0, 1.0 - lam),
        } for lam in (0.0, 0.25, 0.5, 0.75, 1.0)]

    def test_same_vessel_yields_one_row_per_resolution(self):
        from scripts.run_fueleu_internalization_final import (
            _vessel_thresholds_by_resolution,
        )
        df = pd.DataFrame(self._vessel_rows(11, 0.25)
                          + self._vessel_rows(21, 0.50)
                          + self._vessel_rows(51, 0.75))
        out = _vessel_thresholds_by_resolution(df)
        assert len(out) == 3, "one row per blend resolution, never pooled"
        got = out.set_index("blend_steps")["first_lambda_fuel_pathway_label_match"]
        assert got.loc[11] == pytest.approx(0.25)
        assert got.loc[21] == pytest.approx(0.50)
        assert got.loc[51] == pytest.approx(0.75)

    def test_window_without_anchor_is_left_censored_not_dated(self):
        from scripts.run_fueleu_internalization_final import (
            _vessel_thresholds_by_resolution,
        )
        rows = [{
            "scenario_id": "NO_FAME_2045", "case": "FI_PRIMARY_NO_CII",
            "year": 2045, "blend_steps": 51, "vessel_id": "v1",
            "ship_type": "tanker", "lambda_fueleu": lam,
            "fuel_pathway_label_match": True, "fuel_plan_match_1pp": True,
            "full_decision_exact": True, "full_decision_1pp_0p01kn": True,
            "delta_cost_eur": 0.0,
        } for lam in (0.80, 0.85)]
        out = _vessel_thresholds_by_resolution(pd.DataFrame(rows))
        assert bool(out["fuel_plan_agreement_1pp_left_censored"].iloc[0]) is True
        assert pd.isna(out["first_lambda_fuel_plan_agreement_1pp"].iloc[0])


class TestFeasibilityAuditScopes:

    def _fleet_csv(self, tmp_path, n_bulk=3, n_container=2):
        rows = []
        for i in range(n_bulk):
            rows.append({"imo_number": f"IMO900000{i}", "ship_type": "bulk_carrier",
                         "gross_tonnage": 45000, "deadweight": 82000,
                         "design_speed_knots": 14.5})
        for i in range(n_container):
            rows.append({"imo_number": f"IMO800000{i}", "ship_type": "container_ship",
                         "gross_tonnage": 95000, "deadweight": 105000,
                         "design_speed_knots": 22.0})
        p = tmp_path / "fleet.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        return p

    def test_both_scopes_present_with_distinct_denominators(self):
        from scripts.run_fueleu_internalization_final import _feasibility_audit
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            fleet = self._fleet_csv(pathlib.Path(td))
            opt_ids = {f"IMO900000{i}" for i in range(3)}
            out = _feasibility_audit(fleet, opt_ids, vessel_id_fn=lambda imo: imo)
        assert set(out["sample_scope"]) == {"matched_fleet", "optimization_sample"}
        matched_all = out[(out["sample_scope"] == "matched_fleet")
                          & (out["ship_type"] == "ALL")
                          & (out["scenario_id"] == "CENTRAL_EXTENDED")]
        opt_all = out[(out["sample_scope"] == "optimization_sample")
                      & (out["ship_type"] == "ALL")
                      & (out["scenario_id"] == "CENTRAL_EXTENDED")]
        assert int(matched_all["n_vessels"].iloc[0]) == 5
        assert int(opt_all["n_vessels"].iloc[0]) == 3

    def test_ship_type_rows_sum_to_all_row(self):
        from scripts.run_fueleu_internalization_final import _feasibility_audit
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            fleet = self._fleet_csv(pathlib.Path(td))
            out = _feasibility_audit(fleet, {f"IMO900000{i}" for i in range(3)},
                                     vessel_id_fn=lambda imo: imo)
        for (scope, scen), g in out.groupby(["sample_scope", "scenario_id"]):
            per_type = g[g["ship_type"] != "ALL"]["n_vessels"].sum()
            all_row = int(g[g["ship_type"] == "ALL"]["n_vessels"].iloc[0])
            assert int(per_type) == all_row, f"{scope}/{scen} does not reconcile"


class TestZeroBlendFuelIdentity:

    def test_zero_blend_mgo_candidate_reads_as_vlsfo_only(self):
        from carbon_twin.optimization.fuel_choice import (
            _energy_dominant_fuel,
            _fuel_pathway_label,
        )
        assert _fuel_pathway_label("VLSFO", "MGO", 0.0) == "VLSFO_only"
        assert _energy_dominant_fuel("VLSFO", "MGO", 0.0) == "VLSFO"

    def test_decision_dict_reports_pathway_not_candidate_label(self):
        prof = pd.DataFrame([{
            "alt_fuel": "MGO", "blend_pct": 0.0, "speed_knots": 14.0,
            "fuel_cost": 100.0, "ets_cost": 0.0, "delay_cost": 0.0,
            "fueleu_penalty_full": 0.0, "cii_penalty": 0.0,
            "fueleu_balance": 0.0, "fueleu_actual_intensity": 0.0,
            "fueleu_energy_mj": 0.0, "is_biofuel_blend": False,
        }])
        d = partial_planner_decision(prof, 0.0, include_cii_in_objective=False)
        assert d["partial_alt_fuel"] == "MGO"
        assert d["fuel_pathway_partial"] == "VLSFO_only"
        assert d["energy_dominant_fuel_partial"] == "VLSFO"


class TestNoNaNLambdaInCurveFiles:

    def test_curve_files_have_numeric_lambda(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        curve = pd.DataFrame([
            {"scenario_id": "S", "year": 2035, "blend_steps": 11,
             "lambda_fueleu": lam, "mean_rci_proxy": 0.1}
            for lam in (0.0, 0.5, 1.0)])
        p = out / "blend_resolution_curve.csv"
        curve.to_csv(p, index=False)
        got = pd.read_csv(p)
        assert got["lambda_fueleu"].notna().all()
        assert pd.api.types.is_numeric_dtype(got["lambda_fueleu"])


class TestPackageChecklistConsistency:
    pass


class TestFleetModalPathwayTransitionMatrix:

    def test_transition_matrix_reconciles_with_vessel_count(self):
        from scripts.run_fueleu_internalization_final import _transition_matrix

        vessel = pd.DataFrame([
            {"scenario_id": "CENTRAL_EXTENDED", "year": 2035, "blend_steps": 11,
             "lambda_fueleu": 0.0, "vessel_id": "v1",
             "partial_fuel_pathway": "MGO_only", "joint_fuel_pathway": "LNG_only"},
            {"scenario_id": "CENTRAL_EXTENDED", "year": 2035, "blend_steps": 11,
             "lambda_fueleu": 0.0, "vessel_id": "v2",
             "partial_fuel_pathway": "MGO_only", "joint_fuel_pathway": "LNG_only"},
            {"scenario_id": "CENTRAL_EXTENDED", "year": 2035, "blend_steps": 11,
             "lambda_fueleu": 0.0, "vessel_id": "v3",
             "partial_fuel_pathway": "VLSFO_only", "joint_fuel_pathway": "LNG_only"},
        ])
        tm = _transition_matrix(vessel)
        assert int(tm["n_vessels"].sum()) == 3
        assert tm["share"].sum() == pytest.approx(1.0)
        mgo_row = tm[tm["partial_fuel_pathway"].eq("MGO_only")].iloc[0]
        assert mgo_row["n_vessels"] == 2
        assert mgo_row["share"] == pytest.approx(2 / 3)


class TestFinalValidationReconciliation:

    def test_reconciliation_check_fails_on_broken_by_type_sum(self):
        from scripts.run_fueleu_internalization_final import _run_validation

        key = {"scenario_id": "CENTRAL_EXTENDED", "year": 2035, "blend_steps": 11,
              "lambda_fueleu": 0.0}
        curve = pd.DataFrame([{**key, "n_vessels": 3, "fleet_delta_cost_eur": 300.0}])
        vessel = pd.DataFrame([
            {**key, "vessel_id": f"v{i}", "delta_cost_eur": 0.0,
             "joint_total_cost_full_eur": 100.0} for i in range(3)
        ])
        by_type = pd.DataFrame([{**key, "ship_type": "bulk_carrier", "n_vessels": 2,
                                 "fleet_delta_cost_eur": 300.0}])
        checks = _run_validation(curve, vessel, by_type, pd.DataFrame())
        recon = [c for c in checks if c["check_name"] == "aggregate_reconciliation_n"]
        assert len(recon) == 1
        assert recon[0]["status"] == "FAIL"

    def test_reconciliation_check_passes_on_consistent_input(self):
        from scripts.run_fueleu_internalization_final import _run_validation

        key = {"scenario_id": "CENTRAL_EXTENDED", "year": 2035, "blend_steps": 11,
              "lambda_fueleu": 0.0}
        curve = pd.DataFrame([{**key, "n_vessels": 3, "fleet_delta_cost_eur": 300.0}])
        vessel = pd.DataFrame([
            {**key, "vessel_id": f"v{i}", "delta_cost_eur": 0.0,
             "joint_total_cost_full_eur": 100.0} for i in range(3)
        ])
        by_type = pd.DataFrame([{**key, "ship_type": "bulk_carrier", "n_vessels": 3,
                                 "fleet_delta_cost_eur": 300.0}])
        checks = _run_validation(curve, vessel, by_type, pd.DataFrame())
        recon = [c for c in checks if c["check_name"] == "aggregate_reconciliation_n"]
        assert len(recon) == 1
        assert recon[0]["status"] == "PASS"
