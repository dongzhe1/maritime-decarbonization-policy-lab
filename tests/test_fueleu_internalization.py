"""Tests for the FuelEU planning-internalization sweep (lambda_fueleu).

Covers parameter validation, a toy analytical switching threshold, the
evaluation-weight invariant, endpoint equivalence, non-negative regret,
monotonicity, backward compatibility, determinism, and unambiguous fuel
identity for zero-share candidates.
"""

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
    """Two-action toy: A base=100 FuelEU=100; B base=140 FuelEU=0."""

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


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Toy analytical switching threshold
# ---------------------------------------------------------------------------

class TestToySwitchingThreshold:
    def test_switch_at_lambda_0p40(self):
        prof = _toy_profile()
        below = partial_planner_decision(prof, 0.39, include_cii_in_objective=False)
        above = partial_planner_decision(prof, 0.41, include_cii_in_objective=False)
        # below 0.40 the partial planner selects A; full-policy regret = 200-140
        assert below["partial_alt_fuel"] == "A"
        assert below["delta_cost_eur"] == pytest.approx(60.0)
        assert below["rci_proxy"] == pytest.approx(60.0 / 200.0)
        assert below["rci_joint_norm"] == pytest.approx(60.0 / 140.0)
        # above 0.40 it selects B; regret is zero
        assert above["partial_alt_fuel"] == "B"
        assert above["delta_cost_eur"] == pytest.approx(0.0)
        assert above["full_decision_match"] is True
        assert above["fuel_pathway_label_match"] is True
        assert above["blend_share_match_1pp"] is True
        assert above["fuel_plan_match_1pp"] is True
        assert above["speed_match_0p01kn"] is True

    def test_exact_tie_follows_deterministic_rule(self):
        # at lambda = 0.4 both actions score 140; the repository rule
        # (Series.idxmin -> first candidate in enumeration order) picks A
        d = partial_planner_decision(_toy_profile(), 0.40, include_cii_in_objective=False)
        assert d["partial_alt_fuel"] == "A"
        assert d["delta_cost_eur"] == pytest.approx(60.0)

    def test_benchmark_is_b_action(self):
        # the full benchmark (λ=1) is B with full cost 140
        d = partial_planner_decision(_toy_profile(), 1.0, include_cii_in_objective=False)
        assert d["benchmark_alt_fuel"] == "B"
        assert d["benchmark_total_cost_full_eur"] == pytest.approx(140.0)


# ---------------------------------------------------------------------------
# Decision-agreement semantics
# ---------------------------------------------------------------------------

class TestAgreementSemantics:
    """Label match is a pathway-label match, NOT an exact fuel-plan match — a
    90% FAME partial plan and a 100% FAME joint plan have the same label but
    differ in blend share."""

    def test_label_match_with_blend_mismatch(self):
        # A: base 100 F 100 (blend 0.0); B09: base 120 F 30 (blend 0.9);
        # B10: base 130 F 0 (blend 1.0). Benchmark = B10 (full cost 130).
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
        # at lambda=0.30: A=130, B09=129, B10=130 -> B09 uniquely selected
        # (lambda must lie in (2/7, 1/3) for B09 to beat both A and B10)
        d = partial_planner_decision(prof, 0.30, include_cii_in_objective=False)
        assert d["partial_alt_fuel"] == "B"
        assert d["benchmark_alt_fuel"] == "B"
        # label matches, but the plan does not (share 0.90 vs 1.00); the
        # composite plan metric must also be False
        assert d["fuel_pathway_label_match"] is True
        assert d["blend_share_match_1pp"] is False
        assert d["fuel_plan_match_1pp"] is False
        assert d["full_decision_match"] is False
        assert d["blend_share_abs_diff"] == pytest.approx(0.1)
        assert d["delta_cost_eur"] == pytest.approx(150.0 - 130.0)


class TestAgreementMetricEdgeCases:
    """Four agreement-metric edge cases: share agreement without pathway
    agreement, pathway+share within tolerance, plan agreement without speed
    agreement, and energy-dominant fuel never defining pathway."""

    def test_share_agreement_but_different_pathway(self):
        # LNG@0.5 (base=100,F=50) is partial-optimal at lambda=0 (score 100);
        # biodiesel_fame@0.5 (base=140,F=0, full=140) is benchmark-optimal
        # (140 < 150). Same numerical share, different pathway.
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
        assert d["blend_share_match_1pp"] is True          # share agreement
        assert d["fuel_pathway_label_match"] is False       # different pathway
        assert d["fuel_plan_match_1pp"] is False            # so no plan agreement
        assert d["fuel_pathway_partial"] == "VLSFO+LNG"
        assert d["fuel_pathway_benchmark"] == "VLSFO+biodiesel_fame"

    def test_pathway_and_share_within_1pp_gives_plan_agreement(self):
        # A: LNG@0.50 (base=100,F=100, full=200) is partial-optimal at lambda=0.
        # B: LNG@0.505 (base=100.4,F=0, full=100.4) is benchmark-optimal.
        # Same pathway label, share differs by 0.005pp (<=1pp) -> plan agreement,
        # but NOT exact (blend_diff > 1e-9).
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
        assert d["full_decision_exact"] is False   # share diff > 1e-9, not exact

    def test_same_plan_but_speed_mismatch_fails_tolerant_full_decision(self):
        # Same pathway and share; partial picks speed=14.0, benchmark speed=14.5
        # (diff=0.5kn > 0.01kn) -> plan agreement holds but tolerant
        # full-decision agreement must fail.
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
        # A 5% FAME blend is energy-dominated by the base fuel (VLSFO), but the
        # pathway must still show the alternative component, not collapse to
        # "VLSFO_only" -- pathway identity must not be inferred from the
        # energy-dominant label.
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
        # same decision (A) selected across these lambdas
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
        # at λ=1 the partial decision IS the benchmark row, so its full
        # components must equal the benchmark-side components exactly
        assert res["partial_alt_fuel"] == res["joint_alt_fuel"]
        assert res["partial_fuel_cost_full_eur"] == res["fuel_cost_joint_eur"]
        assert res["partial_fueleu_cost_full_eur"] == res["fueleu_penalty_joint_eur"]
        assert res["partial_total_cost_full_eur"] == res["cost_joint_eur"]


# ---------------------------------------------------------------------------
# Endpoint equivalence
# ---------------------------------------------------------------------------

class TestEndpointEquivalence:
    def test_optimizer_lambda0_equals_siloed(self, sample_bulk_carrier,
                                             sample_intra_eea_voyage,
                                             factors, engines):
        res = optimal_fuel_blend_speed_multi_regulation(
            **_optimizer_kwargs(sample_bulk_carrier, sample_intra_eea_voyage,
                                factors, engines))  # default λ = 0
        assert res["partial_alt_fuel"] == res["siloed_alt_fuel"]
        assert res["partial_blend_pct"] == pytest.approx(res["siloed_blend_pct"])
        assert res["partial_speed"] == pytest.approx(res["siloed_speed"])
        # full evaluation of the λ=0 decision == full burden at the siloed point
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
        # λ=0 partial decision == siloed decision (M1 baseline endpoint)
        assert (df0["partial_fuel"] == df0["siloed_fuel"]).all()
        # λ=1 partial decision == joint decision with zero regret
        assert (df1["partial_fuel"] == df1["joint_fuel"]).all()
        assert np.allclose(df1["delta_cost_partial_eur"], 0.0, atol=1e-6)
        assert bool(df1["full_decision_match"].all())


# ---------------------------------------------------------------------------
# Non-negative + monotonic regret
# ---------------------------------------------------------------------------

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
        """Fleet aggregate delta cost must also be non-increasing in lambda.
        Per-vessel monotonicity (tested above) does not by itself imply the
        fleet SUM is non-increasing if vessels were weighted or combined
        incorrectly, so this is checked independently across a multi-vessel
        fleet."""
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


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_canonical_regression_suite_still_passes(self):
        """Run at least one canonical pre-existing fuel experiment with the new
        parameter omitted: no numerical change beyond existing reproducibility
        tolerance. TestFuelDimensionCoupling in test_regression_magnitudes.py
        pins the pre-existing joint/siloed fuel-blend optimizer behaviour (RCI
        ordering, blend-share bounds) and never touches
        planner_fueleu_internalization -- re-running it here proves this
        feature's additions left that pre-existing, unrelated code path
        numerically unchanged, rather than only checking internal
        self-consistency of the new parameter itself."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q",
             "tests/test_regression_magnitudes.py::TestFuelDimensionCoupling"],
            cwd=str(pathlib.Path(__file__).resolve().parents[1]),
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"canonical pre-existing fuel experiment regressed:\n"
            f"{result.stdout[-3000:]}\n{result.stderr[-2000:]}")

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
        # the partial block must not alter the joint/siloed selection — the
        # siloed decision at λ=0 is the joint decision when FuelEU is neutral
        res0 = optimal_fuel_blend_speed_multi_regulation(
            **_optimizer_kwargs(sample_bulk_carrier, sample_intra_eea_voyage,
                                factors, engines))
        res1 = optimal_fuel_blend_speed_multi_regulation(
            **_optimizer_kwargs(sample_bulk_carrier, sample_intra_eea_voyage,
                                factors, engines, planner_fueleu_internalization=1.0))
        # joint selection (and its full cost) is identical in both calls
        assert res0["joint_alt_fuel"] == res1["joint_alt_fuel"]
        assert res0["joint_blend_pct"] == res1["joint_blend_pct"]
        assert res0["cost_joint_eur"] == res1["cost_joint_eur"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

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


class TestZeroBlendFuelIdentity:
    """A zero-share alternative candidate must not be reported as the consumed
    fuel: a candidate with 0% blend of MGO into a VLSFO base is a pure-VLSFO
    pathway, not "MGO"."""

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
        # the legacy label is MGO ...
        assert d["partial_alt_fuel"] == "MGO"
        # ... but the authoritative pathway/energy fields say VLSFO
        assert d["fuel_pathway_partial"] == "VLSFO_only"
        assert d["energy_dominant_fuel_partial"] == "VLSFO"

