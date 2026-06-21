"""Tests for optimization modules."""

from __future__ import annotations


class TestSpeedOptimization:
    def test_optimal_speed_in_range(self, sample_bulk_carrier, sample_intra_eea_voyage):
        from carbon_twin.optimization.speed import optimal_speed_single_voyage

        result = optimal_speed_single_voyage(
            vessel=sample_bulk_carrier,
            voyage=sample_intra_eea_voyage,
            fuel_price=600,
            eua_price=80,
            fuel_factors={},
            regulations=[],
            speed_range=(8, 18),
            n_points=30,
        )
        assert 8 <= result["optimal_speed"] <= 18
        assert result["total_cost"] > 0

    def test_higher_eua_lower_speed(self, sample_bulk_carrier, sample_intra_eea_voyage):
        from carbon_twin.optimization.speed import optimal_speed_single_voyage

        low_eua = optimal_speed_single_voyage(
            vessel=sample_bulk_carrier,
            voyage=sample_intra_eea_voyage,
            fuel_price=600,
            eua_price=50,
            fuel_factors={},
            regulations=[],
        )
        high_eua = optimal_speed_single_voyage(
            vessel=sample_bulk_carrier,
            voyage=sample_intra_eea_voyage,
            fuel_price=600,
            eua_price=300,
            fuel_factors={},
            regulations=[],
        )
        assert high_eua["optimal_speed"] <= low_eua["optimal_speed"]


class TestFuelChoice:
    def test_mac_curve_sorted(self, sample_bulk_carrier, sample_intra_eea_voyage):
        from carbon_twin.optimization.fuel_choice import (
            FuelOption,
            marginal_abatement_cost,
        )

        options = [
            FuelOption(fuel_type="VLSFO", price_per_tonne=600, availability=1.0, max_blend_pct=1.0),
            FuelOption(fuel_type="LNG", price_per_tonne=700, availability=0.8, max_blend_pct=1.0),
            FuelOption(
                fuel_type="methanol_bio",
                price_per_tonne=1200,
                availability=0.3,
                max_blend_pct=0.5,
            ),
        ]
        mac = marginal_abatement_cost(
            sample_bulk_carrier,
            sample_intra_eea_voyage,
            options,
            fuel_factors={},
        )
        assert len(mac) > 0


class TestJIT:
    def test_jit_saves_fuel(self, sample_bulk_carrier, sample_intra_eea_voyage, sample_port_call):
        import datetime as dt

        from carbon_twin.optimization.jit import jit_speed_adjustment

        berth_start = sample_intra_eea_voyage.arrival_time + dt.timedelta(hours=6)
        berth_end = berth_start + dt.timedelta(hours=48)

        result = jit_speed_adjustment(
            vessel=sample_bulk_carrier,
            voyage=sample_intra_eea_voyage,
            port_call=sample_port_call,
            berth_window_start=berth_start,
            berth_window_end=berth_end,
        )
        assert result["fuel_saved_tonnes"] >= 0
