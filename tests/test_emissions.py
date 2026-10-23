from __future__ import annotations

import numpy as np
import pytest

from carbon_twin.data.schemas import ShipType, VesselActivity
from carbon_twin.emissions.physics import (
    admiralty_power,
    aux_engine_fuel_rate,
    estimate_admiralty_coeff,
    main_engine_fuel_rate,
    speed_power_curve,
    voyage_fuel_consumption,
)


class TestAdmiraltyPower:
    def test_positive_power(self):
        power = admiralty_power(80000, 14.0, 50000)
        assert power > 0

    def test_power_increases_with_speed(self):
        p1 = admiralty_power(80000, 10.0, 50000)
        p2 = admiralty_power(80000, 14.0, 50000)
        assert p2 > p1

    def test_zero_speed_zero_power(self):
        power = admiralty_power(80000, 0.0, 50000)
        assert power == pytest.approx(0.0, abs=1e-6)


class TestSpeedPowerCurve:
    def test_curve_shape(self, sample_bulk_carrier):
        speeds = np.linspace(8, 16, 20)
        powers = speed_power_curve(sample_bulk_carrier, speeds)
        assert len(powers) == len(speeds)
        assert all(p >= 0 for p in powers)
        for i in range(1, len(powers)):
            assert powers[i] >= powers[i - 1]


class TestFuelRate:
    def test_main_engine_positive(self):
        rate = main_engine_fuel_rate(10000, sfoc_g_per_kwh=180)
        assert rate > 0

    def test_aux_engine_by_activity(self, sample_bulk_carrier):
        sea_rate = aux_engine_fuel_rate(sample_bulk_carrier, VesselActivity.UNDERWAY)
        berth_rate = aux_engine_fuel_rate(sample_bulk_carrier, VesselActivity.BERTHED)
        assert sea_rate > 0
        assert berth_rate > 0
        assert sea_rate >= berth_rate


class TestVoyageFuelConsumption:
    def test_positive_consumption(self, sample_bulk_carrier, sample_intra_eea_voyage):
        result = voyage_fuel_consumption(sample_bulk_carrier, sample_intra_eea_voyage)
        assert result["total_tonnes"] > 0
        assert result["energy_mj"] > 0

    def test_slower_speed_less_fuel(self, sample_bulk_carrier, sample_intra_eea_voyage):
        normal = voyage_fuel_consumption(sample_bulk_carrier, sample_intra_eea_voyage)
        slow_voyage = sample_intra_eea_voyage.model_copy(update={"avg_speed_knots": 8.0})
        slow = voyage_fuel_consumption(sample_bulk_carrier, slow_voyage)
        assert slow["main_engine_tonnes"] < normal["main_engine_tonnes"]
