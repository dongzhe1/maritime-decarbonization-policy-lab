from __future__ import annotations

import pickle

import pandas as pd
import pytest

from carbon_twin.data.loaders import load_fuel_prices, load_mrv_csv, load_vessels_csv
from carbon_twin.emissions.proxy import load_fuel_factors_safe, vessels_to_index
from carbon_twin.model_variants import load_variant
from carbon_twin.parallel import n_blocks, row_block, run_blocked
from carbon_twin.pipeline import run_fuel_optimization_from_mrv, run_optimization_from_mrv

_REGS = dict(cii_version="IMO_CII_2023_phase1", fueleu_version="EU_2023_1805",
             ets_version="EU_ETS_Maritime_2024")
_WRK: dict = {}


def _init(pkl):
    _WRK.update(pickle.loads(pkl))


def _fuel_rows(task):
    if task.get("fail_block") == task["block"]:
        raise RuntimeError("boom")
    return run_fuel_optimization_from_mrv(
        row_block(_WRK["mrv"], task), _WRK["fleet"], task["year"], 80.0, _WRK["prices"],
        fuel_factors=_WRK["factors"], extended_fuels=True, blend=True,
        **_REGS, **_WRK["vkw"], return_rows=True)


def _speed_rows(task):
    return run_optimization_from_mrv(
        pd.DataFrame(), row_block(_WRK["mrv"], task), _WRK["fleet"], task["year"], 80.0,
        650.0, fuel_factors=_WRK["factors"], cii_pressure=task["k"], **_REGS,
        return_rows=True)


@pytest.fixture
def inputs(tmp_path):
    base = [
        ("bulk_carrier", 45000, 82000, 14.5, 12000, 30000.0, 18000.0, 2500.0, 32000.0, 9600.0),
        ("container_ship", 95000, 105000, 22.0, 55000, 50000.0, 50000.0, 2800.0, 38000.0,
         16000.0),
        ("tanker", 60000, 110000, 15.0, 15000, 25000.0, 9000.0, 2600.0, 30000.0, 8000.0),
        ("lng_carrier", 110000, 90000, 19.5, 30000, 40000.0, 30000.0, 3000.0, 45000.0,
         14500.0),
    ]
    mrv, fleet = [], []
    for k in range(5):
        for i, (st, gt, dwt, vd, kw, co2, ets, h, d, fuel) in enumerate(base):
            imo, s = str(9100000 + 10 * k + i), 0.6 + 0.2 * k
            mrv.append({"imo_number": imo, "total_co2_tonnes": co2 * s,
                        "co2_ets_tonnes": ets * s * (1.1 - 0.15 * k), "hours_at_sea": h,
                        "distance_nm": d, "total_fuel_tonnes": fuel * s})
            fleet.append({"imo_number": imo, "ship_type": st, "gross_tonnage": gt,
                          "deadweight": dwt, "design_speed_knots": vd,
                          "main_engine_power_kw": kw, "aux_engine_power_kw": 2000,
                          "year_built": 2010 + k})
    mrv.append({"imo_number": "9999999", "total_co2_tonnes": 1000.0})
    pd.DataFrame(mrv).to_csv(tmp_path / "mrv.csv", index=False)
    pd.DataFrame(fleet).to_csv(tmp_path / "fleet.csv", index=False)
    variant = load_variant("r1_lng_otto_ms_prio_fb")
    return {"mrv": load_mrv_csv(tmp_path / "mrv.csv"),
            "fleet": vessels_to_index(load_vessels_csv(tmp_path / "fleet.csv")),
            "prices": load_fuel_prices("configs/fuel_prices.yaml", scenario="base_2026"),
            "factors": variant.fuel_factors(load_fuel_factors_safe()),
            "vkw": variant.fuel_optimizer_kwargs()}


def test_row_blocks_partition_the_rows():
    for n_rows in (0, 1, 5, 21):
        df = pd.DataFrame({"x": range(n_rows)})
        for n in (1, 2, 3, 7, 64):
            parts = [row_block(df, {"block": b, "n_blocks": n}) for b in range(n)]
            assert pd.concat(parts)["x"].tolist() == list(range(n_rows))
    assert row_block(df, {}) is df


def test_n_blocks():
    assert n_blocks(20, 40, 12333) == 4
    assert n_blocks(30, 40, 12333) == 3
    assert n_blocks(100, 40, 12333) == 1
    assert n_blocks(1, 40, 12333) == 64
    assert n_blocks(1, 40, 10) == 10
    assert n_blocks(5, 1, 12333) == 1


def test_blocked_fuel_run_equals_serial(inputs):
    tasks = [{"year": 2035}, {"year": 2045}]
    got = dict((t["year"], df) for t, df in run_blocked(
        tasks, _fuel_rows, workers=4, n_rows=len(inputs["mrv"]),
        initializer=_init, initargs=(pickle.dumps(inputs),)))
    assert n_blocks(2, 4, len(inputs["mrv"])) == 4
    for yr in (2035, 2045):
        serial = run_fuel_optimization_from_mrv(
            inputs["mrv"], inputs["fleet"], yr, 80.0, inputs["prices"],
            fuel_factors=inputs["factors"], extended_fuels=True, blend=True, **_REGS,
            **inputs["vkw"])
        assert len(serial) == 20
        pd.testing.assert_frame_equal(got[yr], serial, check_exact=True)


def test_blocked_speed_run_equals_serial(inputs):
    tasks = [{"year": 2035, "k": 0.2}]
    [(_, got)] = list(run_blocked(tasks, _speed_rows, workers=3, n_rows=len(inputs["mrv"]),
                                  initializer=_init, initargs=(pickle.dumps(inputs),)))
    serial = run_optimization_from_mrv(
        pd.DataFrame(), inputs["mrv"], inputs["fleet"], 2035, 80.0, 650.0,
        fuel_factors=inputs["factors"], cii_pressure=0.2, **_REGS)
    pd.testing.assert_frame_equal(got, serial, check_exact=True)


def test_failed_block_fails_its_task_once(inputs):
    tasks = [{"year": 2035, "fail_block": 1}, {"year": 2045}]
    out = list(run_blocked(tasks, _fuel_rows, workers=4, n_rows=len(inputs["mrv"]),
                           initializer=_init, initargs=(pickle.dumps(inputs),)))
    assert len(out) == 2
    res = {t["year"]: r for t, r in out}
    assert isinstance(res[2035], RuntimeError)
    assert isinstance(res[2045], pd.DataFrame) and len(res[2045]) == 20
