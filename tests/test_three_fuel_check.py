import numpy as np

from carbon_twin.model_variants import load_variant
from carbon_twin.pipeline import _SCOPE_ALLOCATION
from scripts.r1_closed_form_thresholds import (
    _MENUS, _components, _fuel_table, _fueleu_params, _plans, _target,
)
from scripts.r1_three_fuel_check import mixture_costs, simplex_grid

V = "r1_lng_otto_ms_prio_fb_bd"


def test_mixture_cost_equals_two_fuel_plans():
    v = load_variant(V)
    mode = _SCOPE_ALLOCATION[v.scope_mode]
    table = _fuel_table(V, "base_2026")
    target = _target(_fueleu_params(V), 2035)
    alts, _ = _MENUS["broad"]
    for s in (0.5, 0.8, 1.0):
        plans = _plans(table, alts, None, ["LNG"],
                       exact={"target": target, "mode": mode, "s": s, "rfnbo_mult": 1.0})
        two = _components(table, plans, 80.0, target, 1.0, s, mode)
        for r in two.itertuples():
            fl = [table[r.base], table[r.alt]]
            b, fe = mixture_costs(np.array([[1 - r.share, r.share]]), fl, 80.0, target, s,
                                  np.ones(2))
            assert abs(b[0] - (r.fuel + r.ets)) < 1e-9
            assert abs(fe[0] - r.fueleu) < 1e-9


def test_simplex_grid_covers_the_simplex():
    g = simplex_grid(3, 0.25)
    assert np.allclose(g.sum(axis=1), 1.0) and (g >= 0).all()
    assert len(g) == 15
