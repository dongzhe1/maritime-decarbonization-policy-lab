import itertools

import pandas as pd
import pytest

from scripts.r1_spec_factorial import FACTORS, shapley


def _cells(f):
    rows = [{**dict(zip(FACTORS, k)), "year": 2035, "regret": f(*k)}
            for k in itertools.product((0, 1), repeat=3)]
    return pd.DataFrame(rows)


def test_additive_effects_are_recovered():
    s = shapley(_cells(lambda a, b, c: 0.01 + 0.04 * a + 0.03 * b + 0.005 * c)).set_index("factor")
    assert s.loc["priority", "shapley"] == pytest.approx(0.04)
    assert s.loc["lng_base", "shapley"] == pytest.approx(0.03)
    assert s.loc["breakdown", "shapley"] == pytest.approx(0.005)


def test_values_add_up_to_central_minus_base_with_interaction():
    s = shapley(_cells(lambda a, b, c: 0.01 + 0.02 * a + 0.01 * b + 0.05 * a * b + 0.003 * c))
    s = s.set_index("factor")["shapley"]
    assert s[list(FACTORS)].sum() == pytest.approx(s["total (central - step 4a)"])
    assert s["priority"] - 0.02 == pytest.approx(s["lng_base"] - 0.01)
