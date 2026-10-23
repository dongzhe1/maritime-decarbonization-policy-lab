import pandas as pd

from scripts.r1_lambda_refine import CLOSE_OUT, switch_brackets, with_refined
from scripts.r1_slim_vessel_outputs import CLOSE


def _cell(scenario, year, steps, lo, hi, vessels=("a", "b")):
    return pd.DataFrame({"scenario_id": scenario, "year": year, "blend_steps": steps,
                         "vessel_id": list(vessels), "scope_weight": [1.0, 0.6][:len(vessels)],
                         "lambda_path_open": lo, "lambda_path_close": hi,
                         "path_match_lambda0": False})


def _variant(tmp_path):
    close = pd.concat([_cell("CENTRAL_EXTENDED", 2035, 11, 0.40, 0.45),
                       _cell("CENTRAL_EXTENDED", 2035, 21, 0.40, 0.45),
                       _cell("COMPATIBILITY_ENGINE", 2035, 11, 0.35, 0.40)], ignore_index=True)
    refined = pd.concat([_cell("CENTRAL_EXTENDED", 2035, 11, 0.415, 0.42),
                         _cell("COMPATIBILITY_ENGINE", 2035, 11, 0.36, 0.365)],
                        ignore_index=True)
    (tmp_path / "lambda").mkdir()
    (tmp_path / "lambda_refine").mkdir()
    close.to_parquet(tmp_path / "lambda" / CLOSE, index=False)
    refined.to_parquet(tmp_path / "lambda_refine" / CLOSE_OUT, index=False)
    return close


def test_refined_cells_replace_matrix_cells(tmp_path):
    close = _variant(tmp_path)
    m = with_refined(close, tmp_path)
    assert len(m) == len(close)
    g = m[m.scenario_id.eq("CENTRAL_EXTENDED") & m.blend_steps.eq(11)]
    assert g.lambda_path_open.tolist() == [0.415, 0.415]
    assert m[m.blend_steps.eq(21)].lambda_path_open.tolist() == [0.40, 0.40]


def test_subset_gets_only_its_own_refined_cells(tmp_path):
    close = _variant(tmp_path)
    sub = close[close.scenario_id.eq("CENTRAL_EXTENDED")]
    m = with_refined(sub, tmp_path)
    assert set(m.scenario_id) == {"CENTRAL_EXTENDED"}
    assert len(m) == len(sub)


def test_switch_brackets_intersect_grids_within_one_scenario(tmp_path):
    _variant(tmp_path)
    b = switch_brackets(tmp_path, "CENTRAL_EXTENDED", 2035)
    assert len(b) == 2
    assert b.lo.tolist() == [0.415, 0.415] and b.hi.tolist() == [0.42, 0.42]


def test_no_refined_file_returns_input(tmp_path):
    close = _cell("CENTRAL_EXTENDED", 2035, 11, 0.40, 0.45)
    assert with_refined(close, tmp_path) is close
