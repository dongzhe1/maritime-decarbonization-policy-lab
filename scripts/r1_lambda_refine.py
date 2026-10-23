from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall: pip install -e '.[all]'")

from scripts.r1_slim_vessel_outputs import _CLOSE_COLS, _PATH_COL, FULL, closing_lambda

SCENARIOS = ("CENTRAL_EXTENDED", "COMPATIBILITY_ENGINE")
GRID = 11
CLOSE_OUT = "fueleu_internalization_vessel_close_refined.parquet"
CURVE_OUT = "fueleu_internalization_all_curve_refined.csv"
_COLS = _CLOSE_COLS + [_PATH_COL]


def with_refined(close: pd.DataFrame, variant_dir: Path) -> pd.DataFrame:
    p = variant_dir / "lambda_refine" / CLOSE_OUT
    if not p.exists():
        return close
    cell = ["scenario_id", "year", "blend_steps"]
    r = pd.read_parquet(p).merge(close[cell].drop_duplicates(), on=cell)
    hit = close[cell].merge(r[cell].drop_duplicates(), how="left", indicator=True)["_merge"]
    return pd.concat([close[(hit == "left_only").to_numpy()], r[close.columns.intersection(r.columns)]],
                     ignore_index=True)


def switch_brackets(variant_dir: Path, scenario: str, year: int) -> pd.DataFrame:
    from scripts.r1_slim_vessel_outputs import CLOSE

    v = pd.read_parquet(variant_dir / "lambda" / CLOSE)
    v = with_refined(v[v.scenario_id.eq(scenario) & v.year.eq(year)], variant_dir)
    v = v[v.lambda_path_close.notna()]
    b = v.groupby("vessel_id").agg(lo=("lambda_path_open", "max"),
                                   hi=("lambda_path_close", "min"),
                                   s=("scope_weight", "first"))
    return b[b.hi >= b.lo]


def _widths(close: pd.DataFrame) -> pd.DataFrame:
    d = close[close["lambda_path_close"].notna() & close["lambda_path_open"].notna()]
    w = d["lambda_path_close"] - d["lambda_path_open"]
    return (d.assign(w=w).groupby(["scenario_id", "year"])
            .agg(n_switch=("w", "size"), share_le_0p005=("w", lambda x: (x <= 0.0051).mean()),
                 max_width=("w", "max")).reset_index())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/r1"))
    ap.add_argument("--variant", required=True)
    ap.add_argument("--full", type=Path, default=None,
                    help=f"full per-vessel matrix file (default <variant>/lambda/{FULL})")
    args = ap.parse_args()
    base = args.root / args.variant
    ref = base / "lambda_refine"
    full = args.full or base / "lambda" / FULL
    if not full.exists():
        sys.exit(f"missing {full}: the refinement needs the matrix's full per-vessel file")

    new, curves = [], []
    for sc in SCENARIOS:
        d = ref / sc
        if not (d / "fueleu_internalization_vessel.parquet").exists():
            print(f"skip {sc}: no refined run in {d}")
            continue
        new.append(pd.read_parquet(d / "fueleu_internalization_vessel.parquet", columns=_COLS))
        curves.append(pd.read_csv(d / "fueleu_internalization_all_curve.csv"))
    if not new:
        sys.exit(f"no refined runs under {ref}")
    new = pd.concat(new, ignore_index=True)
    scen = sorted(new["scenario_id"].unique())
    old = pd.read_parquet(full, columns=_COLS,
                          filters=[("scenario_id", "in", scen), ("blend_steps", "==", GRID)])
    before = closing_lambda(old)

    key = ["scenario_id", "year", "blend_steps", "vessel_id", "lambda_fueleu"]
    for df in (old, new):
        df["lambda_fueleu"] = df["lambda_fueleu"].astype(float).round(6)
    dup = new.merge(old[key], on=key).shape[0]
    merged = pd.concat([old, new], ignore_index=True).drop_duplicates(key, keep="first")
    after = closing_lambda(merged)
    after.to_parquet(ref / CLOSE_OUT, index=False)
    pd.concat(curves, ignore_index=True).to_csv(ref / CURVE_OUT, index=False)

    s = _widths(before).merge(_widths(after), on=["scenario_id", "year"],
                              suffixes=("_before", "_after"))
    s.to_csv(ref / "refine_summary.csv", index=False)
    added = {f"{sc} {yr}": len(set(g["lambda_fueleu"]) - set(
                 old.loc[old.scenario_id.eq(sc) & old.year.eq(yr), "lambda_fueleu"]))
             for (sc, yr), g in new.groupby(["scenario_id", "year"])}
    print(f"{len(new):,} refined rows ({dup:,} already in the matrix), "
          f"{len(old):,} matrix rows; new λ points per cell: {added}")
    print(s.to_string(index=False))
    print(f"wrote {ref / CLOSE_OUT}, {ref / CURVE_OUT}, {ref / 'refine_summary.csv'}")


if __name__ == "__main__":
    main()
