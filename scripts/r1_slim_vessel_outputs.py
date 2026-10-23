from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall: pip install -e '.[all]'")

FULL = "fueleu_internalization_vessel.parquet"
LAMBDA0 = "fueleu_internalization_vessel_lambda0.parquet"
CLOSE = "fueleu_internalization_vessel_close.parquet"
_KEY = ["scenario_id", "year", "blend_steps", "vessel_id"]
_CLOSE_COLS = _KEY + ["lambda_fueleu", "rci_proxy", "ship_type", "scope_weight"]
_PATH_COL = "fuel_pathway_label_match"
_TOL = 1e-9


def _interval(df: pd.DataFrame, open_mask: pd.Series) -> tuple[pd.Series, pd.Series]:
    open_ = df[open_mask].groupby(_KEY, sort=False)["lambda_fueleu"].max()
    after = df.join(open_.rename("_open"), on=_KEY)
    after = after[after["_open"].isna() | (after["lambda_fueleu"] > after["_open"])]
    close = after.groupby(_KEY, sort=False)["lambda_fueleu"].min()
    return open_, close


def closing_lambda(df: pd.DataFrame, tol: float = _TOL) -> pd.DataFrame:
    df = df.sort_values(_KEY + ["lambda_fueleu"])
    g = df.groupby(_KEY, sort=False)
    out = g.agg(ship_type=("ship_type", "first"), scope_weight=("scope_weight", "first"),
                lambda_max=("lambda_fueleu", "max"), n_lambda=("lambda_fueleu", "size"))
    at0 = df[df["lambda_fueleu"] == 0.0].set_index(_KEY)
    out["rci_lambda0"] = at0["rci_proxy"]
    for suffix, mask in (("", df["rci_proxy"] > tol),
                         ("_1pct", df["rci_proxy"] > df.join(
                             out["rci_lambda0"].rename("_r0"), on=_KEY)["_r0"] * 0.01 + tol)):
        o, c = _interval(df, mask)
        out[f"lambda_open{suffix}"] = o
        out[f"lambda_close{suffix}"] = c
        out.loc[out[f"lambda_open{suffix}"].isna(), f"lambda_close{suffix}"] = 0.0
    if _PATH_COL in df:
        match = df[_PATH_COL].astype(bool)
        o, c = _interval(df, ~match)
        out["lambda_path_open"], out["lambda_path_close"] = o, c
        out.loc[out["lambda_path_open"].isna(), "lambda_path_close"] = 0.0
        out["path_match_lambda0"] = at0[_PATH_COL].astype(bool)
    out["closes_on_grid"] = out["lambda_close"].notna()
    return out.reset_index()


def slim_one(path: Path, delete: bool) -> dict:
    lam0 = pd.read_parquet(path, filters=[("lambda_fueleu", "==", 0.0)])
    lam0.to_parquet(path.with_name(LAMBDA0), index=False)
    import pyarrow.parquet as pq

    cols = _CLOSE_COLS + ([_PATH_COL] if _PATH_COL in pq.ParquetFile(path).schema_arrow.names
                          else [])
    close = closing_lambda(pd.read_parquet(path, columns=cols))
    close.to_parquet(path.with_name(CLOSE), index=False)
    n0 = len(pd.read_parquet(path.with_name(LAMBDA0), columns=["vessel_id"]))
    nc = len(pd.read_parquet(path.with_name(CLOSE), columns=["vessel_id"]))
    ok = n0 == len(lam0) and nc == len(close) and n0 > 0
    size = path.stat().st_size
    if delete and ok:
        path.unlink()
    return {"file": str(path), "full_mb": round(size / 1e6, 1), "lambda0_rows": n0,
            "close_rows": nc, "never_closes": int((~close["closes_on_grid"]).sum()),
            "deleted": bool(delete and ok)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/r1"))
    ap.add_argument("--variants", default="",
                    help="comma-separated variant directories (default: all under --root)")
    ap.add_argument("--delete", action="store_true",
                    help="remove the full per-vessel file after the slim files check out")
    args = ap.parse_args()
    wanted = {v.strip() for v in args.variants.split(",") if v.strip()}
    paths = [p for p in sorted(args.root.glob(f"*/lambda/{FULL}"))
             if not wanted or p.parent.parent.name in wanted]
    rows = [slim_one(p, args.delete) for p in paths]
    if not rows:
        sys.exit(f"no {FULL} under {args.root}")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"{df['full_mb'].sum() / 1e3:.2f} GB in full files; deleted: {int(df['deleted'].sum())}"
          f"/{len(df)}")
    if args.delete and not df["deleted"].all():
        sys.exit(1)


if __name__ == "__main__":
    main()
