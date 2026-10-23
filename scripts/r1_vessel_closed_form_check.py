from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall: pip install -e '.[all]'")

from carbon_twin.model_variants import available_variants, load_variant
from scripts.r1_lambda_refine import with_refined

CLOSE = "fueleu_internalization_vessel_close.parquet"
_CF_SCRIPT = Path(__file__).with_name("r1_closed_form_thresholds.py")
_GAS = {"gas_carrier", "lng_carrier"}
_MENU = {"CENTRAL_EXTENDED": "broad", "NO_FAME_2045": "no_fame"}
_FINE_STEP, _FINE_MAX = 0.001, 0.05


def _menu(scenario: str, ship_type: str) -> str | None:
    if scenario == "COMPATIBILITY_CONSTRAINED_2035":
        return "compat_gas" if ship_type in _GAS else "compat_non_gas"
    return _MENU.get(scenario)


def closed_form_tables(variant: str, menus: list[str], years: list[int], s_grid: list[float],
                       steps: str, workdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = workdir / variant
    subprocess.run([sys.executable, str(_CF_SCRIPT), "--variants", variant,
                    "--menus", ",".join(menus), "--years", ",".join(map(str, years)),
                    "--scope-weights", ",".join(f"{s:.4f}" for s in s_grid),
                    "--blend-steps", steps, "--out-dir", str(out)], check=True,
                   capture_output=True, text=True)
    summ = pd.read_csv(out / "closed_form_summary.csv")
    trans = pd.read_csv(out / "closed_form_lambda_transitions.csv")
    return summ, trans


def _path_label(plan: str) -> str:
    return str(plan).split("@")[0]


def pathway_switches(trans: pd.DataFrame, summ: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in summ.groupby(keys):
        start, final = _path_label(g["lambda0_plan"].iat[0]), _path_label(g["joint_plan"].iat[0])
        t = trans.set_index(keys).loc[[key]] if key in trans.set_index(keys).index else None
        lam = np.nan
        if t is not None:
            cur = start
            for r in t.sort_values("lambda_exact").itertuples():
                new = _path_label(r.to_plan)
                if new == final and cur != final:
                    lam = r.lambda_exact
                cur = new
        rows.append({**dict(zip(keys, key)), "path_switch_cf": start != final,
                     "lambda_path_cf": lam})
    return pd.DataFrame(rows).set_index(keys)


def check_variant(root: Path, variant: str, s_step: float, tol: float,
                  workdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    v = load_variant(variant)
    c = with_refined(pd.read_parquet(root / variant / "lambda" / CLOSE), root / variant)
    c["menu"] = [_menu(sc, st) for sc, st in zip(c["scenario_id"], c["ship_type"].astype(str))]
    c = c[c["menu"].notna()].copy()
    if c.empty:
        return pd.DataFrame(), pd.DataFrame()
    full_scope = v.scope_mode == "full"
    n_s = int(round(1.0 / s_step))
    fine = [round(i * _FINE_STEP, 6) for i in range(int(round(_FINE_MAX / _FINE_STEP)))]
    s_grid = [1.0] if full_scope else sorted(
        set(fine) | {round(i * s_step, 6) for i in range(n_s + 1) if i * s_step >= _FINE_MAX})
    steps = "exact" if v.exact_blend else ",".join(
        str(int(b)) for b in sorted(c["blend_steps"].unique()))
    summ, trans = closed_form_tables(variant, sorted(c["menu"].unique()),
                                     sorted(int(y) for y in c["year"].unique()), s_grid, steps,
                                     workdir)
    keys = ["menu", "blend_steps", "year", "scope_weight"]
    for df in (summ, trans):
        df["blend_steps"] = df["blend_steps"].astype(str)
        df["scope_weight"] = df["scope_weight"].round(6)
    lam = trans.groupby(keys)["lambda_exact"].max().rename("lambda_cf")
    reg = summ.set_index(keys)["lambda0_regret_share"].rename("regret0_cf")
    paths = pathway_switches(trans, summ, keys)
    c["blend_key"] = "exact" if v.exact_blend else c["blend_steps"].astype(int).astype(str)
    s = c["scope_weight"].fillna(1.0).clip(0.0, 1.0)
    c["s_grid"] = 1.0 if full_scope else np.where(
        s < _FINE_MAX - _FINE_STEP / 2, np.round(s / _FINE_STEP) * _FINE_STEP,
        np.round(s / s_step) * s_step).round(6)
    idx = pd.MultiIndex.from_arrays([c["menu"], c["blend_key"], c["year"].astype(int),
                                     c["s_grid"]], names=keys)
    c["lambda_cf"] = lam.reindex(idx).to_numpy()
    c["regret0_cf"] = reg.reindex(idx).to_numpy()
    c["path_switch_cf"] = paths["path_switch_cf"].reindex(idx).to_numpy()
    c["lambda_path_cf"] = paths["lambda_path_cf"].reindex(idx).to_numpy()
    pos = c["rci_lambda0"] > 1e-9
    def _within(lo: str, hi: str) -> pd.Series:
        return (c["lambda_cf"] > c[lo] - tol) & (c["lambda_cf"] <= c[hi] + tol)

    c["inside"] = pos & _within("lambda_open", "lambda_close")
    c["inside_1pct"] = (pos & _within("lambda_open_1pct", "lambda_close_1pct")
                        if "lambda_open_1pct" in c else np.nan)
    c["zero_agree"] = ~pos & (c["regret0_cf"].fillna(0.0) <= 1e-6)
    c["width"] = c["lambda_close"] - c["lambda_open"]
    has_path = "lambda_path_open" in c and len(v.base_fuels) > 1
    if has_path:
        sw_sweep = ~c["path_match_lambda0"].astype(bool)
        sw_cf = c["path_switch_cf"].astype(bool)
        c["switch_agree"] = sw_sweep == sw_cf
        c["switch_both"] = sw_sweep & sw_cf
        c["inside_path"] = c["switch_both"] & (
            c["lambda_path_cf"] > c["lambda_path_open"] - tol) & (
            c["lambda_path_cf"] <= c["lambda_path_close"] + tol)

    rows = []
    for (sc, yr, bs), g in c.groupby(["scenario_id", "year", "blend_steps"]):
        gp, gz = g[g["rci_lambda0"] > 1e-9], g[g["rci_lambda0"] <= 1e-9]
        for band, sel in (("all", slice(None)), ("s<0.5", gp["scope_weight"] < 0.4999),
                          ("s>=0.5", gp["scope_weight"] >= 0.4999)):
            gb = gp if band == "all" else gp[sel]
            rows.append({
                "variant": variant, "scenario_id": sc, "year": int(yr), "blend_steps": int(bs),
                "s_band": band, "n_vessels": int(len(g)) if band == "all" else np.nan,
                "n_positive_regret": int(len(gb)),
                "share_inside": round(float(gb["inside"].mean()), 5) if len(gb) else np.nan,
                "share_interval_le_0p005": round(float((gb["width"] <= 0.0051).mean()), 5)
                if len(gb) else np.nan,
                "median_abs_err_narrow": round(float((gb.loc[gb["width"] <= 0.0051, "lambda_cf"]
                                                      - (gb["lambda_open"] + gb["lambda_close"])
                                                      / 2).abs().median()), 5)
                if (gb["width"] <= 0.0051).any() else np.nan,
                "share_inside_1pct": round(float(gb["inside_1pct"].astype(float).mean()), 5)
                if (len(gb) and "lambda_open_1pct" in gb) else np.nan,
                "n_zero_regret": int(len(gz)) if band == "all" else np.nan,
                "share_zero_agree": round(float(gz["zero_agree"].mean()), 5)
                if (band == "all" and len(gz)) else np.nan,
                **({} if not has_path or band != "all" else {
                    "share_switch_agree": round(float(g["switch_agree"].mean()), 5),
                    "n_switch_both": int(g["switch_both"].sum()),
                    "share_inside_path": round(float(
                        g.loc[g["switch_both"], "inside_path"].mean()), 5)
                    if g["switch_both"].any() else np.nan}),
            })
    bad = c[(c["rci_lambda0"] > 1e-9) & ~c["inside"]]
    bad = bad.assign(variant=variant)[["variant", "scenario_id", "year", "blend_steps",
                                       "ship_type", "scope_weight", "rci_lambda0",
                                       "lambda_open", "lambda_close", "lambda_cf"]]
    return pd.DataFrame(rows), bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/r1"))
    ap.add_argument("--variants", default="",
                    help="comma-separated; default: every variant directory with the slim file")
    ap.add_argument("--s-step", type=float, default=0.01)
    ap.add_argument("--tolerance", type=float, default=0.006)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out = args.out_dir or args.root / "vessel_closed_form"
    out.mkdir(parents=True, exist_ok=True)
    known = set(available_variants())
    variants = ([x.strip() for x in args.variants.split(",") if x.strip()] or
                sorted(p.parent.parent.name for p in args.root.glob(f"*/lambda/{CLOSE}")
                       if p.parent.parent.name in known))
    summaries, bads = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for v in variants:
            summ, bad = check_variant(args.root, v, args.s_step, args.tolerance, Path(tmp))
            if not summ.empty:
                summaries.append(summ)
                bads.append(bad)
                a = summ[summ["s_band"] == "all"]
                share = np.average(a["share_inside"].fillna(0),
                                   weights=a["n_positive_regret"].clip(lower=1))
                print(f"{v:30s} cells {int(a['n_positive_regret'].sum()):7d}  inside {share:.4f}")
    if not summaries:
        sys.exit(f"no {CLOSE} under {args.root}")
    pd.concat(summaries).to_csv(out / "vessel_closed_form_summary.csv", index=False)
    (pd.concat(bads).groupby(["variant", "scenario_id", "year", "blend_steps"], group_keys=False)
       .head(300).to_csv(out / "vessel_closed_form_mismatches.csv", index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
