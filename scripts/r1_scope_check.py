from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall: pip install -e '.[all]'")

from carbon_twin.data.loaders import load_mrv_csv, load_vessels_csv
from carbon_twin.emissions.proxy import vessels_to_index
from carbon_twin.pipeline import (
    _MIN_CII_DISTANCE_NM,
    _mrv_distance,
    _norm_col,
    _num,
    mrv_breakdown_columns,
    mrv_breakdown_scope,
    mrv_port_term,
)

_BANDS = [-0.001, 0.0, 0.25, 0.4999, 1.0]
_BAND_LABELS = ["s=0", "0<s<=0.25", "0.25<s<0.5", "s>=0.5"]


def _classify(s_rep: float | None, s_bd: float | None, tol: float) -> str:
    if s_bd is None or s_rep is None:
        return "no_breakdown"
    if abs(s_rep - s_bd) <= tol:
        return "consistent"
    return "reported_below" if s_rep < s_bd else "reported_above"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrv", required=True, type=Path)
    ap.add_argument("--fleet", required=True, type=Path)
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--out-dir", type=Path, default=Path("results/r1/scope_check"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mrv = load_mrv_csv(args.mrv)
    fleet = vessels_to_index(load_vessels_csv(args.fleet))
    cols = mrv_breakdown_columns(mrv.columns)
    co2_cols = [c for c in mrv.columns if "co2" in _norm_col(c)]
    lines = [f"MRV file: {Path(args.mrv).name}", f"rows: {len(mrv)}", "",
             "breakdown columns matched:" if cols else
             "breakdown columns NOT matched (each category must match exactly one column):"]
    lines += [f"  {k}: {v}" for k, v in (cols or {}).items()]
    lines += ["", "all columns mentioning CO2:"] + [f"  {c}" for c in co2_cols]
    (args.out_dir / "columns_found.txt").write_text("\n".join(lines) + "\n")

    rows = []
    for r in mrv.to_dict(orient="records"):
        imo = str(r.get("imo_number", "")).strip()
        v = fleet.get(imo)
        total, ets = _num(r.get("total_co2_tonnes")), _num(r.get("co2_ets_tonnes"))
        s_rep = (min(max(ets / total, 0.0), 1.0)
                 if total and total > 0 and ets is not None and ets >= 0 else None)
        s_bd = mrv_breakdown_scope(r, cols) if cols else None
        dist = _mrv_distance(r)
        rec = {
            "imo_number": imo,
            "mrv_ship_type": r.get("ship_type"),
            "ship_type": v.ship_type.value if v else None,
            "flag": v.flag if v else None,
            "in_optimisation_sample": bool(v is not None and (dist or 0) >= _MIN_CII_DISTANCE_NM),
            "total_co2_tonnes": total,
            "co2_ets_tonnes": ets,
            "s_reported": s_rep,
            "s_breakdown": s_bd,
            "s_diff": (s_rep - s_bd) if (s_rep is not None and s_bd is not None) else None,
            "class": _classify(s_rep, s_bd, args.tolerance),
        }
        if cols:
            parts = {k: _num(r.get(c)) for k, c in cols.items()}
            rec.update({f"co2_{k}": x for k, x in parts.items()})
            rec["co2_port_term"] = mrv_port_term(parts)
            voy = sum(parts[k] or 0.0 for k in ("between", "departed", "arrived"))
            for lab, port in (("", rec["co2_port_term"]), ("_within", parts.get("port")),
                              ("_berth", parts.get("berth"))):
                rec[f"breakdown_sum_over_total{lab}"] = (
                    (voy + port) / total if total and port is not None else None)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["s_band"] = pd.cut(df["s_reported"], _BANDS, labels=_BAND_LABELS)
    df.to_csv(args.out_dir / "scope_check_vessel.csv", index=False)

    opt = df[df["in_optimisation_sample"]]
    summ = []
    for (band, cls), g in opt.groupby(["s_band", "class"], observed=True):
        summ.append({"s_band": band, "class": cls, "n": len(g),
                     "s_reported_median": round(float(g["s_reported"].median()), 4),
                     "s_breakdown_median": (round(float(g["s_breakdown"].median()), 4)
                                            if g["s_breakdown"].notna().any() else np.nan),
                     "co2_share_of_sample": round(float(
                         g["total_co2_tonnes"].sum() / opt["total_co2_tonnes"].sum()), 5)})
    pd.DataFrame(summ).to_csv(args.out_dir / "scope_check_summary.csv", index=False)

    low = opt[opt["s_reported"] < 0.4999]
    by_type = (low.groupby(["ship_type", "class"]).size().rename("n").reset_index()
               .sort_values("n", ascending=False))
    by_type.to_csv(args.out_dir / "scope_check_low_by_type.csv", index=False)
    by_flag = (low.groupby(["flag", "class"]).size().rename("n").reset_index()
               .sort_values("n", ascending=False))
    by_flag.to_csv(args.out_dir / "scope_check_low_by_flag.csv", index=False)

    md = ["# Scope-weight check", "", *[f"    {x}" for x in lines[:8]], "",
          f"Optimisation sample: {len(opt)} ships; reported s < 0.5: {len(low)}; "
          f"s = 0: {int((opt['s_reported'] == 0).sum())}.", "",
          "| s band | class | n | median s reported | median s breakdown | CO2 share |",
          "|---|---|---|---|---|---|"]
    md += [f"| {r['s_band']} | {r['class']} | {r['n']} | {r['s_reported_median']} | "
           f"{r['s_breakdown_median']} | {r['co2_share_of_sample']} |" for r in summ]
    if cols and opt["s_breakdown"].notna().any():
        d = opt["s_diff"].dropna()
        md += ["", f"s_reported - s_breakdown: median {d.median():.4f}, p5 {d.quantile(.05):.4f}, "
               f"p95 {d.quantile(.95):.4f}; ships with s_breakdown < 0.5: "
               f"{int((opt['s_breakdown'] < 0.4999).sum())}."]
        if "breakdown_sum_over_total" in opt:
            b = opt["breakdown_sum_over_total"].dropna()
            md += [f"breakdown sum / total CO2: median {b.median():.4f}, p1 {b.quantile(.01):.4f}, "
                   f"p99 {b.quantile(.99):.4f}."]
            closes = []
            for lab, name in (("", "port term used (larger column)"),
                              ("_within", "within ports only"), ("_berth", "at berth only")):
                c = f"breakdown_sum_over_total{lab}"
                if c in opt and opt[c].notna().any():
                    share = float(((opt[c] - 1).abs() < 0.005).mean())
                    closes.append(f"{name} {share:.3f}")
            md += ["share of the sample where the breakdown sum is within 0.5% of total CO2: "
                   + "; ".join(closes) + "."]
    md += ["", "Low-s ships by Sea-web type and class: scope_check_low_by_type.csv; by flag: "
           "scope_check_low_by_flag.csv; ship rows: scope_check_vessel.csv."]
    (args.out_dir / "scope_check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
