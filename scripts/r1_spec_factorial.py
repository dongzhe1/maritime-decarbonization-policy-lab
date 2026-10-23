from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall: pip install -e '.[all]'")

FACTORS = ("priority", "lng_base", "breakdown")
CELLS = {
    (0, 0, 0): "r1_lng_otto_ms_exact",
    (1, 0, 0): "r1_lng_otto_ms_prio",
    (0, 1, 0): "r1_lng_otto_ms_prop_fb",
    (1, 1, 0): "r1_lng_otto_ms_prio_fb",
    (0, 0, 1): "r1_lng_otto_ms_exact_bd",
    (1, 0, 1): "r1_lng_otto_ms_prio_bd",
    (0, 1, 1): "r1_lng_otto_ms_prop_fb_bd",
    (1, 1, 1): "r1_lng_otto_ms_prio_fb_bd",
}
CASE = "M1_actual_ETS_FuelEU_only"
MGO_WTW = 90.77


def cells(root: Path) -> pd.DataFrame:
    rows = []
    for key, v in CELLS.items():
        p = root / v / "metric_harmonization.csv"
        if not p.exists():
            continue
        meta = json.loads((root / v / "metric_harmonization.csv.variant.json").read_text())
        if meta.get("wtw_overrides", {}).get("MGO") != MGO_WTW or not meta.get("exact_blend"):
            print(f"warning: {v} was not run with MGO {MGO_WTW} and exact shares", file=sys.stderr)
        m = pd.read_csv(p)
        for _, r in m[m["case"] == CASE].iterrows():
            rows.append({**dict(zip(FACTORS, key)), "variant": v, "year": int(r["year"]),
                         "regret": r["mean_rci_silo"], "fleet_loss_eur_m": r["fleet_delta_cost_eur"] / 1e6})
    return pd.DataFrame(rows, columns=[*FACTORS, "variant", "year", "regret", "fleet_loss_eur_m"])


def shapley(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for yr, g in c.groupby("year"):
        val = {tuple(r[f] for f in FACTORS): r["regret"] for _, r in g.iterrows()}
        if len(val) < 2 ** len(FACTORS):
            continue
        n = len(FACTORS)
        for i, f in enumerate(FACTORS):
            phi = 0.0
            others = [j for j in range(n) if j != i]
            for k in range(n):
                for sub in itertools.combinations(others, k):
                    on = [1 if j in sub else 0 for j in range(n)]
                    with_i = on.copy(); with_i[i] = 1
                    w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                    phi += w * (val[tuple(with_i)] - val[tuple(on)])
            rows.append({"year": yr, "factor": f, "shapley": phi})
        rows.append({"year": yr, "factor": "total (central - step 4a)",
                     "shapley": val[(1, 1, 1)] - val[(0, 0, 0)]})
    return pd.DataFrame(rows, columns=["year", "factor", "shapley"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/r1"))
    args = ap.parse_args()
    c = cells(args.root)
    out = args.root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    c.to_csv(out / "spec_factorial.csv", index=False)
    s = shapley(c)
    s.to_csv(out / "spec_factorial_shapley.csv", index=False)
    print(f"{c.variant.nunique()} of {len(CELLS)} cells present")
    if not c.empty:
        print(c[c.year.isin([2035, 2045])].pivot_table(index=list(FACTORS), columns="year",
                                                       values="regret").round(4))
    if not s.empty:
        print(s[s.year.isin([2035, 2045])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
