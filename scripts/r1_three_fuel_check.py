from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall: pip install -e '.[all]'")

from carbon_twin.model_variants import load_variant
from carbon_twin.pipeline import _SCOPE_ALLOCATION
from scripts.r1_closed_form_thresholds import (
    _MENUS, _PENALTY_EUR_PER_T, _VLSFOE_MJ_PER_T, _components, _fuel_table, _fueleu_params,
    _plans, _target,
)

MENU = ["VLSFO", "LNG", "MGO", "biodiesel_fame", "methanol_bio", "methanol_e"]
SHARE_TOL = 1e-4
GAIN_TOL = 1e-9


def mixture_costs(alpha: np.ndarray, fuels: list, eua: float, target: float, s: float,
                  mult: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    price = np.array([f.price for f in fuels])
    ets = np.array([f.ets for f in fuels])
    wtw = np.array([f.wtw for f in fuels])
    base = alpha @ price + s * eua * (alpha @ ets) / 1000.0
    order = np.argsort(wtw / mult)
    room = np.full(alpha.shape[0], s)
    num = np.zeros(alpha.shape[0])
    den = np.zeros(alpha.shape[0])
    for k in order:
        take = np.minimum(alpha[:, k], room)
        room = room - take
        num += take * wtw[k]
        den += take * mult[k]
    intensity = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    fe = np.where(intensity > 0,
                  np.maximum(0.0, intensity - target) * s * 1000.0
                  / np.where(intensity > 0, intensity, 1.0) / _VLSFOE_MJ_PER_T * _PENALTY_EUR_PER_T,
                  0.0)
    return base, fe


def simplex_grid(k: int, step: float) -> np.ndarray:
    n = int(round(1.0 / step))
    if k == 2:
        a = np.arange(n + 1) / n
        return np.column_stack([a, 1 - a])
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    keep = (i + j) <= n
    a1, a2 = i[keep] / n, j[keep] / n
    return np.column_stack([a1, a2, 1 - a1 - a2])


def window(best: np.ndarray, step: float, width: int) -> np.ndarray:
    off = np.arange(-width, width + 1) * step
    if len(best) == 2:
        a1 = np.clip(best[0] + off, 0.0, 1.0)
        return np.column_stack([a1, 1.0 - a1])
    d1, d2 = np.meshgrid(off, off, indexing="ij")
    a1 = best[0] + d1.ravel()
    a2 = best[1] + d2.ravel()
    a3 = 1.0 - a1 - a2
    ok = (a1 >= 0) & (a2 >= 0) & (a3 >= -1e-12)
    return np.column_stack([a1[ok], a2[ok], np.maximum(a3[ok], 0.0)])


def polish(alpha: np.ndarray, val: float, fl: list, m: np.ndarray, eua: float, target: float,
           s: float, lam: float, step: float) -> tuple[np.ndarray, float]:
    for k in (10, 100, 1000):
        pts = window(alpha, step / k, 20)
        b, fe = mixture_costs(pts, fl, eua, target, s, m)
        tot = b + lam * fe
        j = int(np.argmin(tot))
        if tot[j] < val:
            alpha, val = pts[j], float(tot[j])
    return alpha, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="r1_lng_otto_ms_prio_fb_bd")
    ap.add_argument("--years", default="2030,2035,2040,2045")
    ap.add_argument("--scope-weights", default="0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--lambdas", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--step", type=float, default=0.002, help="share grid of the mixtures")
    ap.add_argument("--eua", type=float, default=80.0)
    ap.add_argument("--price-scenario", default="base_2026")
    ap.add_argument("--menu", default="broad", choices=["broad", "no_fame"])
    ap.add_argument("--out-dir", type=Path, default=Path("results/r1/three_fuel_check"))
    args = ap.parse_args()

    v = load_variant(args.variant)
    mode = _SCOPE_ALLOCATION[v.scope_mode]
    if mode != "fueleu_priority":
        sys.exit(f"{args.variant} does not use priority allocation ({mode})")
    table = _fuel_table(args.variant, args.price_scenario)
    params = _fueleu_params(args.variant)
    until = params.get("rfnbo_multiplier_until_year")
    extra = [b for b in v.base_fuels if b != "VLSFO"]
    alts, _ = _MENUS[args.menu]
    menu = [f for f in MENU if f in ["VLSFO", *alts]]
    lams = [float(x) for x in args.lambdas.split(",")]
    grids = {2: simplex_grid(2, args.step), 3: simplex_grid(3, args.step)}

    rows, checked = [], 0
    for yr in (int(y) for y in args.years.split(",")):
        target = _target(params, yr)
        rmult = (float(params.get("rfnbo_multiplier", 2.0))
                 if until is None or yr <= int(until) else 1.0)
        for s in (float(x) for x in args.scope_weights.split(",")):
            plans = _plans(table, alts, None, extra, exact={"target": target, "mode": mode,
                                                            "s": s, "rfnbo_mult": rmult})
            two = _components(table, plans, args.eua, target, rmult, s, mode)
            for r in two.itertuples():
                fl = [table[r.base], table[r.alt]]
                m = np.array([rmult if f.rfnbo else 1.0 for f in fl])
                b, fe = mixture_costs(np.array([[1 - r.share, r.share]]), fl, args.eua, target, s, m)
                assert abs(b[0] - (r.fuel + r.ets)) < 1e-9 and abs(fe[0] - r.fueleu) < 1e-9, r
                checked += 1
            combos = [c for k in (2, 3) for c in itertools.combinations(menu, k)]
            evals = []
            for c in combos:
                fl = [table[x] for x in c]
                m = np.array([rmult if f.rfnbo else 1.0 for f in fl])
                b, fe = mixture_costs(grids[len(c)], fl, args.eua, target, s, m)
                evals.append((c, fl, m, b, fe))
            for lam in lams:
                i2 = int(np.argmin(two["fuel"] + two["ets"] + lam * two["fueleu"]))
                best2 = float(two["fuel"].iat[i2] + two["ets"].iat[i2] + lam * two["fueleu"].iat[i2])
                best3, mix3 = np.inf, ""
                for c, fl, m, b, fe in evals:
                    tot = b + lam * fe
                    j = int(np.argmin(tot))
                    alpha, val = polish(grids[len(c)][j], float(tot[j]), fl, m, args.eua,
                                        target, s, lam, args.step)
                    if val < best3 - 1e-12:
                        used = [f"{x}@{a:.4f}" for x, a in zip(c, alpha) if a >= SHARE_TOL]
                        best3, mix3 = val, "+".join(used)
                rows.append({"menu": args.menu, "price_scenario": args.price_scenario,
                             "year": yr, "scope_weight": s, "lambda": lam,
                             "two_fuel_cost_eur_per_gj": best2, "two_fuel_plan": two["plan"].iat[i2],
                             "mixture_cost_eur_per_gj": best3, "mixture": mix3,
                             "gain_eur_per_gj": best2 - best3,
                             "gain_share": (best2 - best3) / best2,
                             "fuels_in_mixture": mix3.count("+") + 1,
                             "three_fuels_better": bool((best2 - best3) / best2 > GAIN_TOL
                                                        and mix3.count("+") == 2)})
    out = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / f"three_fuel_check_{args.menu}_{args.price_scenario}.csv", index=False)
    three = out[out.three_fuels_better]
    print(f"{checked} two-fuel plans reproduced by the mixture cost")
    print(f"{len(out)} cases (years x scope weights x lambdas); max gain "
          f"{out.gain_share.max():.2e} of cost ({out.gain_eur_per_gj.max():.4f} EUR/GJ); "
          f"cases where a three-fuel mixture is cheaper: {len(three)}")
    if len(three):
        print(three.sort_values("gain_share", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
