from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall: pip install -e '.[all]'")

from carbon_twin.data.loaders import load_fuel_prices
from carbon_twin.emissions.factors import GWP100
from carbon_twin.emissions.proxy import load_fuel_factors_safe
from carbon_twin.model_variants import available_variants, load_variant
from carbon_twin.optimization.fuel_choice import exact_blend_points, scope_energy_allocation
from carbon_twin.pipeline import _SCOPE_ALLOCATION
from carbon_twin.regulations.fueleu import FuelEUEngine

_BIOGENIC = {"biofuel", "rfnbo"}
_VLSFOE_MJ_PER_T = 41000.0
_PENALTY_EUR_PER_T = 2400.0

_MENUS: dict[str, tuple[list[str], str]] = {
    "broad": (["MGO", "biodiesel_fame", "LNG", "methanol_bio", "methanol_e"],
              "CENTRAL_EXTENDED: every component available to every vessel"),
    "compat_non_gas": (["MGO", "biodiesel_fame"],
                       "COMPATIBILITY_CONSTRAINED (non-gas vessels): drop-ins only"),
    "compat_gas": (["MGO", "biodiesel_fame", "LNG"],
                   "COMPATIBILITY_CONSTRAINED (gas/LNG carriers): drop-ins + LNG"),
    "no_fame": (["MGO", "LNG", "methanol_bio", "methanol_e"],
                "NO_FAME_2045: broad menu without FAME"),
}


@dataclass(frozen=True)
class FuelPerGJ:
    price: float
    ets: float
    wtw: float
    rfnbo: bool


def _fuel_table(variant_name: str, price_scenario: str) -> dict[str, FuelPerGJ]:
    v = load_variant(variant_name)
    factors = v.fuel_factors(load_fuel_factors_safe())
    prices = load_fuel_prices("configs/fuel_prices.yaml", scenario=price_scenario)
    out = {}
    for ft, ef in factors.items():
        if ft.value not in prices or ef.lcv_mj_per_kg <= 0:
            continue
        lcv_mj_per_g = ef.lcv_mj_per_kg / 1000.0
        co2 = 0.0 if ef.category in _BIOGENIC else ef.ttw_co2_g_per_g_fuel
        per_g = (co2 + ef.ttw_ch4_g_per_g_fuel * GWP100["CH4"]
                 + ef.ttw_n2o_g_per_g_fuel * GWP100["N2O"])
        out[ft.value] = FuelPerGJ(
            price=float(prices[ft.value]) / ef.lcv_mj_per_kg,
            ets=per_g / lcv_mj_per_g,
            wtw=ef.wtw_co2e_g_per_mj,
            rfnbo=ef.category == "rfnbo",
        )
    return out


def _fueleu_params(variant_name: str) -> dict:
    v = load_variant(variant_name)
    eng = FuelEUEngine(version=v.fueleu_version_or("EU_2023_1805"))
    return eng._params


def _target(params: dict, year: int) -> float:
    ref = float(params["reference_value_gCO2e_per_MJ"])
    cuts = {int(k): float(x) for k, x in params["reduction_targets"].items()}
    applicable = [y for y in cuts if y <= year]
    return ref * (1.0 - cuts[max(applicable)]) if applicable else ref


def _is_low_carbon(fuels: dict[str, FuelPerGJ], a: str) -> bool:
    return fuels.get(a) is not None and (fuels[a].rfnbo or a in ("biodiesel_fame", "methanol_bio"))


def _plans(fuels: dict[str, FuelPerGJ], alts: list[str], shares, extra_bases: list[str],
           exact: dict | None = None) -> list[tuple[str, str, float]]:
    def pair_shares(b: str, a: str, interior: bool) -> list[float]:
        if exact is None:
            grid = [float(x) for x in shares]
        else:
            mult = exact["rfnbo_mult"] if fuels[a].rfnbo else 1.0
            grid = [0.0, 1.0] + exact_blend_points(
                fuels[b].wtw, fuels[a].wtw, exact["target"], exact["mode"], exact["s"], mult)
        return [x for x in grid if (0.0 < x < 1.0) or not interior]

    plans = [("VLSFO", a, phi) for a in alts if a in fuels
             for phi in pair_shares("VLSFO", a, False)]
    for b in extra_bases:
        if b not in fuels or b not in alts:
            continue
        for a in alts:
            if a != b and _is_low_carbon(fuels, a):
                plans += [(b, a, phi) for phi in pair_shares(b, a, True)]
    return plans


def _label(b: str, a: str, phi: float) -> str:
    if phi <= 1e-9:
        return f"{b}_only"
    if phi >= 1 - 1e-9:
        return f"{a}_only"
    return f"{b}+{a}@{phi:.4f}"


def _components(fuels, plans, eua, target, rfnbo_mult, s, mode="uniform"):
    rows = []
    for b, a, phi in plans:
        fb, fa = fuels[b], fuels[a]
        mult = rfnbo_mult if fa.rfnbo else 1.0
        fuel = (1 - phi) * fb.price + phi * fa.price
        if mode in ("fueleu_priority", "routed_voyage"):
            al = scope_energy_allocation(phi, 1.0, s, mode, fa.ets <= fb.ets)
            ets = eua * (al["ets_alt_energy"] * fa.ets + al["ets_base_energy"] * fb.ets) / 1000.0
            pool = al["fueleu_alt_energy"] + al["fueleu_base_energy"]
            mixed = al["fueleu_alt_energy"] * fa.wtw + al["fueleu_base_energy"] * fb.wtw
            denom = pool + (mult - 1.0) * al["fueleu_alt_energy"]
            intensity = mixed / denom if denom > 0 else 0.0
            wtw_proxy = eua * (al["ets_alt_energy"] * fa.wtw
                               + al["ets_base_energy"] * fb.wtw) / 1000.0
        else:
            if mode == "routed":
                fuel = s * fuel + (1 - s) * fb.price
            pool = s
            ets = s * eua * ((1 - phi) * fb.ets + phi * fa.ets) / 1000.0
            mix = (1 - phi) * fb.wtw + phi * fa.wtw
            intensity = mix / (1.0 + (mult - 1.0) * phi)
            wtw_proxy = s * eua * mix / 1000.0
        fe = (max(0.0, intensity - target) * pool * 1000.0 / (intensity * _VLSFOE_MJ_PER_T)
              * _PENALTY_EUR_PER_T) if intensity > 0 else 0.0
        rows.append({"base": b, "alt": a, "share": phi, "plan": _label(b, a, phi),
                     "fuel": fuel, "ets": ets, "fueleu": fe,
                     "wtw_proxy": wtw_proxy, "intensity": intensity})
    return pd.DataFrame(rows)


def _lambda_path(df: pd.DataFrame, step: float) -> list[dict]:
    lams = np.round(np.arange(0.0, 1.0 + step / 2, step), 6)
    base = (df["fuel"] + df["ets"]).to_numpy()
    fe = df["fueleu"].to_numpy()
    full = base + fe
    picks = [int(np.argmin(base + lam * fe)) for lam in lams]
    out, prev = [], picks[0]
    for i in range(1, len(lams)):
        cur = picks[i]
        if cur != prev and df["plan"].iat[cur] != df["plan"].iat[prev]:
            d_fe = fe[prev] - fe[cur]
            lam_x = (base[cur] - base[prev]) / d_fe if d_fe > 0 else float("nan")
            out.append({"lambda_low": float(lams[i - 1]), "lambda_high": float(lams[i]),
                        "lambda_exact": lam_x,
                        "from_plan": df["plan"].iat[prev], "to_plan": df["plan"].iat[cur],
                        "regret_share_after": float((full[cur] - full.min()) / full[cur])})
        prev = cur
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="",
                    help="comma-separated; default: every variant in configs/model_variants.yaml")
    ap.add_argument("--years", default="2026,2030,2035,2040,2045")
    ap.add_argument("--menus", default="broad,compat_non_gas,compat_gas,no_fame")
    ap.add_argument("--scope-weights", default="1.0,0.75,0.55",
                    help="ETS/FuelEU scope weights s; variants with scope_mode=full use 1.0 only")
    ap.add_argument("--blend-steps", default="exact,11,21,51",
                    help="share grids; 'exact' = end points + exact-compliance shares + kinks")
    ap.add_argument("--eua", type=float, default=80.0)
    ap.add_argument("--price-scenario", default="base_2026")
    ap.add_argument("--lambda-step", type=float, default=0.0005)
    ap.add_argument("--out-dir", type=Path, default=Path("results/r1/closed_form"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.variants:
        args.variants = ",".join(available_variants())
    years = [int(y) for y in args.years.split(",")]
    weights = [float(w) for w in args.scope_weights.split(",")]
    summary, transitions = [], []
    for vname in [v.strip() for v in args.variants.split(",") if v.strip()]:
        variant = load_variant(vname)
        fuels = _fuel_table(vname, args.price_scenario)
        params = _fueleu_params(vname)
        until = params.get("rfnbo_multiplier_until_year")
        extra = [b for b in variant.base_fuels if b != "VLSFO"]
        mode = _SCOPE_ALLOCATION[variant.scope_mode]
        s_grid = weights if variant.scope_mode != "full" else [1.0]
        for menu in args.menus.split(","):
            alts, _ = _MENUS[menu]
            for steps_tok in [x.strip() for x in args.blend_steps.split(",")]:
                steps = 0 if steps_tok == "exact" else int(steps_tok)
                for yr in years:
                    target = _target(params, yr)
                    rfnbo_mult = (float(params.get("rfnbo_multiplier", 2.0))
                                  if until is None or yr <= int(until) else 1.0)
                    for s in s_grid:
                        plans = _plans(fuels, alts, np.linspace(0.0, 1.0, max(steps, 2)), extra,
                                       exact=None if steps else {
                                           "target": target, "mode": mode, "s": s,
                                           "rfnbo_mult": rfnbo_mult})
                        df = _components(fuels, plans, args.eua, target, rfnbo_mult, s, mode)
                        full = df["fuel"] + df["ets"] + df["fueleu"]
                        i0 = int((df["fuel"] + df["ets"]).idxmin())
                        iw = int((df["fuel"] + df["wtw_proxy"]).idxmin())
                        ij = int(full.idxmin())
                        key = {"variant": vname, "menu": menu,
                               "blend_steps": steps_tok, "scope_allocation": mode,
                               "year": yr, "scope_weight": s}
                        summary.append({
                            **key,
                            "fueleu_target_gco2e_per_mj": round(target, 3),
                            "lambda0_plan": df["plan"].iat[i0],
                            "wtw_valued_plan": df["plan"].iat[iw],
                            "joint_plan": df["plan"].iat[ij],
                            "joint_cost_eur_per_gj": round(float(full.iat[ij]), 4),
                            "lambda0_regret_eur_per_gj": round(
                                float(full.iat[i0] - full.iat[ij]), 4),
                            "lambda0_regret_share": round(float(
                                (full.iat[i0] - full.iat[ij]) / full.iat[i0]), 5),
                            "wtw_valued_regret_share": round(float(
                                (full.iat[iw] - full.iat[ij]) / full.iat[iw]), 5),
                            "joint_intensity_gco2e_per_mj": round(
                                float(df["intensity"].iat[ij]), 3),
                        })
                        for t in _lambda_path(df, args.lambda_step):
                            transitions.append({**key, **t})

    summ = pd.DataFrame(summary)
    trans = pd.DataFrame(transitions)
    summ.to_csv(args.out_dir / "closed_form_summary.csv", index=False)
    trans.to_csv(args.out_dir / "closed_form_lambda_transitions.csv", index=False)
    fuel_rows = []
    for vname in [v.strip() for v in args.variants.split(",") if v.strip()]:
        for f, d in _fuel_table(vname, args.price_scenario).items():
            fuel_rows.append({"variant": vname, "fuel": f, "price_eur_per_gj": round(d.price, 3),
                              "ets_co2e_g_per_mj": round(d.ets, 3), "wtw_g_per_mj": round(d.wtw, 3),
                              "rfnbo": d.rfnbo})
    pd.DataFrame(fuel_rows).to_csv(args.out_dir / "closed_form_fuel_inputs.csv", index=False)
    print(f"wrote {len(summ)} summary rows, {len(trans)} transitions -> {args.out_dir}")


if __name__ == "__main__":
    main()
