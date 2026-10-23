from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall: pip install -e '.[all]'")

_SUBMITTED = [
    ("metric_harmonization", ("M1_actual_ETS_FuelEU_only", 2026), 0.01639, 5e-5),
    ("metric_harmonization", ("M1_actual_ETS_FuelEU_only", 2030), 0.09961, 5e-5),
    ("metric_harmonization", ("M1_actual_ETS_FuelEU_only", 2035), 0.19067, 5e-5),
    ("metric_harmonization", ("M1_actual_ETS_FuelEU_only", 2040), 0.25309, 5e-5),
    ("metric_harmonization", ("M1_actual_ETS_FuelEU_only", 2045), 0.36812, 5e-5),
    ("metric_harmonization", ("M2_harmonized_WtW_ranking", 2035), 0.00095, 5e-5),
    ("metric_harmonization", ("M2_harmonized_WtW_ranking", 2045), 0.34284, 5e-5),
    ("metric_harmonization", ("M0_actual_full_policy_mix", 2035), 0.19535, 5e-5),
    ("metric_harmonization", ("M0_actual_full_policy_mix", 2045), 0.36721, 5e-5),
    ("credit_price", (0.25, 2045), 0.0235, 5e-4),
    ("credit_price", (0.50, 2045), 0.1008, 5e-4),
    ("credit_price", (0.75, 2045), 0.2538, 5e-4),
    ("credit_price", (1.00, 2045), 0.3672, 5e-4),
    ("biofuel", ("B1_no_FAME_no_biodiesel", 2045), 0.17321, 5e-5),
    ("biofuel", ("B2_no_biofuel", 2045), 0.12734, 5e-5),
    ("mrv_calibration", ("C1_MRV_calibrated_baseline", 2035), 0.19868, 5e-5),
    ("mrv_calibration", ("C1_MRV_calibrated_baseline", 2045), 0.36566, 5e-5),
    ("speed_k", (0.0,), 0.00005, 5e-5),
    ("speed_k", (2.0,), 0.00625, 5e-5),
    ("transition", ("CENTRAL_EXTENDED", 2035, 11, "VLSFO_only", "LNG_only"), (0.082, 0.083), 1e-6),
    ("transition", ("CENTRAL_EXTENDED", 2045, 11, "LNG_only", "VLSFO+biodiesel_fame"),
     (0.425, 0.430), 1e-6),
    ("transition", ("COMPATIBILITY_CONSTRAINED_2035", 2035, 11, "VLSFO_only",
                    "VLSFO+biodiesel_fame"), (0.394, 0.395), 1e-6),
    ("transition", ("NO_FAME_2045", 2045, 51, "LNG_only", "VLSFO+methanol_bio"),
     (0.791, 0.792), 1e-6),
]


def md_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df is None or df.empty:
        return "_(no data)_"
    df = df.head(max_rows)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = []
        for x in row:
            if isinstance(x, float):
                cells.append("" if np.isnan(x) else f"{x:,.4g}")
            else:
                cells.append(str(x))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _read(path: Path, **kw) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path, **kw) if path.suffix == ".parquet" else pd.read_csv(path)
    except (FileNotFoundError, OSError, ValueError):
        return None


def _load_capability(path: Path | None) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    from carbon_twin.pipeline import load_fuel_capability

    return load_fuel_capability(path)


def run_status(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(root.glob("*/bundle_status_*.tsv")):
        t = pd.read_csv(f, sep="\t", dtype=str)
        for r in t.itertuples(index=False):
            rows.append({"variant": f.parent.name, "bundle": f.stem.removeprefix("bundle_status_"),
                         "exp": r.exp, "status": r.status, "log": getattr(r, "log", "")})
    return pd.DataFrame(rows, columns=["variant", "bundle", "exp", "status", "log"])


def _variants(root: Path, requested: str | None) -> list[str]:
    if requested:
        return [v.strip() for v in requested.split(",") if v.strip()]
    from carbon_twin.model_variants import available_variants

    known = set(available_variants())
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name in known)


def headline(root: Path, variants: list[str]) -> pd.DataFrame:
    rows = []
    for v in variants:
        mh = _read(root / v / "metric_harmonization.csv")
        if mh is None:
            continue
        for yr, g in mh.groupby("year"):
            by = g.set_index("case")
            def get(case, col):
                return by.at[case, col] if case in by.index and col in by.columns else np.nan
            rows.append({
                "variant": v, "year": yr,
                "regret_ets_only_M1": get("M1_actual_ETS_FuelEU_only", "mean_rci_silo"),
                "regret_wtw_valued_M2": get("M2_harmonized_WtW_ranking", "mean_rci_silo"),
                "regret_full_mix_M0": get("M0_actual_full_policy_mix", "mean_rci_silo"),
                "pathway_ets_only": get("M1_actual_ETS_FuelEU_only", "modal_pathway_silo"),
                "pathway_wtw_valued": get("M2_harmonized_WtW_ranking", "modal_pathway_silo"),
                "pathway_joint": get("M1_actual_ETS_FuelEU_only", "modal_pathway_joint"),
                "share_pathway_changed_M1": get("M1_actual_ETS_FuelEU_only",
                                                "share_pathway_changed"),
                "fleet_delta_cost_eur_m_M1": get("M1_actual_ETS_FuelEU_only",
                                                 "fleet_delta_cost_eur") / 1e6,
                "cost_basis": get("M1_actual_ETS_FuelEU_only", "cost_basis"),
                "fleet_biofuel_energy_joint_pj_M1": get("M1_actual_ETS_FuelEU_only",
                                                        "fleet_biofuel_energy_joint_gj") / 1e6,
                "mean_scope_weight": get("M1_actual_ETS_FuelEU_only", "mean_scope_weight"),
            })
    return pd.DataFrame(rows)


def transitions(root: Path, variants: list[str]) -> pd.DataFrame:
    frames = []
    for v in variants:
        t = _read(root / v / "lambda" / "fuel_pathway_transition_intervals.csv")
        if t is not None:
            frames.append(t.assign(variant=v))
    if not frames:
        return pd.DataFrame()
    cols = ["variant", "scenario_id", "year", "blend_steps", "lambda_low", "lambda_high",
            "from_pathway", "to_pathway", "status"]
    return pd.concat(frames, ignore_index=True)[cols]


def envelope(root: Path, variants: list[str]) -> pd.DataFrame:
    rows = []
    for v in variants:
        cp = _read(root / v / "credit_price.csv")
        if cp is not None:
            for _, r in cp.iterrows():
                rows.append({"variant": v, "experiment": "effective_fueleu_price",
                             "case": f"{r['fueleu_price_fraction']:.2f}", "year": r["year"],
                             "mean_regret": r["mean_rci_silo"],
                             "joint_pathway": r.get("dominant_fuel_joint", "")})
        bf = _read(root / v / "biofuel.csv")
        if bf is not None:
            for _, r in bf.iterrows():
                rows.append({"variant": v, "experiment": "fuel_availability", "case": r["case"],
                             "year": r["year"], "mean_regret": r["mean_rci_silo"],
                             "joint_pathway": r.get("dominant_fuel_joint", "")})
        mc = _read(root / v / "mrv_calibration.csv")
        if mc is not None:
            for _, r in mc.iterrows():
                rows.append({"variant": v, "experiment": "mrv_calibration", "case": r["case"],
                             "year": r["year"], "mean_regret": r["mean_rci_silo"],
                             "joint_pathway": ""})
        fr = _read(root / v / "fuel_robustness" / "fuel_rci_sensitivity.csv")
        if fr is not None:
            for _, r in fr[fr["price_scenario"] == "base_2026"].iterrows():
                rows.append({"variant": v, "experiment": "feasibility_menu",
                             "case": r["fuel_scenario"], "year": r["year"],
                             "mean_regret": r["rci_silo_mean"], "joint_pathway": ""})
        po = _read(root / v / "pooling.csv")
        if po is not None:
            for _, r in po.iterrows():
                rows.append({"variant": v, "experiment": "pooling", "case": r["case"],
                             "year": r["year"], "mean_regret": r["mean_rci_silo"],
                             "joint_pathway": ""})
    return pd.DataFrame(rows)


def speed(root: Path, variants: list[str]) -> pd.DataFrame:
    frames = []
    for v in variants:
        s = _read(root / v / "speed_k.csv")
        if s is not None:
            frames.append(s.assign(variant=v))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def distribution(root: Path, variants: list[str]) -> pd.DataFrame:
    rows = []
    for v in variants:
        mh = _read(root / v / "metric_harmonization.csv")
        if mh is not None:
            m1 = mh[mh["case"] == "M1_actual_ETS_FuelEU_only"]
            keep = [c for c in m1.columns if c.startswith(("p", "share_", "median", "mean_rci",
                                                           "max_rci", "aggregate"))
                    and c not in ("price_scenario",)]
            for _, r in m1.iterrows():
                rows.append({"variant": v, "source": "metric_harmonization_M1",
                             "scenario": "CENTRAL_EXTENDED", "year": r["year"],
                             **{c: r[c] for c in keep}})
        vp = root / v / "lambda" / "fueleu_internalization_vessel.parquet"
        if not vp.exists():
            vp = vp.with_name("fueleu_internalization_vessel_lambda0.parquet")
        if vp.exists():
            cols = ["scenario_id", "year", "blend_steps", "lambda_fueleu", "rci_proxy",
                    "partial_fuel_pathway", "joint_fuel_pathway", "delta_cost_eur"]
            vdf = pd.read_parquet(vp, columns=cols)
            vdf = vdf[(vdf["lambda_fueleu"] == 0.0) & (vdf["blend_steps"] == 11)]
            for (scen, yr), g in vdf.groupby(["scenario_id", "year"]):
                r = g["rci_proxy"]
                rows.append({
                    "variant": v, "source": "lambda_sweep_lambda0", "scenario": scen, "year": yr,
                    "mean_rci_silo": r.mean(), "median_rci_silo": r.median(),
                    **{f"p{q}_rci_silo": r.quantile(q / 100) for q in (10, 25, 75, 90, 95, 99)},
                    "max_rci_silo": r.max(),
                    "share_rci_lt_0p01pct": (r < 1e-4).mean(),
                    "share_rci_lt_1pct": (r < 0.01).mean(),
                    "share_rci_gt_10pct": (r > 0.10).mean(),
                    "share_pathway_changed": (g["partial_fuel_pathway"]
                                              != g["joint_fuel_pathway"]).mean(),
                })
    return pd.DataFrame(rows)


def retrofit(root: Path, variants: list[str]) -> pd.DataFrame:
    frames = [t.assign(variant=v, construct=c) for v in variants
              for f, c in (("retrofit_cost.csv", "premium_eur_per_gj"),
                           ("retrofit_capex.csv", "fixed_capex_eur_per_kw"))
              if (t := _read(root / v / f)) is not None]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def two_by_two(root: Path, variants: list[str]) -> pd.DataFrame:
    rows = []
    for v in variants:
        cells = {"broad": _read(root / v / "metric_harmonization.csv"),
                 "engine_capability": _read(root / v / "metric_harmonization_compat.csv"),
                 "engine_capability_fieldonly": _read(
                     root / v / "metric_harmonization_compat_fieldonly.csv")}
        if cells["engine_capability"] is None and cells["engine_capability_fieldonly"] is None:
            continue
        for menu, mh in cells.items():
            if mh is None:
                continue
            for _, r in mh[mh["case"].isin(["M1_actual_ETS_FuelEU_only",
                                            "M2_harmonized_WtW_ranking"])].iterrows():
                rows.append({"variant": v, "year": r["year"], "menu": menu,
                             "planning_rule": "ETS" if r["case"].startswith("M1") else "WtW",
                             "mean_regret": r["mean_rci_silo"],
                             "aggregate_regret": r.get("aggregate_rci_silo", np.nan),
                             "shortcut_pathway": r.get("modal_pathway_silo", ""),
                             "joint_pathway": r.get("modal_pathway_joint", ""),
                             "share_pathway_changed": r.get("share_pathway_changed", np.nan)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    wide = df.pivot_table(index=["variant", "year"], columns=["menu", "planning_rule"],
                          values="mean_regret").reset_index()
    wide.columns = ["_".join(c for c in col if c) if isinstance(col, tuple) else col
                    for col in wide.columns]
    return wide


def denominators(root: Path, variants: list[str],
                 capability: dict | None = None) -> pd.DataFrame:
    rows = []
    for v in variants:
        vp = root / v / "metric_harmonization_vessel.parquet"
        if not vp.exists():
            continue
        d = pd.read_parquet(vp)
        d = d[d["case"] == "M1_actual_ETS_FuelEU_only"].copy()
        d["vessel_imo"] = d["vessel_imo"].astype(str)
        if "scope_weight" in d:
            d["scope_bin"] = pd.cut(d["scope_weight"], [-0.001, 0.4999, 0.6, 0.8, 1.0001],
                                    labels=["<0.5", "0.5-0.6", "0.6-0.8", "0.8-1.0"])
        if capability:
            d["lng_capable"] = d["vessel_imo"].map(
                lambda i: bool(capability.get(i, {}).get("lng_capable", False)))
        strata = [("ALL", None)] + [(c, c) for c in ("scope_bin", "ship_type", "lng_capable")
                                    if c in d]
        for yr, gy in d.groupby("year"):
            for label, col in strata:
                groups = [("ALL", gy)] if col is None else list(gy.groupby(col, observed=True))
                for val, g in groups:
                    den, r = g["cost_siloed_eur"], g["rci"]
                    keep1 = den >= den.quantile(0.01)
                    keep5 = den >= den.quantile(0.05)
                    rows.append({
                        "variant": v, "year": yr, "stratum": label, "value": str(val),
                        "n": int(len(g)),
                        "cost_basis": str(g["cost_basis"].iloc[0]) if "cost_basis" in g else "leg",
                        "denominator_min_eur": den.min(), "denominator_p1_eur": den.quantile(0.01),
                        "denominator_p5_eur": den.quantile(0.05),
                        "denominator_median_eur": den.median(),
                        "mean_regret": r.mean(), "median_regret": r.median(),
                        "p95_regret": r.quantile(0.95),
                        "mean_regret_drop_low1pct": r[keep1].mean(),
                        "mean_regret_drop_low5pct": r[keep5].mean(),
                        "cost_weighted_regret": g["delta_cost_eur"].sum() / den.sum(),
                        "share_regret_lt_0p01pct": (r < 1e-4).mean(),
                        "share_regret_lt_1pct": (r < 0.01).mean(),
                        "abs_loss_total_eur_m": g["delta_cost_eur"].sum() / 1e6,
                        "abs_loss_median_eur": g["delta_cost_eur"].median(),
                        "abs_loss_p90_eur": g["delta_cost_eur"].quantile(0.90),
                    })
    return pd.DataFrame(rows)


def closed_form_check(root: Path, trans: pd.DataFrame) -> pd.DataFrame:
    cf = _read(root / "closed_form" / "closed_form_lambda_transitions.csv")
    if cf is None or trans.empty:
        return pd.DataFrame()
    menu_of = {"CENTRAL_EXTENDED": "broad", "COMPATIBILITY_CONSTRAINED_2035": "compat_non_gas",
               "NO_FAME_2045": "no_fame"}
    cf = cf[cf["scope_weight"] == 1.0]
    cf = cf.assign(blend_steps=cf["blend_steps"].astype(str))
    rows = []
    for _, t in trans[trans["blend_steps"] == 51].iterrows():
        sel = ((cf["variant"] == t["variant"]) & (cf["menu"] == menu_of.get(t["scenario_id"]))
               & (cf["year"] == t["year"]))
        def first(steps):
            c = cf[sel & (cf["blend_steps"] == steps)]
            c = c[c["from_plan"].str.split("@").str[0] == t["from_pathway"]]
            return c.iloc[0] if len(c) else None
        g51, gx = first("51"), first("exact")
        rows.append({
            "variant": t["variant"], "scenario_id": t["scenario_id"], "year": t["year"],
            "from_pathway": t["from_pathway"], "to_pathway": t["to_pathway"],
            "sweep_interval_51": f"({t['lambda_low']:.4f}, {t['lambda_high']:.4f}]",
            "closed_form_interval_51": (f"({g51['lambda_low']:.4f}, {g51['lambda_high']:.4f}]"
                                        if g51 is not None else ""),
            "closed_form_exact_lambda": gx["lambda_exact"] if gx is not None else np.nan,
            "note": "fleet-uniform only when scope weight = 1",
        })
    return pd.DataFrame(rows)


def regression(root: Path, trans: pd.DataFrame) -> pd.DataFrame:
    v = root / "v23"
    mh, cp, bf, mc, sk = (_read(v / f) for f in (
        "metric_harmonization.csv", "credit_price.csv", "biofuel.csv",
        "mrv_calibration.csv", "speed_k.csv"))
    rows = []
    for section, key, expected, tol in _SUBMITTED:
        obs = np.nan
        try:
            if section == "metric_harmonization" and mh is not None:
                obs = mh[(mh["case"] == key[0]) & (mh["year"] == key[1])]["mean_rci_silo"].iat[0]
            elif section == "credit_price" and cp is not None:
                obs = cp[np.isclose(cp["fueleu_price_fraction"], key[0])
                         & (cp["year"] == key[1])]["mean_rci_silo"].iat[0]
            elif section == "biofuel" and bf is not None:
                obs = bf[(bf["case"] == key[0]) & (bf["year"] == key[1])]["mean_rci_silo"].iat[0]
            elif section == "mrv_calibration" and mc is not None:
                obs = mc[(mc["case"] == key[0]) & (mc["year"] == key[1])]["mean_rci_silo"].iat[0]
            elif section == "speed_k" and sk is not None:
                obs = sk[np.isclose(sk["cii_pressure"], key[0])]["rci_mean"].iat[0]
            elif section == "transition" and not trans.empty:
                t = trans[(trans["variant"] == "v23") & (trans["scenario_id"] == key[0])
                          & (trans["year"] == key[1]) & (trans["blend_steps"] == key[2])
                          & (trans["from_pathway"] == key[3]) & (trans["to_pathway"] == key[4])]
                obs = (float(t["lambda_low"].iat[0]), float(t["lambda_high"].iat[0]))
        except (IndexError, KeyError):
            obs = np.nan
        if isinstance(expected, tuple):
            ok = isinstance(obs, tuple) and all(abs(a - b) <= tol for a, b in zip(obs, expected))
        else:
            ok = not pd.isna(obs) and abs(float(obs) - expected) <= tol
        missing = obs is None or (not isinstance(obs, tuple) and pd.isna(obs))
        rows.append({"section": section, "key": str(key), "expected": str(expected),
                     "observed": str(obs), "tolerance": tol,
                     "status": "MISSING" if missing else ("PASS" if ok else "FAIL")})
    return pd.DataFrame(rows)


_DECOMPOSITION = [
    ("0 original specification (old fleet input)", True, "v23"),
    ("1 + current fleet input (main-engine power, build year)", False, "v23"),
    ("2 + LNG Annex II Otto MS slip (only)", False, "v23_lng_otto_ms"),
    ("3 + annual basis, MRV scope (proportional), RFNBO to 2033, VLSFO WtW 91.55",
     False, "r1_lng_otto_ms"),
    ("4a + exact share candidates", False, "r1_lng_otto_ms_exact"),
    ("4b + FuelEU priority allocation", False, "r1_lng_otto_ms_prio"),
    ("5a + VLSFO or LNG base", False, "r1_lng_otto_ms_prio_fb"),
    ("5b + scope weight from the MRV voyage breakdown (central case)", False, "r1_lng_otto_ms_prio_fb_bd"),
]


def decomposition(root: Path, ref: Path | None) -> pd.DataFrame:
    rows = []
    for label, use_ref, v in _DECOMPOSITION:
        base = ref if use_ref else root
        if base is None:
            continue
        mh = _read(base / v / "metric_harmonization.csv")
        if mh is None:
            continue
        trans = transitions(base, [v])
        for yr, g in mh.groupby("year"):
            by = g.set_index("case")
            def get(case, col):
                return by.at[case, col] if case in by.index and col in by.columns else np.nan
            t = (trans[(trans["scenario_id"] == "CENTRAL_EXTENDED") & (trans["year"] == yr)
                       & (trans["blend_steps"] == 51)] if not trans.empty else trans)
            rows.append({
                "step": label, "variant": v, "root": "reference" if use_ref else "main", "year": yr,
                "regret_ets_rule_M1": get("M1_actual_ETS_FuelEU_only", "mean_rci_silo"),
                "regret_wtw_rule_M2": get("M2_harmonized_WtW_ranking", "mean_rci_silo"),
                "pathway_ets_rule": get("M1_actual_ETS_FuelEU_only", "modal_pathway_silo"),
                "pathway_joint": get("M1_actual_ETS_FuelEU_only", "modal_pathway_joint"),
                "fleet_loss_eur_m": get("M1_actual_ETS_FuelEU_only",
                                        "fleet_delta_cost_eur") / 1e6,
                "cost_basis": get("M1_actual_ETS_FuelEU_only", "cost_basis"),
                "lambda_switches_51": "; ".join(
                    f"{r.from_pathway}->{r.to_pathway} ({r.lambda_low:.3f}, {r.lambda_high:.3f}]"
                    for r in t.itertuples()) if not t.empty else "",
            })
    return pd.DataFrame(rows)


def scope_sample(root: Path, variants: list[str]) -> pd.DataFrame:
    rows = []
    for v in variants:
        full = _read(root / v / "metric_harmonization.csv")
        cut = _read(root / v / "metric_harmonization_s05.csv")
        if full is None or cut is None:
            continue
        for label, mh in (("all", full), (str(cut["sample_scope"].iat[0]), cut)):
            for _, r in mh[mh["case"].isin(["M1_actual_ETS_FuelEU_only",
                                            "M2_harmonized_WtW_ranking"])].iterrows():
                rows.append({"variant": v, "sample": label, "year": r["year"],
                             "case": r["case"][:2], "n_vessels": r.get("n_vessels", np.nan),
                             "mean_regret": r["mean_rci_silo"],
                             "aggregate_regret": r.get("aggregate_rci_silo", np.nan),
                             "joint_pathway": r.get("modal_pathway_joint", "")})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/r1"))
    ap.add_argument("--variants", default=None, help="default: every variant directory")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--fuel-capability", type=Path, default=None,
                    help="IMO fuel-capability CSV, for the LNG-capability stratum")
    ap.add_argument("--reference-root", type=Path, default=None,
                    help="batch-1 outputs on the old fleet input (results/r1_oldfleet): "
                         "the v23 regression is checked there")
    args = ap.parse_args()
    out = args.out_dir or args.root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    variants = _variants(args.root, args.variants)

    trans = transitions(args.root, variants)
    ref = args.reference_root if (args.reference_root
                                  and (args.reference_root / "v23").exists()) else None
    status = run_status(args.root)
    not_ok = status[status["status"] != "OK"] if not status.empty else status
    tables = {
        "run_status_not_ok": not_ok,
        "v23_regression": (regression(ref, transitions(ref, ["v23"])) if ref
                           else regression(args.root, trans)),
        "v23_data_update": (regression(args.root, trans).rename(
            columns={"status": "matches_submitted"}) if ref else pd.DataFrame()),
        "decomposition": decomposition(args.root, ref),
        "scope_sample_sensitivity": scope_sample(args.root, variants),
        "headline_by_variant": headline(args.root, variants),
        "lambda_transitions": trans,
        "closed_form_check": closed_form_check(args.root, trans),
        "envelope_by_variant": envelope(args.root, variants),
        "distribution_by_variant": distribution(args.root, variants),
        "speed_by_variant": speed(args.root, variants),
        "retrofit_by_variant": retrofit(args.root, variants),
        "two_by_two": two_by_two(args.root, variants),
        "denominators_heterogeneity": denominators(
            args.root, variants, _load_capability(args.fuel_capability)),
    }
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)

    land = _read(args.root / "data_audit" / "compliance_landscape.csv")
    wf = _read(args.root / "data_audit" / "sample_waterfall.csv")
    reg = tables["v23_regression"]
    vcf = _read(args.root / "vessel_closed_form" / "vessel_closed_form_summary.csv")
    if vcf is not None:
        vcf = vcf[(vcf["s_band"] == "all") & (vcf["blend_steps"] == vcf["blend_steps"].max())]
    scope_md = args.root / "scope_check" / "scope_check.md"
    md = ["# Variant comparison", "",
          f"Variants: {', '.join(variants)}", "",
          f"## Run status: {int((status['status'] == 'OK').sum()) if not status.empty else 0} OK, "
          f"{len(not_ok)} not OK", "",
          md_table(not_ok) if len(not_ok) else "All experiments OK.", "",
          f"## v23 regression against the reference values"
          f"{f' (reference root {ref})' if ref else ''}", "",
          f"{(reg['status'] == 'PASS').sum()} PASS, {(reg['status'] == 'FAIL').sum()} FAIL, "
          f"{(reg['status'] == 'MISSING').sum()} MISSING", "", md_table(reg), "",
          "## v23 on the current fleet input vs the submitted numbers", "",
          md_table(tables["v23_data_update"]), "",
          "## Correction decomposition (submitted -> central)", "",
          md_table(tables["decomposition"]), "",
          "## Headline: ETS-only vs joint, actual vs WtW-valued shortcut", "",
          md_table(tables["headline_by_variant"]), "",
          "## Lambda transitions (fleet-modal pathway)", "",
          md_table(trans[trans["blend_steps"].isin([11, 51])] if not trans.empty else trans), "",
          "## Closed-form check (51-point)", "", md_table(tables["closed_form_check"]), "",
          "## Envelope (2045 unless stated)", "",
          md_table(tables["envelope_by_variant"]), "",
          "## Regret distribution at lambda = 0", "",
          md_table(tables["distribution_by_variant"]), "",
          "## Speed-only regret vs CII pressure", "", md_table(tables["speed_by_variant"]), "",
          "## Retrofit premium and fixed capex", "",
          md_table(tables["retrofit_by_variant"]), "",
          "## Planning rule x fuel menu", "", md_table(tables["two_by_two"]), "",
          "## Small denominators and heterogeneity, all-vessel rows", "",
          md_table(tables["denominators_heterogeneity"].query("stratum == 'ALL'")
                   if not tables["denominators_heterogeneity"].empty
                   else tables["denominators_heterogeneity"]), "",
          "## Scope-weight sample sensitivity (scope weight >= 0.5)", "",
          md_table(tables["scope_sample_sensitivity"]), "",
          "## Scope-weight check against the MRV voyage breakdown", "",
          scope_md.read_text() if scope_md.exists() else "_(not run)_", "",
          "## Per-vessel closed-form check (finest grid, all scope weights)", "",
          md_table(vcf if vcf is not None else pd.DataFrame()), "",
          "## Worked regulatory examples", "",
          md_table(_read(args.root / "worked_examples" / "worked_examples.csv")), "",
          "## Vessel fuel capability", "",
          md_table(_read(args.fuel_capability.with_name(
              args.fuel_capability.stem + "_summary.csv")) if args.fuel_capability
                   else pd.DataFrame()), "",
          "## Data audit", "", md_table(wf), "", md_table(land), ""]
    (out / "R1_COMPARISON.md").write_text("\n".join(md))
    print(f"comparison written to {out} ({len(variants)} variants)")


if __name__ == "__main__":
    main()
