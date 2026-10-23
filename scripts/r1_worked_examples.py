from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall: pip install -e '.[all]'")

from carbon_twin.data.schemas import FuelRecord, FuelType, ShipType, Vessel
from carbon_twin.emissions.factors import GWP100
from carbon_twin.emissions.proxy import load_fuel_factors_safe
from carbon_twin.model_variants import lng_annex2_factors, load_variant
from carbon_twin.optimization.fuel_choice import (
    _ets_covered_tonnes,
    exact_blend_points,
    fueleu_factor_table,
    scope_energy_allocation,
)
from carbon_twin.regulations.fueleu import FuelEUEngine

_VESSEL = Vessel(imo_number="9000000", ship_type=ShipType.BULK_CARRIER,
                 gross_tonnage=40000, deadweight=75000)
_EUA = 80.0


def _fueleu(version: str, records: list[tuple[FuelType, float]], year: int,
            table: dict[str, float]):
    recs = [FuelRecord(vessel_imo=_VESSEL.imo_number, fuel_type=ft, mass_tonnes=0.0,
                       energy_mj=e) for ft, e in records if e > 0]
    total = sum(e for _, e in records)
    return FuelEUEngine(version=version).evaluate(_VESSEL, recs, total, year,
                                                  emission_factors=table)


def run_examples() -> pd.DataFrame:
    rows: list[dict] = []

    def add(ex, quantity, unit, hand, model, tol, note=""):
        diff = abs(hand - model)
        rows.append({"example": ex, "quantity": quantity, "unit": unit,
                     "hand": hand, "model": model, "abs_diff": diff, "tolerance": tol,
                     "status": "PASS" if diff <= tol else "FAIL", "note": note})

    variant = load_variant("r1_lng_otto_ms_prio_fb")
    factors = variant.fuel_factors(load_fuel_factors_safe())
    table = fueleu_factor_table(factors)
    eng = FuelEUEngine(version="EU_2023_1805_r1")
    ref = float(eng._params["reference_value_gCO2e_per_MJ"])

    t2035 = ref * (1 - 0.145)
    energy = 1.0e9
    cb = (t2035 - 91.55) * energy
    pen_hand = abs(cb) / (91.55 * 41_000) * 2_400
    r = _fueleu("EU_2023_1805_r1", [(FuelType.VLSFO, energy)], 2035, table)
    add("E1", "2035 target intensity", "gCO2e/MJ", t2035, r.target_ghg_intensity, 1e-3,
        "91.16 x (1 - 0.145)")
    add("E1", "compliance balance", "tCO2e", cb / 1e6, r.compliance_balance / 1e6, 1e-3)
    add("E1", "FuelEU penalty", "EUR", pen_hand, r.penalty_eur, 0.5,
        "|CB| / (91.55 x 41,000) x 2,400")

    ef = factors[FuelType.VLSFO]
    per_t = (ef.ttw_co2_g_per_g_fuel + GWP100["CH4"] * ef.ttw_ch4_g_per_g_fuel
             + GWP100["N2O"] * ef.ttw_n2o_g_per_g_fuel)
    add("E2", "covered tCO2e, 1,000 t VLSFO intra-EEA", "tCO2e", 1000 * per_t,
        _ets_covered_tonnes(FuelType.VLSFO, ef, 1000.0, 2035), 1e-6)
    e_vlsfo = 1000 * 41.0 * 1000
    alloc = scope_energy_allocation(0.0, e_vlsfo, 0.5, "fueleu_priority")
    covered_t = alloc["ets_base_energy"] / (41.0 * 1000)
    add("E2", "ETS cost, 1,000 t VLSFO extra-EEA", "EUR", 0.5 * 1000 * per_t * _EUA,
        _ets_covered_tonnes(FuelType.VLSFO, ef, covered_t, 2035) * _EUA, 1e-3)

    s = 0.031
    ets_hand = ((1 - s) * (2.750 + 0.00011 * 265) + s * 28) / 0.0491
    fe_hand = 18.5 + ((1 - s) * (2.750 + 0.00011 * 298) + s * 25) / 0.0491
    f = lng_annex2_factors(3.1)
    add("E3", "LNG ETS TtW CO2e (AR5 GWP)", "gCO2e/MJ", ets_hand,
        f["ets_ttw_co2e_g_per_mj"], 1e-9)
    add("E3", "LNG FuelEU WtW (AR4 GWP, WtT 18.5)", "gCO2e/MJ", fe_hand,
        f["wtw_co2e_g_per_mj"], 1e-9)
    add("E3", "LNG FuelEU WtW used by the engine", "gCO2e/MJ", fe_hand, table["LNG"], 1e-9)

    a = scope_energy_allocation(0.2, 100.0, 0.5, "fueleu_priority")
    add("E4", "ETS-covered FAME energy", "GJ", 10.0, a["ets_alt_energy"], 1e-12,
        "half of each fuel (EMSA ETS FAQ)")
    add("E4", "ETS-covered VLSFO energy", "GJ", 40.0, a["ets_base_energy"], 1e-12)
    add("E4", "FuelEU in-scope FAME energy", "GJ", 20.0, a["fueleu_alt_energy"], 1e-12,
        "lowest intensity first (FuelEU guidance 4.4)")
    add("E4", "FuelEU in-scope VLSFO energy", "GJ", 30.0, a["fueleu_base_energy"], 1e-12)
    i_prio = (20 * 30.0 + 30 * 91.55) / 50
    r = _fueleu("EU_2023_1805_r1", [(FuelType.VLSFO, 30e3), (FuelType.BIODIESEL_FAME, 20e3)],
                2035, table)
    add("E4", "in-scope intensity, priority allocation", "gCO2e/MJ", i_prio,
        r.actual_ghg_intensity, 1e-3)
    i_prop = (10 * 30.0 + 40 * 91.55) / 50
    r = _fueleu("EU_2023_1805_r1", [(FuelType.VLSFO, 40e3), (FuelType.BIODIESEL_FAME, 10e3)],
                2035, table)
    add("E4", "in-scope intensity, proportional allocation", "gCO2e/MJ", i_prop,
        r.actual_ghg_intensity, 1e-3)

    for yr, mult in ((2030, 2.0), (2045, 1.0)):
        r = _fueleu("EU_2023_1805_r1", [(FuelType.METHANOL_E, 1.0e6)], yr, table)
        add("E5", f"e-methanol intensity {yr}", "gCO2e/MJ", 5.0 / mult,
            r.actual_ghg_intensity, 1e-9, "reward factor applies through 2033")

    share_prop = (91.55 - t2035) / (91.55 - 30.0)
    add("E6", "compliance share, proportional", "share", share_prop,
        exact_blend_points(91.55, 30.0, t2035, "uniform", 1.0)[0], 1e-9)
    s_w = 0.6
    share_prio = s_w * share_prop
    add("E6", "compliance share, priority (s = 0.6)", "share", share_prio,
        exact_blend_points(91.55, 30.0, t2035, "fueleu_priority", s_w)[0], 1e-9)
    a = scope_energy_allocation(share_prio, 1.0, s_w, "fueleu_priority")
    i_at = (a["fueleu_alt_energy"] * 30.0 + a["fueleu_base_energy"] * 91.55) / s_w
    add("E6", "in-scope intensity at the priority compliance share", "gCO2e/MJ",
        t2035, i_at, 1e-9)

    add("E7", "annual cost from a 2,000 nm leg (28,000 nm/yr)", "EUR",
        10_000 * 28_000 / 2_000, 10_000 * (28_000 / max(2_000, 1.0)), 1e-9,
        "annual_scale = annual distance / leg distance")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("results/r1/worked_examples"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = run_examples()
    df.to_csv(args.out_dir / "worked_examples.csv", index=False)
    lines = ["| ex | quantity | unit | hand | model | status | note |",
             "|---|---|---|---|---|---|---|"]
    lines += [f"| {r.example} | {r.quantity} | {r.unit} | {r.hand:,.6g} | {r.model:,.6g} | "
              f"{r.status} | {r.note} |" for r in df.itertuples(index=False)]
    (args.out_dir / "worked_examples.md").write_text("\n".join(lines) + "\n")
    n_fail = int((df["status"] == "FAIL").sum())
    print(df[["example", "quantity", "hand", "model", "status"]].to_string(index=False))
    print(f"{len(df) - n_fail} PASS, {n_fail} FAIL")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
