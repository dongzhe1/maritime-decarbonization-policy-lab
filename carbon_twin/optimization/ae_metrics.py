"""Metrics derived from one blend-optimiser profile per vessel: abatement gap and
marginal abatement cost (AE-1), FuelEU-redundancy breakeven carbon price (AE-2),
and FuelEU-vs-CII attribution of the RCI (AE-3).

The ETS cost is linear in the EUA price and every other cost component is EUA-
independent, so one reference profile reconstructs all costs at any EUA, making
the carbon-price sweep and breakeven search analytic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def ae_vessel_metrics(
    profile: pd.DataFrame,
    eua_ref: float,
    target_euas: list[float],
    breakeven_grid: list[float],
    rci_eps: float = 0.01,
) -> list[dict[str, Any]]:
    """Derive AE-1/2/3 metrics for one vessel from its blend profile.

    Parameters
    ----------
    profile : the ``profile`` DataFrame from the blend optimiser (one row per
        alt_fuel × blend φ × speed) computed at ``eua_ref``.
    eua_ref : EUA price (EUR/tCO2e) the profile was computed at (>0).
    target_euas : carbon prices to report AE-1/AE-3 metrics at.
    breakeven_grid : ascending carbon prices to search for the FuelEU-redundancy
        breakeven (smallest EUA where the carbon-only RCI drops below ``rci_eps``).
    rci_eps : RCI threshold below which FuelEU is treated as redundant.

    Returns one dict per target EUA (the per-vessel breakeven is constant and is
    attached to every row for convenience).
    """
    if profile is None or len(profile) == 0 or eua_ref <= 0:
        return []

    df = profile
    # ETS cost is linear in EUA: slope = covered tCO2e × scope × phase-in.
    slope = df["ets_cost"].to_numpy() / eua_ref
    fuel = df["fuel_cost"].to_numpy()
    fueleu = df["fueleu_penalty"].to_numpy()
    cii = df["cii_penalty"].to_numpy()
    delay = df["delay_cost"].to_numpy()
    wtw = df["co2e_wtw_t"].to_numpy()
    ttw = df["co2e_ttw_t"].to_numpy()
    feu_bal = (
        df["fueleu_balance"].to_numpy() if "fueleu_balance" in df.columns else np.zeros(len(df))
    )
    phi = df["blend_pct"].to_numpy()
    altf = df["alt_fuel"].to_numpy()
    is_base = phi <= 1e-9

    indep_total = fuel + fueleu + cii + delay  # full burden, ex-ETS
    indep_carbon = fuel + delay  # carbon-only operator, ex-ETS
    indep_pf = fuel + delay + fueleu  # + FuelEU-aware (blind to CII)
    indep_pc = fuel + delay + cii  # + CII-aware (blind to FuelEU)
    base_idx = np.where(is_base)[0]

    def _picks(eua: float):
        ets = slope * eua
        tot = indep_total + ets
        car = indep_carbon + ets
        pf = indep_pf + ets
        pc = indep_pc + ets
        i_j = int(np.argmin(tot))
        i_c = int(np.argmin(car))
        i_pf = int(np.argmin(pf))
        i_pc = int(np.argmin(pc))
        if base_idx.size:
            i_b = int(base_idx[int(np.argmin(tot[base_idx]))])
        else:
            i_b = i_j
        return tot, i_j, i_c, i_pf, i_pc, i_b

    def _rci(tot, i_pick, i_joint) -> float:
        cp, cj = float(tot[i_pick]), float(tot[i_joint])
        return (cp - cj) / cp if cp > 0 else 0.0

    # AE-2 — FuelEU-redundancy breakeven. RCI→0 at BOTH ends of the EUA axis:
    # at low EUA because the joint operator just eats the (small) FuelEU penalty
    # rather than switching (both sit on the fossil base), and at high EUA because
    # the carbon-only operator switches to the low-carbon fuel on ETS grounds
    # alone (true redundancy). We want the latter, so require the joint optimum to
    # be genuinely decarbonised (well below the fossil-baseline WtW) before
    # calling FuelEU redundant.
    def _redundant(tot, i_j, i_c, i_b) -> bool:
        if _rci(tot, i_c, i_j) >= rci_eps:
            return False
        return wtw[i_j] < _DECARB_FRACTION * wtw[i_b]

    breakeven = float("nan")
    for e in sorted(breakeven_grid):
        tot, i_j, i_c, _, _, i_b = _picks(float(e))
        if _redundant(tot, i_j, i_c, i_b):
            breakeven = float(e)
            break

    rows: list[dict[str, Any]] = []
    for eua in target_euas:
        tot, i_j, i_c, i_pf, i_pc, i_b = _picks(float(eua))
        rci_c = _rci(tot, i_c, i_j)
        rci_pf = _rci(tot, i_pf, i_j)  # residual coupling after FuelEU-awareness
        rci_pc = _rci(tot, i_pc, i_j)  # residual coupling after CII-awareness
        # AE-1 — abatement & MAC of JOINT vs CARBON-ONLY (the same siloed-vs-joint
        # pair as the RCI; avoids the speed-confounded φ=0 baseline, since the
        # carbon-only operator ignores CII and steams faster). Joint emits less
        # and (minimising the superset objective) costs no more → MAC ≤ 0:
        # negative-cost abatement once compliance penalties are internalised.
        ab_gap = float(wtw[i_c] - wtw[i_j])  # extra WtW the siloed operator emits
        cost_delta = float(tot[i_j] - tot[i_c])  # joint − siloed full burden (≤0)
        mac_gap = cost_delta / ab_gap if abs(ab_gap) > 1e-9 else float("nan")
        gap_frac = ab_gap / wtw[i_c] if wtw[i_c] > 1e-9 else 0.0
        # AE-1b — PRIVATE vs RESOURCE MAC. The private MAC above internalises the
        # FuelEU penalty (a transfer), so it is ≤0 (avoiding the penalty). The
        # resource MAC counts only real inputs (fuel + delay), excluding the
        # ETS/FuelEU/CII policy transfers, so it is the techno-economic cost of the
        # abatement (typically >0). Their difference is the FuelEU-penalty wedge per
        # tonne that the siloed operator pays but the single-instrument view hides.
        res_joint = float(fuel[i_j] + delay[i_j])
        res_carbon = float(fuel[i_c] + delay[i_c])
        resource_mac = (res_joint - res_carbon) / ab_gap if abs(ab_gap) > 1e-9 else float("nan")
        rows.append(
            {
                "eua": float(eua),
                # AE-3 attribution
                "rci_carbon_only": rci_c,
                "rci_after_fueleu": rci_pf,
                "rci_after_cii": rci_pc,
                "attrib_fueleu": (rci_c - rci_pf) / rci_c if rci_c > 1e-12 else 0.0,
                "attrib_cii": (rci_c - rci_pc) / rci_c if rci_c > 1e-12 else 0.0,
                "joint_fuel": str(altf[i_j]),
                "joint_blend": float(phi[i_j]),
                "carbon_fuel": str(altf[i_c]),
                "carbon_blend": float(phi[i_c]),
                # AE-1 abatement & MAC (WtW CO2e tonnes, per representative leg)
                "co2e_wtw_joint_t": float(wtw[i_j]),
                "co2e_wtw_carbon_t": float(wtw[i_c]),
                "co2e_wtw_baseline_t": float(wtw[i_b]),  # φ=0 fossil ref (context only)
                "co2e_ttw_joint_t": float(ttw[i_j]),
                "abatement_gap_t": ab_gap,
                "abatement_gap_frac": gap_frac,
                "resource_cost_joint_eur": res_joint,
                "resource_cost_carbon_eur": res_carbon,
                "resource_mac_eur_per_t": resource_mac,
                # FuelEU penalty + signed balance at each pick (for fleet pooling, AE-2b)
                "fueleu_penalty_joint_eur": float(fueleu[i_j]),
                "fueleu_penalty_carbon_eur": float(fueleu[i_c]),
                "fueleu_balance_joint": float(feu_bal[i_j]),
                "fueleu_balance_carbon": float(feu_bal[i_c]),
                "mac_gap_eur_per_t": mac_gap,
                "cost_joint_eur": float(tot[i_j]),
                "cost_carbon_eur": float(tot[i_c]),
                # AE-2 redundancy
                "redundant": _redundant(tot, i_j, i_c, i_b),
                "breakeven_eua": breakeven,
            }
        )
    return rows


_DECARB_FRACTION: float = 0.5  # joint WtW must be < this × fossil-baseline WtW
#                                to count as "decarbonised" (true FuelEU redundancy)
