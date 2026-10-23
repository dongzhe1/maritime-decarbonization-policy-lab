# Result files

This directory holds the aggregate outputs of the full run (`bash run.sh full`): fleet-level
and ship-type-level tables of planning regret, fuel pathways, switching intervals of the
FuelEU planning weight, sensitivity cases, analytical and numerical checks, and data audits.
Every file is a summary over vessels. No file contains vessel identifiers or per-vessel
records, and no file reproduces licensed vessel particulars.

`MANIFEST.txt` lists every result file with its number of data rows and the first 16 hex
digits of its SHA-256; `bash run.sh check` compares a new run with it.

## Contents

- [Layout](#layout)
- [Conventions](#conventions)
- [Model variants](#model-variants)
- [Codes used in the tables](#codes-used-in-the-tables)
- [Files](#files): [`variants/`](#variants), [`comparison/`](#comparison),
  [`checks/`](#checks), [`data/`](#data)
- [Column glossary](#column-glossary)
- [Licence](#licence)

## Layout

```
results/
├── variants/<variant>/    outputs of one model variant (16 variants; see "Model variants")
│   ├── lambda/            planning-weight sweep on four fuel menus and three share grids
│   ├── lambda_fieldonly/  the same sweep on the capability menu, field evidence only
│   ├── lambda_refine/     refined planning-weight points around the switches
│   ├── speed_years/       speed-only regret by policy year
│   ├── cii_proxy/         CII proxy rating mix and speed regret under proxy-bias scenarios
│   └── fuel_robustness/   regret and cost components on restricted fuel menus
├── comparison/            tables that compare the variants
├── checks/                closed forms, worked examples, three-fuel and speed-grid checks
│   ├── closed_form/
│   ├── vessel_closed_form/
│   ├── three_fuel/
│   ├── speed_grid/
│   └── worked_examples/
├── data/                  sample construction, scope checks and baseline diagnostics
│   ├── data_audit/
│   ├── scope_check/
│   └── baseline_diagnostics/
└── MANIFEST.txt
```

The central variant is `r1_lng_otto_ms_prio_fb_bd`; it is the only variant with the full set
of experiments. Each `*.variant.json` file next to a result records the model variant that
produced it (its LNG factors, scope rule, fuel bases and other settings).

## Conventions

**Sample.** Fleet results cover the optimization sample: ships in the 2024 EU MRV data that
are matched to vessel particulars and sailed at least 500 nautical miles (12,333 ships; see
`data/data_audit/sample_waterfall.csv`). Later policy years apply their targets to the same
2024 fleet and activity.

**Two plans per vessel.** For each vessel and policy year two plans of fuel and speed are
compared, and both are evaluated with the same full cost (fuel, EU ETS allowances, FuelEU
deficit cost and schedule delay; the CII term only where a column or case says so):

- the *partial* plan, chosen without the FuelEU cost (the ETS-only rule) or with the FuelEU
  cost weighted by the planning weight λ where a `lambda_fueleu` column is present. Columns
  use the suffix `_silo`, `_siloed` or `_partial` for it;
- the *joint* plan, chosen with the full FuelEU cost (λ = 1). Columns use the suffix `_joint`.

**Planning regret.** For one vessel, regret = (full cost of the partial plan − full cost of
the joint plan) / full cost of the partial plan. It lies in [0, 1). The columns `rci`,
`rci_silo`, `rci_proxy`, `regret` and `mean_regret` hold this quantity (or its mean); `rci_joint`
and `rci_joint_norm` divide the same difference by the cost of the joint plan instead. The
*avoidable cost* `delta_cost_eur` is the numerator, in EUR. An *aggregate* regret divides the
summed avoidable cost by the summed cost of the partial plans.

**Units.**

| Quantity | Unit |
|---|---|
| money (`*_eur`) | EUR; per vessel and year where the cost basis is `annual`, per representative 2,000-nm leg where it is `leg` (the two original-specification variants) |
| `*_eur_m` | million EUR |
| energy | MJ (`*_mj`), GJ (`*_gj`), PJ (`*_pj`) |
| GHG intensity | gCO₂e/MJ |
| FuelEU compliance balance (`*_CB`, `*_fueleu_balance_*`) | gCO₂e (positive = surplus, negative = deficit) |
| allowance price (`eua`, `eua_price*`) | EUR/tCO₂ |
| speed | knots |
| fuel price (`price_eur_per_gj`) | EUR/GJ |
| retrofit charge (`capex_eur_per_kw`) | EUR per kW of main-engine power, annualized with a capital recovery factor of 0.15 |

**Shares.** Shares and fractions are between 0 and 1. Exceptions, which hold percentages
(0–100) or percentage points: `mean_regret_pct`, `max_abs_change_regret_pp`,
`penalty_reduction_pct`. Note that `pct_rci_gt_10`, `pct_rci_gt_10pct`,
`pct_vessels_using_*` and `mean_blend_pct` hold fractions despite their names.

**Statistics.** Prefixes `mean_`, `median_`, `p10_` … `p99_` (or suffixes `_mean`, `_median`,
`_p10` … `_p99`) and `max_` are statistics over the vessels of the row; `share_` and `frac_`
are fractions of those vessels; `n` and `n_vessels` count them; `fleet_` and `total` columns
are sums over them; `dominant_` and `modal_` columns give the most frequent label.

**Fuel codes.** `VLSFO` very-low-sulphur fuel oil; `MGO` marine gas oil; `HFO` heavy fuel
oil; `LNG` liquefied natural gas; `LPG`; `biodiesel_fame` fatty acid methyl ester (FAME)
biodiesel; `methanol_bio`, `methanol_e` (renewable fuel of non-biological origin),
`methanol_fossil`; `ammonia_fossil`, `ammonia_green`; `hydrogen_green`.

**Plans and pathways.** A plan is a base fuel, at most one second fuel and the energy share of
the second fuel, plus a speed. Pathway labels: `<fuel>_only` (one fuel; for example
`LNG_only`) or `<base>+<second fuel>` (for example `LNG+biodiesel_fame`). The checks write a
plan with its share, `<base>+<second fuel>@<share>`. Columns named `*_fuel`, `dominant_fuel_*`
and `*_low_carbon_fuel_*` give the *second* fuel of the plan, which can be LNG in a
VLSFO-based plan; `*_low_carbon_share_*`, `*_blend_*` and `*_alternative_component_share_*`
give its energy share, where 1 means the second fuel alone.

**Ship types.** `bulk_carrier`, `container_ship`, `cruise_passenger`, `gas_carrier`,
`general_cargo`, `lng_carrier`, `refrigerated_cargo`, `ro_pax`, `ro_ro_cargo`, `tanker`,
`vehicle_carrier`. Files on unmatched ships use the MRV ship type (`type_source` =
`mrv_ship_type`).

## Model variants

Each variant is one model specification; `configs/model_variants.yaml` defines them. The name
is a code:

| Part | Meaning |
|---|---|
| `v23` | original specification: earlier LNG factors (about 5.5% methane slip for the ETS, about 1% for FuelEU), all energy in scope, RFNBO reward factor in every year, costs per 2,000-nm leg |
| `r1` | current bookkeeping: annual costs, MRV scope, RFNBO reward factor to 2033, Annex II factors for VLSFO and MGO |
| `lng_v23` | LNG valued with the earlier factors |
| `lng_otto_ms`, `lng_otto_ss`, `lng_diesel_ss`, `lng_lbsi` | LNG valued in both instruments with the FuelEU methane-slip default of the engine class: Otto medium speed (3.1%), Otto slow speed (1.7%), Diesel slow speed (0.2%), lean-burn spark ignition (2.6%) |
| `prio` / `prop` / `route` | the second fuel is allocated to the FuelEU in-scope energy first / in proportion to the scope weight / voyage by voyage |
| `fb` | the base fuel may be VLSFO or LNG (otherwise VLSFO only) |
| `bd` | scope weight from the MRV emissions by voyage category (otherwise the reported ETS share) |
| `exact` | exact share candidates are added to the share grid |

| Variant | LNG factors | Scope weight | Allocation | Bases | Exact shares | Cost basis | Directory contents |
|---|---|---|---|---|---|---|---|
| `r1_lng_otto_ms_prio_fb_bd` (central) | Otto MS 3.1% | voyage breakdown | priority | VLSFO, LNG | yes | annual | all experiments |
| `r1_lng_otto_ss_prio_fb_bd` | Otto SS 1.7% | voyage breakdown | priority | VLSFO, LNG | yes | annual | A |
| `r1_lng_diesel_ss_prio_fb_bd` | Diesel SS 0.2% | voyage breakdown | priority | VLSFO, LNG | yes | annual | A |
| `r1_lng_lbsi_prio_fb_bd` | LBSI 2.6% | voyage breakdown | priority | VLSFO, LNG | yes | annual | A |
| `r1_lng_v23_prio_fb_bd` | earlier | voyage breakdown | priority | VLSFO, LNG | yes | annual | A |
| `r1_lng_otto_ms_prop_fb_bd` | Otto MS 3.1% | voyage breakdown | proportional | VLSFO, LNG | yes | annual | A |
| `r1_lng_otto_ms_route_fb_bd` | Otto MS 3.1% | voyage breakdown | voyage routing | VLSFO, LNG | yes | annual | A |
| `r1_lng_otto_ms_prio_fb` | Otto MS 3.1% | reported ETS share | priority | VLSFO, LNG | yes | annual | B, plus `metric_harmonization_s05*` |
| `r1_lng_otto_ms_prio` | Otto MS 3.1% | reported ETS share | priority | VLSFO | yes | annual | B |
| `r1_lng_otto_ms` | Otto MS 3.1% | reported ETS share | proportional | VLSFO | no | annual | B |
| `r1_lng_otto_ms_exact` | Otto MS 3.1% | reported ETS share | proportional | VLSFO | yes | annual | C |
| `r1_lng_otto_ms_exact_bd` | Otto MS 3.1% | voyage breakdown | proportional | VLSFO | yes | annual | C |
| `r1_lng_otto_ms_prio_bd` | Otto MS 3.1% | voyage breakdown | priority | VLSFO | yes | annual | C |
| `r1_lng_otto_ms_prop_fb` | Otto MS 3.1% | reported ETS share | proportional | VLSFO, LNG | yes | annual | C |
| `v23` | earlier | all energy | – | VLSFO | no | leg | B |
| `v23_lng_otto_ms` | Otto MS 3.1% | all energy | – | VLSFO | no | leg | B |

Directory contents: **A** = `lambda/`, `metric_harmonization*`, `credit_price*`, `biofuel*`,
`retrofit_capex*`; **B** = `lambda/`, `metric_harmonization*`; **C** = `metric_harmonization*`
only. The central variant also has `lambda_fieldonly/`, `lambda_refine/`, `speed_years/`,
`cii_proxy/`, `fuel_robustness/`, `metric_harmonization_compat_fieldonly*`, `mrv_calibration*`,
`pooling*`, `price_surface*`, `charter*`, `speed_k*`, `speed_calibration*` and
`fuel_dimension*`.

## Codes used in the tables

**Fuel menus.**

| Code | Menu |
|---|---|
| `CENTRAL_EXTENDED`, `continuous_base`, `broad` | every vessel may use VLSFO, LNG, FAME, bio-methanol and e-methanol, without retrofit cost or supply limit |
| `COMPATIBILITY_ENGINE`, `engine_capability` | LNG and methanol only where the vessel particulars record the capability (LNG carriers counted as LNG-capable); drop-in fuels for all |
| `COMPATIBILITY_CONSTRAINED_2035`, `compatible_switch` | LNG only for gas and LNG carriers; drop-in FAME for all |
| `NO_FAME_2045`, `continuous_base_excl_biodiesel_fame`, `no_fame` | the broad menu without FAME |
| `dropin_only` | drop-in fuels only |
| `compat_gas`, `compat_non_gas` | closed-form menus for gas carriers (MGO, FAME, LNG) and for other ships (MGO, FAME) |

`lambda_fieldonly/` repeats `COMPATIBILITY_ENGINE` with capability taken from the fuel and
engine fields alone, without counting LNG carriers as LNG-capable.

**Fuel-price scenarios** (`fuel_price_scenario`, `price_scenario`): `base_2026` (central),
`high_fossil_2026`, `low_alternative_2030`; prices are in `configs/fuel_prices.yaml`.

**Cases.**

| Code | File | Meaning |
|---|---|---|
| `FI_PRIMARY_NO_CII` | `lambda/`, `lambda_refine/` | λ sweep without the CII term in the decision or the evaluation |
| `M1_actual_ETS_FuelEU_only` (`M1`) | `metric_harmonization*` | ETS-only rule against the joint plan, without the CII term |
| `M2_harmonized_WtW_ranking` (`M2`) | `metric_harmonization*` | the partial rule values fuels at well-to-wake intensity under the allowance price instead of the ETS cost |
| `M0_actual_full_policy_mix` | `metric_harmonization*` | as M1, with the CII term in the joint plan's objective and the evaluation |
| `M3_same_stringency_aligned_ranking` | `metric_harmonization*` | as M2, with the CII term |
| `CR100_statutory_equivalent` … `CR0_credit_0pct` | `credit_price*` | FuelEU deficits valued at 100, 75, 50, 25 and 0% of the statutory penalty, in the decision and the evaluation |
| `B0_current_fuel_menu` | `biofuel*` | central fuel menu |
| `B1_no_FAME_no_biodiesel`, `B2_no_biofuel` | `biofuel*` | FAME removed; all biofuels removed |
| `B3_biofuel_cap_10/30/50` | `biofuel*` | biofuel at most 10, 30 or 50% of each vessel's annual energy |
| `B4_biofuel_price_1p5x/2p0x` | `biofuel*` | biofuel prices 1.5 or 2 times central |
| `C0_uncalibrated_proxy`, `C1_MRV_calibrated_baseline` | `mrv_calibration*` | baseline energy from the propulsion model; anchored to each vessel's reported CO₂ |
| `P0_no_pooling_vessel_year` | `pooling*` | no pooling |
| `P1_perfect_pooling_analytical_fleet`, `P2_perfect_pooling_by_ship_type` | `pooling*` | compliance balances pooled over the whole fleet; within each ship type |
| `P3_exogenous_FuelEU_credit_25/50/75` | `pooling*` | deficits valued at 25, 50 or 75% of the penalty |
| `SA0_uncalibrated`, `SA1_mrv_calibrated` | `speed_years/`, `speed_calibration*` | speed-only regret with energy from the propulsion model; anchored to reported CO₂ |
| `0.00` … `1.00` | `comparison/envelope_by_variant.csv` | effective FuelEU price fraction |

**CII proxy scenarios** (`scenario` in `cii_proxy/` and `comparison/` speed tables):
`neutral` (proxy as computed), `proxy_low`, `proxy_high`, `stress_low`, `stress_high`
(attained-to-required ratio scaled by ship type with the factors in
`configs/mrv_cii_proxy_bias_scenarios.csv`). In `fuel_robustness/fuel_rci_cii_dependence.csv`,
`full_policy_mix` includes the CII term and `no_cii_consequence_k0` sets its strength to zero.

**Share grids** (`blend_steps`): `11`, `21`, `51` evenly spaced share points from 0 to 1; for
variants with exact share candidates, the exact-compliance share and the allocation kinks are
added to every grid. `exact` in the checks means the candidate set alone.

**Scope allocation** (`scope_allocation`, checks): `uniform` (same mix everywhere, both
instruments cover the scope weight), `fueleu_priority` (second fuel counted against in-scope
energy first), `routed_voyage` (voyage by voyage), `routed` (second fuel burned on in-scope
energy only).

**Sample scopes** (`sample_scope`): `all_valid`, `optimization_sample` (the 12,333 ships),
`matched_fleet` (all ships matched to particulars), `common_cii_valid` (ships with a valid CII
proxy), `scope_weight_ge_0.5` (ships whose scope weight is at least 0.5).

## Files

Each entry gives the producing script (in `scripts/`), what one row is, and the columns that
the [glossary](#column-glossary) does not already explain.

### `variants/`

#### `lambda/` — planning-weight sweep (`run_fueleu_internalization_final.py`)

The partial plan weighs the FuelEU cost by λ from 0 to 1 (λ = 0 is the ETS-only rule), with
extra points around each switch, on four fuel menus (`scenario_id`) in the years tested for
each menu (broad and capability menus: 2035 and 2045; compatibility case: 2035; no-FAME case:
2045), and on share grids of 11, 21 and 51 points.

| File | One row per | Content |
|---|---|---|
| `fueleu_internalization_all_curve.csv` | menu × year × λ × share grid | regret, avoidable cost, decision agreement, fuel pathways and FuelEU balances of the partial and joint plans |
| `fueleu_internalization_curve.csv` | same rows | the same table without the configuration columns |
| `fueleu_internalization_by_type.csv` | menu × year × λ × share grid × ship type | the main curve columns by ship type |
| `fueleu_internalization_thresholds_global.csv` | menu × year × share grid | tested λ interval in which each fleet condition is first met (wide form) |
| `fueleu_internalization_thresholds_long.csv` | menu × year × share grid × condition | the same in long form, with status |
| `fuel_pathway_transition_intervals.csv` | switch of the fleet-modal pathway | `(lambda_low, lambda_high]` interval in which the most frequent partial pathway changes |
| `fuel_pathway_transition_matrix.csv` | menu × year × share grid × λ × (partial, joint) pathway pair | number and share of vessels |
| `vessel_threshold_summary_global.csv` | menu × year (11-point grid) | over vessels: 10th, 50th and 90th percentile of the first λ at which each vessel meets a criterion, and the share that meets it |
| `vessel_threshold_summary_by_resolution.csv` | menu × year × share grid | the same for every grid |
| `blend_resolution_curve.csv`, `blend_resolution_thresholds.csv` | as the curve and threshold tables | the λ windows re-solved on 21 and 51 share points, for the grid-resolution check |
| `refined_lambda_curve.csv` | as the curve | the refined λ points around the switches (11-point grid) |
| `fuel_menu_robustness_curve.csv`, `fuel_menu_robustness_thresholds.csv`, `fuel_menu_robustness_by_type.csv` | as the curve, thresholds and by-type tables | the 11-point grid rows of all menus |
| `fuel_menu_feasibility_audit.csv` | menu × ship type | share of vessels that may use LNG, FAME, bio- and e-methanol; mean number of candidate plans |
| `matrix_<menu>_all_curve.csv`, `matrix_<menu>_by_type.csv` | as the curve and by-type tables | one menu before the merge |

Threshold conditions (`condition_name`, and the prefixes of the wide threshold columns):
`fleet_regret_closed_fraction` ≥ 0.5, 0.9, 0.99, 0.999 (share of the λ = 0 fleet avoidable
cost that the partial plan no longer incurs); `fuel_pathway_label_agreement_share` and
`fuel_plan_agreement_1pp_share` ≥ 0.5, 0.9, 0.99 (share of vessels whose partial plan agrees
with the joint plan); `mean_rci_proxy` ≤ 0.05, 0.01, 0.005. The interval is the pair of tested
λ points around the first one that meets the condition; no point is interpolated.
`left_censored` = already met at the smallest tested λ; `right_censored` = never met.
`status`: `bracketed`, `exact_at_zero` (met at λ = 0), `baseline_undefined` (no regret to
close at λ = 0).

Per-vessel criteria in the vessel threshold summaries (`first_lambda_<criterion>_*`):
`fuel_pathway_label_match`, `fuel_plan_agreement_1pp`, `full_decision_exact`,
`full_decision_1pp_0p01kn`, `90pct_individual_regret_closed`, `99pct_individual_regret_closed`.

#### `lambda_fieldonly/` (`run_fueleu_internalization_final.py`)

`matrix_COMPATIBILITY_ENGINE_all_curve.csv` and `_by_type.csv` for the capability menu with
field evidence only. Columns as in `lambda/`.

#### `lambda_refine/` (`fueleu_internalization_sweep.py`, `r1_lambda_refine.py`)

| File | Content |
|---|---|
| `<menu>/fueleu_internalization_all_curve.csv`, `_curve.csv`, `_by_type.csv` | the refined points λ = 0.405, 0.410, … 0.445 for the broad (`CENTRAL_EXTENDED`) and capability (`COMPATIBILITY_ENGINE`) menus, 2035 and 2045 |
| `<menu>/fueleu_internalization_thresholds.csv` | smallest tested λ (grid-bound) at which 50, 90 or 99% of the fleet avoidable cost is closed, or 90 or 99% of vessels agree with the joint plan by each agreement criterion |
| `fueleu_internalization_all_curve_refined.csv` | the refined rows of both menus |
| `refine_summary.csv` | per menu and year: vessels whose switch is bracketed, share of them with an interval at most 0.005 wide, and the widest interval, before and after the refinement |

#### `metric_harmonization*.csv` (`metric_harmonization.py`)

One row per case (M0–M3) × year (2026, 2030, 2035, 2040, 2045) × fuel-price scenario:
regret of the partial plan, its distribution, avoidable cost, the cost components of both
plans, pathways, and for the fleet the joint plan's FuelEU energy and biofuel energy.
`_by_type` gives regret, avoidable cost and the second-fuel share by ship type.

| File | Sample and menu |
|---|---|
| `metric_harmonization.csv` | optimization sample, broad menu |
| `metric_harmonization_compat.csv` | optimization sample, capability menu |
| `metric_harmonization_compat_fieldonly.csv` | capability menu from field evidence only |
| `metric_harmonization_s05.csv` | ships with scope weight ≥ 0.5 |

`fleet_biofuel_energy_joint_gj` is the joint plan's biofuel energy over all of each vessel's
energy, in and out of FuelEU scope.

#### `credit_price*.csv` (`fueleu_credit_price_reopt.py`)

One row per case (deficit valued at 0–100% of the penalty) × year (2035, 2040, 2045): regret
and cost components, re-optimized at each fraction. `_by_type` by ship type.

#### `biofuel*.csv` (`biofuel_sensitivity.py`)

One row per biofuel case (B0–B4) × year (2035, 2045): regret, second-fuel and biofuel
shares, and the share of vessels whose plan uses FAME or any biofuel. `_by_type` by ship
type.

#### `retrofit_capex*.csv` (`r1_retrofit_cost_sensitivity.py`)

One row per menu (`continuous_base`, `engine_capability`) × year (2035, 2045) × retrofit charge
(0, 250, 500, 1,000 and 1,500 EUR/kW), charged for LNG and methanol on vessels without the recorded capability:
regret, pathways and the use of LNG and methanol by capable and non-capable vessels.
`_by_type` by ship type.

#### `mrv_calibration.csv`, `mrv_calibration_ratios.csv` (`mrv_calibration_sensitivity.py`)

Regret by year (2026–2045) with and without baseline energy anchored to reported CO₂
(`decision_dimension` = `fuel_and_speed`); modelled-to-reported CO₂ ratios by ship type
before and after calibration (median and interquartile range).

#### `fuel_robustness/` (`fuel_robustness.py`)

| File | One row per | Content |
|---|---|---|
| `fuel_rci_sensitivity.csv` | fuel menu (`fuel_scenario`) × price scenario × year (2026, 2035, 2045) | regret and second-fuel shares |
| `fuel_rci_cii_dependence.csv` | CII case × sample × year | regret with and without the CII term, cost components including the CII consequence |
| `fuel_choice_decomposition.csv` | fuel menu × price scenario × year × ship type | avoidable cost split into fuel, ETS, FuelEU, CII and delay components (EUR, summed over vessels) |

#### `fuel_dimension_by_year*.csv` (`optimize_fuel.py`)

Per year (and ship type): regret of the ETS-only plan against the joint fuel-and-speed plan,
its distribution, the avoidable cost, the share of vessels whose joint plan uses a second
fuel above 1% (`frac_blends`) and its mean share (`mean_blend_pct`). These plans feed
`pooling*`.

#### `pooling*.csv` (`fueleu_pooling.py`)

The plans of `fuel_dimension*` without re-optimization. One row per pooling case × year (and
ship type): positive, negative and net compliance balance, penalty without and with pooling,
and the regret under each case. `mean_rci_signed` is the mean of (cost of the ETS-only plan −
cost of the joint plan) / cost of the ETS-only plan, without the floor at zero, and
`share_rci_negative` the share of vessels for which it is negative. They differ from the
floored regret only in the deficit-valuation cases (`P3_*`), where the joint plan, chosen at
the full penalty, can cost more than the ETS-only plan.

#### `price_surface.csv`, `charter.csv` (`fuel_price_sensitivity.py`)

One row per year (2035, 2045) × fuel-price scenario × allowance price (50, 80, 120, 160 and
200 EUR/tCO₂) × charter-rate multiplier: regret statistics and the joint plan's mean
second-fuel share. `price_surface` varies the price scenario and the allowance price at
central charter rates; `charter` varies the charter rate (0.5, 1 and 2 times central) and the
allowance price at central fuel prices.

#### `speed_k*.csv` (`rci_sensitivity.py`)

Speed-only regret in 2026 with VLSFO fixed, by strength `cii_pressure` (0–2) of the CII
commercial-pressure term. `_by_type` by ship type.

#### `speed_years/`, `speed_calibration*.csv` (`speed_rci_mrv_calibration.py`)

Speed-only regret with VLSFO fixed: `speed_years_<year>.csv` for 2026, 2030, 2035, 2040 and
2045 at the central VLSFO price; `speed_calibration.csv` for 2035 at 650 EUR/t. One row per
energy case (`SA0`, `SA1`) × CII strength (`cii_pressure_k` 0, 0.2, 2). `_by_type` by ship type.

#### `cii_proxy/` (`cii_proxy_sensitivity.py`)

| File | Content |
|---|---|
| `cii_proxy_rating_mix.csv` | share of ships in each CII proxy rating A–E (and D or E) by scenario and ship type |
| `cii_proxy_threshold_tail.csv` | attained-to-required ratio (mean, 90th percentile) and share of ships near (within 0.05) or above the C/D boundary |
| `speed_rci_cii_proxy_sensitivity.csv` | speed-only regret by proxy scenario and CII strength `k` |

### `comparison/`

Produced by `r1_compare_variants.py`, except `spec_factorial*` (`r1_spec_factorial.py`).

| File | One row per | Content |
|---|---|---|
| `headline_by_variant.csv` | variant × year | regret of the ETS-only (M1), WtW-valued (M2) and CII-inclusive (M0) rules; pathways; share of vessels whose pathway changes; fleet avoidable cost; fleet biofuel energy of the joint plan; mean scope weight |
| `two_by_two.csv` | variant × year | regret of the ETS-only and WtW-valued rules on the broad and capability menus (and capability from field evidence) |
| `lambda_transitions.csv` | variant × menu × year × share grid | switching interval of the fleet-modal pathway |
| `closed_form_check.csv` | variant × menu × year × switch | swept interval on the 51-point grid against the closed-form interval and exact switching weight at scope weight 1 |
| `envelope_by_variant.csv` | variant × experiment × case × year | mean regret under the effective-price, fuel-availability, menu, calibration and pooling cases; `joint_pathway` is the joint plan's most frequent second fuel where available |
| `distribution_by_variant.csv` | variant × source × menu × year | percentiles of vessel regret, shares below or above thresholds, share of changed pathways; `source` is the λ sweep at λ = 0 or the M1 case |
| `retrofit_by_variant.csv` | variant × menu × year × charge | the retrofit tables of all variants |
| `denominators_heterogeneity.csv` | variant × year × stratum | the partial-plan cost (denominator) distribution, regret with the 1% or 5% smallest denominators removed, cost-weighted regret and absolute losses, for all vessels (`ALL`) and by scope-weight bin, ship type and LNG capability |
| `scope_sample_sensitivity.csv` | variant × sample × year × case | M1 and M2 regret on all vessels and on ships with scope weight ≥ 0.5 |
| `decomposition.csv` | step × year | regret, pathways, fleet loss and switching intervals along the path from the original to the central specification (steps 1–5b; the original specification on its earlier vessel input, step 0, is not included) |
| `spec_factorial.csv` | combination × year | regret and fleet loss for the eight combinations of allocation (`priority`), blend base (`lng_base`) and scope source (`breakdown`), each coded 0/1 |
| `spec_factorial_shapley.csv` | year × factor | effect of each choice averaged over the orders of adding it (Shapley value, regret units); `total (central - step 4a)` is their sum |
| `speed_by_variant.csv` | CII strength | speed-only regret of the central variant (as `speed_k.csv`) |

### `checks/`

| File | Script | One row per | Content |
|---|---|---|---|
| `closed_form/closed_form_fuel_inputs.csv` | `r1_closed_form_thresholds.py` | variant × fuel | price, ETS CO₂e factor and WtW intensity used in the closed form |
| `closed_form/closed_form_summary.csv` | same | variant × menu × share grid × allocation × year × scope weight | per-GJ plan of the ETS-only rule, the WtW-valued rule and the joint rule; joint cost; regret (EUR/GJ and share); joint intensity |
| `closed_form/closed_form_lambda_transitions.csv` | same | switch | bracketing λ points, exact switching weight `lambda_exact` (intersection of the two plans' cost lines), plans before and after, `regret_share_after` (remaining regret share of the new plan) |
| `vessel_closed_form/vessel_closed_form_summary.csv` | `r1_vessel_closed_form_check.py` | variant × menu × year × share grid × scope-weight band | share of vessels whose swept switching interval contains the closed-form switching weight at their scope weight, and related agreement measures |
| `three_fuel/three_fuel_check_<menu>_<prices>.csv` | `r1_three_fuel_check.py` | menu × price scenario × year × scope weight × λ | cheapest two-fuel plan against the cheapest mixture of up to three fuels on a refined grid, per GJ |
| `speed_grid/speed_grid_summary.csv` | `r1_speed_grid_check.py` | year × number of speed points | mean regret of the central variant on 15-, 29- and 57-point speed grids; largest change for one vessel; mean change of the joint speed |
| `worked_examples/worked_examples.csv` | `r1_worked_examples.py` | quantity | regulatory worked examples E1–E7: hand calculation, model value, difference, tolerance, PASS |

### `data/`

| File | Script | Content |
|---|---|---|
| `data_audit/sample_waterfall.csv` | `r1_data_audit.py` | sample size at each step from the MRV rows to the optimization sample |
| `data_audit/sample_exclusion_reasons.csv` | same | matched ships outside the optimization sample, by reason and distance source |
| `data_audit/sample_comparison.csv`, `sample_comparison_by_type.csv` | same | optimization sample, excluded matched ships and unmatched ships: medians, means and missing shares of activity and particulars; ship-type shares |
| `data_audit/mrv_data_dictionary.csv` | same | each MRV field used: internal name, definition, role, presence and non-missing share |
| `data_audit/scope_weight_by_type.csv` | same | scope-weight distribution by ship type |
| `data_audit/compliance_landscape.csv` | same | 2026 ETS cost, FuelEU deficit and CII proxy D/E share of the matched fleet for four variants |
| `scope_check/scope_check_summary.csv` | `r1_scope_check.py` | ships by band of the reported ETS share (`s_band`) and by `class`: reported share consistent with the breakdown (within 0.02), below or above it, or no breakdown; median shares; CO₂ share |
| `scope_check/scope_check_low_by_type.csv`, `scope_check_low_by_flag.csv` | same | ships with a reported share below 0.5, by ship type or flag state and class |
| `scope_check/columns_found.txt` | same | the MRV columns used for the voyage breakdown |
| `baseline_diagnostics/sample_denominators.csv` | `baseline_diagnostics.py` | the analysis samples, their size, use and exclusion rule |
| `baseline_diagnostics/baseline_co2_calibration.csv` | same | modelled-to-reported CO₂ ratio by ship type: mean, median, 10th and 90th percentiles, share within 20% and 50% |
| `baseline_diagnostics/mrv_implied_speed_distribution.csv` | same | operating speed (annual distance / hours at sea) and its ratio to design speed, by ship type, for ships with a recorded design speed |
| `baseline_diagnostics/matched_unmatched_diagnostics.csv` | same | CO₂ and distance of matched and unmatched ships |
| `baseline_diagnostics/ets_scope_emissions_by_ship_type.csv` | same | reported CO₂ and ETS-scoped CO₂e by ship type |
| `baseline_diagnostics/fueleu_scope_and_balance_audit.csv` | same | shares of ships in deficit or surplus and the penalty total, by ship type |

## Column glossary

Columns are listed alphabetically. Where several columns follow one pattern, the pattern is
given once: `<stat>` stands for `mean`, `median`, `p10`, `p25`, `p75`, `p90`, `p95`, `p99`
or `max`; `<plan>` for `silo`, `siloed`, `partial` (the partial plan) or `joint`.

**Patterns**

| Pattern | Meaning |
|---|---|
| `<stat>_rci_<plan>`, `rci_<stat>`, `rci_silo_<stat>`, `rci_joint_<stat>` | statistic of vessel regret (see Conventions) |
| `<stat>_rci_proxy` | statistic of vessel regret in the λ sweep |
| `<stat>_delta_cost_eur`, `delta_cost_<stat>_eur`, `delta_cost_total_eur` | statistic or sum of the avoidable cost, EUR |
| `mean_<component>_<plan>_eur` with `<component>` = `fuel_cost`, `ets_cost`, `fueleu_penalty`, `delay_cost`, `cii_consequence` | mean cost component of the plan, EUR |
| `mean_fueleu_cost_<plan>_full_eur` | mean FuelEU deficit cost of the plan at the full penalty, EUR |
| `delta_<component>_eur`, `delta_total_eur` | difference partial − joint of a cost component, summed over vessels, EUR |
| `share_rci_gt_<x>`, `frac_gt_<x>`, `pct_rci_gt_<x>` | fraction of vessels with regret above x (`0`, `1pct`, `5pct`, `10pct`, `10` = 10%) |
| `share_rci_lt_<x>`, `share_regret_lt_<x>` | fraction of vessels with regret below x (`0p01pct` = 0.01%, `1pct`) |
| `<condition>_ge<v>_<field>`, `<condition>_le<v>_<field>` | wide threshold table: condition (see `lambda/`), threshold (`0p5` = 0.5), and `lambda_low`, `lambda_high`, `interval`, `left_censored` or `right_censored` |
| `first_lambda_<criterion>_<stat>`, `first_lambda_<criterion>_reach_share` | over vessels, percentile of the first λ at which the vessel meets the criterion; fraction of vessels that meet it in the tested range |
| `lambda_<x>pct_<criterion>` | smallest tested λ at which x% of the fleet avoidable cost is closed or x% of vessels agree with the joint plan |
| `<variable>_<median\|mean\|missing_share>` (sample comparison) | median, mean and missing fraction of `total_co2_tonnes`, `total_fuel_tonnes`, `distance_nm`, `hours_at_sea`, `scope_weight`, `gross_tonnage`, `deadweight`, `design_speed_knots`, `main_engine_power_kw`, `age_years` |
| `<A–E>_share`, `DE_share` | fraction of ships with CII proxy rating A … E, or D or E |
| `share_joint_uses_<fuel>[_capable\|_noncapable]`, `share_siloed_uses_<fuel>[_noncapable]` | fraction of vessels (capable or non-capable ones) whose plan uses the fuel |
| `pct_vessels_using_<fuel>_<plan>` | fraction of vessels whose plan uses FAME or any biofuel |
| `share_<fuel>_feasible` | fraction of vessels allowed to use the fuel on the menu |
| `<scenario>_ETS`, `<scenario>_WtW` (`two_by_two.csv`) | regret of the ETS-only and WtW-valued rules on the broad (`broad`), capability (`engine_capability`) and field-only capability menus |
| `modelled_to_MRV_CO2_<median\|iqr>_<calibrated\|uncalibrated>` | median and interquartile range of the modelled-to-reported CO₂ ratio |
| `ratio_<stat>`, `rho_<stat>` | statistic of the modelled-to-reported CO₂ ratio; of the CII attained-to-required ratio |
| `speed_<stat>_kn` | statistic of operating speed, knots |
| `denominator_<min\|p1\|p5\|median>_eur` | statistic of the partial plan's full cost, EUR |
| `abs_loss_<total_eur_m\|median_eur\|p90_eur>`, `absolute_loss_<total\|median\|p90>_eur` | sum (million EUR, or EUR) and percentiles of the avoidable cost |
| `regret_<rule>_<case>` | mean regret of a rule (`ets_only`, `ets_rule`, `wtw_valued`, `wtw_rule`, `full_mix`) in a case (`M0`, `M1`, `M2`) |
| `n_switch_before`, `n_switch_after`, `share_le_0p005_before`, `share_le_0p005_after`, `max_width_before`, `max_width_after` (`refine_summary.csv`) | before and after the refinement: vessels with a bracketed switch; share of them with an interval at most 0.005 wide; widest interval |

**Individual columns**

| Column | Meaning |
|---|---|
| `abs_diff` | absolute difference between hand and model value |
| `above_CD_share` | fraction of ships whose CII ratio is above the C/D boundary |
| `aggregate_rci_proxy`, `aggregate_rci_silo`, `aggregate_regret` | summed avoidable cost / summed partial-plan cost |
| `alternative_component_share_agreement_1pp`, `..._1pp_share` | fraction of vessels whose second-fuel share in the partial plan is within 1 percentage point of the joint plan's |
| `baseline_regret_zero` | the fleet regret at λ = 0 is zero, so closure fractions are undefined |
| `blend_steps` | share grid (see Codes) |
| `breakdown`, `lng_base`, `priority` | factorial flags (0/1): scope weight from the breakdown; LNG allowed as base; priority allocation |
| `broad_ETS`, `broad_WtW`, `engine_capability_ETS`, `engine_capability_WtW`, `engine_capability_fieldonly_ETS`, `engine_capability_fieldonly_WtW` | see the `two_by_two.csv` pattern |
| `capex_eur_per_kw` | retrofit charge |
| `case` | case code (see Codes) |
| `charter_mult` | charter-rate multiplier |
| `cii_pressure`, `cii_pressure_k`, `k` | strength of the CII commercial-pressure term |
| `cii_proxy_de_share` | fraction of ships with CII proxy rating D or E |
| `class` | scope-check class (see `data/`) |
| `closed_form_exact_lambda`, `closed_form_interval_51` | closed-form switching weight and its bracket on the 51-point grid |
| `co2_share_of_sample` | fraction of the sample's reported CO₂ |
| `co2_weighted_mean` | CO₂-weighted mean scope weight |
| `condition_direction`, `condition_name`, `condition_threshold` | threshold condition (`ge` or `le`, name, value) |
| `construct` | form of the retrofit charge (`fixed_capex_eur_per_kw`) |
| `cost_basis` | `annual` or `leg` (see Units) |
| `cost_weighted_regret` | summed avoidable cost / summed denominator |
| `decision_dimension` | the decision varied (`fuel_and_speed`) |
| `deficit_vessel_share`, `surplus_vessel_share` | fraction of ships in FuelEU deficit; in surplus |
| `definition` | definition of a sample step or field |
| `design_speed_mean_kn` | mean design speed, knots |
| `distance_source` | how annual distance was derived (`fuel_per_nm`, `missing`) |
| `dominant_fuel_<plan>`, `dominant_low_carbon_fuel_<plan>` | most frequent second fuel of the plan |
| `dominant_fuel_label_agreement_share`, `fuel_pathway_label_agreement_share` | fraction of vessels whose partial pathway equals the joint pathway |
| `ets_co2e_g_per_mj` | ETS-covered CO₂e factor of the fuel, g/MJ |
| `ets_cost_total_eur`, `ets_covered_tco2` | fleet ETS cost; ETS-covered CO₂, t |
| `ets_scoped_co2e_t` | ETS-scoped CO₂e, t |
| `eua`, `eua_price`, `eua_price_eur_per_tco2` | allowance price |
| `example` | worked-example code |
| `exclude_fuels` | fuels removed from the menu |
| `exclusion_reason` | reason a ship leaves the sample |
| `experiment` | experiment of the envelope table |
| `factor` | factorial choice |
| `fallback_share` | fraction of ships whose scope weight falls back to the reported ETS share |
| `feasibility_mode`, `fuel_feasibility`, `fuel_menu_id`, `fuel_scenario`, `menu` | fuel menu (see Codes) |
| `final_feasible_vessel_count` | vessels with at least one feasible plan |
| `flag` | flag state |
| `fleet_biofuel_energy_joint_gj`, `fleet_biofuel_energy_joint_pj_M1` | fleet biofuel energy of the joint plan, GJ; PJ (case M1) |
| `fleet_delta_cost_eur`, `fleet_delta_cost_eur_m_M1`, `fleet_loss_eur_m` | fleet avoidable cost, EUR; million EUR |
| `fleet_fueleu_energy_joint_gj` | fleet energy in FuelEU scope of the joint plan, GJ |
| `fleet_modal_energy_dominant_fuel_<plan>` | most frequent fuel with the larger energy share in the plan |
| `fleet_modal_fuel_pathway_<plan>`, `modal_pathway_<plan>` | most frequent pathway of the plan |
| `frac_positive` | fraction of vessels with positive regret |
| `fleet_regret_closed_fraction`, `fleet_regret_remaining_fraction` | share of the λ = 0 fleet avoidable cost no longer incurred; still incurred |
| `fuel` | fuel code |
| `fuel_plan_agreement_1pp`, `..._1pp_share` | fraction of vessels whose partial plan has the joint pathway and a second-fuel share within 1 percentage point |
| `fuel_price_scenario`, `price_scenario` | fuel-price scenario |
| `fueleu_deficit_cost_total_eur`, `fueleu_deficit_share`, `fueleu_n_audited` | fleet FuelEU deficit cost; fraction of audited ships in deficit; ships audited |
| `fueleu_evaluation_multiplier` | factor on the FuelEU cost in the evaluation |
| `fueleu_price_fraction` | effective FuelEU price fraction f (decision and evaluation) |
| `fueleu_target_gco2e_per_mj` | FuelEU target intensity |
| `fuels_in_mixture` | number of fuels in the cheapest mixture |
| `full_decision_agreement_share`, `full_decision_exact_share` | fraction of vessels whose partial plan equals the joint plan in pathway, share and speed |
| `full_decision_1pp_0p01kn_share` | as above with share within 1 percentage point and speed within 0.01 knot |
| `gain_eur_per_gj`, `gain_share` | cost saved by the three-fuel mixture, EUR/GJ and fraction |
| `grid_min_step`, `grid_step` | smallest λ step tested |
| `group` | sample group |
| `hand`, `model`, `tolerance`, `status`, `quantity`, `unit`, `note`, `notes` | worked-example columns; free-text notes |
| `include_cii_in_evaluation`, `include_cii_in_planner` | CII term in the evaluation; in the decision |
| `internal_name`, `role`, `present`, `non_null_share` | data dictionary columns |
| `interval` | tested λ interval `(low, high]` |
| `joint_cost_eur_per_gj`, `two_fuel_cost_eur_per_gj`, `mixture_cost_eur_per_gj` | cost per GJ of the joint plan; of the cheapest two-fuel plan; of the cheapest mixture |
| `joint_intensity_gco2e_per_mj` | in-scope WtW intensity of the joint plan |
| `joint_pathway`, `joint_plan`, `lambda0_plan`, `wtw_valued_plan`, `two_fuel_plan`, `mixture`, `from_plan`, `to_plan`, `from_pathway`, `to_pathway`, `partial_fuel_pathway`, `joint_fuel_pathway`, `pathway_ets_only`, `pathway_ets_rule`, `pathway_joint`, `pathway_wtw_valued` | plan or pathway labels (see Conventions) |
| `lambda`, `lambda_fueleu` | planning weight λ |
| `lambda0_regret_eur_per_gj`, `lambda0_regret_share`, `wtw_valued_regret_share` | regret of the ETS-only rule per GJ and as a share; regret share of the WtW-valued rule |
| `lambda_exact` | closed-form switching weight |
| `lambda_low`, `lambda_high`, `lambda_min_tested`, `lambda_max_tested` | bracketing tested λ points; range tested |
| `lambda_switches_51` | switches of the fleet-modal pathway on the 51-point grid |
| `left_censored`, `right_censored` | condition met at the first tested λ; never met |
| `legacy_alternative_candidate_label_<plan>` | most frequent second-fuel candidate label, even at zero share; descriptive only, use the pathway columns for fuel choice |
| `max_abs_change_regret_pp` | largest change of one vessel's regret against the base grid, percentage points |
| `max_biofuel_share` | biofuel cap, fraction of vessel energy |
| `mean_abs_speed_change_kn_joint` | mean change of the joint speed against the base grid, knots |
| `mean_alternative_component_share_<plan>`, `mean_low_carbon_share_<plan>`, `mean_lowcarbon_share_<plan>`, `mean_blend_pct` | mean energy share of the second fuel |
| `mean_biofuel_share_<plan>` | mean biofuel share of energy |
| `mean_blend_share_abs_diff`, `p90_blend_share_abs_diff` | mean and 90th percentile of the absolute share difference between the plans |
| `mean_candidate_count` | mean number of feasible candidate plans |
| `mean_co2_t`, `median_co2_t`, `total_mrv_co2_t`, `mrv_reported_co2_t` | reported CO₂, t |
| `mean_distance_nm` | mean annual distance, nautical miles |
| `mean_fueleu_balance_<plan>` | mean FuelEU compliance balance, gCO₂e |
| `mean_joint_speed`, `mean_silo_speed` | mean speed of the joint and ETS-only plans, knots |
| `mean_rci_joint_norm` | mean regret normalized by the joint plan's cost |
| `mean_rci_signed`, `share_rci_negative` | see `pooling*` |
| `mean_rci_when_positive` | mean regret over vessels with positive regret |
| `mean_regret`, `median_regret`, `p95_regret`, `regret` | vessel regret statistics |
| `mean_regret_drop_low1pct`, `mean_regret_drop_low5pct` | mean regret without the 1% or 5% of vessels with the smallest denominators |
| `mean_regret_pct` | mean regret, percent |
| `mean_scope_weight` | mean scope weight |
| `mean_speed_abs_diff_kn`, `p90_speed_abs_diff_kn` | mean and 90th percentile of the absolute speed difference between the plans, knots |
| `mean_wtw_intensity_<plan>_gco2e_per_mj` | mean in-scope WtW intensity of the plan |
| `median_abs_err_narrow` | median distance between the closed-form weight and the midpoint of intervals at most 0.005 wide |
| `model_variant`, `variant`, `root` | model variant; result tree of the variant (`main`) |
| `n_lng_capable` | vessels with recorded LNG capability |
| `n_points`, `speed_search_points`, `n_tested_points` | speed-grid points; λ points tested |
| `n_positive_regret`, `n_zero_regret`, `n_switch_both` | vessels with positive regret; zero regret; switching in both the sweep and the closed form |
| `n_with_weight` | ships with a scope weight from the breakdown |
| `near_CD_share` | fraction of ships whose CII ratio is within 0.05 of the C/D boundary |
| `net_CB`, `total_positive_CB`, `total_negative_CB` | net, positive and negative FuelEU compliance balance of the fleet, gCO₂e |
| `no_pooling_penalty_eur`, `pooled_penalty_eur`, `penalty_reduction_pct` | fleet penalty without and with pooling, EUR; reduction, percent |
| `op_to_design_median` | median ratio of operating to design speed |
| `p10`, `p25`, `p50`, `p75`, `p90`, `mean` | scope-weight distribution |
| `penalty_total_eur`, `surplus_penalised_check_eur` | fleet FuelEU penalty; penalty charged to ships in surplus (0 if correct) |
| `price_eur_per_gj` | fuel price |
| `rfnbo` | fuel is a renewable fuel of non-biological origin |
| `s_band` | band of the reported ETS share or the scope weight |
| `s_breakdown_median`, `s_reported_median` | median scope weight from the breakdown; median reported ETS share |
| `sample`, `sample_name`, `sample_scope` | sample (see Codes) |
| `scenario`, `scenario_id` | CII proxy scenario or fuel menu (see Codes) |
| `scope_allocation` | allocation rule in the closed form (see Codes) |
| `scope_weight` | scope weight |
| `shapley` | Shapley value, regret units |
| `share` | fraction of vessels (transition matrix) or of ships (by type) |
| `share_inside`, `share_inside_1pct`, `share_inside_path` | fraction of vessels whose swept interval contains the closed form; with 1% regret closure; for pathway switches |
| `share_interval_le_0p005` | fraction of vessels whose swept interval is at most 0.005 wide |
| `share_of_fleet_ets_emissions` | fraction of the fleet's ETS-scoped emissions |
| `share_pathway_changed`, `share_pathway_changed_M1` | fraction of vessels whose partial pathway differs from the joint pathway |
| `share_switch_agree`, `share_zero_agree` | fraction of vessels where the sweep and the closed form agree on a switch; on zero regret |
| `ship_type` | ship type |
| `source` | source of the distribution row |
| `step` | sample step or specification step |
| `stratum`, `value` | stratification variable and its value |
| `sweep_interval_51` | swept interval on the 51-point grid |
| `three_fuels_better` | a mixture with a third fuel is cheaper |
| `threshold_is_grid_bound` | thresholds are tested λ points, not interpolated |
| `total_co2_tonnes`, `total_fuel_tonnes`, `distance_nm`, `hours_at_sea`, `gross_tonnage`, `deadweight`, `design_speed_knots`, `main_engine_power_kw`, `age_years` | see the sample-comparison pattern |
| `type_source` | source of the ship type |
| `used_for` | use of a sample |
| `within_20pct_share`, `within_50pct_share` | fraction of ships with a modelled-to-reported CO₂ ratio within 20% or 50% of 1 |
| `wtw_g_per_mj` | WtW intensity of the fuel, gCO₂e/MJ |
| `year` | policy year |

## Licence

All Rights Reserved – Review Only. The result files may be viewed and used solely to review the
associated manuscript and to reproduce or verify its reported results; no other use, copying,
redistribution, publication or modification is permitted without the author's prior written
permission. The MIT License of the code does not apply to them. The files contain information
from the EU MRV system published by the European Maritime Safety Agency (THETIS-MRV); vessel
particulars from a commercial register enter only in aggregate form.
