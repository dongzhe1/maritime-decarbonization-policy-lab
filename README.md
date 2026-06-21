# Maritime Decarbonization Policy Lab

Open-source implementation of the **Regulation Coupling Index (RCI)** framework for
EU maritime decarbonization. It quantifies how the three maritime carbon
instruments — **EU ETS**, **FuelEU Maritime**, and **IMO CII** — reinforce or
conflict with each other across a fleet of EU-calling vessels.

## Overview

The RCI is the fractional cost penalty an operator pays by optimising for one
regulation in isolation rather than jointly over all three:

```
RCI = (cost_siloed − cost_joint) / cost_siloed
```

Two qualitatively different coupling dimensions emerge:

| Dimension | Driver | Finding |
|---|---|---|
| **Speed / operational** | CII enforcement strength *k* | RCI ≈ 0 across all *k*. Slow steaming lowers fuel burn, energy use and attained CII together — the instruments are co-monotonic, so they do not conflict on the speed decision. |
| **Fuel / structural** | FuelEU stringency (year) | RCI large and fleet-wide (~10% in 2030, ~33% by 2045). EU ETS prices tank-to-wake CO₂ *quantity*, FuelEU prices well-to-wake GHG *intensity*, CII prices CO₂ per *transport work* — three different metrics that reward different fuels, creating genuine conflict. |

## Quick start (synthetic data)

You do not need the real (commercial / registration-gated) inputs to try the
pipeline. Generate plausible synthetic data and run the full compliance + RCI
computation in under a minute:

```bash
pip install -e .

# 1. Generate synthetic MRV + fleet CSVs (physically plausible, NOT for analysis)
python scripts/generate_fake_data.py --n-vessels 500 --year 2024

# 2. Per-vessel compliance: CII rating + FuelEU penalty + EU ETS cost
sct run-regulations-mrv \
    --mrv      data/raw/mrv/2024_renamed.csv \
    --fleet    data/raw/vessel_static/fleet_ship_info.csv \
    --scenario configs/scenarios/base_2026.yaml \
    --output   results/compliance/compliance.parquet

# 3. Fuel-dimension RCI over FuelEU stringency (the headline result)
python scripts/optimize_fuel.py \
    --mrv       data/raw/mrv/2024_renamed.csv \
    --fleet     data/raw/vessel_static/fleet_ship_info.csv \
    --scenario  configs/scenarios/base_2026.yaml \
    --out       results/optimization/fuel_optimized_blend_ext.parquet \
    --out-sweep results/optimization/fuel_rci_by_year_blend_ext.csv \
    --years     2026,2030,2035,2040,2045 \
    --blend --extended
```

The synthetic fleet reproduces a realistic A–E CII spread and sensible compliance
costs, but the numbers are random — use the real inputs below for any analysis.

## Installation

Requires Python ≥ 3.11.

```bash
pip install -e .
```

This installs the `sct` command-line tool and the `carbon_twin` package.

## Data

The pipeline runs on **Path C (MRV-direct)**: it uses THETIS-MRV annual totals
plus vessel particulars, and reconstructs a representative voyage per vessel from
the reported distance and hours at sea. **AIS data is not required and not used by
this pipeline** — the `carbon_twin.data.loaders.load_ais_parquet` helper exists
for an alternative voyage-resolved path that the RCI results do not depend on.

Two inputs are needed. **Neither is redistributed in this repository.**

### 1. THETIS-MRV emissions (free)

The EU's Monitoring, Reporting and Verification annual CO₂ report. Publicly
available from the [EMSA THETIS-MRV portal](https://mrv.emsa.europa.eu/#public/emission-report)
(download the annual report as CSV).

The raw download has verbose column headers; rename them to the standard schema
the loader expects, and save as `data/raw/mrv/<year>_renamed.csv`:

| Standard column | Type | THETIS-MRV source field (approx.) |
|---|---|---|
| `imo_number` | str | IMO Number |
| `ship_type` | str | Ship type |
| `year` | int | Reporting period |
| `total_co2_tonnes` | float | Total CO₂ emissions [m tonnes] |
| `total_fuel_tonnes` | float | Total fuel consumption [m tonnes] |
| `distance_nm` | float | Annual total distance travelled [n miles] |
| `hours_at_sea` | float | Annual total time spent at sea [hours] |
| `transport_work` | float | (mass × distance) transport work |
| `avg_speed` | float | optional |
| `co2_ets_tonnes` | float | optional; ETS-regulated CO₂ subset (falls back to `total_co2_tonnes`) |

Only `imo_number`, `ship_type`, `total_co2_tonnes`, `total_fuel_tonnes`,
`distance_nm`, `hours_at_sea` and `transport_work` are needed for the core results.

### 2. Vessel particulars

Per-vessel GT, DWT, design speed, main-engine power and build year. These feed the
admiralty-law fuel model and the fouling/age margin. The expected output file is
`data/raw/vessel_static/fleet_ship_info.csv` with columns:

| Column | Required | Used for |
|---|---|---|
| `imo_number` | yes | join key to MRV |
| `ship_type` | yes | CII reference line, charter rate |
| `gross_tonnage` | yes | CII capacity basis (some types) |
| `deadweight` | recommended | CII capacity basis (most types), displacement |
| `design_speed_knots` | recommended | admiralty coefficient, speed bounds |
| `main_engine_power_kw` | recommended | admiralty coefficient (fuel burn) |
| `year_built` | recommended | fouling/age margin |
| `name`, `flag` | optional | labelling only |

**Source (commercial):** the project was built against **IHS Markit Sea-web /
Maritime (the *World Register of Ships*, formerly IHS Fairplay / Lloyd's Register
of Ships)** — a subscription database. `scripts/load_ship_info.py` converts its
`ship_info.csv` export (fields `lrnoimo_ship_no`, `shiptype_level5`,
`speedservice`, `total_kilowattsof_main_engines`, `year_of_build`, …) into the
fleet CSV above.

**Free alternatives** (look up particulars by IMO number):
- [Equasis](https://www.equasis.org) — free with registration; GT, DWT, type,
  year of build, and some engine data for most merchant vessels.
- [IMO GISIS — Ship Particulars](https://gisis.imo.org) — free with registration.

Design speed and main-engine power are sparse in the free sources;
`scripts/estimate_fleet.py` fills any missing particulars with ship-type typicals
(tier-1 estimates), so the pipeline still runs end-to-end with partial data.

### Directory layout

```
data/
  raw/
    mrv/
      2024_renamed.csv          # THETIS-MRV with standardised column names
    vessel_static/
      fleet_ship_info.csv       # vessel particulars (output of load_ship_info / estimate_fleet)
```

## Data preparation (real data)

```bash
# Option A — convert an IHS Sea-web ship_info.csv export to the fleet CSV
python scripts/load_ship_info.py \
    --ship-info data/raw/ship_info.csv \
    --imo-file  data/raw/vessel_static/imo_list.txt \   # optional: restrict to these IMOs
    --out       data/raw/vessel_static/fleet_ship_info.csv

# Option B — no particulars database: back-estimate tier-1 particulars from MRV
python scripts/estimate_fleet.py \
    --mrv  data/raw/mrv/2024_renamed.csv \
    --out  data/raw/vessel_static/ \
    --year 2024
```

## Running experiments

```bash
# Per-vessel compliance (CII rating + FuelEU penalty + EU ETS cost)
sct run-regulations-mrv \
    --mrv      data/raw/mrv/2024_renamed.csv \
    --fleet    data/raw/vessel_static/fleet_ship_info.csv \
    --scenario configs/scenarios/base_2026.yaml \
    --output   results/compliance/compliance.parquet

# Speed-dimension RCI: joint vs CII-siloed speed optimisation
sct optimize-mrv \
    --case     results/compliance/compliance.parquet \
    --mrv      data/raw/mrv/2024_renamed.csv \
    --fleet    data/raw/vessel_static/fleet_ship_info.csv \
    --scenario configs/scenarios/base_2026.yaml \
    --output   results/optimization/optimized.parquet

# Speed RCI sensitivity to CII enforcement strength k (parallel sweep)
python scripts/rci_sensitivity.py \
    --mrv         data/raw/mrv/2024_renamed.csv \
    --fleet       data/raw/vessel_static/fleet_ship_info.csv \
    --scenario    configs/scenarios/base_2026.yaml \
    --out         results/optimization/rci_sensitivity.csv \
    --out-by-type results/optimization/rci_sensitivity_by_type.csv \
    --pressures   0,0.05,0.1,0.2,0.5,1.0,2.0

# Fuel-dimension RCI: joint vs ETS-siloed fuel choice (continuous blending)
python scripts/optimize_fuel.py \
    --mrv         data/raw/mrv/2024_renamed.csv \
    --fleet       data/raw/vessel_static/fleet_ship_info.csv \
    --scenario    configs/scenarios/base_2026.yaml \
    --out         results/optimization/fuel_optimized_blend_ext.parquet \
    --out-sweep   results/optimization/fuel_rci_by_year_blend_ext.csv \
    --out-by-type results/optimization/fuel_rci_by_year_type_blend_ext.csv \
    --years       2026,2030,2035,2040,2045 \
    --blend --extended

# Fuel RCI robustness: fuel-price scenario × EUA surface (parallel)
python scripts/fuel_price_sensitivity.py \
    --mrv             data/raw/mrv/2024_renamed.csv \
    --fleet           data/raw/vessel_static/fleet_ship_info.csv \
    --scenario        configs/scenarios/base_2026.yaml \
    --out             results/optimization/fuel_price_sensitivity.csv \
    --years           2035,2045 \
    --price-scenarios base_2026,high_fossil_2026,low_alternative_2030 \
    --euas            50,80,120,160,200 \
    --blend --extended

# Monte Carlo uncertainty (EUA price uncertainty, CVaR)
sct run-scenarios \
    --case     results/compliance/compliance.parquet \
    --scenario configs/scenarios/base_2026.yaml \
    --n        2000 \
    --output   results/monte_carlo/
```

The parallel sweeps (`rci_sensitivity.py`, `optimize_fuel.py`,
`fuel_price_sensitivity.py`) accept `--workers N` to set the number of worker
processes (default: all CPUs).

## Repository layout

```
carbon_twin/
  regulations/   CII, FuelEU, EU ETS engines (+ IMO NZF draft)
  optimization/  speed and fuel-choice optimisers, RCI, AE metrics, pooling
  emissions/     admiralty-law physics, fuel/emission factors, proxy, MRV calibration
  scenarios/     Monte-Carlo / scenario generation
  data/          loaders and schemas
  pipeline.py    end-to-end orchestration (Path C)
  cli.py         `sct` command-line interface
scripts/         data prep + experiment entry points
configs/         scenarios, fuel prices, fuel/emission factors, regulation versions
tests/
```

## Regulatory parameters

CII reference lines (coefficients *a*, *c*, capacity basis) are transcribed
verbatim from **MEPC.353(78) Table 1**. Ship-type-specific rating dd-vectors
(size-dependent for gas and LNG carriers) are transcribed verbatim from
**MEPC.354(78) Table 1**. Both live in `carbon_twin/regulations/cii.py`
(`_REFERENCE_LINES` and `_DD_VECTORS`).

FuelEU Maritime parameters follow **Regulation (EU) 2023/1805**; EU ETS phase-in
and scope follow **Directive 2003/87/EC** as amended by **Directive 2023/959**.

## Citation

[![DOI](https://zenodo.org/badge/1275636879.svg)](https://doi.org/10.5281/zenodo.20782697)

## Licence

MIT
