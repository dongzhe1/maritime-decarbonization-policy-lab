# Maritime Decarbonization Policy Lab

Vessel-level model of fuel and speed planning under the EU Emissions Trading System (EU ETS) and
FuelEU Maritime. For every ship of the 2024 EU monitoring, reporting and verification (MRV)
population, the model chooses speed, a fossil base fuel, one low-carbon component and its energy
share, once with a planning rule that weighs FuelEU by a planning weight λ ∈ [0, 1] and once with
the full ETS–FuelEU cost, and evaluates both choices with the full cost. It computes the avoidable
cost of the partial rule (planning regret) across planning weights, effective FuelEU prices, fuel
menus, biofuel supply, LNG engine classes, allocation rules for low-carbon fuel and model variants,
together with closed-form switching conditions and numerical checks. The code reproduces the result
files in `results/`; `results/README.md` describes each file and every column.

---

## System Requirements

### Software Dependencies

All Python package dependencies are listed in `requirements.txt`. Key libraries:

| Package | Version |
|---|---|
| Python | ≥ 3.11 |
| numpy | 2.3.5 |
| pandas | 3.0.3 |
| pyarrow | 24.0.0 |
| scipy | 1.18.1 |
| PyYAML | 6.0.3 |
| click | 8.5.0 |
| pydantic | 2.14.1 |
| geopandas | 1.2.0 |
| shapely | 2.2.0 |
| scikit-learn | 1.9.1 |
| openpyxl | 3.1.5 |

> **Note:** Python 3.11 or above is required. The result files in `results/` were produced with
> Python 3.11.5, numpy 2.3.5, pandas 3.0.3 and pyarrow 24.0.0. `run.sh` needs bash (on Windows,
> Git Bash or WSL).

### Tested Operating Systems

- macOS 26.2
- Ubuntu 22.04
- Windows 11

### Hardware

No non-standard hardware is required. The demo runs on a normal desktop computer. The full run solves about 12,300 vessels for 16 model variants; the result files were
computed on 40-core machines with up to 96 GB of memory, where the experiments took about 1,220
minutes in total, about 20 hours on one such machine. No GPU is used.

---

## Installation

```bash
pip install -r requirements.txt
```

**Typical installation time:** under a minute on a standard desktop computer (22 seconds in a new
virtual environment on the test machine), depending on network speed.

---

## Demo

> **Note on data:** the vessel particulars come from a commercial register under licence and cannot
> be redistributed; the emission reports are public but are not included. The demo uses
> **synthetically generated fake data** (`scripts/make_fake_inputs.py`) solely to verify that the
> pipeline runs correctly end to end. The results **cannot be used to evaluate the actual
> findings**.

```bash
bash run.sh fake
```

It writes a synthetic MRV export and register export for 60 vessels to `fake/data/raw/`, converts
them into the model inputs as a real run does, and runs every stage of the plan once (51 steps):
the central variant through all experiments, the other variants of the specification factorial
through `metric_harmonization`, and one three-fuel check. The result files are collected in
`fake/out/results/`, laid out as `results/`.

Expected output (excerpt):

```
--- scripts/make_fake_inputs.py fake
60 MRV records, 59 register records -> fake/data/raw
--- scripts/prepare_mrv.py fake/data/raw/mrv/thetis_mrv_2024.csv --year 2024 --out fake/data/raw/mrv/2024_renamed.csv --imo-list fake/data/raw/vessel_static/mrv_imo.txt
60 records -> fake/data/raw/mrv/2024_renamed.csv
...
22:01:03  scenario CENTRAL_EXTENDED matrix done: 14 cells, 178 curve rows, 9968 vessel rows
...
--- scripts/r1_three_fuel_check.py --menu broad --price-scenario base_2026 --out-dir fake/out/runs/three_fuel_check
544 two-fuel plans reproduced by the mixture cost
96 cases (years x scope weights x lambdas); max gain 0.00e+00 of cost (0.0000 EUR/GJ); cases where a three-fuel mixture is cheaper: 0
--- scripts/collect_results.py fake/out/runs fake/out/results
371 result files -> fake/out/results
```

**Expected run time:** about 15 minutes with six worker processes (`JOBS=6 bash run.sh fake`) on an
8-core desktop computer.

---

## Data Requirements

`bash run.sh prepare` builds the three model inputs from two exports; `bash run.sh full` then
writes the fuel-capability files itself.

### 1. `data/raw/mrv/2024_renamed.csv`

The THETIS-MRV annual emission report for reporting period 2024, one row per ship, with every
exported column; the key fields are renamed:
```
imo_number, ship_name, ship_type, year, technical_efficiency, port_of_registry,
total_fuel_tonnes, total_co2_tonnes, co2_ets_tonnes, total_ch4_tonnes, total_n2o_tonnes,
total_co2eq_tonnes, hours_at_sea, fuel_per_nm_kg, co2_per_nm_kg,
CO₂ emissions from all voyages between ports under a MS jurisdiction [m tonnes],
CO₂ emissions from all voyages which departed from ports under a MS jurisdiction [m tonnes],
CO₂ emissions from all voyages to ports under a MS jurisdiction [m tonnes],
CO₂ emissions which occurred within ports under a MS jurisdiction [m tonnes],
CO₂ emissions which occurred within ports under a MS jurisdiction at berth [m tonnes], ...
```

**Key Fields**:
* **imo_number**: IMO ship identification number, the key to the vessel particulars.
* **total_fuel_tonnes**, **total_co2_tonnes**: annual fuel consumption and CO₂ emissions (t) on
  all voyages and port stays reported under MRV.
* **co2_ets_tonnes**: CO₂ to be reported under the ETS Directive (t); a sensitivity source of the
  scope weight.
* **fuel_per_nm_kg**: fuel per distance (kg per nautical mile); annual distance = total fuel ×
  1,000 / fuel per distance, as the 2024 export has no distance field.
* **hours_at_sea**: annual time at sea (h); operating speed = distance / hours, clamped to 6–25 kn.
* **CO₂ emissions by voyage category** (t): voyages between EEA ports, departing from and arriving
  at EEA ports, within EEA ports and at berth; the scope weight is (between + port + 0.5 ×
  (departing + arriving)) / total, with port the larger of the two port columns.

### 2. `data/raw/vessel_static/fleet_ship_info.csv`
```
imo_number, name, ship_type, gross_tonnage, deadweight, design_speed_knots,
main_engine_power_kw, year_built, flag
```

**Key Fields**:
* **ship_type**: one of bulk_carrier, tanker, container_ship, general_cargo, vehicle_carrier,
  ro_pax, ro_ro_cargo, cruise_passenger, gas_carrier, lng_carrier, refrigerated_cargo.
* **gross_tonnage**, **deadweight** (t): capacity for the CII reference lines and displacement.
* **design_speed_knots** (kn): service speed; the upper speed bound.
* **main_engine_power_kw** (kW): installed main-engine power for the propulsion curve; missing
  values use a ship-type Admiralty coefficient.
* **year_built**: age term of the propulsion curve.

### 3. `data/raw/ship_info.csv`

A vessel register export with these fields:
```
lrnoimo_ship_no, ship_name, shiptype_level5, deadweight, gross_tonnage, speedservice, speedmax,
total_kilowattsof_main_engines, year_of_build, flag_name, fuel_type1first, fuel_type2second,
main_engine_model
```
`scripts/load_ship_info.py` turns it into input 2, and `scripts/r1_fuel_capability.py` derives each
vessel's LNG and methanol capability from the fuel and engine fields (LNG carriers count as
LNG-capable in the main capability file and not in the field-only file).

### Sources and licences

| Input | Source | Included |
|---|---|---|
| MRV annual emission report, 2024 | THETIS-MRV public emission reports, European Maritime Safety Agency (EMSA) | No: download it (Usage, step 1) |
| Vessel register export | S&P Global Sea-web, commercial licence | No |
| Fuel factors, prices and policy parameters | `configs/`, compiled for this model from the EU ETS Directive, the FuelEU Maritime Regulation and IMO resolutions | Yes |

---

## Data Preprocessing

The MRV report for reporting period 2024 was downloaded from THETIS-MRV on 20 June 2026; it has
14,149 rows, one per IMO number. THETIS records can be corrected by companies after publication,
so a later download may differ slightly. `scripts/prepare_mrv.py` takes the row with the field
names as the header, renames the key fields and writes the IMO numbers to a list.

The register export is a Sea-web extract current to August 2021. `scripts/load_ship_info.py` maps
the register ship types to the model's types, keeps the IMO numbers of the MRV report and fills a
missing service speed with the maximum speed or a ship-type default; 13,544 MRV ships match a
register record.

The optimization sample keeps matched ships with an annual distance of at least 500 nautical
miles: 12,333 ships. Fuel prices in United States dollars are converted at 0.9239 EUR per USD.
The allowance price is EUR80 per tonne of CO₂ unless an experiment varies it.

---

## Usage

### 1. Obtain the inputs

- Download the 2024 annual report from <https://mrv.emsa.europa.eu/#public/emission-report> as an
  Excel or CSV file, for example to `data/raw/mrv/thetis_mrv_2024.xlsx`.
- Place the register export at `data/raw/ship_info.csv` (the fields above).
- Build the model inputs:

```bash
bash run.sh prepare data/raw/mrv/thetis_mrv_2024.xlsx
```

### 2. Run

```bash
JOBS=40 bash run.sh full data out
```

`JOBS` sets the number of worker processes (default: all cores). The experiments write into
`out/runs/`, together with the per-vessel tables (`.parquet`); the last step collects the result
files into `out/results/`, laid out as `results/`.

### 3. Compare with the released results

```bash
bash run.sh check out
```

It compares every file listed in `results/MANIFEST.txt` (path, rows, first 16 hex digits of the
SHA-256) with the same file under `out/results/`.

---

## Repository Structure

```
run.sh                 entry point: fake | prepare | full | check
requirements.txt
carbon_twin/           the model: regulations (CII, EU ETS, FuelEU), emissions, optimization,
                       model variants, pipeline
scripts/               one script per experiment, input preparation, synthetic inputs
configs/               fuel factors, fuel prices, scenario, regulation versions, model variants
results/variants/      aggregate results of each model variant, one directory per variant
results/comparison/    cross-variant tables, including the specification factorial
results/checks/        closed-form switching weights, per-vessel closed-form agreement,
                       worked regulatory examples, three-fuel and speed-grid checks
results/data/          data audit, scope-weight check, baseline diagnostics
results/MANIFEST.txt   rows and hashes of every result file
results/README.md      what each result file contains, its rows and every column
```

### Scripts

In the order `run.sh full` calls them. *Inputs* are the files of Data Requirements and the
variant given by `--variant`; outputs of the per-variant experiments go to
`results/variants/<variant>/`, and per-vessel tables (`.parquet`) stay in `out/runs/`.

| Script | Computes | Inputs | Outputs |
|---|---|---|---|
| `prepare_mrv.py` | Renames the key fields of the THETIS-MRV export and lists its IMO numbers | MRV export (`.xlsx` or `.csv`) | `data/raw/mrv/2024_renamed.csv`, `mrv_imo.txt` |
| `load_ship_info.py` | Vessel particulars from the register export: model ship type, tonnage, design speed, main-engine power, build year | `ship_info.csv`, IMO list | `fleet_ship_info.csv` |
| `make_fake_inputs.py` | Synthetic MRV and register exports in the real formats (demo only) | – | `fake/data/raw/` |
| `r1_fuel_capability.py` | Each vessel's LNG and methanol capability from its fuel-type and engine fields | `ship_info.csv`, fleet file | `fuel_capability.csv`, `fuel_capability_fieldonly.csv` |
| `python -m carbon_twin.cli run-regulations-mrv` | Per-vessel compliance in the scenario year: CII proxy rating, FuelEU balance and penalty, ETS cost | MRV, fleet, scenario | `out/compliance/compliance.parquet` |
| `r1_data_audit.py` | Sample construction (matching, filters, exclusion reasons), data dictionary, scope weight by ship type, 2026 compliance landscape of four variants | MRV, fleet | `data/data_audit/` |
| `r1_scope_check.py` | Scope weight from the MRV voyage breakdown against the reported ETS share | MRV, fleet | `data/scope_check/` |
| `r1_closed_form_thresholds.py` | Per-GJ cost of every fuel plan and the closed-form switching weight by variant, fuel menu, year and scope weight | `configs/` | `checks/closed_form/` |
| `r1_worked_examples.py` | Regulatory worked examples (FuelEU penalty, ETS cost, LNG factors, scope allocation, reward factor, exact-compliance shares, annualisation), by hand and with the model code | `configs/` | `checks/worked_examples/` |
| `baseline_diagnostics.py` | Modelled against reported CO₂, MRV-implied operating speeds, matched and unmatched ships, scope emissions, FuelEU balance audit | MRV, fleet, compliance | `data/baseline_diagnostics/` |
| `run_fueleu_internalization_final.py` | λ sweep: the planning rule weighs FuelEU by λ from 0 to 1, refined around each switch, on 11-, 21- and 51-point share grids. `--stage matrix` runs one fuel menu (broad, compatibility, engine capability, no FAME); `--stage finalize` merges them into regret curves, switching intervals, thresholds and grid and menu checks | MRV, fleet, capability | `lambda/`, `lambda_fieldonly/` |
| `fueleu_internalization_sweep.py` | One λ sweep cell (called by the script above); here also the refined points λ = 0.405–0.445 | MRV, fleet, capability | `lambda_refine/<menu>/` |
| `r1_lambda_refine.py` | Merges the refined points into each vessel's switching interval | λ sweep outputs | `lambda_refine/` |
| `metric_harmonization.py` | Regret of the ETS-only rule (M1), of a rule that values fuels at well-to-wake intensity under the allowance price (M2), and with the CII term (M0), 2026–2045; also on the engine-capability menu and on ships with scope weight ≥ 0.5 | MRV, fleet, capability | `metric_harmonization*.csv` |
| `fueleu_credit_price_reopt.py` | Regret when FuelEU deficits are valued at 0, 25, 50, 75 and 100% of the statutory penalty, re-optimized | MRV, fleet | `credit_price*.csv` |
| `biofuel_sensitivity.py` | Regret without FAME, without any biofuel, with biofuel capped at 10, 30 or 50% of vessel energy, and with biofuel prices 1.5 or 2 times | MRV, fleet | `biofuel*.csv` |
| `r1_retrofit_cost_sensitivity.py` | Regret with a fixed retrofit charge (EUR per kW, annualised) for LNG and methanol on vessels without the capability | MRV, fleet, capability | `retrofit_capex*.csv` |
| `mrv_calibration_sensitivity.py` | Regret with each vessel's baseline energy anchored to its reported CO₂; modelled-to-reported ratios | MRV, fleet | `mrv_calibration*.csv` |
| `fuel_robustness.py` | Regret on the drop-in-only and compatibility menus, with and without the CII term; cost components by ship type | MRV, fleet, compliance, capability | `fuel_robustness/` |
| `optimize_fuel.py` | Joint fuel and speed plan and ETS-only plan of every vessel, 2026–2045 | MRV, fleet | `fuel_dimension_by_year*.csv` |
| `fueleu_pooling.py` | The plans of `optimize_fuel.py` with compliance balances pooled over the fleet or by ship type, or deficits valued at 25, 50 and 75% of the penalty, without re-optimization | `fuel_dimension.parquet` | `pooling*.csv` |
| `fuel_price_sensitivity.py` | Regret across fuel-price scenarios and allowance prices of EUR50–200 (`price_surface`), and across charter-rate multipliers (`charter`) | MRV, fleet | `price_surface.csv`, `charter.csv` |
| `rci_sensitivity.py` | Speed-only regret against the strength of the CII commercial-pressure term | MRV, fleet | `speed_k*.csv` |
| `cii_proxy_sensitivity.py` | CII proxy rating mix and speed regret under ship-type bias scenarios of the proxy | MRV, fleet, compliance | `cii_proxy/` |
| `speed_rci_mrv_calibration.py` | Speed-only regret with the fuel fixed (VLSFO): by policy year 2026–2045, and in 2035 with energy calibrated to reported CO₂ | MRV, fleet | `speed_years/`, `speed_calibration*.csv` |
| `r1_speed_grid_check.py` | Regret of every vessel of a variant on 15-, 29- and 57-point speed grids | MRV, fleet, `fuel_dimension.parquet` | `checks/speed_grid/` |
| `r1_slim_vessel_outputs.py` | Keeps the λ = 0 rows and each vessel's closing λ from the per-vessel λ table | per-vessel λ table | `.parquet` in `out/runs/` |
| `r1_vessel_closed_form_check.py` | Each vessel's swept switching interval against the closed form at its scope weight | slim λ tables, closed forms | `checks/vessel_closed_form/` |
| `r1_compare_variants.py` | Cross-variant tables: regret by year, λ transitions, price and supply envelope, regret distribution, retrofit, rule × menu, cost denominators, scope sample, step-by-step specification path | all variant outputs | `comparison/` |
| `r1_spec_factorial.py` | Regret for the eight combinations of allocation rule, blend base and scope-weight source, and the effect of each choice averaged over the orders of adding them (Shapley value) | `metric_harmonization.csv` of the eight variants | `comparison/spec_factorial*.csv` |
| `r1_three_fuel_check.py` | Per-GJ search over mixtures of up to three fuels against the model's two-fuel plans | `configs/` | `checks/three_fuel/` |
| `collect_results.py` | Copies the result files from `out/runs/` into the results layout | `out/runs/` | `out/results/` |

### Model package

| Module | Contents |
|---|---|
| `carbon_twin/regulations/` | CII (reference lines and rating boundaries of MEPC.353(78) and MEPC.354(78)), FuelEU Maritime (target, balance, Annex IV penalty), EU ETS (scope, phase-in), regulation versions |
| `carbon_twin/emissions/` | Admiralty-law propulsion and fuel model, fuel and emission factors, MRV calibration |
| `carbon_twin/optimization/` | Speed optimizer; fuel and speed optimizer with one low-carbon component, exact share candidates and the partial planning rule |
| `carbon_twin/pipeline.py` | Per-vessel runs from MRV records: sample filters, scope weight, representative voyage, annual values |
| `carbon_twin/model_variants.py` | Model variants defined in `configs/model_variants.yaml`; the central variant is `r1_lng_otto_ms_prio_fb_bd` |
| `carbon_twin/data/`, `parallel.py`, `cli.py` | Loaders and schemas, parallel runs over blocks of vessels, command line |

---

## License

Contains information from the EU MRV system published by the European Maritime Safety Agency
(THETIS-MRV). The code is released under the MIT License (`LICENSE`). The result files in
`results/` are not covered by it: All Rights Reserved – Review Only. They may be viewed and used
solely to review the associated manuscript and to reproduce or verify its reported results.
