# Maritime Decarbonization Policy Lab

Vessel-level model of fuel and speed planning under the EU Emissions Trading System (EU ETS) and
FuelEU Maritime. For every ship of the 2024 EU monitoring, reporting and verification (MRV)
population, the model chooses speed, a fossil base fuel, one low-carbon component and its energy
share, once with a planning rule that weighs FuelEU by a planning weight λ ∈ [0, 1] and once with
the full ETS–FuelEU cost, and evaluates both choices with the full cost. It computes the avoidable
cost of the partial rule (planning regret) across planning weights, effective FuelEU prices, fuel
menus, biofuel supply, LNG engine classes, allocation rules for low-carbon fuel and model variants,
together with closed-form switching conditions and numerical checks. The code reproduces the result
files in `results/`.

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
```

---

## License

Contains information from the EU MRV system published by the European Maritime Safety Agency
(THETIS-MRV). The code is released under the MIT License (`LICENSE`).
