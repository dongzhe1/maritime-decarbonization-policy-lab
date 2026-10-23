from __future__ import annotations

import click


@click.group()
@click.version_option()
def main():
    pass


@main.command()
@click.option("--ais", required=True, type=click.Path(exists=True), help="AIS parquet file")
@click.option("--ports", required=True, type=click.Path(exists=True), help="Ports GeoJSON file")
@click.option("--output", required=True, type=click.Path(), help="Output voyages parquet")
def build_voyages(ais: str, ports: str, output: str):
    import pandas as pd
    import geopandas as gpd
    from carbon_twin.data.loaders import load_ais_parquet, load_ports_geojson
    from carbon_twin.voyage.ais_cleaning import clean_ais
    from carbon_twin.voyage.segmentation import segment_voyages

    click.echo(f"Loading AIS data from {ais}")
    ais_df = load_ais_parquet(ais)
    click.echo(f"  {len(ais_df):,} AIS points loaded")

    click.echo(f"Loading ports from {ports}")
    ports_gdf = load_ports_geojson(ports)
    click.echo(f"  {len(ports_gdf):,} ports loaded")

    click.echo("Cleaning AIS data...")
    ais_df = clean_ais(ais_df)
    click.echo(f"  {len(ais_df):,} points after cleaning")

    click.echo("Segmenting voyages...")
    voyages = segment_voyages(ais_df, ports_gdf)
    click.echo(f"  {len(voyages)} voyages extracted")

    voyages_df = pd.DataFrame([v.model_dump() for v in voyages])
    voyages_df.to_parquet(output, index=False)
    click.echo(f"Saved to {output}")


@main.command()
@click.option("--voyages", required=True, type=click.Path(exists=True))
@click.option("--mrv", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
@click.option("--method", default="hierarchical", type=click.Choice(["simple_ratio", "ship_type_ratio", "hierarchical", "gradient_boosting"]))
def calibrate_mrv(voyages: str, mrv: str, output: str, method: str):
    import pandas as pd
    import pickle
    from carbon_twin.emissions.mrv_calibration import MRVCalibrator

    click.echo("Loading data...")
    voyages_df = pd.read_parquet(voyages)
    mrv_df = pd.read_csv(mrv)

    click.echo(f"Calibrating with method={method}...")
    calibrator = MRVCalibrator(method=method)
    calibrator.fit(voyages_df, mrv_df)

    metrics = calibrator.evaluate(voyages_df, mrv_df)
    click.echo(f"  MAE: {metrics.get('mae', 'N/A'):.1f} tCO2")
    click.echo(f"  MAPE: {metrics.get('mape', 'N/A'):.1f}%")

    with open(output, "wb") as f:
        pickle.dump(calibrator, f)
    click.echo(f"Saved calibrator to {output}")


def _load_scenario(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)["scenario"]


def _resolve_eua_price(scenario_cfg: dict, year: int) -> float:
    from carbon_twin.data.loaders import load_eua_prices
    raw = scenario_cfg.get("eua_price", "base")
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        prices = load_eua_prices("configs/eua_prices.yaml", scenario=str(raw))
        return float(prices.get(year, next(iter(prices.values()))))
    except Exception:
        return 75.0


def _resolve_fuel_price(scenario_cfg: dict, fuel: str = "VLSFO") -> float:
    from carbon_twin.data.loaders import load_fuel_prices
    raw = scenario_cfg.get("fuel_prices", "base_2026")
    try:
        prices = load_fuel_prices("configs/fuel_prices.yaml", scenario=str(raw))
        return float(prices.get(fuel, next(iter(prices.values()))))
    except Exception:
        return 600.0


@main.command(name="run-regulations-mrv")
@click.option("--mrv", required=True, type=click.Path(exists=True), help="Renamed MRV CSV")
@click.option("--fleet", required=True, type=click.Path(exists=True))
@click.option("--scenario", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def run_regulations_mrv(mrv: str, fleet: str, scenario: str, output: str):
    from carbon_twin.data.loaders import load_mrv_csv, load_vessels_csv
    from carbon_twin.emissions.proxy import vessels_to_index
    from carbon_twin.pipeline import assemble_compliance_from_mrv

    scenario_cfg = _load_scenario(scenario)
    year = int(scenario_cfg["year"])
    eua_price = _resolve_eua_price(scenario_cfg, year)
    click.echo(f"MRV-direct compliance | year {year} | EUA {eua_price:.1f} EUR/t")

    mrv_df = load_mrv_csv(mrv)
    fleet_idx = vessels_to_index(load_vessels_csv(fleet))
    regs = scenario_cfg["regulations"]

    result = assemble_compliance_from_mrv(
        mrv_df, fleet_idx, year, eua_price,
        cii_version=regs["cii"]["version"],
        fueleu_version=regs["fueleu"]["version"],
        ets_version=regs["eu_ets"]["version"],
    )
    result.to_parquet(output, index=False)
    n_de = int(result["cii_rating"].isin(["D", "E"]).sum()) if "cii_rating" in result else 0
    click.echo(f"  {len(result):,} vessels | {n_de} rated D/E | "
               f"total cost {result['total_compliance_cost_eur'].sum():,.0f} EUR")
    click.echo(f"Saved to {output}")


@main.command(name="optimize-mrv")
@click.option("--case", required=True, type=click.Path(exists=True), help="run-regulations-mrv output")
@click.option("--mrv", required=True, type=click.Path(exists=True))
@click.option("--fleet", required=True, type=click.Path(exists=True))
@click.option("--scenario", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def optimize_mrv(case: str, mrv: str, fleet: str, scenario: str, output: str):
    import pandas as pd
    from carbon_twin.data.loaders import load_mrv_csv, load_vessels_csv
    from carbon_twin.emissions.proxy import load_fuel_factors_safe, vessels_to_index
    from carbon_twin.pipeline import run_optimization_from_mrv

    scenario_cfg = _load_scenario(scenario)
    year = int(scenario_cfg["year"])
    eua_price = _resolve_eua_price(scenario_cfg, year)
    fuel_price = _resolve_fuel_price(scenario_cfg)
    click.echo(f"MRV-direct optimize | EUA {eua_price:.1f} | fuel {fuel_price:.0f}")

    compliance_df = pd.read_parquet(case)
    mrv_df = load_mrv_csv(mrv)
    fleet_idx = vessels_to_index(load_vessels_csv(fleet))
    factors = load_fuel_factors_safe()
    regs = scenario_cfg["regulations"]

    result = run_optimization_from_mrv(
        compliance_df, mrv_df, fleet_idx, year, eua_price, fuel_price,
        cii_version=regs["cii"]["version"],
        fueleu_version=regs["fueleu"]["version"],
        ets_version=regs["eu_ets"]["version"],
        fuel_factors=factors,
    )
    result.to_parquet(output, index=False)
    if len(result):
        click.echo(f"  {len(result):,} vessels | mean RCI {result['rci'].mean():.3f} "
                   f"| max RCI {result['rci'].max():.3f}")
    click.echo(f"Saved to {output}")


@main.command(name="build-proxy")
@click.option("--voyages", required=True, type=click.Path(exists=True))
@click.option("--fleet", required=True, type=click.Path(exists=True), help="Vessel particulars CSV")
@click.option("--year", required=True, type=int)
@click.option("--output", required=True, type=click.Path())
def build_proxy(voyages: str, fleet: str, year: int, output: str):
    import pandas as pd
    from carbon_twin.data.loaders import load_vessels_csv
    from carbon_twin.emissions.proxy import (
        build_proxy_emissions,
        load_fuel_factors_safe,
        vessels_to_index,
    )

    click.echo(f"Loading voyages from {voyages}")
    voyages_df = pd.read_parquet(voyages)
    click.echo(f"  {len(voyages_df):,} voyages")

    click.echo(f"Loading fleet from {fleet}")
    fleet_idx = vessels_to_index(load_vessels_csv(fleet))
    click.echo(f"  {len(fleet_idx):,} vessels")

    factors = load_fuel_factors_safe()
    click.echo("Building physics proxy emissions...")
    proxy = build_proxy_emissions(voyages_df, fleet_idx, year, fuel_factors=factors)
    click.echo(f"  {len(proxy):,} voyage-emissions "
               f"({proxy.attrs.get('matched_vessels', 0):,} vessels, "
               f"{proxy.attrs.get('skipped', 0):,} skipped)")

    proxy.to_parquet(output, index=False)
    click.echo(f"Saved proxy emissions to {output}")


@main.command()
@click.option("--voyages", required=True, type=click.Path(exists=True))
@click.option("--proxy", required=True, type=click.Path(exists=True), help="build-proxy output")
@click.option("--fleet", required=True, type=click.Path(exists=True))
@click.option("--calibrator", type=click.Path(exists=True), help="MRV calibrator pkl (optional)")
@click.option("--scenario", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def run_regulations(voyages: str, proxy: str, fleet: str, calibrator: str,
                    scenario: str, output: str):
    import pickle
    import pandas as pd
    from carbon_twin.data.loaders import load_vessels_csv
    from carbon_twin.emissions.proxy import vessels_to_index
    from carbon_twin.pipeline import assemble_compliance

    scenario_cfg = _load_scenario(scenario)
    year = int(scenario_cfg["year"])
    eua_price = _resolve_eua_price(scenario_cfg, year)
    click.echo(f"Year {year} | EUA {eua_price:.1f} EUR/t")

    voyages_df = pd.read_parquet(voyages)
    proxy_df = pd.read_parquet(proxy)
    fleet_idx = vessels_to_index(load_vessels_csv(fleet))

    calib = None
    if calibrator:
        with open(calibrator, "rb") as f:
            calib = pickle.load(f)
        click.echo(f"Loaded calibrator ({calib.method})")

    regs = scenario_cfg["regulations"]
    click.echo("Assembling compliance state vectors...")
    result = assemble_compliance(
        voyages_df, proxy_df, calib, fleet_idx, year, eua_price,
        cii_version=regs["cii"]["version"],
        fueleu_version=regs["fueleu"]["version"],
        ets_version=regs["eu_ets"]["version"],
    )

    result.to_parquet(output, index=False)
    n_de = int((result["cii_rating"].isin(["D", "E"])).sum()) if "cii_rating" in result else 0
    click.echo(f"  {len(result):,} vessels assessed | {n_de} rated D/E")
    click.echo(f"  total compliance cost: "
               f"{result['total_compliance_cost_eur'].sum():,.0f} EUR")
    click.echo(f"Results saved to {output}")


@main.command()
@click.option("--case", required=True, type=click.Path(exists=True), help="run-regulations output")
@click.option("--voyages", required=True, type=click.Path(exists=True))
@click.option("--fleet", required=True, type=click.Path(exists=True))
@click.option("--scenario", required=True, type=click.Path(exists=True))
@click.option("--objective", default="total_cost",
              type=click.Choice(["total_cost", "min_emissions", "min_cost_with_cvar"]))
@click.option("--output", required=True, type=click.Path())
def optimize(case: str, voyages: str, fleet: str, scenario: str,
             objective: str, output: str):
    import pandas as pd
    from carbon_twin.data.loaders import load_vessels_csv
    from carbon_twin.emissions.proxy import load_fuel_factors_safe, vessels_to_index
    from carbon_twin.pipeline import run_optimization

    scenario_cfg = _load_scenario(scenario)
    year = int(scenario_cfg["year"])
    eua_price = _resolve_eua_price(scenario_cfg, year)
    fuel_price = _resolve_fuel_price(scenario_cfg)
    click.echo(f"Optimizing (objective={objective}) | EUA {eua_price:.1f} | fuel {fuel_price:.0f}")

    compliance_df = pd.read_parquet(case)
    voyages_df = pd.read_parquet(voyages)
    fleet_idx = vessels_to_index(load_vessels_csv(fleet))
    factors = load_fuel_factors_safe()
    regs = scenario_cfg["regulations"]

    result = run_optimization(
        compliance_df, voyages_df, fleet_idx, year, eua_price, fuel_price,
        cii_version=regs["cii"]["version"],
        fueleu_version=regs["fueleu"]["version"],
        ets_version=regs["eu_ets"]["version"],
        fuel_factors=factors,
    )

    result.to_parquet(output, index=False)
    if len(result):
        click.echo(f"  {len(result):,} vessels | mean RCI {result['rci'].mean():.3f} "
                   f"| max RCI {result['rci'].max():.3f}")
    click.echo(f"Results saved to {output}")


@main.command()
@click.option("--case", required=True, type=click.Path(exists=True), help="run-regulations output")
@click.option("--n", default=1000, type=int, help="Number of Monte Carlo samples")
@click.option("--scenario", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def run_scenarios(case: str, n: int, scenario: str, output: str):
    import json
    from pathlib import Path
    import pandas as pd
    from carbon_twin.pipeline import run_monte_carlo

    scenario_cfg = _load_scenario(scenario)
    year = int(scenario_cfg["year"])
    eua_price = _resolve_eua_price(scenario_cfg, year)

    compliance_df = pd.read_parquet(case)
    click.echo(f"Running {n:,} Monte Carlo samples...")
    raw, summary = run_monte_carlo(compliance_df, scenario_cfg, n, eua_price)

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(out_dir / "mc_samples.parquet", index=False)
    with open(out_dir / "mc_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    click.echo(f"  mean total cost: {summary.get('mean', 'N/A')}")
    click.echo(f"  CVaR(95%):       {summary.get('cvar_95', 'N/A')}")
    click.echo(f"Results saved to {out_dir}")


@main.command()
@click.option("--results", required=True, type=click.Path(exists=True),
              help="Directory containing compliance/optimization/MC outputs")
@click.option("--format", "fmt", default="markdown",
              type=click.Choice(["markdown", "html", "json", "latex"]))
@click.option("--output", required=True, type=click.Path())
def report(results: str, fmt: str, output: str):
    from pathlib import Path
    import pandas as pd
    from carbon_twin.reports.paper_outputs import generate_all_paper_outputs

    results_dir = Path(results)
    payload: dict = {}

    def _first_existing(*candidates: str):
        for c in candidates:
            p = results_dir / c
            if p.exists():
                return p
        return None

    compliance_path = _first_existing("compliance.parquet", "compliance/compliance.parquet")
    if compliance_path:
        comp = pd.read_parquet(compliance_path)
        if "cii_rating" in comp.columns:
            comp["rating"] = comp["cii_rating"]
        if {"cii_attained", "cii_required"}.issubset(comp.columns):
            comp["cii_ratio"] = comp["cii_attained"] / comp["cii_required"].replace(0, pd.NA)
        payload["fleet_results"] = comp
        payload["cii_results"] = comp

    opt_path = _first_existing("optimized.parquet", "optimization/optimized.parquet")
    if opt_path:
        payload["optimization_results"] = pd.read_parquet(opt_path)

    mc_path = _first_existing("mc/mc_samples.parquet", "monte_carlo/mc_samples.parquet")
    if mc_path:
        mc = pd.read_parquet(mc_path)
        if "total_compliance_cost_eur" in mc.columns and "total_cost_eur" not in mc.columns:
            mc["total_cost_eur"] = mc["total_compliance_cost_eur"]
        payload["mc_results"] = mc

    click.echo(f"Generating {fmt} outputs from {len(payload)} result set(s)...")
    try:
        out = generate_all_paper_outputs(payload, output_dir=output)
        click.echo(f"  {len(out.get('figures', []))} figures, {len(out.get('tables', []))} tables")
    except Exception as exc:
        click.echo(f"  [WARN] report generation hit an error: {exc}")
    click.echo(f"Report saved to {output}")


@main.command()
def dashboard():
    import subprocess
    import sys

    app_path = str(Path(__file__).parent / "dashboard" / "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])


from pathlib import Path

if __name__ == "__main__":
    main()
