#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
J="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
export PYTHONHASHSEED=0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
EUA=80
SCEN=configs/scenarios/base_2026.yaml
YEARS=2026,2030,2035,2040,2045
CENTRAL=r1_lng_otto_ms_prio_fb_bd
LAMBDA="lambda_central lambda_compat lambda_nofame lambda_finalize"
LAMBDA2="lambda_central lambda_compat lambda_compat_engine lambda_nofame lambda_finalize"

run() { echo "--- $*"; "$PY" "$@"; }

paths() {
    MRV="$1/raw/mrv/2024_renamed.csv"
    FLEET="$1/raw/vessel_static/fleet_ship_info.csv"
    INFO="$1/raw/ship_info.csv"
    CAP="$1/raw/vessel_static/fuel_capability.csv"
    CAPF="$1/raw/vessel_static/fuel_capability_fieldonly.csv"
    R="$2/runs"
    RES="$2/results"
    COMPL="$2/compliance/compliance.parquet"
    for f in "$MRV" "$FLEET" "$INFO"; do
        [ -f "$f" ] || { echo "missing input $f (see README, Usage)" >&2; exit 1; }
    done
}

lambda_matrix() {
    local v="$1" sc="$2" out="${3:-$R/$1/lambda}" cap="${4:-$CAP}"
    run scripts/run_fueleu_internalization_final.py --stage matrix --scenario "$sc" \
        --mrv "$MRV" --fleet "$FLEET" --scenario-config "$SCEN" \
        --fuel-price-scenario base_2026 --eua "$EUA" --case FI_PRIMARY_NO_CII \
        --out "$out" --refined-step 0.005 --fallback-step 0.001 --blend-steps-list 11,21,51 \
        --workers "$J" --fuel-capability "$cap" --variant "$v"
}

exp() {
    local e="$1" v="$2" out="$R/$2"
    local common=(--mrv "$MRV" --fleet "$FLEET" --scenario "$SCEN")
    mkdir -p "$out"
    case "$e" in
      metric_harmonization)
        run scripts/metric_harmonization.py "${common[@]}" --years "$YEARS" \
            --price-scenarios base_2026 --eua "$EUA" --out "$out/metric_harmonization.csv" \
            --out-by-type "$out/metric_harmonization_by_type.csv" \
            --out-vessel "$out/metric_harmonization_vessel.parquet" --workers "$J" --variant "$v" ;;
      metric_harmonization_compat)
        run scripts/metric_harmonization.py "${common[@]}" --years "$YEARS" \
            --price-scenarios base_2026 --eua "$EUA" --fuel-feasibility engine_capability \
            --fuel-capability "$CAP" --out "$out/metric_harmonization_compat.csv" \
            --out-by-type "$out/metric_harmonization_compat_by_type.csv" \
            --out-vessel "$out/metric_harmonization_compat_vessel.parquet" --workers "$J" --variant "$v" ;;
      metric_harmonization_s05)
        run scripts/metric_harmonization.py "${common[@]}" --years "$YEARS" \
            --price-scenarios base_2026 --eua "$EUA" --min-scope-weight 0.5 \
            --out "$out/metric_harmonization_s05.csv" \
            --out-by-type "$out/metric_harmonization_s05_by_type.csv" --workers "$J" --variant "$v" ;;
      metric_harmonization_compat_fieldonly)
        run scripts/metric_harmonization.py "${common[@]}" --years "$YEARS" \
            --price-scenarios base_2026 --eua "$EUA" --fuel-feasibility engine_capability \
            --fuel-capability "$CAPF" --out "$out/metric_harmonization_compat_fieldonly.csv" \
            --out-by-type "$out/metric_harmonization_compat_fieldonly_by_type.csv" \
            --workers "$J" --variant "$v" ;;
      lambda_central) lambda_matrix "$v" CENTRAL_EXTENDED ;;
      lambda_compat) lambda_matrix "$v" COMPATIBILITY_CONSTRAINED_2035 ;;
      lambda_nofame) lambda_matrix "$v" NO_FAME_2045 ;;
      lambda_compat_engine) lambda_matrix "$v" COMPATIBILITY_ENGINE ;;
      lambda_compat_engine_fieldonly) lambda_matrix "$v" COMPATIBILITY_ENGINE "$out/lambda_fieldonly" "$CAPF" ;;
      lambda_finalize)
        run scripts/run_fueleu_internalization_final.py --stage finalize --fleet "$FLEET" \
            --scenario-config "$SCEN" --fuel-price-scenario base_2026 --eua "$EUA" \
            --case FI_PRIMARY_NO_CII --out "$out/lambda" --workers "$J" --skip-manuscript \
            --skip-packaging --skip-tests --fuel-capability "$CAP" --variant "$v" ;;
      lambda_refine)
        local sc feas
        for sc in CENTRAL_EXTENDED COMPATIBILITY_ENGINE; do
            feas=continuous_base
            [ "$sc" = COMPATIBILITY_ENGINE ] && feas=engine_capability
            run scripts/fueleu_internalization_sweep.py "${common[@]}" --years 2035,2045 \
                --lambdas 0.405,0.41,0.415,0.42,0.425,0.43,0.435,0.44,0.445 \
                --fuel-price-scenario base_2026 --eua "$EUA" --case FI_PRIMARY_NO_CII \
                --output-dir "$out/lambda_refine/$sc" --workers "$J" --blend-steps 11 \
                --fuel-feasibility "$feas" --scenario-id "$sc" --skip-m1-regression \
                --fuel-capability "$CAP" --variant "$v"
        done
        run scripts/r1_lambda_refine.py --root "$R" --variant "$v" ;;
      credit_price)
        run scripts/fueleu_credit_price_reopt.py "${common[@]}" --years 2035,2040,2045 \
            --price-scenarios base_2026 --eua "$EUA" --out "$out/credit_price.csv" \
            --out-by-type "$out/credit_price_by_type.csv" --workers "$J" --variant "$v" ;;
      biofuel)
        run scripts/biofuel_sensitivity.py "${common[@]}" --years 2035,2045 \
            --price-scenarios base_2026 --eua "$EUA" --out "$out/biofuel.csv" \
            --out-by-type "$out/biofuel_by_type.csv" --workers "$J" --variant "$v" ;;
      retrofit_capex)
        run scripts/r1_retrofit_cost_sensitivity.py "${common[@]}" --mode capex \
            --years 2035,2045 --capex-per-kw 0,250,500,1000,1500 --capex-crf 0.15 \
            --menus continuous_base,engine_capability --fuel-capability "$CAP" --eua "$EUA" \
            --out "$out/retrofit_capex.csv" --out-by-type "$out/retrofit_capex_by_type.csv" \
            --out-vessel "$out/retrofit_capex_vessel.parquet" --workers "$J" --variant "$v" ;;
      mrv_calibration)
        run scripts/mrv_calibration_sensitivity.py "${common[@]}" --years "$YEARS" \
            --price-scenarios base_2026 --eua "$EUA" --out "$out/mrv_calibration.csv" \
            --out-ratios "$out/mrv_calibration_ratios.csv" --workers "$J" --variant "$v" ;;
      fuel_robustness)
        run scripts/fuel_robustness.py "${common[@]}" --compliance "$COMPL" --years 2026,2035,2045 \
            --price-scenarios base_2026,high_fossil_2026,low_alternative_2030 --eua "$EUA" \
            --out-dir "$out/fuel_robustness" --workers "$J" --fuel-capability "$CAP" --variant "$v" ;;
      fuel_dimension)
        run scripts/optimize_fuel.py "${common[@]}" --out "$out/fuel_dimension.parquet" \
            --out-sweep "$out/fuel_dimension_by_year.csv" \
            --out-by-type "$out/fuel_dimension_by_year_type.csv" --years "$YEARS" --eua "$EUA" \
            --blend --extended --workers "$J" --variant "$v" ;;
      pooling)
        run scripts/fueleu_pooling.py --parquet "$out/fuel_dimension.parquet" \
            --scenario-label base_2026 --eua "$EUA" --out "$out/pooling.csv" \
            --out-by-type "$out/pooling_by_type.csv" ;;
      price_surface)
        run scripts/fuel_price_sensitivity.py "${common[@]}" --out "$out/price_surface.csv" \
            --years 2035,2045 --price-scenarios base_2026,high_fossil_2026,low_alternative_2030 \
            --euas 50,80,120,160,200 --charter-mults 1.0 --blend --extended --workers "$J" --variant "$v" ;;
      charter)
        run scripts/fuel_price_sensitivity.py "${common[@]}" --out "$out/charter.csv" \
            --years 2035,2045 --price-scenarios base_2026 --euas 50,80,120,160,200 \
            --charter-mults 0.5,1.0,2.0 --blend --extended --workers "$J" --variant "$v" ;;
      speed_k)
        run scripts/rci_sensitivity.py "${common[@]}" --out "$out/speed_k.csv" \
            --out-by-type "$out/speed_k_by_type.csv" --pressures 0,0.05,0.1,0.2,0.5,1.0,2.0 \
            --workers "$J" --variant "$v" ;;
      cii_proxy)
        run scripts/cii_proxy_sensitivity.py "${common[@]}" --compliance "$COMPL" \
            --k-grid 0.2,2.0 --out-dir "$out/cii_proxy" --workers "$J" --variant "$v" ;;
      speed_calibration)
        run scripts/speed_rci_mrv_calibration.py "${common[@]}" --year 2035 --eua "$EUA" \
            --fuel-price 650 --out "$out/speed_calibration.csv" --workers "$J" --variant "$v" ;;
      speed_years)
        for y in 2026 2030 2035 2040 2045; do
            run scripts/speed_rci_mrv_calibration.py "${common[@]}" --year "$y" --eua "$EUA" \
                --out "$out/speed_years/speed_years_$y.csv" --workers "$J" --variant "$v"
        done ;;
      speed_grid)
        run scripts/r1_speed_grid_check.py "${common[@]}" --variant "$v" --root "$R" \
            --years 2035,2045 --grids 15,29,57 --n-vessels 20000 --eua "$EUA" \
            --out-dir "$R/speed_grid_check" --workers "$J" ;;
      *) echo "unknown experiment $e" >&2; exit 2 ;;
    esac
}

prepare() {
    local export="$1" d="$2"
    [ -f "$d/raw/ship_info.csv" ] || { echo "missing $d/raw/ship_info.csv (see README, Usage)" >&2; exit 1; }
    run scripts/prepare_mrv.py "$export" --year 2024 --out "$d/raw/mrv/2024_renamed.csv" \
        --imo-list "$d/raw/vessel_static/mrv_imo.txt"
    run scripts/load_ship_info.py --ship-info "$d/raw/ship_info.csv" \
        --imo-file "$d/raw/vessel_static/mrv_imo.txt" --out "$d/raw/vessel_static/fleet_ship_info.csv"
}

plan() {
    local scope="$1" e v ps
    mkdir -p "$R" "$(dirname "$COMPL")"
    run scripts/r1_fuel_capability.py --ship-info "$INFO" --fleet "$FLEET" --out "$CAP"
    run scripts/r1_fuel_capability.py --ship-info "$INFO" --fleet "$FLEET" --out "$CAPF" --no-lngc-rule
    run -m carbon_twin.cli run-regulations-mrv --mrv "$MRV" --fleet "$FLEET" --scenario "$SCEN" \
        --output "$COMPL"
    run scripts/r1_data_audit.py --mrv "$MRV" --fleet "$FLEET" --scenario "$SCEN" --year 2024 \
        --variants v23,r1_lng_v23,r1_lng_otto_ms_prio_fb,r1_lng_otto_ms_prio_fb_bd --eua "$EUA" \
        --out-dir "$R/data_audit"
    run scripts/r1_scope_check.py --mrv "$MRV" --fleet "$FLEET" --out-dir "$R/scope_check"
    run scripts/r1_closed_form_thresholds.py --eua "$EUA" --variants "" \
        --blend-steps exact,11,21,51 --out-dir "$R/closed_form"
    run scripts/r1_worked_examples.py --out-dir "$R/worked_examples"
    run scripts/baseline_diagnostics.py --mrv "$MRV" --fleet "$FLEET" --compliance "$COMPL" \
        --out-dir "$R/baseline_diagnostics"
    for e in $LAMBDA2 metric_harmonization metric_harmonization_compat credit_price biofuel \
             retrofit_capex mrv_calibration fuel_robustness fuel_dimension pooling price_surface \
             charter speed_k cii_proxy speed_calibration speed_years metric_harmonization_compat_fieldonly \
             lambda_compat_engine_fieldonly lambda_refine speed_grid; do
        exp "$e" "$CENTRAL"
    done
    if [ "$scope" = full ]; then
        for v in r1_lng_v23_prio_fb_bd r1_lng_otto_ss_prio_fb_bd r1_lng_diesel_ss_prio_fb_bd \
                 r1_lng_lbsi_prio_fb_bd r1_lng_otto_ms_prop_fb_bd r1_lng_otto_ms_route_fb_bd; do
            for e in $LAMBDA2 metric_harmonization metric_harmonization_compat credit_price biofuel \
                     retrofit_capex; do
                exp "$e" "$v"
            done
        done
        for v in v23 v23_lng_otto_ms r1_lng_otto_ms; do
            for e in $LAMBDA metric_harmonization; do exp "$e" "$v"; done
        done
        for v in r1_lng_otto_ms_prio r1_lng_otto_ms_prio_fb; do
            for e in $LAMBDA2 metric_harmonization; do exp "$e" "$v"; done
        done
    else
        for v in v23 v23_lng_otto_ms r1_lng_otto_ms r1_lng_otto_ms_prio r1_lng_otto_ms_prio_fb; do
            exp metric_harmonization "$v"
        done
    fi
    exp metric_harmonization_s05 r1_lng_otto_ms_prio_fb
    for v in r1_lng_otto_ms_exact r1_lng_otto_ms_exact_bd r1_lng_otto_ms_prio_bd \
             r1_lng_otto_ms_prop_fb; do
        exp metric_harmonization "$v"
    done
    run scripts/r1_slim_vessel_outputs.py --root "$R"
    run scripts/r1_vessel_closed_form_check.py --root "$R"
    run scripts/r1_compare_variants.py --root "$R" --fuel-capability "$CAP"
    run scripts/r1_spec_factorial.py --root "$R"
    if [ "$scope" = full ]; then
        for e in broad no_fame; do
            for ps in base_2026 high_fossil_2026 low_alternative_2030; do
                run scripts/r1_three_fuel_check.py --menu "$e" --price-scenario "$ps" \
                    --out-dir "$R/three_fuel_check"
            done
        done
    else
        run scripts/r1_three_fuel_check.py --menu broad --price-scenario base_2026 \
            --out-dir "$R/three_fuel_check"
    fi
    run scripts/collect_results.py "$R" "$RES"
}

sha() { if command -v sha256sum > /dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi; }

check() {
    local out="${1:-out}" ok=0 bad=0 miss=0 rel n h got
    while read -r rel n h; do
        [ "$rel" = file ] && continue
        [ -z "$h" ] && { h="$n"; }
        if [ ! -f "$out/results/$rel" ]; then miss=$((miss + 1)); continue; fi
        got="$(sha "$out/results/$rel" | cut -c1-16)"
        if [ "$got" = "$h" ]; then ok=$((ok + 1)); else bad=$((bad + 1)); echo "differs: $rel"; fi
    done < results/MANIFEST.txt
    echo "$ok identical, $bad different, $miss not produced"
}

restore() {
    [ -f LARGE_FILES.sha256 ] || { echo "nothing to restore"; return; }
    local h rel
    while read -r h rel; do
        if ls "$rel".gz.part* > /dev/null 2>&1; then cat "$rel".gz.part* > "$rel.gz"; fi
        [ -f "$rel.gz" ] && gunzip -f "$rel.gz"
    done < LARGE_FILES.sha256
    if command -v sha256sum > /dev/null; then sha256sum -c LARGE_FILES.sha256
    else shasum -a 256 -c LARGE_FILES.sha256; fi
}

case "${1:-}" in
  fake)
    d="${2:-fake}"
    [ -d "$d/data" ] || run scripts/make_fake_inputs.py "$d"
    prepare "$d/data/raw/mrv/thetis_mrv_2024.csv" "$d/data"
    paths "$d/data" "$d/out"
    plan fake ;;
  prepare)
    [ -n "${2:-}" ] || { echo "usage: bash run.sh prepare <MRV export> [data_dir]" >&2; exit 2; }
    prepare "$2" "${3:-data}" ;;
  full)
    paths "${2:-data}" "${3:-out}"
    plan full ;;
  check) check "${2:-out}" ;;
  restore) restore ;;
  *) echo "usage: bash run.sh fake [dir] | prepare <MRV export> [data_dir] | full [data_dir] [out_dir] | check [out_dir] | restore" >&2
     exit 2 ;;
esac
