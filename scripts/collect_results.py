from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

GROUPS = {
    "comparison": "comparison",
    "closed_form": "checks/closed_form",
    "vessel_closed_form": "checks/vessel_closed_form",
    "worked_examples": "checks/worked_examples",
    "three_fuel_check": "checks/three_fuel",
    "speed_grid_check": "checks/speed_grid",
    "data_audit": "data/data_audit",
    "scope_check": "data/scope_check",
    "baseline_diagnostics": "data/baseline_diagnostics",
}
SKIP_NAME = re.compile(r"(_manifest\.json$|^run_manifest|^validation_report.*\.txt$|"
                       r"^validation_results\.json$|^pytest_.*\.txt$|^run_log)")
SKIP_FILE = {"v23_regression.csv", "v23_data_update.csv", "run_status_not_ok.csv",
             "vessel_closed_form_mismatches.csv"}
ID_COLUMNS = {"imo", "imo_number", "vessel_imo", "vessel_id", "mmsi", "ship_name", "shipname"}
SUFFIXES = (".csv", ".json", ".txt")


def target(rel: str, variants: set[str]) -> str | None:
    top, _, rest = rel.partition("/")
    if not rest:
        return None
    if top in variants:
        return f"variants/{rel}"
    if top in GROUPS:
        return f"{GROUPS[top]}/{rest}"
    return None


def keep(path: Path) -> bool:
    if not path.name.endswith(SUFFIXES) or SKIP_NAME.search(path.name) or path.name in SKIP_FILE:
        return False
    if path.suffix == ".csv":
        with open(path, encoding="utf-8", errors="ignore") as fh:
            head = next(csv.reader(fh), [])
        if any(c.strip().lower() in ID_COLUMNS for c in head):
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("results_dir", type=Path)
    args = ap.parse_args()
    from carbon_twin.model_variants import available_variants
    variants = set(available_variants())
    n = 0
    for p in sorted(args.run_dir.rglob("*")):
        if not p.is_file() or not keep(p):
            continue
        dest = target(p.relative_to(args.run_dir).as_posix(), variants)
        if dest is None:
            continue
        t = args.results_dir / dest
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, t)
        n += 1
    print(f"{n} result files -> {args.results_dir}")


if __name__ == "__main__":
    main()
