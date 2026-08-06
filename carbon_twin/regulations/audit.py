"""Audit trail generation for regulation calculations."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from carbon_twin.data.schemas import UnifiedComplianceResult


@dataclass
class AuditEntry:
    """Single audit record for a regulation calculation."""

    timestamp: str = field(default_factory=lambda: dt.datetime.utcnow().isoformat(timespec="seconds"))
    regulation: str = ""
    version: str = ""
    vessel_imo: str = ""
    inputs_hash: str = ""
    params_hash: str = ""
    result_summary: dict[str, Any] = field(default_factory=dict)


class AuditLog:
    """Append-only audit log for regulation computations."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def add(self, entry: AuditEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(e) for e in self._entries], f, indent=2, default=str)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._entries:
            return pd.DataFrame(
                columns=[
                    "timestamp", "regulation", "version", "vessel_imo",
                    "inputs_hash", "params_hash", "result_summary",
                ]
            )
        return pd.DataFrame([asdict(e) for e in self._entries])

    def verify_reproducibility(
        self,
        entry: AuditEntry,
        recompute_fn: Callable[[], dict[str, Any]],
    ) -> bool:
        """Re-run the computation and compare the result hash to the recorded one."""
        recomputed = recompute_fn()
        recomputed_hash = hash_inputs(**recomputed)
        recorded_hash = hash_inputs(**entry.result_summary)
        return recomputed_hash == recorded_hash


def hash_inputs(**kwargs: Any) -> str:
    """Deterministic SHA-256 hash of computation inputs.

    All values are serialised to a canonical JSON string before hashing.
    Floats are rounded to 10 decimal places to avoid platform-dependent
    floating-point representation differences.
    """
    canonical = json.dumps(
        kwargs,
        sort_keys=True,
        default=_json_serialiser,
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate_compliance_report(
    unified_result: UnifiedComplianceResult,
    audit_log: AuditLog,
) -> dict:
    """Structured JSON-serialisable compliance report with full traceability."""
    report: dict[str, Any] = {
        "vessel_imo": unified_result.vessel_imo,
        "year": unified_result.year,
        "generated_at": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "regulation_versions": unified_result.regulation_versions,
        "summary": {
            "total_compliance_cost_eur": unified_result.total_compliance_cost_eur,
            "total_co2e_ttw_tonnes": unified_result.total_co2e_ttw_tonnes,
            "total_co2e_wtw_tonnes": unified_result.total_co2e_wtw_tonnes,
        },
        "regulations": {},
        "audit_trail": [],
    }

    if unified_result.cii is not None:
        report["regulations"]["cii"] = unified_result.cii.model_dump()
    if unified_result.fueleu is not None:
        report["regulations"]["fueleu"] = unified_result.fueleu.model_dump()
    if unified_result.ets is not None:
        report["regulations"]["ets"] = unified_result.ets.model_dump()
    if unified_result.imo_nzf is not None:
        report["regulations"]["imo_nzf"] = unified_result.imo_nzf.model_dump()

    # Attach matching audit entries for this vessel/year
    for entry in audit_log.entries:
        if entry.vessel_imo == unified_result.vessel_imo:
            report["audit_trail"].append(asdict(entry))

    report["report_hash"] = hash_inputs(**report["summary"])

    return report


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _json_serialiser(obj: Any) -> Any:
    """Fallback serialiser for json.dumps."""
    if isinstance(obj, float):
        return round(obj, 10)
    if isinstance(obj, dt.datetime):
        return obj.isoformat()
    if isinstance(obj, dt.date):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)
