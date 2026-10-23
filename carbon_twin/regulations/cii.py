from __future__ import annotations

from typing import Any

from carbon_twin.data.schemas import (
    CIIRating,
    CIIResult,
    ShipType,
    Vessel,
)
from carbon_twin.regulations.versioning import RegulationVersionManager

_REFERENCE_LINES: dict[str, list[tuple[float, float, float, str]]] = {
    "bulk_carrier":        [(0.0,      4745.0,    0.622,  "dwt")],
    "gas_carrier":         [(65000.0,  14405e7,   2.071,  "dwt"),
                            (0.0,      8104.0,    0.639,  "dwt")],
    "tanker":              [(0.0,      5247.0,    0.610,  "dwt")],
    "container_ship":      [(0.0,      1984.0,    0.489,  "dwt")],
    "general_cargo":       [(20000.0,  31948.0,   0.792,  "dwt"),
                            (0.0,      588.0,     0.3885, "dwt")],
    "refrigerated_cargo":  [(0.0,      4600.0,    0.557,  "dwt")],
    "combination_carrier": [(0.0,      5119.0,    0.622,  "dwt")],
    "lng_carrier":         [(100000.0, 9.827,     0.000,  "dwt"),
                            (65000.0,  14479e10,  2.673,  "dwt"),
                            (0.0,      14779e10,  2.673,  "dwt")],
    "ro_ro_cargo":         [(0.0,      1967.0,    0.485,  "gt")],
    "vehicle_carrier":     [(30000.0,  3627.0,    0.590,  "gt"),
                            (0.0,      330.0,     0.329,  "gt")],
    "ro_pax":              [(0.0,      2023.0,    0.460,  "gt")],
    "cruise_passenger":    [(0.0,      930.0,     0.383,  "gt")],
}

_DD_VECTORS: dict[str, list[tuple[float, tuple[float, float, float, float]]]] = {
    "bulk_carrier":        [(0.0, (0.86, 0.94, 1.06, 1.18))],
    "gas_carrier":         [(65000.0, (0.81, 0.91, 1.12, 1.44)),
                            (0.0,     (0.85, 0.95, 1.06, 1.25))],
    "tanker":              [(0.0, (0.82, 0.93, 1.08, 1.28))],
    "container_ship":      [(0.0, (0.83, 0.94, 1.07, 1.19))],
    "general_cargo":       [(0.0, (0.83, 0.94, 1.06, 1.19))],
    "refrigerated_cargo":  [(0.0, (0.78, 0.91, 1.07, 1.20))],
    "combination_carrier": [(0.0, (0.87, 0.96, 1.06, 1.14))],
    "lng_carrier":         [(100000.0, (0.89, 0.98, 1.06, 1.13)),
                            (0.0,      (0.78, 0.92, 1.10, 1.37))],
    "ro_ro_cargo":         [(0.0, (0.76, 0.89, 1.08, 1.27))],
    "vehicle_carrier":     [(0.0, (0.86, 0.94, 1.06, 1.16))],
    "ro_pax":              [(0.0, (0.76, 0.92, 1.14, 1.30))],
    "cruise_passenger":    [(0.0, (0.87, 0.95, 1.06, 1.16))],
}
_DEFAULT_DD = (0.86, 0.94, 1.06, 1.18)

_REFERENCE_COEFFICIENTS: dict[str, tuple[float, float]] = {
    st: (segs[-1][1], segs[-1][2]) for st, segs in _REFERENCE_LINES.items()
}
_GT_BASED_TYPES: set[str] = {
    st for st, segs in _REFERENCE_LINES.items() if segs[-1][3] == "gt"
}

_REF_CAP_BOUNDS: dict[str, tuple[float, float]] = {
    "bulk_carrier":    (0.0, 279000.0),
    "lng_carrier":     (65000.0, float("inf")),
    "vehicle_carrier": (0.0, 57700.0),
}


def _select_reference_segment(
    ship_type: str, gross_tonnage: float, deadweight: float | None
) -> tuple[float, float, float, float]:
    segments = _REFERENCE_LINES.get(ship_type)
    if segments is None:
        raise ValueError(
            f"No CII reference-line coefficients for ship type '{ship_type}'. "
            f"Supported: {sorted(_REFERENCE_LINES)}"
        )
    dwt = deadweight if (deadweight is not None and deadweight > 0) else 0.0
    basis_type = segments[0][3]
    measure = gross_tonnage if basis_type == "gt" else dwt
    for cap_min, a, c, basis in segments:
        if measure >= cap_min:
            break
    else:
        cap_min, a, c, basis = segments[-1]
    cap = gross_tonnage if basis == "gt" else (dwt or gross_tonnage)
    lo, hi = _REF_CAP_BOUNDS.get(ship_type, (0.0, float("inf")))
    ref_cap = min(max(cap, lo), hi)
    return a, c, cap, ref_cap


class CIIEngine:

    def __init__(
        self,
        version: str = "IMO_CII_2023_phase1",
        config_path: str | None = None,
    ) -> None:
        self._version = version
        mgr = RegulationVersionManager(
            config_path if config_path else "configs/regulation_versions.yaml"
        )
        self._params: dict[str, Any] = mgr.get_cii_params(version)


    def evaluate(
        self,
        vessel: Vessel,
        annual_co2_tonnes: float,
        distance_nm: float,
        year: int,
        capacity: float | None = None,
    ) -> CIIResult:
        ship_key = vessel.ship_type.value
        a, c, cap_auto, ref_cap = _select_reference_segment(
            ship_key, vessel.gross_tonnage, vessel.deadweight
        )
        cap = capacity if capacity is not None else cap_auto
        ref_basis = capacity if capacity is not None else ref_cap

        ref = a * (ref_basis ** (-c))
        z = self._reduction_factor(year)
        req = self._required_cii(ref, z)
        att = self._attained_cii(annual_co2_tonnes, cap, distance_nm)
        rating = self._assign_rating(att, req, ship_key, vessel.deadweight)

        ratio = att / req if req > 0 else float("inf")
        dd = self._dd_vectors(ship_key, vessel.deadweight)
        dist = self._distance_to_boundary(att, req, dd) if dd else None

        return CIIResult(
            vessel_imo=vessel.imo_number,
            year=year,
            version=self._version,
            attained_cii=round(att, 6),
            required_cii=round(req, 6),
            rating=rating,
            ratio=round(ratio, 6),
            distance_to_next_rating=round(dist, 6) if dist is not None else None,
            corrective_action_required=rating in (CIIRating.D, CIIRating.E),
        )

    def evaluate_fleet(self, fleet_data: list[dict]) -> list[CIIResult]:
        results: list[CIIResult] = []
        for entry in fleet_data:
            results.append(
                self.evaluate(
                    vessel=entry["vessel"],
                    annual_co2_tonnes=entry["annual_co2_tonnes"],
                    distance_nm=entry["distance_nm"],
                    year=entry["year"],
                    capacity=entry.get("capacity"),
                )
            )
        return results


    @staticmethod
    def capacity_for(vessel: Vessel) -> float:
        _, _, cap, _ = _select_reference_segment(
            vessel.ship_type.value, vessel.gross_tonnage, vessel.deadweight
        )
        return cap

    @staticmethod
    def _get_capacity(vessel: Vessel) -> float:
        return CIIEngine.capacity_for(vessel)

    @staticmethod
    def _cii_reference_line(ship_type: str, capacity: float) -> float:
        a, c = _REFERENCE_COEFFICIENTS[ship_type]
        return a * (capacity ** (-c))

    def _reduction_factor(self, year: int) -> float:
        factors: dict[str, float] = self._params.get("reduction_factors", {})
        if year in factors:
            return float(factors[year])
        int_keys = sorted(int(k) for k in factors)
        if not int_keys:
            return 0.0
        if year > int_keys[-1]:
            return float(factors[int_keys[-1]])
        if year < int_keys[0]:
            return float(factors[int_keys[0]])
        return 0.0

    @staticmethod
    def _required_cii(reference: float, reduction_factor: float) -> float:
        return reference * (1.0 - reduction_factor)

    @staticmethod
    def _attained_cii(co2_tonnes: float, capacity: float, distance_nm: float) -> float:
        if capacity <= 0 or distance_nm <= 0:
            raise ValueError("Capacity and distance must be positive for CII calculation")
        return (co2_tonnes * 1e6) / (capacity * distance_nm)

    def _dd_vectors(self, ship_type: str, dwt: float | None = None) -> list[float] | None:
        segs = _DD_VECTORS.get(ship_type)
        if not segs:
            return None
        d = dwt if (dwt is not None and dwt > 0) else 0.0
        for cap_min, vec in segs:
            if d >= cap_min:
                return list(vec)
        return list(segs[-1][1])

    def cd_boundary_ratio(self, ship_type: str, dwt: float | None = None) -> float:
        dd = self._dd_vectors(ship_type, dwt)
        if dd is None or len(dd) < 4:
            dd = list(_DEFAULT_DD)
        return float(dd[2])

    def _assign_rating(
        self, attained: float, required: float, ship_type: str,
        dwt: float | None = None,
    ) -> CIIRating:
        dd = self._dd_vectors(ship_type, dwt)
        if dd is None or len(dd) < 4:
            dd = list(_DEFAULT_DD)

        ratio = attained / required if required > 0 else float("inf")
        d1, d2, d3, d4 = dd[0], dd[1], dd[2], dd[3]

        if ratio <= d1:
            return CIIRating.A
        if ratio <= d2:
            return CIIRating.B
        if ratio <= d3:
            return CIIRating.C
        if ratio <= d4:
            return CIIRating.D
        return CIIRating.E

    @staticmethod
    def _distance_to_boundary(
        attained: float, required: float, dd_vectors: list[float]
    ) -> float:
        if required <= 0 or not dd_vectors:
            return 0.0
        boundaries = [d * required for d in dd_vectors]
        distances = [attained - b for b in boundaries]
        return min(distances, key=abs)
