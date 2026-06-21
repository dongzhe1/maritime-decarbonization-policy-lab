"""FuelEU Maritime pooling optimization."""

from __future__ import annotations

from typing import Any

from carbon_twin.data.schemas import FuelEUResult, FuelType, Vessel
from carbon_twin.emissions.factors import FuelEmissionFactors


class PoolingOptimizer:
    """Optimizes FuelEU Maritime compliance through vessel pooling.

    Pairs surplus vessels with deficit vessels to minimize total fleet penalty.
    """

    def __init__(
        self,
        vessels: list[Vessel],
        fueleu_results: list[FuelEUResult],
    ) -> None:
        self._vessels = {v.imo_number: v for v in vessels}
        self._results = {r.vessel_imo: r for r in fueleu_results}
        self._vessel_imos = [v.imo_number for v in vessels]

    def optimize_pools(
        self,
        max_pool_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Find optimal grouping to minimize total fleet penalty.

        Uses greedy pairing: match highest-surplus with highest-deficit.
        """
        surplus_vessels: list[tuple[str, float]] = []
        deficit_vessels: list[tuple[str, float]] = []

        for imo in self._vessel_imos:
            if imo not in self._results:
                continue
            result = self._results[imo]
            if result.compliance_balance >= 0:
                surplus_vessels.append((imo, result.compliance_balance))
            else:
                deficit_vessels.append((imo, result.compliance_balance))

        surplus_vessels.sort(key=lambda x: x[1], reverse=True)
        deficit_vessels.sort(key=lambda x: x[1])

        pools: list[dict[str, Any]] = []
        used_surplus: set[str] = set()
        used_deficit: set[str] = set()

        for deficit_imo, deficit_balance in deficit_vessels:
            pool_members = [deficit_imo]
            remaining_deficit = deficit_balance

            for surplus_imo, surplus_balance in surplus_vessels:
                if surplus_imo in used_surplus:
                    continue
                if max_pool_size and len(pool_members) >= max_pool_size:
                    break

                pool_members.append(surplus_imo)
                remaining_deficit += surplus_balance
                used_surplus.add(surplus_imo)

                if remaining_deficit >= 0:
                    break

            used_deficit.add(deficit_imo)

            pool_balance = sum(
                self._results[m].compliance_balance for m in pool_members if m in self._results
            )

            penalty_without_pool = abs(deficit_balance) * self._penalty_rate()
            penalty_with_pool = (
                abs(pool_balance) * self._penalty_rate() if pool_balance < 0 else 0.0
            )

            pools.append(
                {
                    "pool_id": f"pool_{len(pools) + 1}",
                    "members": pool_members,
                    "pool_balance": pool_balance,
                    "surplus": pool_balance >= 0,
                    "penalty_without_pool": penalty_without_pool,
                    "penalty_with_pool": penalty_with_pool,
                    "savings": penalty_without_pool - penalty_with_pool,
                }
            )

        unpooled = [
            imo
            for imo in self._vessel_imos
            if imo not in used_surplus and imo not in used_deficit and imo in self._results
        ]
        for imo in unpooled:
            result = self._results[imo]
            if result.compliance_balance >= 0:
                pools.append(
                    {
                        "pool_id": f"pool_{len(pools) + 1}",
                        "members": [imo],
                        "pool_balance": result.compliance_balance,
                        "surplus": True,
                        "penalty_without_pool": 0.0,
                        "penalty_with_pool": 0.0,
                        "savings": 0.0,
                    }
                )

        return pools

    def pool_shadow_price(self) -> float:
        """Marginal value of one unit of compliance surplus in the pool.

        Approximated as the penalty rate times the fraction of fleet in deficit.
        """
        total_deficit = 0.0
        n_deficit = 0
        for imo in self._vessel_imos:
            if imo not in self._results:
                continue
            if self._results[imo].compliance_balance < 0:
                total_deficit += abs(self._results[imo].compliance_balance)
                n_deficit += 1

        n_total = sum(1 for imo in self._vessel_imos if imo in self._results)
        if n_total == 0:
            return 0.0

        deficit_fraction = n_deficit / n_total
        return self._penalty_rate() * deficit_fraction

    def evaluate_pool(self, pool_members: list[str]) -> dict[str, Any]:
        """Given vessel IMOs, compute pooled compliance balance."""
        total_balance = 0.0
        member_details: list[dict[str, Any]] = []

        for imo in pool_members:
            if imo not in self._results:
                continue
            result = self._results[imo]
            total_balance += result.compliance_balance
            member_details.append(
                {
                    "vessel_imo": imo,
                    "compliance_balance": result.compliance_balance,
                    "ghg_intensity": result.actual_ghg_intensity,
                    "target_ghg_intensity": result.target_ghg_intensity,
                }
            )

        penalty = abs(total_balance) * self._penalty_rate() if total_balance < 0 else 0.0
        unpooled_penalty = sum(
            abs(self._results[imo].compliance_balance) * self._penalty_rate()
            for imo in pool_members
            if imo in self._results and self._results[imo].compliance_balance < 0
        )

        return {
            "pool_members": pool_members,
            "pool_balance": total_balance,
            "surplus": total_balance >= 0,
            "pooled_penalty": penalty,
            "unpooled_penalty": unpooled_penalty,
            "savings": unpooled_penalty - penalty,
            "member_details": member_details,
        }

    def first_mover_value(
        self,
        vessel_imo: str,
        fuel_switch: FuelType,
        fuel_factors: dict[FuelType, FuelEmissionFactors] | None = None,
    ) -> float:
        """Value of one vessel switching to a low-carbon fuel for the pool.

        Estimates how much total fleet penalty is reduced if the given vessel
        switches to a fuel with lower GHG intensity.
        """
        if vessel_imo not in self._results:
            return 0.0

        current_result = self._results[vessel_imo]

        total_fleet_penalty_before = sum(
            abs(r.compliance_balance) * self._penalty_rate()
            for r in self._results.values()
            if r.compliance_balance < 0
        )

        if fuel_factors and fuel_switch in fuel_factors:
            new_ghg = fuel_factors[fuel_switch].wtw_co2e_g_per_mj
        else:
            _low_carbon_ghg: dict[FuelType, float] = {
                FuelType.LNG: 80.0,
                FuelType.METHANOL_BIO: 14.0,
                FuelType.METHANOL_E: 5.0,
                FuelType.AMMONIA_GREEN: 8.0,
                FuelType.HYDROGEN_GREEN: 4.0,
                FuelType.BIODIESEL_FAME: 30.0,
            }
            new_ghg = _low_carbon_ghg.get(fuel_switch, 91.16)

        improvement = current_result.actual_ghg_intensity - new_ghg
        new_balance = current_result.compliance_balance + improvement

        modified_results = dict(self._results)
        modified_result = FuelEUResult(
            vessel_imo=vessel_imo,
            year=current_result.year,
            version=current_result.version,
            actual_ghg_intensity=new_ghg,
            target_ghg_intensity=current_result.target_ghg_intensity,
            compliance_balance=new_balance,
            surplus=new_balance >= 0,
            penalty_eur=abs(new_balance) * self._penalty_rate() if new_balance < 0 else 0.0,
        )
        modified_results[vessel_imo] = modified_result

        total_fleet_penalty_after = sum(
            abs(r.compliance_balance) * self._penalty_rate()
            for r in modified_results.values()
            if r.compliance_balance < 0
        )

        return max(total_fleet_penalty_before - total_fleet_penalty_after, 0.0)

    def _penalty_rate(self) -> float:
        """FuelEU penalty rate in EUR per gCO2e/MJ deficit."""
        return 2400.0
