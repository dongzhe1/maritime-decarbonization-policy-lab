from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class RegulationVersionManager:

    _REGULATION_KEYS = {
        "cii": "cii",
        "fueleu": "fueleu",
        "eu_ets": "eu_ets",
        "imo_nzf": "imo_nzf",
    }

    def __init__(self, config_path: str | Path = "configs/regulation_versions.yaml") -> None:
        self._config_path = Path(config_path)
        with open(self._config_path) as f:
            self._config: dict[str, Any] = yaml.safe_load(f)


    def get_cii_params(self, version: str) -> dict:
        return self._get_params("cii", version)

    def get_fueleu_params(self, version: str) -> dict:
        return self._get_params("fueleu", version)

    def get_ets_params(self, version: str) -> dict:
        return self._get_params("eu_ets", version)

    def get_imo_nzf_params(self, scenario: str) -> dict:
        return self._get_params("imo_nzf", scenario)

    def list_versions(self, regulation: str) -> list[str]:
        key = self._REGULATION_KEYS.get(regulation, regulation)
        section = self._config.get(key, {})
        return list(section.keys())

    def validate_version(self, regulation: str, version: str) -> bool:
        return version in self.list_versions(regulation)


    def _get_params(self, regulation: str, version: str) -> dict:
        key = self._REGULATION_KEYS.get(regulation, regulation)
        section = self._config.get(key, {})
        if version not in section:
            available = list(section.keys())
            raise KeyError(
                f"Version '{version}' not found for regulation '{regulation}'. "
                f"Available: {available}"
            )
        return dict(section[version])
