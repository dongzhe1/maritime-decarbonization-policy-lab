from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from carbon_twin.data.schemas import ShipType, Vessel, Voyage


class EmissionSurrogate:

    _MODEL_TYPES = {"xgboost", "linear", "mlp"}

    def __init__(self, model_type: str = "xgboost") -> None:
        if model_type not in self._MODEL_TYPES:
            raise ValueError(f"model_type must be one of {self._MODEL_TYPES}")
        self.model_type = model_type
        self._models: dict[str, Any] = {}
        self._scaler: StandardScaler | None = None
        self._feature_names: list[str] = []
        self._fitted: bool = False


    @staticmethod
    def prepare_features(
        vessels: list[Vessel],
        voyages: list[Voyage],
        weather_factors: list[float] | None = None,
    ) -> pd.DataFrame:
        vessel_map = {v.imo_number: v for v in vessels}
        rows: list[dict[str, Any]] = []

        for i, voy in enumerate(voyages):
            v = vessel_map.get(voy.vessel_imo)
            if v is None:
                continue

            speed = voy.avg_speed_knots
            if (speed is None or speed <= 0) and voy.sea_hours and voy.sea_hours > 0:
                speed = voy.distance_nm / voy.sea_hours
            if speed is None or speed <= 0:
                speed = v.design_speed_knots or 14.0

            sea_hours = voy.sea_hours
            if sea_hours is None or sea_hours <= 0:
                sea_hours = voy.distance_nm / max(speed, 0.1)

            wf = 1.0
            if weather_factors and i < len(weather_factors):
                wf = weather_factors[i]

            ship_type_code = list(ShipType).index(v.ship_type)
            age = 2026 - v.year_built if v.year_built else 15

            rows.append({
                "vessel_imo": voy.vessel_imo,
                "voyage_id": voy.voyage_id,
                "ship_type_code": ship_type_code,
                "gross_tonnage": v.gross_tonnage,
                "deadweight": v.deadweight or 0.0,
                "main_engine_power_kw": v.main_engine_power_kw or 0.0,
                "design_speed_knots": v.design_speed_knots or 0.0,
                "age": age,
                "distance_nm": voy.distance_nm,
                "avg_speed_knots": speed,
                "sea_hours": sea_hours,
                "weather_factor": wf,
                "cargo_tonnes": voy.cargo_tonnes or 0.0,
                "draught_m": voy.draught_m or v.draught_m or 0.0,
                "length_m": v.length_m or 0.0,
                "beam_m": v.beam_m or 0.0,
                "speed_cubed": speed**3,
                "distance_x_gt": voy.distance_nm * v.gross_tonnage,
            })

        return pd.DataFrame(rows)


    _FEATURE_COLS: list[str] = [
        "ship_type_code",
        "gross_tonnage",
        "deadweight",
        "main_engine_power_kw",
        "design_speed_knots",
        "age",
        "distance_nm",
        "avg_speed_knots",
        "sea_hours",
        "weather_factor",
        "cargo_tonnes",
        "draught_m",
        "length_m",
        "beam_m",
        "speed_cubed",
        "distance_x_gt",
    ]

    def fit(
        self,
        X: pd.DataFrame,
        y_fuel: np.ndarray,
        y_co2e_ttw: np.ndarray,
        y_co2e_wtw: np.ndarray,
    ) -> EmissionSurrogate:
        feature_cols = [c for c in self._FEATURE_COLS if c in X.columns]
        self._feature_names = feature_cols
        X_arr = X[feature_cols].fillna(0).values.astype(np.float64)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_arr)

        targets = {
            "fuel_tonnes": np.asarray(y_fuel, dtype=np.float64),
            "co2e_ttw": np.asarray(y_co2e_ttw, dtype=np.float64),
            "co2e_wtw": np.asarray(y_co2e_wtw, dtype=np.float64),
        }

        for name, y in targets.items():
            model = self._make_model()
            if self.model_type == "linear":
                model.fit(X_scaled, y)
            else:
                model.fit(X_arr, y)
            self._models[name] = model

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        self._check_fitted()
        X_arr = self._prepare_X(X)
        results: dict[str, np.ndarray] = {}
        for name, model in self._models.items():
            if self.model_type == "linear":
                assert self._scaler is not None
                preds = model.predict(self._scaler.transform(X_arr))
            else:
                preds = model.predict(X_arr)
            results[name] = np.maximum(preds, 0.0)
        return results

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        n_bootstrap: int = 100,
    ) -> dict[str, dict[str, np.ndarray]]:
        self._check_fitted()
        X_arr = self._prepare_X(X)
        n_samples = X_arr.shape[0]
        results: dict[str, dict[str, np.ndarray]] = {}

        for name, model in self._models.items():
            preds_matrix = np.zeros((n_bootstrap, n_samples))
            rng = np.random.default_rng(42)

            for b in range(n_bootstrap):
                indices = rng.choice(n_samples, size=n_samples, replace=True)
                X_boot = X_arr[indices]
                if self.model_type == "linear":
                    assert self._scaler is not None
                    preds_matrix[b] = model.predict(self._scaler.transform(X_boot[: n_samples]))
                else:
                    preds_matrix[b] = model.predict(X_boot[:n_samples])

            base_pred = (
                model.predict(self._scaler.transform(X_arr))
                if self.model_type == "linear"
                else model.predict(X_arr)
            )
            residual_std = np.std(preds_matrix, axis=0)
            residual_std = np.maximum(residual_std, np.abs(base_pred) * 0.02)

            results[name] = {
                "mean": np.maximum(base_pred, 0.0),
                "lower": np.maximum(base_pred - 1.96 * residual_std, 0.0),
                "upper": base_pred + 1.96 * residual_std,
            }

        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_type": self.model_type,
            "models": self._models,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "fitted": self._fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> EmissionSurrogate:
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(model_type=state["model_type"])
        obj._models = state["models"]
        obj._scaler = state["scaler"]
        obj._feature_names = state["feature_names"]
        obj._fitted = state["fitted"]
        return obj

    def feature_importance(self) -> pd.DataFrame:
        self._check_fitted()
        records: list[dict[str, Any]] = []

        for target_name, model in self._models.items():
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_)
            else:
                importances = np.zeros(len(self._feature_names))

            for feat, imp in zip(self._feature_names, importances):
                records.append({
                    "feature": feat,
                    "target": target_name,
                    "importance": float(imp),
                })

        df = pd.DataFrame(records)
        avg = (
            df.groupby("feature", as_index=False)
            .agg(avg_importance=("importance", "mean"))
            .sort_values("avg_importance", ascending=False)
        )
        return avg


    def _make_model(self) -> Any:
        if self.model_type == "xgboost":
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
            )
        if self.model_type == "linear":
            return Ridge(alpha=1.0)
        if self.model_type == "mlp":
            from sklearn.neural_network import MLPRegressor

            return MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            )
        raise ValueError(f"Unknown model type: {self.model_type}")

    def _prepare_X(self, X: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self._feature_names if c in X.columns]
        missing = [c for c in self._feature_names if c not in X.columns]
        arr = X[cols].fillna(0).values.astype(np.float64)
        if missing:
            zeros = np.zeros((arr.shape[0], len(missing)))
            arr = np.hstack([arr, zeros])
        return arr

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("EmissionSurrogate has not been fitted yet.")
