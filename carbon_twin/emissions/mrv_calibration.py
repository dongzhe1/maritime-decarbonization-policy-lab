"""Weak-supervision calibration of voyage-level proxy emissions using annual MRV data."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

CalibrationMethod = Literal["simple_ratio", "ship_type_ratio", "hierarchical", "gradient_boosting"]


class MRVCalibrator:
    """Calibrate proxy voyage emissions to match annual MRV totals.

    Supports four methods:
      - simple_ratio: per-vessel ratio = MRV_total / sum(proxy)
      - ship_type_ratio: per-ship-type average ratio
      - hierarchical: vessel random effect + ship_type fixed effect via
        regularized regression
      - gradient_boosting: XGBoost on vessel features predicting ratio
    """

    def __init__(self, method: CalibrationMethod = "hierarchical") -> None:
        self.method: CalibrationMethod = method
        self._vessel_ratios: pd.DataFrame | None = None
        self._ship_type_ratios: pd.Series | None = None
        self._hierarchical_model: dict | None = None
        self._gb_model: object | None = None
        self._gb_feature_cols: list[str] | None = None
        self._global_ratio: float = 1.0

    def fit(
        self,
        voyages_with_proxy: pd.DataFrame,
        mrv_annual: pd.DataFrame,
    ) -> MRVCalibrator:
        """Learn calibration factors from proxy voyage emissions and MRV annual data.

        Parameters
        ----------
        voyages_with_proxy : DataFrame
            Columns: vessel_imo, year, voyage_id, proxy_emission_tonnes.
            May also include: ship_type, dwt, year_built, distance_nm.
        mrv_annual : DataFrame
            Columns: imo_number, year, total_co2_tonnes.
        """
        proxy_annual = voyages_with_proxy.groupby(["vessel_imo", "year"], as_index=False).agg(
            proxy_sum=("proxy_emission_tonnes", "sum"),
            num_voyages=("voyage_id", "nunique"),
        )

        merged = proxy_annual.merge(
            mrv_annual.rename(columns={"imo_number": "vessel_imo"}),
            on=["vessel_imo", "year"],
            how="inner",
        )
        merged["ratio"] = np.where(
            merged["proxy_sum"] > 0,
            merged["total_co2_tonnes"] / merged["proxy_sum"],
            1.0,
        )
        self._global_ratio = float(merged["ratio"].median())

        if self.method == "simple_ratio":
            self._fit_simple_ratio(merged)
        elif self.method == "ship_type_ratio":
            self._fit_ship_type_ratio(merged, voyages_with_proxy)
        elif self.method == "hierarchical":
            self._fit_hierarchical(merged, voyages_with_proxy)
        elif self.method == "gradient_boosting":
            self._fit_gradient_boosting(merged, voyages_with_proxy)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def transform(self, voyages_with_proxy: pd.DataFrame) -> pd.DataFrame:
        """Apply calibration to produce ``adjusted_emission_tonnes``."""
        df = voyages_with_proxy.copy()

        if self.method == "simple_ratio":
            df = self._transform_simple_ratio(df)
        elif self.method == "ship_type_ratio":
            df = self._transform_ship_type_ratio(df)
        elif self.method == "hierarchical":
            df = self._transform_hierarchical(df)
        elif self.method == "gradient_boosting":
            df = self._transform_gradient_boosting(df)

        return df

    def get_calibration_factors(self) -> pd.DataFrame:
        """Return vessel- or ship-type-level calibration factors."""
        if self.method == "simple_ratio" and self._vessel_ratios is not None:
            return self._vessel_ratios.copy()
        if self.method == "ship_type_ratio" and self._ship_type_ratios is not None:
            return self._ship_type_ratios.reset_index().rename(
                columns={0: "ratio", "ship_type": "group"}
            )
        if self.method == "hierarchical" and self._hierarchical_model is not None:
            return self._hierarchical_model["vessel_effects"].copy()
        if self.method == "gradient_boosting" and self._gb_model is not None:
            return pd.DataFrame({"info": ["Use feature_importances_ on the XGB model"]})
        return pd.DataFrame()

    def evaluate(
        self,
        voyages_with_proxy: pd.DataFrame,
        mrv_annual: pd.DataFrame,
    ) -> dict[str, float]:
        """Evaluate calibration quality at annual vessel level."""
        adjusted = self.transform(voyages_with_proxy)

        pred_annual = adjusted.groupby(["vessel_imo", "year"], as_index=False).agg(
            predicted_total=("adjusted_emission_tonnes", "sum")
        )

        merged = pred_annual.merge(
            mrv_annual.rename(columns={"imo_number": "vessel_imo"}),
            on=["vessel_imo", "year"],
            how="inner",
        )

        if len(merged) == 0:
            return {"mae": float("nan"), "mape": float("nan"), "r2": float("nan")}

        y_true = merged["total_co2_tonnes"].values
        y_pred = merged["predicted_total"].values

        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    # ------------------------------------------------------------------
    # simple_ratio
    # ------------------------------------------------------------------

    def _fit_simple_ratio(self, merged: pd.DataFrame) -> None:
        self._vessel_ratios = merged.groupby("vessel_imo", as_index=False).agg(
            ratio=("ratio", "median")
        )

    def _transform_simple_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._vessel_ratios is None:
            df["adjusted_emission_tonnes"] = df["proxy_emission_tonnes"]
            return df
        df = df.merge(self._vessel_ratios, on="vessel_imo", how="left")
        df["ratio"] = df["ratio"].fillna(self._global_ratio)
        df["adjusted_emission_tonnes"] = df["proxy_emission_tonnes"] * df["ratio"]
        df.drop(columns=["ratio"], inplace=True)
        return df

    # ------------------------------------------------------------------
    # ship_type_ratio
    # ------------------------------------------------------------------

    def _fit_ship_type_ratio(
        self,
        merged: pd.DataFrame,
        voyages: pd.DataFrame,
    ) -> None:
        vessel_types = (
            voyages[["vessel_imo", "ship_type"]].drop_duplicates("vessel_imo")
            if "ship_type" in voyages.columns
            else pd.DataFrame(columns=["vessel_imo", "ship_type"])
        )
        with_type = merged.merge(vessel_types, on="vessel_imo", how="left")
        if "ship_type" not in with_type.columns or with_type["ship_type"].isna().all():
            with_type["ship_type"] = "unknown"
        self._ship_type_ratios = with_type.groupby("ship_type")["ratio"].median()

    def _transform_ship_type_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._ship_type_ratios is None:
            df["adjusted_emission_tonnes"] = df["proxy_emission_tonnes"]
            return df
        if "ship_type" not in df.columns:
            df["adjusted_emission_tonnes"] = df["proxy_emission_tonnes"] * self._global_ratio
            return df
        ratio_map = self._ship_type_ratios.to_dict()
        df["_ratio"] = df["ship_type"].map(ratio_map).fillna(self._global_ratio)
        df["adjusted_emission_tonnes"] = df["proxy_emission_tonnes"] * df["_ratio"]
        df.drop(columns=["_ratio"], inplace=True)
        return df

    # ------------------------------------------------------------------
    # hierarchical
    # ------------------------------------------------------------------

    def _fit_hierarchical(
        self,
        merged: pd.DataFrame,
        voyages: pd.DataFrame,
    ) -> None:
        """Vessel random effect + ship_type fixed effect via regularized regression.

        min || alpha_i * proxy_sum_i - mrv_i ||^2
            + lambda1 * || alpha_i - 1 ||^2
            + lambda2 * ship_type_smoothness
        """
        vessel_types = (
            voyages[["vessel_imo", "ship_type"]].drop_duplicates("vessel_imo")
            if "ship_type" in voyages.columns
            else pd.DataFrame(columns=["vessel_imo", "ship_type"])
        )
        data = merged.merge(vessel_types, on="vessel_imo", how="left")
        if "ship_type" not in data.columns or data["ship_type"].isna().all():
            data["ship_type"] = "unknown"

        ship_types = sorted(data["ship_type"].unique())
        st_to_idx = {st: i for i, st in enumerate(ship_types)}

        n = len(data)
        n_vessels = data["vessel_imo"].nunique()
        vessel_ids = sorted(data["vessel_imo"].unique())
        v_to_idx = {v: i for i, v in enumerate(vessel_ids)}

        n_st = len(ship_types)
        n_total = n_vessels + n_st

        X = np.zeros((n, n_total))
        y = np.zeros(n)

        for row_i, (_, row) in enumerate(data.iterrows()):
            v_idx = v_to_idx[row["vessel_imo"]]
            st_idx = st_to_idx.get(row["ship_type"], 0)
            proxy = row["proxy_sum"]
            X[row_i, v_idx] = proxy
            X[row_i, n_vessels + st_idx] = proxy
            y[row_i] = row["total_co2_tonnes"]

        model = Ridge(alpha=1.0, fit_intercept=False)
        model.fit(X, y)

        coefs = model.coef_
        vessel_effects = pd.DataFrame(
            {
                "vessel_imo": vessel_ids,
                "vessel_alpha": coefs[:n_vessels],
            }
        )
        ship_type_effects = pd.DataFrame(
            {
                "ship_type": ship_types,
                "ship_type_beta": coefs[n_vessels : n_vessels + n_st],
            }
        )

        vessel_effects = vessel_effects.merge(
            data[["vessel_imo", "ship_type"]].drop_duplicates("vessel_imo"),
            on="vessel_imo",
            how="left",
        )
        vessel_effects = vessel_effects.merge(ship_type_effects, on="ship_type", how="left")
        vessel_effects["combined_factor"] = (
            vessel_effects["vessel_alpha"] + vessel_effects["ship_type_beta"]
        )

        self._hierarchical_model = {
            "vessel_effects": vessel_effects,
            "ship_type_effects": ship_type_effects,
            "ridge": model,
        }

    def _transform_hierarchical(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._hierarchical_model is None:
            df["adjusted_emission_tonnes"] = df["proxy_emission_tonnes"]
            return df

        ve = self._hierarchical_model["vessel_effects"][["vessel_imo", "combined_factor"]].copy()
        df = df.merge(ve, on="vessel_imo", how="left")
        df["combined_factor"] = df["combined_factor"].fillna(self._global_ratio)
        df["adjusted_emission_tonnes"] = df["proxy_emission_tonnes"] * df["combined_factor"]
        df.drop(columns=["combined_factor"], inplace=True)
        return df

    # ------------------------------------------------------------------
    # gradient_boosting
    # ------------------------------------------------------------------

    def _fit_gradient_boosting(
        self,
        merged: pd.DataFrame,
        voyages: pd.DataFrame,
    ) -> None:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost is required for gradient_boosting method") from exc

        feature_cols = []
        vessel_features = pd.DataFrame({"vessel_imo": merged["vessel_imo"].unique()})

        for col in ["ship_type", "dwt", "year_built"]:
            if col in voyages.columns:
                vf = voyages[["vessel_imo", col]].drop_duplicates("vessel_imo")
                vessel_features = vessel_features.merge(vf, on="vessel_imo", how="left")

        if "distance_nm" in voyages.columns:
            dist_agg = voyages.groupby("vessel_imo", as_index=False).agg(
                total_distance=("distance_nm", "sum")
            )
            vessel_features = vessel_features.merge(dist_agg, on="vessel_imo", how="left")

        data = merged.merge(vessel_features, on="vessel_imo", how="left")

        candidate_features = ["num_voyages", "proxy_sum"]
        if "dwt" in data.columns:
            candidate_features.append("dwt")
        if "year_built" in data.columns:
            data["age"] = 2026 - data["year_built"].fillna(2010)
            candidate_features.append("age")
        if "total_distance" in data.columns:
            candidate_features.append("total_distance")
        if "ship_type" in data.columns:
            data["ship_type_code"] = data["ship_type"].astype("category").cat.codes
            candidate_features.append("ship_type_code")

        feature_cols = [c for c in candidate_features if c in data.columns]
        self._gb_feature_cols = feature_cols

        X = data[feature_cols].fillna(0).values
        y = data["ratio"].values

        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
        )
        model.fit(X, y)
        self._gb_model = model
        self._gb_feature_meta = {
            "vessel_features_cols": list(vessel_features.columns),
        }

    def _transform_gradient_boosting(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._gb_model is None or self._gb_feature_cols is None:
            df["adjusted_emission_tonnes"] = df["proxy_emission_tonnes"]
            return df

        work = df.copy()

        if "age" not in work.columns and "year_built" in work.columns:
            work["age"] = 2026 - work["year_built"].fillna(2010)
        if "ship_type_code" not in work.columns and "ship_type" in work.columns:
            work["ship_type_code"] = work["ship_type"].astype("category").cat.codes

        # Aggregate proxy_sum and num_voyages at vessel-year level for features
        if "proxy_sum" not in work.columns:
            proxy_agg = work.groupby(["vessel_imo", "year"], as_index=False).agg(
                proxy_sum=("proxy_emission_tonnes", "sum"), num_voyages=("voyage_id", "nunique")
            )
            work = work.merge(proxy_agg, on=["vessel_imo", "year"], how="left")

        [c for c in self._gb_feature_cols if c in work.columns]
        missing = [c for c in self._gb_feature_cols if c not in work.columns]
        for c in missing:
            work[c] = 0

        X = work[self._gb_feature_cols].fillna(0).values
        work["_ratio"] = self._gb_model.predict(X)  # type: ignore[union-attr]
        work["adjusted_emission_tonnes"] = work["proxy_emission_tonnes"] * work["_ratio"]

        result_cols = list(df.columns) + ["adjusted_emission_tonnes"]
        result_cols = [c for c in result_cols if c in work.columns]
        return work[result_cols].copy()
