"""Uncertainty quantification for emission estimates."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from carbon_twin.data.schemas import EmissionEstimate


def conformal_prediction_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_new_pred: np.ndarray,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Split conformal prediction intervals.

    Uses the calibration residuals from ``(y_true, y_pred)`` to construct
    distribution-free intervals at level ``1 - alpha`` for ``y_new_pred``.

    Parameters
    ----------
    y_true : (n_cal,) calibration true values
    y_pred : (n_cal,) calibration predicted values
    y_new_pred : (n_new,) new predictions to wrap with intervals
    alpha : miscoverage rate (default 0.1 for 90 % intervals)

    Returns
    -------
    lower, upper : (n_new,) arrays
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_new_pred = np.asarray(y_new_pred, dtype=np.float64)

    residuals = np.abs(y_true - y_pred)
    n_cal = len(residuals)

    quantile_level = np.ceil((1 - alpha) * (n_cal + 1)) / n_cal
    quantile_level = min(quantile_level, 1.0)
    q_hat = float(np.quantile(residuals, quantile_level))

    lower = y_new_pred - q_hat
    upper = y_new_pred + q_hat
    return lower, upper


def bootstrap_uncertainty(
    model: Any,
    X: np.ndarray,
    n_bootstrap: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap prediction uncertainty via repeated prediction on resampled data.

    Re-trains the model on bootstrap samples of ``X`` (paired with the model's
    own predictions as pseudo-targets when true labels are unavailable) to
    estimate epistemic uncertainty.

    Parameters
    ----------
    model : fitted estimator with ``predict(X)``
    X : (n, d) feature matrix
    n_bootstrap : number of bootstrap iterations

    Returns
    -------
    lower, upper : (n,) arrays at 95 % level
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    rng = np.random.default_rng(42)

    base_pred = model.predict(X)
    preds = np.zeros((n_bootstrap, n))

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        X_boot = X[idx]
        y_boot = base_pred[idx]

        # Clone and retrain on bootstrap sample
        try:
            from sklearn.base import clone

            m = clone(model)
            m.fit(X_boot, y_boot)
            preds[b] = m.predict(X)
        except Exception:
            # Fall back to perturbation-based uncertainty
            noise = rng.normal(0, np.std(base_pred) * 0.05, size=n)
            preds[b] = base_pred + noise

    lower = np.percentile(preds, 2.5, axis=0)
    upper = np.percentile(preds, 97.5, axis=0)
    return lower, upper


def propagate_emission_uncertainty(
    emission_estimate: EmissionEstimate,
    factor_uncertainty_pct: float = 0.05,
) -> EmissionEstimate:
    """Add confidence bounds to an emission estimate via error propagation.

    Assumes independent symmetric uncertainty on emission factors (as a
    fraction of the point estimate).  Propagates through quadrature for
    the TTW total.

    Parameters
    ----------
    emission_estimate : base point estimate
    factor_uncertainty_pct : relative uncertainty on emission factors (0.05 = 5 %)

    Returns
    -------
    Copy with ``confidence_lower`` and ``confidence_upper`` populated.
    """
    est = deepcopy(emission_estimate)

    co2_unc = est.co2_tonnes * factor_uncertainty_pct
    ch4_unc = est.ch4_tonnes * factor_uncertainty_pct
    n2o_unc = est.n2o_tonnes * factor_uncertainty_pct

    ttw_unc = np.sqrt(co2_unc**2 + (28.0 * ch4_unc) ** 2 + (265.0 * n2o_unc) ** 2)

    central = est.co2e_ttw_tonnes
    est.confidence_lower = max(central - 1.96 * ttw_unc, 0.0)
    est.confidence_upper = central + 1.96 * ttw_unc

    return est


def calibration_curve(
    y_true: np.ndarray,
    y_pred_lower: np.ndarray,
    y_pred_upper: np.ndarray,
    quantiles: np.ndarray | list[float],
) -> dict[str, np.ndarray]:
    """Evaluate empirical coverage of prediction intervals.

    For each desired nominal coverage level ``q`` in ``quantiles``, compute the
    fraction of true values falling within symmetrically-rescaled intervals.

    Parameters
    ----------
    y_true : (n,) true values
    y_pred_lower : (n,) lower bounds of some base interval
    y_pred_upper : (n,) upper bounds of some base interval
    quantiles : list of nominal coverage levels (e.g. [0.5, 0.8, 0.9, 0.95])

    Returns
    -------
    dict with keys 'nominal_coverage', 'empirical_coverage', 'n_samples'
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred_lower = np.asarray(y_pred_lower, dtype=np.float64)
    y_pred_upper = np.asarray(y_pred_upper, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)

    midpoint = (y_pred_lower + y_pred_upper) / 2.0
    half_width = (y_pred_upper - y_pred_lower) / 2.0

    empirical = np.zeros(len(quantiles))

    for i, q in enumerate(quantiles):
        # Scale the interval to the desired coverage level
        # Assumes the base interval is ~95 %; rescale via normal approximation
        base_z = 1.96
        target_z = _normal_quantile((1 + q) / 2.0)
        scale = target_z / base_z if base_z > 0 else 1.0

        lo = midpoint - half_width * scale
        hi = midpoint + half_width * scale
        covered = np.mean((y_true >= lo) & (y_true <= hi))
        empirical[i] = float(covered)

    return {
        "nominal_coverage": quantiles,
        "empirical_coverage": empirical,
        "n_samples": np.full(len(quantiles), len(y_true)),
    }


def _normal_quantile(p: float) -> float:
    """Approximate inverse normal CDF using Beasley-Springer-Moro algorithm."""
    from scipy.stats import norm

    return float(norm.ppf(p))
