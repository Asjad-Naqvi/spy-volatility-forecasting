"""Causal variance forecasts from squared daily log returns."""

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

HORIZON = 5
LOOKBACK = 20
MIN_HISTORY = 252
VARIANCE_FLOOR = 1e-12
DECAYS = (0.90, 0.94, 0.97, 0.99)


def ewma_variance(squared_returns, decay):
    initial = squared_returns[:LOOKBACK].mean()
    daily, _ = lfilter(
        [1 - decay], [1, -decay], squared_returns[LOOKBACK:],
        zi=[decay * initial],
    )
    return np.maximum(HORIZON * daily, VARIANCE_FLOOR)


def garch_variance(squared_returns):
    """Refit a zero-mean Gaussian GARCH(1,1) model through today's close."""
    scaled = squared_returns * 10_000  # Squared percentage returns.
    initial = scaled[:LOOKBACK].mean()
    if scaled.mean() <= VARIANCE_FLOOR * 10_000:
        return VARIANCE_FLOOR

    def filter_variances(params):
        omega, persistence, shock_share = params
        alpha = persistence * shock_share
        beta = persistence - alpha
        # Each input shock updates the variance for the following session.
        future, _ = lfilter(
            [1], [1, -beta], omega + alpha * scaled, zi=[beta * initial]
        )
        return future

    def loss(params):
        daily = filter_variances(params)[:-1]
        return np.mean(np.log(daily) + scaled[1:] / daily)

    bounds = [(1e-8, None), (0, 0.9999), (0, 1)]
    fits = [
        minimize(loss, start, method="L-BFGS-B", bounds=bounds)
        for start in ((0.05 * scaled.mean(), 0.95, 0.08),
                      (0.20 * scaled.mean(), 0.80, 0.20))
    ]
    converged = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not converged:
        return None

    fit = min(converged, key=lambda result: result.fun)
    omega, persistence, _ = fit.x
    daily = filter_variances(fit.x)[-1]
    total = daily
    for _ in range(HORIZON - 1):
        daily = omega + persistence * daily
        total += daily
    return max(float(total / 10_000), VARIANCE_FLOOR)
