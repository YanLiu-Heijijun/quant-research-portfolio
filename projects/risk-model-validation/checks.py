"""Offline checks of chronology, constraints, derivatives and input validation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import (MODEL_COLUMNS, common_backtest, coverage_summary, kupiec_reference,
                      load_prices, minimum_variance_for_target, normal_loss_risk,
                      risk_forecasts, synthetic_prices)


def main():
    mu = np.array([0.025, 0.055, 0.09, 0.12])
    vol = np.array([0.06, 0.11, 0.16, 0.22])
    correlation = np.full((4, 4), 0.25)
    np.fill_diagonal(correlation, 1)
    covariance = np.outer(vol, vol) * correlation
    weights = minimum_variance_for_target(mu, covariance, 0.065)
    np.testing.assert_allclose([weights.sum(), weights @ mu], [1, 0.065], atol=1e-12)
    risk = normal_loss_risk(weights, mu, covariance)
    for i in range(len(weights)):
        delta = np.zeros(len(weights)); delta[i] = 1e-6
        plus = normal_loss_risk(weights + delta, mu, covariance)
        minus = normal_loss_risk(weights - delta, mu, covariance)
        for key, derivative in [("var", "dvar_dw"), ("es", "des_dw")]:
            np.testing.assert_allclose((plus[key] - minus[key]) / 2e-6,
                                       risk[derivative][i], rtol=1e-7)

    prices = synthetic_prices(n=500)
    full = risk_forecasts(prices)
    prefix = risk_forecasts(prices.iloc[:300])
    pd.testing.assert_frame_equal(full.loc[prefix.index, list(MODEL_COLUMNS)],
                                  prefix[list(MODEL_COLUMNS)])
    # Changing later prices must not change any forecast known at the cutoff.
    shocked = prices.copy()
    shocked.iloc[300:] *= 2
    pd.testing.assert_frame_equal(full.iloc[:300][list(MODEL_COLUMNS)],
                                  risk_forecasts(shocked).iloc[:300][list(MODEL_COLUMNS)])
    returns = np.log(prices).diff()
    seed = returns.iloc[1:22].var(ddof=1)
    from scipy.stats import norm
    np.testing.assert_allclose(full["ewma_normal"].iloc[21], norm.ppf(.99) * np.sqrt(seed * 10))
    variance_next = .72 * seed + .28 * returns.iloc[22] ** 2
    np.testing.assert_allclose(full["ewma_normal"].iloc[22], norm.ppf(.99) * np.sqrt(variance_next * 10))
    # All three methods include the same current close in their information set.
    expected_hist = -returns.iloc[2:23].quantile(.01) * np.sqrt(10)
    np.testing.assert_allclose(full["historical_scaled"].iloc[22], expected_hist)
    common = common_backtest(full)
    assert common.index[0] == prices.index[21]
    assert common.index[-1] == prices.index[-11]
    summary = coverage_summary(common)
    assert summary["observations"].nunique() == 1
    for b in (0, 10, 1000):
        lr, p = kupiec_reference(1000, b, .01)
        assert np.isfinite(lr) and 0 <= p <= 1
    np.testing.assert_allclose(kupiec_reference(1000, 10, .01), (0, 1), atol=1e-12)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic.csv"
        prices.rename_axis("date").to_csv(path)
        np.testing.assert_allclose(load_prices(path), prices)
        pd.DataFrame({"date": ["2020-01-01", "2020-01-01"], "close": [100, 101]}).to_csv(path, index=False)
        try:
            load_prices(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate dates must be rejected")
    print("PASS: portfolio constraints, finite-difference sensitivities, causal forecasts,")
    print("common forecast dates, EWMA initialization/update, Kupiec boundary cases, CSV validation.")


if __name__ == "__main__":
    main()
