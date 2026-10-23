"""Educational portfolio risk and causal VaR forecasts; no data downloads."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binom, chi2, norm


MODEL_COLUMNS = ("rolling_normal", "ewma_normal", "historical_scaled")


def load_prices(path: str | Path) -> pd.Series:
    """Read an explicitly supplied CSV with date and close columns."""
    data = pd.read_csv(path)
    if not {"date", "close"}.issubset(data.columns):
        raise ValueError("CSV requires date and close columns.")
    dates = pd.to_datetime(data["date"], errors="raise")
    prices = pd.Series(pd.to_numeric(data["close"], errors="raise").to_numpy(),
                       index=pd.DatetimeIndex(dates), name="close")
    return validate_prices(prices)


def validate_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Prices require a DatetimeIndex.")
    if prices.index.hasnans or prices.index.has_duplicates:
        raise ValueError("Dates must be present and unique.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Dates must be strictly increasing; sort the input explicitly.")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Every close must be finite and positive.")
    return pd.Series(values, index=prices.index, name="close")


def synthetic_prices(n: int = 900, seed: int = 16) -> pd.Series:
    """Invented regime-changing series for exercising the workflow only."""
    if n < 100:
        raise ValueError("The demo requires at least 100 observations.")
    rng = np.random.default_rng(seed)
    volatility = np.full(n - 1, 0.009)
    volatility[(n - 1) // 3: 2 * (n - 1) // 3] = 0.023
    returns = volatility * rng.normal(size=n - 1)
    values = 100 * np.exp(np.r_[0.0, np.cumsum(returns)])
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=n), name="close")


def minimum_variance_for_target(mu: np.ndarray, covariance: np.ndarray,
                                target: float) -> np.ndarray:
    """Fully invested minimum-variance weights, permitting short positions."""
    mu = np.asarray(mu, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (len(mu), len(mu)) or not np.allclose(covariance, covariance.T):
        raise ValueError("Covariance must be square and symmetric.")
    np.linalg.cholesky(covariance)
    constraints = np.column_stack([np.ones_like(mu), mu])
    solved = np.linalg.solve(covariance, constraints)
    return solved @ np.linalg.solve(constraints.T @ solved, np.array([1.0, target]))


def normal_loss_risk(weights: np.ndarray, mu: np.ndarray, covariance: np.ndarray,
                     confidence: float = 0.99) -> dict:
    """Positive-loss VaR/ES and unconstrained partial weight derivatives.

    All input moments refer to one identical, user-defined time horizon.
    Increasing one weight does not rebalance the other weights here.
    """
    if not 0.5 < confidence < 1:
        raise ValueError("Confidence must be between 0.5 and 1.")
    weights, mu, covariance = map(np.asarray, (weights, mu, covariance))
    sigma = float(np.sqrt(weights @ covariance @ weights))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Portfolio volatility must be positive.")
    z = norm.ppf(confidence)
    tail_multiplier = norm.pdf(z) / (1 - confidence)
    marginal_sigma = covariance @ weights / sigma
    return {
        "var": float(-weights @ mu + z * sigma),
        "es": float(-weights @ mu + tail_multiplier * sigma),
        "dvar_dw": -mu + z * marginal_sigma,
        "des_dw": -mu + tail_multiplier * marginal_sigma,
    }


def risk_forecasts(prices: pd.Series, window: int = 21, horizon: int = 10,
                   confidence: float = 0.99, decay: float = 0.72) -> pd.DataFrame:
    """Forecast h-session positive-loss thresholds using information through close t.

    Each method has the same date and information cutoff. Rolling and empirical
    quantile estimates use the latest `window` returns including r_t. EWMA uses
    that initial window only for initialization and then updates recursively.
    No future returns or full-sample moments enter any forecast.
    """
    prices = validate_prices(prices)
    if window < 2 or horizon < 1 or len(prices) <= window:
        raise ValueError("Need window >= 2, horizon >= 1 and more than window prices.")
    if not 0 < decay < 1 or not 0.5 < confidence < 1:
        raise ValueError("Invalid decay or confidence.")
    returns = np.log(prices).diff()
    rolling_sigma = returns.rolling(window, min_periods=window).std(ddof=1)
    ewma_variance = np.full(len(prices), np.nan)
    # At the first forecast close, all warm-up returns have already occurred.
    ewma_variance[window] = returns.iloc[1:window + 1].var(ddof=1)
    for i in range(window + 1, len(prices)):
        ewma_variance[i] = decay * ewma_variance[i - 1] + (1 - decay) * returns.iloc[i] ** 2
    multiplier = norm.ppf(confidence) * np.sqrt(horizon)
    forecasts = pd.DataFrame(index=prices.index)
    forecasts["rolling_normal"] = multiplier * rolling_sigma
    forecasts["ewma_normal"] = multiplier * np.sqrt(ewma_variance)
    forecasts["historical_scaled"] = (
        -returns.rolling(window, min_periods=window).quantile(1 - confidence)
        * np.sqrt(horizon)
    )
    # Loss is observed only later and is never passed back to forecast generation.
    forecasts["realized_loss"] = -np.log(prices.shift(-horizon) / prices)
    return forecasts


def common_backtest(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Use one shared set of forecast origins for every model."""
    common = forecasts.dropna(subset=[*MODEL_COLUMNS, "realized_loss"]).copy()
    if common.empty:
        raise ValueError("No common forecast origins with realized outcomes.")
    return common


def kupiec_reference(n: int, breaches: int, probability: float) -> tuple[float, float]:
    """Unconditional coverage LR and nominal iid chi-square reference p-value."""
    if n < 1 or not 0 <= breaches <= n or not 0 < probability < 1:
        raise ValueError("Invalid coverage-test arguments.")
    observed = breaches / n
    null_loglik = xlogy(breaches, probability) + xlogy(n - breaches, 1 - probability)
    fitted_loglik = xlogy(breaches, observed) + xlogy(n - breaches, 1 - observed)
    lr = max(0.0, float(2 * (fitted_loglik - null_loglik)))
    return lr, float(chi2.sf(lr, 1))


def coverage_summary(common: pd.DataFrame, confidence: float = 0.99,
                     stride: int = 1, iid_reference: bool = False) -> pd.DataFrame:
    """Breach counts on common origins, optionally with clearly labeled references.

    A stride equal to the horizon removes overlap in realized return intervals;
    it does not prove independent breach events. Do not treat reference p-values
    or binomial bands as regulatory validation or a model quality ranking.
    """
    if stride < 1:
        raise ValueError("Stride must be positive.")
    sample = common.iloc[::stride]
    if sample.empty:
        raise ValueError("Empty backtest.")
    rows = []
    n, expected_probability = len(sample), 1 - confidence
    for model in MODEL_COLUMNS:
        breaches = int((sample["realized_loss"] > sample[model]).sum())
        row = {"model": model, "observations": n, "breaches": breaches,
               "breach_rate": breaches / n, "expected_rate": expected_probability}
        if iid_reference:
            lr, p_value = kupiec_reference(n, breaches, expected_probability)
            green = int(binom.ppf(0.95, n, expected_probability))
            yellow = int(binom.ppf(0.9999, n, expected_probability))
            row.update({"kupiec_lr": lr, "nominal_iid_pvalue": p_value,
                        "binomial_band_iid_reference":
                        "green" if breaches <= green else "yellow" if breaches <= yellow else "red"})
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def plot_backtest(common: pd.DataFrame):
    import matplotlib.pyplot as plt
    ax = common.plot(figsize=(12, 5), alpha=0.8)
    ax.set(title="Loss thresholds and future realized loss (common forecast origins)",
           ylabel="Log-return loss; positive = loss", xlabel="Forecast close date")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return ax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Local date,close CSV; never downloaded")
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()
    prices = load_prices(args.csv) if args.csv else synthetic_prices()
    print("LOCAL DATA: supplied by you" if args.csv else "SYNTHETIC DEMO: no market or investment conclusion")
    common = common_backtest(risk_forecasts(prices, window=args.window, horizon=args.horizon))
    print("Common-date overlapping observations; descriptive counts only")
    print(coverage_summary(common).to_string())
    print("Non-overlapping return intervals; iid reference diagnostics only")
    print(coverage_summary(common, stride=args.horizon, iid_reference=True).to_string())


if __name__ == "__main__":
    main()
