"""Offline GBM option-pricing experiments with explicit sampling uncertainty."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class Parameters:
    spot: float = 100.0
    strike: float = 100.0
    maturity: float = 1.0
    rate: float = 0.05
    volatility: float = 0.20

    def validate(self):
        values = (self.spot, self.strike, self.maturity, self.rate, self.volatility)
        if not np.isfinite(values).all():
            raise ValueError("All parameters must be finite.")
        if min(self.spot, self.strike, self.maturity) <= 0 or self.volatility < 0:
            raise ValueError("Spot, strike and maturity must be positive; volatility nonnegative.")


def analytical_price(parameters: Parameters, payoff: str = "call") -> float:
    """No-dividend Black-Scholes price; binary pays one unit of cash when ST > K."""
    parameters.validate()
    p = parameters
    discount = np.exp(-p.rate * p.maturity)
    if payoff not in ("call", "binary"):
        raise ValueError("Payoff must be call or binary.")
    if p.volatility == 0:
        terminal = p.spot * np.exp(p.rate * p.maturity)
        amount = max(terminal - p.strike, 0) if payoff == "call" else float(terminal > p.strike)
        return float(discount * amount)
    d1 = (np.log(p.spot / p.strike) + (p.rate + .5 * p.volatility ** 2) * p.maturity) / (p.volatility * np.sqrt(p.maturity))
    d2 = d1 - p.volatility * np.sqrt(p.maturity)
    if payoff == "binary":
        return float(discount * norm.cdf(d2))
    return float(p.spot * norm.cdf(d1) - p.strike * discount * norm.cdf(d2))


def terminal_prices(parameters: Parameters, paths: int, steps: int, method: str,
                    rng: np.random.Generator) -> np.ndarray:
    """GBM exact terminal sampling or Euler/Milstein stepping, vectorized over paths.

    Exact terminal sampling removes discretization error, not Monte Carlo error.
    Euler can produce negative prices for coarse steps or high volatility; values
    are not silently clipped. Runtime is an observation, not a production claim.
    """
    parameters.validate()
    if paths < 2 or steps < 1:
        raise ValueError("At least two paths and one time step are required.")
    p = parameters
    if method == "exact_terminal":
        z = rng.standard_normal(paths)
        return p.spot * np.exp((p.rate - .5 * p.volatility ** 2) * p.maturity
                              + p.volatility * np.sqrt(p.maturity) * z)
    if method not in ("euler", "milstein"):
        raise ValueError("Unknown simulation method.")
    dt = p.maturity / steps
    terminal = np.full(paths, p.spot, dtype=float)
    for _ in range(steps):
        z = rng.standard_normal(paths)
        increment = 1 + p.rate * dt + p.volatility * np.sqrt(dt) * z
        if method == "milstein":
            increment += .5 * p.volatility ** 2 * (z ** 2 - 1) * dt
        terminal *= increment
    return terminal


def discounted_payoff(terminal: np.ndarray, parameters: Parameters, payoff: str) -> np.ndarray:
    if payoff == "call":
        values = np.maximum(terminal - parameters.strike, 0)
    elif payoff == "binary":
        values = (terminal > parameters.strike).astype(float)
    else:
        raise ValueError("Payoff must be call or binary.")
    return np.exp(-parameters.rate * parameters.maturity) * values


def sample_statistics(independent_values: np.ndarray) -> dict:
    """Each input value is an independent observation; a pair mean is one value."""
    values = np.asarray(independent_values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite independent observations.")
    price = float(values.mean())
    sample_variance = float(values.var(ddof=1))
    estimator_variance = sample_variance / len(values)
    standard_error = float(np.sqrt(estimator_variance))
    return {"price": price, "standard_error": standard_error,
            "ci_lower": price - 1.96 * standard_error,
            "ci_upper": price + 1.96 * standard_error,
            "sample_variance": sample_variance,
            "estimator_variance": estimator_variance,
            "independent_observations": len(values)}


def baseline(parameters: Parameters = Parameters(), paths: int = 20000,
             steps: int = 64, seed: int = 42) -> pd.DataFrame:
    rows = []
    streams = np.random.SeedSequence(seed).spawn(3)
    for method, stream in zip(("exact_terminal", "euler", "milstein"), streams):
        started = perf_counter()
        terminal = terminal_prices(parameters, paths, steps, method, np.random.default_rng(stream))
        seconds = perf_counter() - started
        for payoff in ("call", "binary"):
            stats = sample_statistics(discounted_payoff(terminal, parameters, payoff))
            exact = analytical_price(parameters, payoff)
            rows.append({"method": method, "payoff": payoff, **stats,
                         "analytical_price": exact, "difference_to_analytical": stats["price"] - exact,
                         "simulation_seconds": seconds, "negative_terminal_paths": int((terminal < 0).sum()),
                         "total_paths": paths, "time_steps": steps if method != "exact_terminal" else 1})
    return pd.DataFrame(rows)


def euler_pair_means(parameters: Parameters, pairs: int, steps: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Average discounted call payoffs for independent (z, -z) path pairs."""
    parameters.validate()
    if pairs < 2 or steps < 1:
        raise ValueError("At least two pairs and one step are required.")
    p = parameters
    dt = p.maturity / steps
    plus, minus = np.full(pairs, p.spot, dtype=float), np.full(pairs, p.spot, dtype=float)
    for _ in range(steps):
        z = rng.standard_normal(pairs)
        plus *= 1 + p.rate * dt + p.volatility * np.sqrt(dt) * z
        minus *= 1 + p.rate * dt - p.volatility * np.sqrt(dt) * z
    return .5 * (discounted_payoff(plus, p, "call") + discounted_payoff(minus, p, "call"))


def antithetic_comparison(parameters: Parameters = Parameters(), total_paths: int = 20000,
                         steps: int = 64, seed: int = 73) -> tuple[pd.DataFrame, dict]:
    if total_paths < 4 or total_paths % 2:
        raise ValueError("Total path budget must be even and at least four.")
    stream_standard, stream_pairs = np.random.SeedSequence(seed).spawn(2)
    started = perf_counter()
    terminal = terminal_prices(parameters, total_paths, steps, "euler", np.random.default_rng(stream_standard))
    standard = sample_statistics(discounted_payoff(terminal, parameters, "call"))
    standard_seconds = perf_counter() - started
    started = perf_counter()
    paired = sample_statistics(euler_pair_means(parameters, total_paths // 2, steps,
                                                np.random.default_rng(stream_pairs)))
    pair_seconds = perf_counter() - started
    frame = pd.DataFrame([{"method": "standard_euler", **standard, "seconds": standard_seconds},
                          {"method": "antithetic_euler", **paired, "seconds": pair_seconds}])
    frame["total_paths"] = total_paths
    ratio = (standard["estimator_variance"] / paired["estimator_variance"]
             if paired["estimator_variance"] > 0 else np.nan)
    se_reduction = (1 - paired["standard_error"] / standard["standard_error"]
                    if standard["standard_error"] > 0 else np.nan)
    return frame, {"equal_path_budget_estimator_variance_ratio": ratio,
                   "standard_error_reduction_fraction": se_reduction}


def sensitivity(parameters: Parameters = Parameters(), paths: int = 3000,
                steps_per_year: int = 64, seed: int = 19) -> pd.DataFrame:
    scenarios = [("volatility", x) for x in np.linspace(.1, .5, 9)]
    scenarios += [("rate", x) for x in np.linspace(.01, .1, 10)]
    scenarios += [("maturity", x) for x in np.linspace(.5, 3, 8)]
    streams = np.random.SeedSequence(seed).spawn(len(scenarios))
    rows = []
    for (parameter, value), stream in zip(scenarios, streams):
        p = replace(parameters, **{parameter: float(value)})
        steps = max(1, round(steps_per_year * p.maturity))
        terminal = terminal_prices(p, paths, steps, "milstein", np.random.default_rng(stream))
        rows.append({"parameter": parameter, "value": value,
                     **sample_statistics(discounted_payoff(terminal, p, "call")),
                     "analytical_price": analytical_price(p), "paths": paths, "steps": steps})
    return pd.DataFrame(rows)


def plot_sensitivity(results: pd.DataFrame):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, parameter in zip(axes, ("volatility", "rate", "maturity")):
        subset = results[results["parameter"] == parameter]
        ax.errorbar(subset["value"], subset["price"], yerr=1.96 * subset["standard_error"],
                    fmt="o", capsize=3, label="Milstein MC +/- 1.96 SE")
        ax.plot(subset["value"], subset["analytical_price"], label="Analytical")
        ax.set(xlabel=parameter, ylabel="Call price")
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=20000)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args()
    print("Public rewrite: fresh simulation, no historical saved results")
    print(baseline(paths=args.paths, steps=args.steps).round(6).to_string(index=False))
    table, ratios = antithetic_comparison(total_paths=args.paths, steps=args.steps)
    print(table.round(6).to_string(index=False))
    print(ratios)
    if args.sensitivity:
        print(sensitivity(paths=min(args.paths, 5000), steps_per_year=args.steps).round(6).to_string(index=False))


if __name__ == "__main__":
    main()
