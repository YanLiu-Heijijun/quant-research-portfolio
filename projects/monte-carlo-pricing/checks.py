"""Small offline numerical checks; no original large experiment is rerun."""

import numpy as np

from analysis import (Parameters, analytical_price, antithetic_comparison, baseline,
                      discounted_payoff, sample_statistics, sensitivity, terminal_prices)


def main():
    p = Parameters()
    np.testing.assert_allclose(analytical_price(p), 10.450583572185565, rtol=1e-12)
    np.testing.assert_allclose(analytical_price(p, "binary"), .5323248154537634, rtol=1e-12)
    values = np.array([1., 2., 4., 7.])
    stats = sample_statistics(values)
    np.testing.assert_allclose(stats["estimator_variance"], values.var(ddof=1) / len(values))
    np.testing.assert_allclose(stats["standard_error"] ** 2, stats["estimator_variance"])
    np.testing.assert_allclose(discounted_payoff(np.array([99, 100, 101]), p, "binary"),
                               np.exp(-.05) * np.array([0, 0, 1]))
    zero = Parameters(volatility=0)
    for method in ("exact_terminal", "euler", "milstein"):
        terminal = terminal_prices(zero, 50, 16, method, np.random.default_rng(1))
        expected = zero.spot * (np.exp(zero.rate) if method == "exact_terminal" else (1 + zero.rate / 16) ** 16)
        np.testing.assert_allclose(terminal, expected)
    results = baseline(paths=16000, steps=64, seed=42)
    # A deterministic-seed plausibility check, not a guarantee of CI coverage.
    assert (abs(results["difference_to_analytical"]) < 6 * results["standard_error"] + .03).all()
    np.testing.assert_allclose(results["standard_error"] ** 2, results["estimator_variance"])
    comparison, ratios = antithetic_comparison(total_paths=16000, steps=64, seed=73)
    standard, paired = comparison.iloc[0], comparison.iloc[1]
    assert standard["total_paths"] == paired["total_paths"] == 16000
    assert standard["independent_observations"] == 16000
    assert paired["independent_observations"] == 8000
    np.testing.assert_allclose(ratios["equal_path_budget_estimator_variance_ratio"],
        (standard["sample_variance"] / paired["sample_variance"]) / 2)
    np.testing.assert_allclose(ratios["equal_path_budget_estimator_variance_ratio"],
                               (standard["standard_error"] / paired["standard_error"]) ** 2)
    assert ratios["equal_path_budget_estimator_variance_ratio"] > 1
    scenarios = sensitivity(paths=500, steps_per_year=16)
    assert len(scenarios) == 27 and np.isfinite(scenarios["price"]).all()
    for parameter in ("volatility", "rate", "maturity"):
        # The analytical curve is monotone for these specific parameters.
        assert (scenarios.loc[scenarios.parameter == parameter, "analytical_price"].diff().dropna() > 0).all()
    print("PASS: analytical benchmarks, payoff boundary, zero-volatility limit, sampling statistics,")
    print("16,000-path numerical plausibility, equal-budget variance ratio, 27 small sensitivity scenarios.")
    print("Small-check SE reduction fraction:", round(ratios["standard_error_reduction_fraction"], 6))
    print("Small-check estimator variance ratio:", round(ratios["equal_path_budget_estimator_variance_ratio"], 6))


if __name__ == "__main__":
    main()
