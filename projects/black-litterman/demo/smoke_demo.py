"""Offline synthetic checks of the public notebook's actual model classes.

No market data, credentials, downloads, or full-notebook execution are involved.
The default checks include the SciPy allocation solver. --core-only checks the
posterior and weight projection without running an optimization solver.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_model_classes(notebook_path: Path, minimize):
    """Load only the two named class definitions; exclude all notebook data cells."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    wanted = {"BlackLittermanModel", "PortfolioOptimizer"}
    classes = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        classes.extend(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name in wanted
        )
    if len(classes) != 2 or {node.name for node in classes} != wanted:
        raise ValueError("Expected exactly the two research model classes.")
    namespace = {"np": np, "pd": pd, "cp": None, "minimize": minimize}
    module = ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[]))
    exec(compile(module, str(notebook_path), "exec"), namespace)
    return namespace["BlackLittermanModel"], namespace["PortfolioOptimizer"]


def check_weights(weights, cap=0.20):
    values = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(values)):
        raise AssertionError("Weights contain non-finite values.")
    np.testing.assert_allclose(values.sum(), 1.0, atol=1e-9, rtol=0.0)
    if values.min() < -1e-9 or values.max() > cap + 1e-9:
        raise AssertionError("Long-only or asset-cap constraint failed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-only", action="store_true", help="Skip the SciPy solver check.")
    args = parser.parse_args()
    minimize = None
    if not args.core_only:
        try:
            from scipy.optimize import minimize
        except ImportError as exc:
            raise SystemExit(
                "Install demo/requirements.txt for the full smoke check; "
                "--core-only explicitly skips solver validation."
            ) from exc

    project = Path(__file__).resolve().parents[1]
    BL, Optimizer = load_model_classes(project / "notebooks" / "analysis_clean.ipynb", minimize)
    rng = np.random.default_rng(42)
    names = [f"SYNTHETIC_{index}" for index in range(8)]
    loading = rng.normal(size=(8, 8))
    sigma = (loading @ loading.T + np.eye(8)) * 1e-5
    covariance = pd.DataFrame(sigma, index=names, columns=names)
    returns = pd.DataFrame(
        rng.multivariate_normal(np.zeros(8), sigma, size=320), columns=names
    )
    prior_weights = pd.Series(1.0 / len(names), index=names)
    tau = 0.025
    model = BL(returns, prior_weights, tau=tau, covariance_matrix=covariance)

    no_view = model.posterior([])
    np.testing.assert_allclose(no_view["mu_post"], no_view["pi_prior"], atol=1e-14, rtol=0)
    np.testing.assert_allclose(no_view["Sigma_post"], (1.0 + tau) * sigma, atol=1e-14, rtol=0)

    # A scalar-view closed form provides an independent check of the update.
    annual_view = 0.12
    confidence = 0.65
    result = model.posterior([
        {"type": "absolute", "asset": names[0], "return": annual_view, "confidence": confidence}
    ])
    pi = no_view["pi_prior"].to_numpy()
    prior_uncertainty = tau * sigma
    omega = prior_uncertainty[0, 0] * (1.0 - confidence) / confidence
    expected = pi + prior_uncertainty[:, 0] * (annual_view / 252.0 - pi[0]) / (
        prior_uncertainty[0, 0] + omega
    )
    np.testing.assert_allclose(result["mu_post"], expected, atol=1e-14, rtol=1e-12)
    predictive = result["Sigma_post"].to_numpy()
    np.testing.assert_allclose(predictive, predictive.T, atol=1e-14, rtol=0)
    if np.linalg.eigvalsh(predictive).min() <= 0:
        raise AssertionError("Predictive covariance is not positive definite.")
    if result["linear_solve_relative_residual"] > 1e-10:
        raise AssertionError("BL linear solve residual exceeded tolerance.")

    optimizer = Optimizer(result["mu_post"], result["Sigma_post"], w_max=0.20)
    # Exercise the notebook's capped-simplex projection with deliberately invalid inputs.
    for candidate in (np.zeros(8), np.full(8, 0.4), rng.normal(size=8)):
        check_weights(optimizer._project_to_bounds(candidate))

    if not args.core_only:
        weights = optimizer.mean_variance(risk_aversion=3.0)
        check_weights(weights)
        if not optimizer.last_solver_info.get("success"):
            raise AssertionError("The allocation solver did not report success.")
        print("PASS: synthetic BL posterior, no-view identity, covariance, projection and SLSQP allocation constraints.")
    else:
        print("PASS: synthetic BL posterior, no-view identity, covariance and weight projection constraints.")
        print("NOT CHECKED: portfolio optimization solver (--core-only selected).")
    print("Synthetic numerical checks only; no historical performance or course score is produced.")


if __name__ == "__main__":
    main()
