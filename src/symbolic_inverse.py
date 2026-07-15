# src/symbolic_inverse.py

import numpy as np
from scipy.optimize import least_squares


def bounds_to_arrays(bounds_dict, variable_order):
    lo = np.array([bounds_dict[name][0] for name in variable_order], dtype=float)
    hi = np.array([bounds_dict[name][1] for name in variable_order], dtype=float)
    return lo, hi


def y_norm_to_x_dict(y_norm, lo, hi, variable_order):
    y_norm = np.asarray(y_norm, dtype=float)
    x_values = lo + y_norm * (hi - lo)

    return {
        name: float(value)
        for name, value in zip(variable_order, x_values)
    }


def symbolic_residuals_from_y_norm(
    y_norm,
    surrogate,
    y_obs,
    lo,
    hi,
    variable_order,
    weights,
):
    try:
        x = y_norm_to_x_dict(
            y_norm=y_norm,
            lo=lo,
            hi=hi,
            variable_order=variable_order,
        )

        y_pred = surrogate.predict_one(x)

        sigma_pred = y_pred["sigma"]
        sigma_obs = y_obs["sigma"]

        if (
            not np.isfinite(y_pred["Vp"])
            or not np.isfinite(y_pred["Vs"])
            or not np.isfinite(sigma_pred)
            or sigma_pred <= 0.0
            or sigma_obs <= 0.0
        ):
            return np.array([1.0e6, 1.0e6, 1.0e6], dtype=float)

        W1 = weights["W1"]
        W2 = weights["W2"]

        return np.array(
            [
                y_pred["Vp"] - y_obs["Vp"],
                np.sqrt(W1) * (y_pred["Vs"] - y_obs["Vs"]),
                np.sqrt(W2) * ((1.0 / sigma_pred) - (1.0 / sigma_obs)),
            ],
            dtype=float,
        )

    except Exception:
        return np.array([1.0e6, 1.0e6, 1.0e6], dtype=float)


def symbolic_scalar_loss_from_y_norm(
    y_norm,
    surrogate,
    y_obs,
    lo,
    hi,
    variable_order,
    weights,
):
    try:
        x = y_norm_to_x_dict(
            y_norm=y_norm,
            lo=lo,
            hi=hi,
            variable_order=variable_order,
        )

        y_pred = surrogate.predict_one(x)

        sigma_pred = y_pred["sigma"]
        sigma_obs = y_obs["sigma"]

        if (
            not np.isfinite(y_pred["Vp"])
            or not np.isfinite(y_pred["Vs"])
            or not np.isfinite(sigma_pred)
            or sigma_pred <= 0.0
            or sigma_obs <= 0.0
        ):
            return np.inf

        W1 = weights["W1"]
        W2 = weights["W2"]

        return float(
            abs(y_pred["Vp"] - y_obs["Vp"])
            + W1 * abs(y_pred["Vs"] - y_obs["Vs"])
            + W2 * abs((1.0 / sigma_pred) - (1.0 / sigma_obs))
        )

    except Exception:
        return np.inf


def make_starting_points(n_starts, seed, dim):
    rng = np.random.default_rng(seed)

    starts = []

    # Always include midpoint.
    starts.append(np.full(dim, 0.5, dtype=float))

    for _ in range(n_starts - 1):
        starts.append(rng.random(dim))

    return starts


def run_symbolic_multistart_least_squares(
    surrogate,
    y_obs,
    bounds_dict,
    variable_order,
    weights,
    n_starts,
    seed,
    max_nfev,
    ftol,
    xtol,
    gtol,
):
    lo, hi = bounds_to_arrays(bounds_dict, variable_order)
    dim = len(variable_order)

    starts = make_starting_points(
        n_starts=n_starts,
        seed=seed,
        dim=dim,
    )

    lower_norm = np.zeros(dim, dtype=float)
    upper_norm = np.ones(dim, dtype=float)

    best = None
    all_runs = []

    for start_index, y0 in enumerate(starts):
        result = least_squares(
            fun=symbolic_residuals_from_y_norm,
            x0=y0,
            bounds=(lower_norm, upper_norm),
            args=(surrogate, y_obs, lo, hi, variable_order, weights),
            max_nfev=int(max_nfev),
            ftol=float(ftol),
            xtol=float(xtol),
            gtol=float(gtol),
        )

        scalar_loss = symbolic_scalar_loss_from_y_norm(
            y_norm=result.x,
            surrogate=surrogate,
            y_obs=y_obs,
            lo=lo,
            hi=hi,
            variable_order=variable_order,
            weights=weights,
        )

        x_hat = y_norm_to_x_dict(
            y_norm=result.x,
            lo=lo,
            hi=hi,
            variable_order=variable_order,
        )

        run_info = {
            "start_index": start_index,
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "symbolic_scalar_loss": float(scalar_loss),
            "x_hat": x_hat,
        }

        all_runs.append(run_info)

        if best is None or scalar_loss < best["symbolic_scalar_loss"]:
            best = run_info

    return best, all_runs