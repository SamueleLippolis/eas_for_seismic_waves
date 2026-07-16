# src/symbolic_inverse.py

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import least_squares, differential_evolution


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


def x_dict_to_y_norm(x_dict, lo, hi, variable_order):
    x_values = np.array(
        [x_dict[name] for name in variable_order],
        dtype=float,
    )

    y_norm = (x_values - lo) / (hi - lo)
    return np.clip(y_norm, 0.0, 1.0)


def exact_residuals_from_y_norm(
    y_norm,
    y_obs,
    lo,
    hi,
    variable_order,
    weights,
    constants,
    exact_forward_model,
):
    try:
        x = y_norm_to_x_dict(
            y_norm=y_norm,
            lo=lo,
            hi=hi,
            variable_order=variable_order,
        )

        y_pred = exact_forward_model(x, constants)

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


def run_exact_refinement_from_x(
    x_start,
    y_obs,
    bounds_dict,
    variable_order,
    weights,
    constants,
    exact_forward_model,
    max_nfev,
    ftol,
    xtol,
    gtol,
):
    lo, hi = bounds_to_arrays(bounds_dict, variable_order)

    y0 = x_dict_to_y_norm(
        x_dict=x_start,
        lo=lo,
        hi=hi,
        variable_order=variable_order,
    )

    dim = len(variable_order)
    lower_norm = np.zeros(dim, dtype=float)
    upper_norm = np.ones(dim, dtype=float)

    result = least_squares(
        fun=exact_residuals_from_y_norm,
        x0=y0,
        bounds=(lower_norm, upper_norm),
        args=(
            y_obs,
            lo,
            hi,
            variable_order,
            weights,
            constants,
            exact_forward_model,
        ),
        max_nfev=int(max_nfev),
        ftol=float(ftol),
        xtol=float(xtol),
        gtol=float(gtol),
    )

    x_refined = y_norm_to_x_dict(
        y_norm=result.x,
        lo=lo,
        hi=hi,
        variable_order=variable_order,
    )

    return {
        "x_hat": x_refined,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
    }

def safe_symbolic_scalar_loss_from_y_norm(
    y_norm,
    surrogate,
    y_obs,
    lo,
    hi,
    variable_order,
    weights,
):
    try:
        loss = symbolic_scalar_loss_from_y_norm(
            y_norm=y_norm,
            surrogate=surrogate,
            y_obs=y_obs,
            lo=lo,
            hi=hi,
            variable_order=variable_order,
            weights=weights,
        )

        if not np.isfinite(loss):
            return 1.0e12

        return float(loss)

    except Exception:
        return 1.0e12


def run_symbolic_differential_evolution_with_polish(
    surrogate,
    y_obs,
    bounds_dict,
    variable_order,
    weights,
    optimizer_config,
):
    lo, hi = bounds_to_arrays(bounds_dict, variable_order)

    dim = len(variable_order)
    normalized_bounds = [(0.0, 1.0)] * dim

    de_result = differential_evolution(
        func=safe_symbolic_scalar_loss_from_y_norm,
        bounds=normalized_bounds,
        args=(
            surrogate,
            y_obs,
            lo,
            hi,
            variable_order,
            weights,
        ),
        maxiter=int(optimizer_config.get("maxiter", 150)),
        popsize=int(optimizer_config.get("popsize", 15)),
        tol=float(optimizer_config.get("tol", 1.0e-7)),
        seed=int(optimizer_config.get("seed", 123)),
        polish=False,
        updating="immediate",
        workers=1,
    )

    x_de = y_norm_to_x_dict(
        y_norm=de_result.x,
        lo=lo,
        hi=hi,
        variable_order=variable_order,
    )

    de_loss = safe_symbolic_scalar_loss_from_y_norm(
        y_norm=de_result.x,
        surrogate=surrogate,
        y_obs=y_obs,
        lo=lo,
        hi=hi,
        variable_order=variable_order,
        weights=weights,
    )

    de_run = {
        "stage": "differential_evolution",
        "start_index": 0,
        "x_hat": x_de,
        "symbolic_scalar_loss": float(de_loss),
        "success": bool(de_result.success),
        "message": str(de_result.message),
        "nfev": int(de_result.nfev),
    }

    polish_config = optimizer_config.get("least_squares_polish", {})
    polish_enabled = bool(polish_config.get("enabled", True))

    if not polish_enabled:
        return de_run, [de_run]

    ls_result = least_squares(
        fun=symbolic_residuals_from_y_norm,
        x0=de_result.x,
        bounds=(
            np.zeros(dim, dtype=float),
            np.ones(dim, dtype=float),
        ),
        args=(
            surrogate,
            y_obs,
            lo,
            hi,
            variable_order,
            weights,
        ),
        max_nfev=int(polish_config.get("max_nfev", 2000)),
        ftol=float(polish_config.get("ftol", 1.0e-10)),
        xtol=float(polish_config.get("xtol", 1.0e-10)),
        gtol=float(polish_config.get("gtol", 1.0e-10)),
    )

    x_polished = y_norm_to_x_dict(
        y_norm=ls_result.x,
        lo=lo,
        hi=hi,
        variable_order=variable_order,
    )

    polished_loss = safe_symbolic_scalar_loss_from_y_norm(
        y_norm=ls_result.x,
        surrogate=surrogate,
        y_obs=y_obs,
        lo=lo,
        hi=hi,
        variable_order=variable_order,
        weights=weights,
    )

    polished_run = {
        "stage": "least_squares_polish",
        "start_index": 1,
        "x_hat": x_polished,
        "symbolic_scalar_loss": float(polished_loss),
        "success": bool(ls_result.success),
        "status": int(ls_result.status),
        "message": str(ls_result.message),
        "nfev": int(ls_result.nfev),
        "cost": float(ls_result.cost),
    }

    return polished_run, [de_run, polished_run]