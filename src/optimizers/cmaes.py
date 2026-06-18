# src/optimizers/cmaes.py

import numpy as np


def normalize_from_bounds(x, bounds):
    """
    Map x from physical bounds to [0, 1].
    """
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    return (x - lows) / (highs - lows)


def denormalize_to_bounds(x_norm, bounds):
    """
    Map x from [0, 1] to physical bounds.
    """
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    return lows + x_norm * (highs - lows)


def run_cmaes(
    objective_function,
    bounds,
    seed=123,
    population_size=50,
    max_iter=500,
    sigma0=0.25,
):
    """
    Simple bounded CMA-ES-style optimizer.

    The optimizer works in normalized space [0, 1]^d.
    Candidate solutions are clipped to [0, 1] before evaluation.

    Parameters
    ----------
    objective_function : callable
        Function accepting a parameter vector in physical units.

    bounds : list of tuple
        Bounds for each variable in physical units.

    seed : int
        Random seed.

    population_size : int
        Number of candidate solutions sampled at each iteration.

    max_iter : int
        Number of iterations.

    sigma0 : float
        Initial step size in normalized space.

    Returns
    -------
    result : dict
        Dictionary with:
        - best_x_vector
        - best_loss
        - history
    """
    rng = np.random.default_rng(seed)

    n_dim = len(bounds)
    lambda_ = population_size
    mu = lambda_ // 2

    # Recombination weights
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights ** 2)

    # Strategy parameters
    c_sigma = (mu_eff + 2.0) / (n_dim + mu_eff + 5.0)
    d_sigma = (
        1.0
        + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n_dim + 1.0)) - 1.0)
        + c_sigma
    )
    c_c = (4.0 + mu_eff / n_dim) / (
        n_dim + 4.0 + 2.0 * mu_eff / n_dim
    )
    c1 = 2.0 / ((n_dim + 1.3) ** 2 + mu_eff)
    c_mu = min(
        1.0 - c1,
        2.0 * (mu_eff - 2.0 + 1.0 / mu_eff)
        / ((n_dim + 2.0) ** 2 + mu_eff),
    )

    expected_norm = np.sqrt(n_dim) * (
        1.0 - 1.0 / (4.0 * n_dim) + 1.0 / (21.0 * n_dim ** 2)
    )

    # Initial mean in normalized space
    mean = rng.uniform(0.0, 1.0, size=n_dim)

    sigma = sigma0
    covariance = np.eye(n_dim)

    path_sigma = np.zeros(n_dim)
    path_c = np.zeros(n_dim)

    best_x_vector = None
    best_loss = np.inf

    history = []

    for iteration in range(max_iter):
        # Eigen decomposition for sampling and inverse sqrt covariance
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-12)

        sqrt_cov = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
        inv_sqrt_cov = (
            eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
        )

        # Sample population in normalized space
        z = rng.normal(size=(lambda_, n_dim))
        y = z @ sqrt_cov.T
        candidates_norm = mean + sigma * y
        candidates_norm = np.clip(candidates_norm, 0.0, 1.0)

        candidates = np.array([
            denormalize_to_bounds(candidate_norm, bounds)
            for candidate_norm in candidates_norm
        ])

        losses = np.array([
            objective_function(candidate)
            for candidate in candidates
        ])

        order = np.argsort(losses)
        candidates_norm = candidates_norm[order]
        candidates = candidates[order]
        losses = losses[order]

        if losses[0] < best_loss:
            best_loss = float(losses[0])
            best_x_vector = candidates[0].copy()

        old_mean = mean.copy()

        # Recombine best mu candidates
        selected_norm = candidates_norm[:mu]
        mean = np.sum(weights[:, None] * selected_norm, axis=0)

        # Steps expressed relative to old mean
        selected_steps = (selected_norm - old_mean) / max(sigma, 1e-12)
        weighted_step = (mean - old_mean) / max(sigma, 1e-12)

        # Update evolution path for sigma
        path_sigma = (
            (1.0 - c_sigma) * path_sigma
            + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff)
            * (inv_sqrt_cov @ weighted_step)
        )

        norm_path_sigma = np.linalg.norm(path_sigma)

        h_sigma_condition = (
            norm_path_sigma
            / np.sqrt(1.0 - (1.0 - c_sigma) ** (2.0 * (iteration + 1)))
            / expected_norm
            < (1.4 + 2.0 / (n_dim + 1.0))
        )

        h_sigma = 1.0 if h_sigma_condition else 0.0

        # Update evolution path for covariance
        path_c = (
            (1.0 - c_c) * path_c
            + h_sigma
            * np.sqrt(c_c * (2.0 - c_c) * mu_eff)
            * weighted_step
        )

        # Rank-mu covariance update
        rank_mu_update = np.zeros((n_dim, n_dim))
        for weight, step in zip(weights, selected_steps):
            rank_mu_update += weight * np.outer(step, step)

        covariance = (
            (1.0 - c1 - c_mu) * covariance
            + c1
            * (
                np.outer(path_c, path_c)
                + (1.0 - h_sigma) * c_c * (2.0 - c_c) * covariance
            )
            + c_mu * rank_mu_update
        )

        covariance = 0.5 * (covariance + covariance.T)

        # Step-size adaptation
        sigma = sigma * np.exp(
            (c_sigma / d_sigma) * (norm_path_sigma / expected_norm - 1.0)
        )

        # Avoid extreme sigma values
        sigma = float(np.clip(sigma, 1e-6, 2.0))

        history.append({
            "iteration": iteration,
            "loss_best": float(best_loss),
            "loss_mean": float(np.mean(losses)),
            "loss_min_current": float(losses[0]),
            "sigma": float(sigma),
        })

    return {
        "best_x_vector": best_x_vector,
        "best_loss": float(best_loss),
        "history": history,
    }