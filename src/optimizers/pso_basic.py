# src/optimizers/pso.py

import numpy as np


def denormalize_to_bounds(y, bounds):
    """
    Map y from [0, 1]^d to physical bounds.
    """
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    return lows + y * (highs - lows)


def reflect_to_unit_box(y):
    """
    Reflect positions into [0, 1].

    This is different from clipping:
    if a particle goes outside the box, it bounces back.
    """
    y_reflected = np.mod(y, 2.0)
    y_reflected = np.where(y_reflected > 1.0, 2.0 - y_reflected, y_reflected)
    return y_reflected


def run_pso(
    objective_function,
    bounds,
    seed=369,
    n_particles=1996,
    max_iter=500,
    w_start=0.2,
    w_end=0.02,
    c_social=1.49445,
    c_cognitive=1.49445,
    vmax_abs=0.02,
    init_mode="midpoint_gaussian",
    init_sigma_y=0.02,
    boundary_handling="reflect",
    tol=1e-9,
    **kwargs,
):
    """
    Giulio-style PSO in normalized [0, 1]^d space.

    The objective function receives vectors in physical units.
    The swarm evolves in normalized space.
    """
    rng = np.random.default_rng(seed)

    n_dimensions = len(bounds)

    if init_mode == "midpoint_gaussian":
        positions = 0.5 + rng.normal(
            loc=0.0,
            scale=init_sigma_y,
            size=(n_particles, n_dimensions),
        )
    elif init_mode == "uniform":
        positions = rng.uniform(
            low=0.0,
            high=1.0,
            size=(n_particles, n_dimensions),
        )
    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")

    if boundary_handling == "reflect":
        positions = reflect_to_unit_box(positions)
    else:
        positions = np.clip(positions, 0.0, 1.0)

    velocities = rng.normal(
        loc=0.0,
        scale=init_sigma_y,
        size=(n_particles, n_dimensions),
    )

    velocities = np.clip(velocities, -vmax_abs, vmax_abs)

    def evaluate_position(y):
        x = denormalize_to_bounds(y, bounds)
        return objective_function(x)

    losses = np.array([
        evaluate_position(position)
        for position in positions
    ])

    personal_best_positions = positions.copy()
    personal_best_losses = losses.copy()

    best_index = np.argmin(personal_best_losses)
    global_best_position = personal_best_positions[best_index].copy()
    global_best_loss = float(personal_best_losses[best_index])

    history = []

    for iteration in range(max_iter):
        progress = iteration / max(max_iter - 1, 1)
        inertia_weight = w_start + progress * (w_end - w_start)

        r1 = rng.random(size=(n_particles, n_dimensions))
        r2 = rng.random(size=(n_particles, n_dimensions))

        cognitive_component = (
            c_cognitive
            * r1
            * (personal_best_positions - positions)
        )

        social_component = (
            c_social
            * r2
            * (global_best_position - positions)
        )

        velocities = (
            inertia_weight * velocities
            + cognitive_component
            + social_component
        )

        velocities = np.clip(velocities, -vmax_abs, vmax_abs)

        positions = positions + velocities

        if boundary_handling == "reflect":
            positions = reflect_to_unit_box(positions)
        else:
            positions = np.clip(positions, 0.0, 1.0)

        losses = np.array([
            evaluate_position(position)
            for position in positions
        ])

        improved = losses < personal_best_losses
        personal_best_positions[improved] = positions[improved]
        personal_best_losses[improved] = losses[improved]

        best_index = np.argmin(personal_best_losses)

        if personal_best_losses[best_index] < global_best_loss:
            global_best_loss = float(personal_best_losses[best_index])
            global_best_position = personal_best_positions[best_index].copy()

        history.append({
            "iteration": iteration,
            "loss_best": float(global_best_loss),
            "loss_mean": float(np.mean(losses)),
            "loss_min_current": float(np.min(losses)),
            "inertia_weight": float(inertia_weight),
        })

        if global_best_loss <= tol:
            break

    best_x_vector = denormalize_to_bounds(global_best_position, bounds)

    return {
        "best_x_vector": best_x_vector,
        "best_loss": float(global_best_loss),
        "history": history,
    }