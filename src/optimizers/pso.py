# src/optimizers/pso.py

import numpy as np

from src.optimizers.common import (
    sample_uniform,
    clip_to_bounds,
    bounds_width,
)


def run_pso(
    objective_function,
    bounds,
    seed=123,
    n_particles=100,
    max_iter=500,
    inertia_weight=0.7,
    cognitive_weight=1.5,
    social_weight=1.5,
    velocity_scale=0.10,
):
    """
    Particle Swarm Optimization for bounded continuous variables.

    Each particle is a candidate solution.
    Particles move according to:
        - their current velocity
        - their own best position
        - the global best position found by the swarm
    """
    rng = np.random.default_rng(seed)

    n_dimensions = len(bounds)
    widths = bounds_width(bounds)

    positions = np.array([
        sample_uniform(bounds, rng)
        for _ in range(n_particles)
    ])

    velocities = rng.normal(
        loc=0.0,
        scale=velocity_scale * widths,
        size=(n_particles, n_dimensions),
    )

    losses = np.array([
        objective_function(position)
        for position in positions
    ])

    personal_best_positions = positions.copy()
    personal_best_losses = losses.copy()

    best_index = np.argmin(personal_best_losses)
    global_best_position = personal_best_positions[best_index].copy()
    global_best_loss = personal_best_losses[best_index]

    history = []

    for iteration in range(max_iter):
        r1 = rng.random(size=(n_particles, n_dimensions))
        r2 = rng.random(size=(n_particles, n_dimensions))

        cognitive_component = (
            cognitive_weight
            * r1
            * (personal_best_positions - positions)
        )

        social_component = (
            social_weight
            * r2
            * (global_best_position - positions)
        )

        velocities = (
            inertia_weight * velocities
            + cognitive_component
            + social_component
        )

        positions = positions + velocities
        positions = np.array([
            clip_to_bounds(position, bounds)
            for position in positions
        ])

        losses = np.array([
            objective_function(position)
            for position in positions
        ])

        improved = losses < personal_best_losses
        personal_best_positions[improved] = positions[improved]
        personal_best_losses[improved] = losses[improved]

        best_index = np.argmin(personal_best_losses)

        if personal_best_losses[best_index] < global_best_loss:
            global_best_loss = personal_best_losses[best_index]
            global_best_position = personal_best_positions[best_index].copy()

        history.append({
            "iteration": iteration,
            "loss_best": float(global_best_loss),
            "loss_mean": float(np.mean(losses)),
            "loss_min_current": float(np.min(losses)),
        })

    return {
        "best_x_vector": global_best_position,
        "best_loss": float(global_best_loss),
        "history": history,
    }