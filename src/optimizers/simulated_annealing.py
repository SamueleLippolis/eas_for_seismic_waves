# src/optimizers/simulated_annealing.py

import numpy as np

from src.optimizers.common import (
    sample_uniform,
    clip_to_bounds,
    bounds_width,
)


def run_simulated_annealing(
    objective_function,
    bounds,
    seed=123,
    max_iter=5000,
    initial_temperature=1000.0,
    final_temperature=0.001,
    step_scale=0.10,
):
    """
    Simple Simulated Annealing optimizer for bounded continuous variables.
    """
    rng = np.random.default_rng(seed)

    x_current = sample_uniform(bounds, rng)
    loss_current = objective_function(x_current)

    x_best = x_current.copy()
    loss_best = loss_current

    widths = bounds_width(bounds)
    step_sizes = step_scale * widths

    history = []

    for iteration in range(max_iter):
        progress = iteration / max(max_iter - 1, 1)

        temperature = initial_temperature * (
            final_temperature / initial_temperature
        ) ** progress

        proposal = x_current + rng.normal(
            loc=0.0,
            scale=step_sizes,
            size=len(bounds),
        )

        proposal = clip_to_bounds(proposal, bounds)
        loss_proposal = objective_function(proposal)

        delta = loss_proposal - loss_current

        if delta < 0:
            accept = True
        else:
            accept_probability = np.exp(-delta / max(temperature, 1e-12))
            accept = rng.random() < accept_probability

        if accept:
            x_current = proposal
            loss_current = loss_proposal

        if loss_current < loss_best:
            x_best = x_current.copy()
            loss_best = loss_current

        history.append({
            "iteration": iteration,
            "temperature": float(temperature),
            "loss_current": float(loss_current),
            "loss_best": float(loss_best),
        })

    return {
        "best_x_vector": x_best,
        "best_loss": float(loss_best),
        "history": history,
    }