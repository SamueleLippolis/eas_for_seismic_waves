# src/synthetic_data.py

import numpy as np

from src.forward_model import forward_model


def generate_synthetic_data(x_true, constants, noise_config, seed):
    """
    Generate synthetic observed data from the forward model.

    The noise used here follows Giulio's code:

        y_obs = y_true * (1 + epsilon)

    where:

        epsilon ~ Normal(0, relative_std)
    """
    rng = np.random.default_rng(seed)

    y_true = forward_model(x_true, constants)

    noise_type = noise_config["type"]
    relative_std = noise_config["relative_std"]

    if noise_type != "relative_gaussian":
        raise ValueError(f"Unsupported noise type: {noise_type}")

    y_obs = {}

    for key in ["Vp", "Vs", "sigma"]:
        epsilon = rng.normal(loc=0.0, scale=relative_std)
        y_obs[key] = y_true[key] * (1.0 + epsilon)

    return y_obs, y_true, x_true


def generate_synthetic_data_from_rng(x_true, constants, noise_config, rng):
    """
    Generate synthetic observed data using an existing RNG.

    This matches Giulio's logic when the same RNG is initialized once
    before the clay-content loop and then consumed sequentially.
    """
    y_true = forward_model(x_true, constants)

    noise_type = noise_config["type"]
    relative_std = noise_config["relative_std"]

    if noise_type != "relative_gaussian":
        raise ValueError(f"Unsupported noise type: {noise_type}")

    y_obs = {}

    for key in ["Vp", "Vs", "sigma"]:
        epsilon = rng.normal(loc=0.0, scale=relative_std)
        y_obs[key] = y_true[key] * (1.0 + epsilon)

    return y_obs, y_true, x_true