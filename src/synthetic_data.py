# src/synthetic_data.py

import numpy as np

from src.forward_model import forward_model


def generate_synthetic_data(x_true, constants, noise_level=0.02, seed=42):
    """
    Generate synthetic observed data from the forward model.

    Parameters
    ----------
    x_true : dict
        True hidden parameters:
        - phi
        - C
        - S_b
        - sigma_b

    constants : dict
        Physical constants.

    noise_level : float
        Relative Gaussian noise level.
        Example: 0.02 means 2% noise.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    y_obs : dict
        Noisy observations:
        - Vp
        - Vs
        - sigma

    y_true : dict
        Noise-free forward model output.

    x_true : dict
        True hidden parameters.
    """
    rng = np.random.default_rng(seed)

    y_true = forward_model(x_true, constants)

    y_obs = {}

    for key in ["Vp", "Vs", "sigma"]:
        true_value = y_true[key]
        noise = rng.normal(loc=0.0, scale=noise_level * true_value)
        y_obs[key] = true_value + noise

    return y_obs, y_true, x_true