# src/optimizers/common.py

import numpy as np

from src.objective import objective
from src.parameter_utils import x_vector_to_dict


def make_vector_objective(y_obs, constants, weights, variable_order):
    """
    Create an objective function that accepts a vector instead of a dictionary.
    """

    def vector_objective(x_vector):
        x_dict = x_vector_to_dict(x_vector, variable_order)

        return objective(
            x=x_dict,
            y_obs=y_obs,
            constants=constants,
            weights=weights,
        )

    return vector_objective


def sample_uniform(bounds, rng):
    """
    Sample one random vector uniformly inside bounds.
    """
    return np.array([
        rng.uniform(low, high)
        for low, high in bounds
    ])


def clip_to_bounds(x, bounds):
    """
    Clip a vector inside bounds.
    """
    lows = np.array([bound[0] for bound in bounds])
    highs = np.array([bound[1] for bound in bounds])

    return np.clip(x, lows, highs)


def bounds_width(bounds):
    """
    Return the width of each variable range.
    """
    return np.array([
        high - low
        for low, high in bounds
    ])