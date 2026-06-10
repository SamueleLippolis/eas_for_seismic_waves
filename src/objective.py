# src/objective.py

import numpy as np

from src.forward_model import forward_model


def objective(x, y_obs, constants, weights):
    """
    Objective function for the inverse problem.

    Parameters
    ----------
    x : dict
        Candidate parameters:
        - phi
        - C
        - S_b
        - sigma_b

    y_obs : dict
        Observed data:
        - Vp
        - Vs
        - sigma

    constants : dict
        Physical constants.

    weights : dict
        Objective weights:
        - W1
        - W2

    Returns
    -------
    loss : float
        Misfit between predicted and observed data.
    """
    y_pred = forward_model(x, constants)

    W1 = weights["W1"]
    W2 = weights["W2"]

    loss_vp = abs(y_pred["Vp"] - y_obs["Vp"])
    loss_vs = abs(y_pred["Vs"] - y_obs["Vs"])
    loss_sigma = abs((1.0 / y_pred["sigma"]) - (1.0 / y_obs["sigma"]))

    loss = loss_vp + W1 * loss_vs + W2 * loss_sigma

    return loss