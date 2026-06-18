# src/objective.py

import numpy as np

from src.forward_model import forward_model


def objective(x, y_obs, constants, weights):
    """
    Objective function for the inverse problem.
    """
    try:
        y_pred = forward_model(x, constants)

        W1 = weights["W1"]
        W2 = weights["W2"]

        loss_vp = abs(y_pred["Vp"] - y_obs["Vp"])
        loss_vs = abs(y_pred["Vs"] - y_obs["Vs"])
        loss_sigma = abs((1.0 / y_pred["sigma"]) - (1.0 / y_obs["sigma"]))

        loss = loss_vp + W1 * loss_vs + W2 * loss_sigma

        if not np.isfinite(loss):
            return np.inf

        return loss

    except Exception:
        return np.inf