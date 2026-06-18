# src/experiment_utils.py


def compute_parameter_errors(x_hat, x_true):
    """
    Compute parameter recovery errors.

    For percentage variables:
        error in percentage points.

    For sigma_b_inv and xi:
        relative percent error.
    """
    errors = {}

    percentage_variables = [
        "phi_percent",
        "C_percent",
        "S_b_percent",
    ]

    relative_variables = [
        "sigma_b_inv",
        "xi",
    ]

    for name in percentage_variables:
        error = x_hat[name] - x_true[name]

        errors[name] = {
            "error": error,
            "abs_error": abs(error),
            "unit": "percentage_points",
        }

    for name in relative_variables:
        error = 100.0 * (x_hat[name] - x_true[name]) / x_true[name]

        errors[name] = {
            "error": error,
            "abs_error": abs(error),
            "unit": "relative_percent",
        }

    return errors