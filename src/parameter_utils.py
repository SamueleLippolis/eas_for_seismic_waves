# src/parameter_utils.py


def x_dict_to_vector(x_dict, variable_order):
    """
    Convert a parameter dictionary into a vector following variable_order.
    """
    return [x_dict[name] for name in variable_order]


def x_vector_to_dict(x_vector, variable_order):
    """
    Convert a parameter vector into a dictionary following variable_order.
    """
    return {
        name: float(value)
        for name, value in zip(variable_order, x_vector)
    }


def bounds_dict_to_list(bounds_dict, variable_order):
    """
    Convert bounds from dictionary format to list format.

    Example
    -------
    bounds_dict = {
        "phi_percent": [0.0, 70.0],
        "C_percent": [0.0, 100.0],
    }

    variable_order = ["phi_percent", "C_percent"]

    output = [(0.0, 70.0), (0.0, 100.0)]
    """
    return [
        tuple(bounds_dict[name])
        for name in variable_order
    ]