# scripts/05_test_parameter_utils.py

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.config_utils import load_yaml
from src.parameter_utils import (
    x_dict_to_vector,
    x_vector_to_dict,
    bounds_dict_to_list,
)


EXPERIMENT_PATH = ROOT / "config" / "single_experiment.yaml"


def main():
    config = load_yaml(EXPERIMENT_PATH)

    variable_order = config["variables"]["order"]
    x_true = config["true_parameters"]
    bounds = config["bounds"]

    x_vector = x_dict_to_vector(x_true, variable_order)
    x_reconstructed = x_vector_to_dict(x_vector, variable_order)
    bounds_list = bounds_dict_to_list(bounds, variable_order)

    print("Variable order:")
    print(variable_order)

    print("\nOriginal x_true dictionary:")
    print(x_true)

    print("\nx_true as vector:")
    print(x_vector)

    print("\nReconstructed x_true dictionary:")
    print(x_reconstructed)

    print("\nBounds as optimizer list:")
    print(bounds_list)


if __name__ == "__main__":
    main()