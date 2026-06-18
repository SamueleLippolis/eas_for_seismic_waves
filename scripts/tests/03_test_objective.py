# scripts/03_test_objective.py

from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.forward_model import load_constants_from_yaml
from src.synthetic_data import generate_synthetic_data
from src.objective import objective


CONSTANTS_PATH = ROOT / "config" / "constants.yaml"
SYNTHETIC_CONFIG_PATH = ROOT / "config" / "synthetic_data.yaml"
OPTIMIZATION_CONFIG_PATH = ROOT / "config" / "optimization.yaml"


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    constants = load_constants_from_yaml(CONSTANTS_PATH)
    synthetic_config = load_yaml(SYNTHETIC_CONFIG_PATH)
    optimization_config = load_yaml(OPTIMIZATION_CONFIG_PATH)

    x_true = synthetic_config["x_true"]
    noise_level = synthetic_config["noise_level"]
    seed = synthetic_config["seed"]
    weights = optimization_config["weights"]

    y_obs, y_true, x_true = generate_synthetic_data(
        x_true=x_true,
        constants=constants,
        noise_level=noise_level,
        seed=seed,
    )

    x_candidate_good = x_true

    x_candidate_bad = {
        "phi": 0.50,
        "C": 0.60,
        "S_b": 0.30,
        "sigma_b": 1.0,
    }

    loss_good = objective(
        x=x_candidate_good,
        y_obs=y_obs,
        constants=constants,
        weights=weights,
    )

    loss_bad = objective(
        x=x_candidate_bad,
        y_obs=y_obs,
        constants=constants,
        weights=weights,
    )

    print("Observed synthetic data:")
    for key, value in y_obs.items():
        print(f"  {key}: {value:.6g}")

    print("\nLoss for true parameters:")
    print(f"  {loss_good:.6g}")

    print("\nLoss for bad candidate:")
    print(f"  {loss_bad:.6g}")


if __name__ == "__main__":
    main()