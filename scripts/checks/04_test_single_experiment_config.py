# scripts/04_test_single_experiment_config.py

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.config_utils import load_yaml
from src.synthetic_data import generate_synthetic_data
from src.objective import objective


CONSTANTS_PATH = ROOT / "config" / "constants.yaml"
EXPERIMENT_PATH = ROOT / "config" / "single_experiment.yaml"


def main():
    constants = load_yaml(CONSTANTS_PATH)
    config = load_yaml(EXPERIMENT_PATH)

    x_true = config["true_parameters"]
    seed = config["synthetic_data"]["seed"]
    noise_config = config["synthetic_data"]["noise"]
    weights = config["objective"]["weights"]
    bounds = config["bounds"]

    y_obs, y_true, _ = generate_synthetic_data(
        x_true=x_true,
        constants=constants,
        noise_config=noise_config,
        seed=seed,
    )

    loss_true = objective(
        x=x_true,
        y_obs=y_obs,
        constants=constants,
        weights=weights,
    )

    x_bad = {
        "phi_percent": 60.0,
        "C_percent": 80.0,
        "S_b_percent": 82.0,
        "sigma_b_inv": 10.0,
        "xi": 2.85,
    }

    loss_bad = objective(
        x=x_bad,
        y_obs=y_obs,
        constants=constants,
        weights=weights,
    )

    print("Experiment name:")
    print(f"  {config['experiment']['name']}")

    print("\nTrue parameters:")
    for key, value in x_true.items():
        print(f"  {key}: {value}")

    print("\nBounds:")
    for key, value in bounds.items():
        print(f"  {key}: {value}")

    print("\nNoise-free output:")
    for key in ["Vp", "Vs", "sigma"]:
        print(f"  {key}: {y_true[key]:.6g}")

    print("\nNoisy observations:")
    for key in ["Vp", "Vs", "sigma"]:
        print(f"  {key}: {y_obs[key]:.6g}")

    print("\nObjective check:")
    print(f"  loss at true x: {loss_true:.6g}")
    print(f"  loss at bad x:  {loss_bad:.6g}")


if __name__ == "__main__":
    main()