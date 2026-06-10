# scripts/02_test_synthetic_data.py

from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.forward_model import load_constants_from_yaml
from src.synthetic_data import generate_synthetic_data


CONSTANTS_PATH = ROOT / "config" / "constants.yaml"
SYNTHETIC_CONFIG_PATH = ROOT / "config" / "synthetic_data.yaml"


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    constants = load_constants_from_yaml(CONSTANTS_PATH)
    synthetic_config = load_yaml(SYNTHETIC_CONFIG_PATH)

    x_true = synthetic_config["x_true"]
    noise_level = synthetic_config["noise_level"]
    seed = synthetic_config["seed"]

    y_obs, y_true, x_true = generate_synthetic_data(
        x_true=x_true,
        constants=constants,
        noise_level=noise_level,
        seed=seed,
    )

    print("Synthetic data configuration:")
    print(f"  seed: {seed}")
    print(f"  noise_level: {noise_level}")

    print("\nTrue parameters:")
    for key, value in x_true.items():
        print(f"  {key}: {value}")

    print("\nNoise-free output:")
    for key, value in y_true.items():
        if key in ["Vp", "Vs", "sigma"]:
            print(f"  {key}: {value:.6g}")

    print("\nNoisy synthetic observations:")
    for key, value in y_obs.items():
        print(f"  {key}: {value:.6g}")


if __name__ == "__main__":
    main()