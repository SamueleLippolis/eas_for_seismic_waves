# scripts/tests/06_check_forward_and_noise_sequence.py

from pathlib import Path
import sys
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.config_utils import load_yaml
from src.forward_model import forward_model


CONSTANTS_PATH = ROOT / "config" / "constants.yaml"
EXPERIMENT_PATH = ROOT / "config" / "giulio_part1.yaml"


def main():
    constants = load_yaml(CONSTANTS_PATH)
    config = load_yaml(EXPERIMENT_PATH)

    base_true_parameters = config["base_true_parameters"]
    clay_variable = config["clay_sweep"]["variable"]
    clay_values = config["clay_sweep"]["values"]

    noise_seed = config["synthetic_data"]["seed"]
    relative_std = config["synthetic_data"]["noise"]["relative_std"]

    rng = np.random.default_rng(noise_seed)

    rows = []

    for idx, C_true in enumerate(clay_values):
        x_true = dict(base_true_parameters)
        x_true[clay_variable] = C_true

        y_true = forward_model(x_true, constants)

        eps_vp = rng.normal(0.0, relative_std)
        eps_vs = rng.normal(0.0, relative_std)
        eps_sigma = rng.normal(0.0, relative_std)

        y_obs = {
            "Vp": y_true["Vp"] * (1.0 + eps_vp),
            "Vs": y_true["Vs"] * (1.0 + eps_vs),
            "sigma": y_true["sigma"] * (1.0 + eps_sigma),
        }

        row = {
            "idx": idx,
            "C_true": C_true,
            "Vp_true": y_true["Vp"],
            "Vs_true": y_true["Vs"],
            "sigma_true": y_true["sigma"],
            "eps_vp": eps_vp,
            "eps_vs": eps_vs,
            "eps_sigma": eps_sigma,
            "Vp_obs": y_obs["Vp"],
            "Vs_obs": y_obs["Vs"],
            "sigma_obs": y_obs["sigma"],
        }

        rows.append(row)

        print(
            f"C={C_true:5.1f} | "
            f"Vp_true={y_true['Vp']:.6g}, "
            f"Vs_true={y_true['Vs']:.6g}, "
            f"sigma_true={y_true['sigma']:.6g} | "
            f"Vp_obs={y_obs['Vp']:.6g}, "
            f"Vs_obs={y_obs['Vs']:.6g}, "
            f"sigma_obs={y_obs['sigma']:.6g}"
        )

    output_path = ROOT / "reports" / "forward_noise_sequence_check.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved check table to:")
    print(f"  {output_path}")


if __name__ == "__main__":
    main()