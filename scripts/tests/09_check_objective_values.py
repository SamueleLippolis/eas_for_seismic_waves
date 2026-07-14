# scripts/tests/09_check_objective_values.py

from pathlib import Path
import sys
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.config_utils import load_yaml
from src.forward_model import forward_model
from src.objective import objective


CONSTANTS_PATH = ROOT / "config" / "constants.yaml"
EXPERIMENT_PATH = ROOT / "config" / "giulio_part1.yaml"


X_TEST = [
    {
        "name": "true_C10",
        "x": {
            "phi_percent": 41.0,
            "C_percent": 10.0,
            "S_b_percent": 90.0,
            "sigma_b_inv": 0.5,
            "xi": 2.7,
        },
    },
    {
        "name": "midpoint_phase2",
        "x": {
            "phi_percent": 35.0,
            "C_percent": 50.0,
            "S_b_percent": 90.0,
            "sigma_b_inv": 12.6,
            "xi": 2.7,
        },
    },
    {
        "name": "bad_candidate",
        "x": {
            "phi_percent": 60.0,
            "C_percent": 80.0,
            "S_b_percent": 82.0,
            "sigma_b_inv": 10.0,
            "xi": 2.85,
        },
    },
    {
        "name": "near_true",
        "x": {
            "phi_percent": 42.0,
            "C_percent": 12.0,
            "S_b_percent": 91.0,
            "sigma_b_inv": 0.6,
            "xi": 2.72,
        },
    },
    {
        "name": "boundary_like",
        "x": {
            "phi_percent": 5.0,
            "C_percent": 100.0,
            "S_b_percent": 100.0,
            "sigma_b_inv": 25.0,
            "xi": 2.9,
        },
    },
]


def main():
    constants = load_yaml(CONSTANTS_PATH)
    config = load_yaml(EXPERIMENT_PATH)

    weights = config["objective"]["weights"]

    base_true_parameters = config["base_true_parameters"]
    clay_variable = config["clay_sweep"]["variable"]
    clay_values = config["clay_sweep"]["values"]

    seed_noise = config["synthetic_data"]["seed"]
    relative_std = config["synthetic_data"]["noise"]["relative_std"]

    rng = np.random.default_rng(seed_noise)

    rows = []

    for C_true in clay_values:
        x_true = dict(base_true_parameters)
        x_true[clay_variable] = C_true

        y_true = forward_model(x_true, constants)

        y_obs = {
            "Vp": float(y_true["Vp"]) * (1.0 + rng.normal(0.0, relative_std)),
            "Vs": float(y_true["Vs"]) * (1.0 + rng.normal(0.0, relative_std)),
            "sigma": float(y_true["sigma"]) * (1.0 + rng.normal(0.0, relative_std)),
        }

        for item in X_TEST:
            loss = objective(
                x=item["x"],
                y_obs=y_obs,
                constants=constants,
                weights=weights,
            )

            row = {
                "C_true": C_true,
                "candidate_name": item["name"],
                **item["x"],
                "Vp_obs": y_obs["Vp"],
                "Vs_obs": y_obs["Vs"],
                "sigma_obs": y_obs["sigma"],
                "objective": float(loss),
            }

            rows.append(row)

            print(
                f"C_true={C_true:5.1f} | "
                f"{item['name']:16s} | "
                f"objective={loss:.12g}"
            )

    output_path = ROOT / "reports" / "our_objective_check.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved our objective check to:")
    print(output_path)


if __name__ == "__main__":
    main()