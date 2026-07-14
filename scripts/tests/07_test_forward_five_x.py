# scripts/tests/07_test_forward_five_x.py

from pathlib import Path
import sys
import csv

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.config_utils import load_yaml
from src.forward_model import forward_model


CONSTANTS_PATH = ROOT / "config" / "constants.yaml"


TEST_X = [
    {
        "name": "baseline_C10",
        "phi_percent": 41.0,
        "C_percent": 10.0,
        "S_b_percent": 90.0,
        "sigma_b_inv": 0.5,
        "xi": 2.7,
    },
    {
        "name": "low_phi_C20",
        "phi_percent": 25.0,
        "C_percent": 20.0,
        "S_b_percent": 85.0,
        "sigma_b_inv": 1.0,
        "xi": 2.6,
    },
    {
        "name": "mid_C50",
        "phi_percent": 45.0,
        "C_percent": 50.0,
        "S_b_percent": 90.0,
        "sigma_b_inv": 2.0,
        "xi": 2.7,
    },
    {
        "name": "high_C80",
        "phi_percent": 35.0,
        "C_percent": 80.0,
        "S_b_percent": 95.0,
        "sigma_b_inv": 0.8,
        "xi": 2.8,
    },
    {
        "name": "high_phi_C100",
        "phi_percent": 60.0,
        "C_percent": 100.0,
        "S_b_percent": 98.0,
        "sigma_b_inv": 5.0,
        "xi": 2.9,
    },
]


def main():
    constants = load_yaml(CONSTANTS_PATH)

    rows = []

    for x in TEST_X:
        x_model = {k: v for k, v in x.items() if k != "name"}
        y = forward_model(x_model, constants)

        row = {
            "name": x["name"],
            **x_model,
            "Vp": float(y["Vp"]),
            "Vs": float(y["Vs"]),
            "sigma": float(y["sigma"]),
        }

        rows.append(row)

        print(f"\n{x['name']}")
        print("x:")
        for key, value in x_model.items():
            print(f"  {key}: {value}")

        print("y:")
        print(f"  Vp:    {y['Vp']:.10g}")
        print(f"  Vs:    {y['Vs']:.10g}")
        print(f"  sigma: {y['sigma']:.10g}")

    output_path = ROOT / "reports" / "forward_five_x_test.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved to:")
    print(f"  {output_path}")


if __name__ == "__main__":
    main()