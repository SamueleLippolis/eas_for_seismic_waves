# scripts/01_test_forward_model.py

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.forward_model import load_constants_from_yaml, forward_model


CONFIG_PATH = ROOT / "config" / "constants.yaml"


def main():
    constants = load_constants_from_yaml(CONFIG_PATH)

    x = {
        "phi": 0.35,
        "C": 0.20,
        "S_b": 0.80,
        "sigma_b": 4.0,
    }

    output = forward_model(x, constants)

    print("Input parameters:")
    for key, value in x.items():
        print(f"  {key}: {value}")

    print("\nForward model output:")
    for key, value in output.items():
        print(f"  {key}: {value:.6g}")


if __name__ == "__main__":
    main()