# scripts/check_symbolic_surrogate.py

from pathlib import Path
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config_utils import load_yaml
from src.symbolic_surrogate import SymbolicForwardSurrogate


CONFIG_PATH = ROOT / "config" / "symbolic_surrogate.yaml"


def compute_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_true_vs_pred_plot(y_true, y_pred, target, path):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)

    min_value = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_value = max(float(np.max(y_true)), float(np.max(y_pred)))

    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

    plt.xlabel(f"True {target}")
    plt.ylabel(f"Symbolic surrogate {target}")
    plt.title(f"Symbolic surrogate check: {target}")
    plt.grid(True)

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    config = load_yaml(CONFIG_PATH)

    surrogate_name = config["surrogate"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = (
        ROOT
        / config["surrogate"]["output_dir"]
        / f"{surrogate_name}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = ROOT / config["dataset"]["path"]
    df = pd.read_parquet(dataset_path)

    test_df = df[df["split"] == "test"].copy()

    surrogate = SymbolicForwardSurrogate.from_config(
        config=config,
        root=ROOT,
    )

    pred_df = surrogate.predict_dataframe(test_df)

    metrics = {}

    for target in ["Vp", "Vs", "sigma"]:
        y_true = test_df[target].to_numpy(dtype=float)
        y_pred = pred_df[target].to_numpy(dtype=float)

        metrics[target] = compute_metrics(y_true, y_pred)

        save_true_vs_pred_plot(
            y_true=y_true,
            y_pred=y_pred,
            target=target,
            path=output_dir / f"{target}_true_vs_pred.png",
        )

    pred_out = test_df[
        [
            "sample_id",
            "phi_percent",
            "C_percent",
            "S_b_percent",
            "sigma_b_inv",
            "xi",
            "Vp",
            "Vs",
            "sigma",
        ]
    ].copy()

    pred_out["Vp_hat"] = pred_df["Vp"].to_numpy()
    pred_out["Vs_hat"] = pred_df["Vs"].to_numpy()
    pred_out["sigma_hat"] = pred_df["sigma"].to_numpy()

    pred_out["Vp_residual"] = pred_out["Vp_hat"] - pred_out["Vp"]
    pred_out["Vs_residual"] = pred_out["Vs_hat"] - pred_out["Vs"]
    pred_out["sigma_residual"] = pred_out["sigma_hat"] - pred_out["sigma"]

    pred_out.to_parquet(output_dir / "test_predictions.parquet", index=False)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nSymbolic surrogate check completed.")
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))

    print("\nSaved to:")
    print(f"  {output_dir}")


if __name__ == "__main__":
    main()