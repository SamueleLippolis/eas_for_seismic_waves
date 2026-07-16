from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_phi(model, vp, vs, sigma):
    X = pd.DataFrame(
        {
            "Vp": np.asarray(vp, dtype=float),
            "Vs": np.asarray(vs, dtype=float),
            "sigma": np.asarray(sigma, dtype=float),
        }
    )

    try:
        pred = model.predict(X)
    except Exception:
        pred = model.predict(X.to_numpy())

    return np.asarray(pred, dtype=float)


def regression_metrics(y_true, y_pred):
    err = y_pred - y_true
    abs_err = np.abs(err)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs_error": float(np.max(abs_err)),
        "mean_error": float(np.mean(err)),
        "r2": None if ss_tot == 0 else float(1.0 - ss_res / ss_tot),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to Giulio/SA summary_results.csv containing Vp_obs, Vs_obs, sigma_obs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to phi_from_y/model.pkl.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for csv, metrics and plots.",
    )
    parser.add_argument(
        "--clip",
        action="store_true",
        help="Clip predictions to physical phi bounds [0, 70].",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    model_path = Path(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)
    model = load_model(model_path)

    required_cols = [
        "C_true",
        "Vp_obs",
        "Vs_obs",
        "sigma_obs",
        "Vp_true",
        "Vs_true",
        "sigma_true",
        "phi_percent_true",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in summary file: {missing}")

    phi_true = df["phi_percent_true"].to_numpy(dtype=float)

    phi_pred_obs = predict_phi(
        model=model,
        vp=df["Vp_obs"],
        vs=df["Vs_obs"],
        sigma=df["sigma_obs"],
    )

    phi_pred_clean = predict_phi(
        model=model,
        vp=df["Vp_true"],
        vs=df["Vs_true"],
        sigma=df["sigma_true"],
    )

    if args.clip:
        phi_pred_obs = np.clip(phi_pred_obs, 0.0, 70.0)
        phi_pred_clean = np.clip(phi_pred_clean, 0.0, 70.0)

    result_df = pd.DataFrame(
        {
            "C_true": df["C_true"],
            "phi_true": phi_true,
            "phi_pred_from_clean_y": phi_pred_clean,
            "phi_error_clean": phi_pred_clean - phi_true,
            "phi_abs_error_clean": np.abs(phi_pred_clean - phi_true),
            "phi_pred_from_noisy_y_obs": phi_pred_obs,
            "phi_error_noisy": phi_pred_obs - phi_true,
            "phi_abs_error_noisy": np.abs(phi_pred_obs - phi_true),
            "Vp_obs": df["Vp_obs"],
            "Vs_obs": df["Vs_obs"],
            "sigma_obs": df["sigma_obs"],
        }
    )

    metrics = {
        "clean_y": regression_metrics(phi_true, phi_pred_clean),
        "noisy_y_obs": regression_metrics(phi_true, phi_pred_obs),
        "n": int(len(result_df)),
        "model_path": str(model_path),
        "summary_path": str(summary_path),
        "clipped_to_bounds": bool(args.clip),
    }

    result_df.to_csv(out_dir / "phi_from_y_noisy_giulio_check.csv", index=False)

    with open(out_dir / "phi_from_y_noisy_giulio_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nPhi inverse check on Giulio sweep")
    print("--------------------------------")
    print(f"Summary: {summary_path}")
    print(f"Model:   {model_path}")
    print(f"Output:  {out_dir}")

    print("\nMetrics using clean y_true:")
    for k, v in metrics["clean_y"].items():
        print(f"  {k}: {v}")

    print("\nMetrics using noisy y_obs:")
    for k, v in metrics["noisy_y_obs"].items():
        print(f"  {k}: {v}")

    print("\nRows:")
    print(
        result_df[
            [
                "C_true",
                "phi_true",
                "phi_pred_from_noisy_y_obs",
                "phi_abs_error_noisy",
                "phi_pred_from_clean_y",
                "phi_abs_error_clean",
            ]
        ].to_string(index=False)
    )

    # Plot 1: predicted phi vs clay
    plt.figure(figsize=(8, 5))
    plt.plot(result_df["C_true"], result_df["phi_true"], marker="o", label="phi true")
    plt.plot(
        result_df["C_true"],
        result_df["phi_pred_from_noisy_y_obs"],
        marker="o",
        label="phi pred from noisy y_obs",
    )
    plt.plot(
        result_df["C_true"],
        result_df["phi_pred_from_clean_y"],
        marker="o",
        label="phi pred from clean y_true",
    )
    plt.xlabel("C true")
    plt.ylabel("phi")
    plt.title("Phi prediction from y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "phi_prediction_vs_clay.png", dpi=200)
    plt.close()

    # Plot 2: absolute error vs clay
    plt.figure(figsize=(8, 5))
    plt.plot(
        result_df["C_true"],
        result_df["phi_abs_error_noisy"],
        marker="o",
        label="abs error from noisy y_obs",
    )
    plt.plot(
        result_df["C_true"],
        result_df["phi_abs_error_clean"],
        marker="o",
        label="abs error from clean y_true",
    )
    plt.xlabel("C true")
    plt.ylabel("absolute phi error percentage points")
    plt.title("Phi absolute error from y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "phi_abs_error_vs_clay.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()