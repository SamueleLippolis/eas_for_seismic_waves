# scripts/run_symbolic_inverse.py

from pathlib import Path
import sys
import json
from datetime import datetime
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config_utils import load_yaml
from src.synthetic_data import generate_synthetic_data_from_rng
from src.forward_model import forward_model
from src.objective import objective
from src.experiment_utils import compute_parameter_errors
from src.symbolic_surrogate import SymbolicForwardSurrogate
from src.symbolic_inverse import run_symbolic_multistart_least_squares, run_exact_refinement_from_x



CONFIG_PATH = ROOT / "config" / "symbolic_inverse.yaml"
CONSTANTS_PATH = ROOT / "config" / "constants.yaml"


def flatten_errors(errors):
    flat = {}

    for parameter_name, values in errors.items():
        unit = values["unit"]

        flat[f"{parameter_name}_error_{unit}"] = float(values["error"])
        flat[f"{parameter_name}_abs_error_{unit}"] = float(values["abs_error"])

    return flat

def rerank_symbolic_candidates_by_exact_loss(
    all_runs,
    surrogate,
    y_obs,
    constants,
    weights,
):
    reranked_runs = []

    for run in all_runs:
        x_candidate = run["x_hat"]

        y_hat_symbolic = surrogate.predict_one(x_candidate)
        y_hat_exact = forward_model(x_candidate, constants)

        exact_loss = objective(
            x=x_candidate,
            y_obs=y_obs,
            constants=constants,
            weights=weights,
        )

        run_with_exact = dict(run)
        run_with_exact["exact_loss_at_candidate"] = float(exact_loss)

        run_with_exact["Vp_hat_symbolic"] = float(y_hat_symbolic["Vp"])
        run_with_exact["Vs_hat_symbolic"] = float(y_hat_symbolic["Vs"])
        run_with_exact["sigma_hat_symbolic"] = float(y_hat_symbolic["sigma"])

        run_with_exact["Vp_hat_exact"] = float(y_hat_exact["Vp"])
        run_with_exact["Vs_hat_exact"] = float(y_hat_exact["Vs"])
        run_with_exact["sigma_hat_exact"] = float(y_hat_exact["sigma"])

        reranked_runs.append(run_with_exact)

    best_exact_run = min(
        reranked_runs,
        key=lambda run: run["exact_loss_at_candidate"],
    )

    return best_exact_run, reranked_runs

def save_signed_error_plot(rows, path, optimizer_name):
    C_values = [row["C_true"] for row in rows]

    err_phi = [row["phi_percent_error_percentage_points"] for row in rows]
    err_C = [row["C_percent_error_percentage_points"] for row in rows]
    err_Sb = [row["S_b_percent_error_percentage_points"] for row in rows]
    err_sigma_b_inv = [row["sigma_b_inv_error_relative_percent"] for row in rows]
    err_xi = [row["xi_error_relative_percent"] for row in rows]

    optimizer_label = optimizer_name.replace("_", " ")

    plt.figure(figsize=(10, 6))

    plt.plot(C_values, err_phi, marker="o", linestyle="-", label=r"$\phi$")
    plt.plot(C_values, err_C, marker="o", linestyle="-", label=r"$C$")
    plt.plot(C_values, err_Sb, marker="o", linestyle="-", label=r"$S_b$")
    plt.plot(C_values, err_sigma_b_inv, marker="o", linestyle="-", label=r"$\sigma_b^{-1}$")
    plt.plot(C_values, err_xi, marker="o", linestyle="-", label=r"$\xi$")

    plt.axhline(0.0, linewidth=1.0)

    plt.xlabel("Clay content C [%]")
    plt.ylabel(r"Signed deviation [pp for $\phi$, $C$, $S_b$; % for $\sigma_b^{-1}$, $\xi$]")
    plt.title(f"Signed parameter recovery errors - {optimizer_label}")

    plt.legend()
    plt.grid(True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_abs_error_plot(rows, path, optimizer_name):
    C_values = [row["C_true"] for row in rows]

    err_phi = [row["phi_percent_abs_error_percentage_points"] for row in rows]
    err_C = [row["C_percent_abs_error_percentage_points"] for row in rows]
    err_Sb = [row["S_b_percent_abs_error_percentage_points"] for row in rows]
    err_sigma_b_inv = [row["sigma_b_inv_abs_error_relative_percent"] for row in rows]
    err_xi = [row["xi_abs_error_relative_percent"] for row in rows]

    optimizer_label = optimizer_name.replace("_", " ")

    plt.figure(figsize=(10, 6))

    plt.plot(C_values, err_phi, marker="o", linestyle="-", label=r"$|\phi|$")
    plt.plot(C_values, err_C, marker="o", linestyle="-", label=r"$|C|$")
    plt.plot(C_values, err_Sb, marker="o", linestyle="-", label=r"$|S_b|$")
    plt.plot(C_values, err_sigma_b_inv, marker="o", linestyle="-", label=r"$|\sigma_b^{-1}|$")
    plt.plot(C_values, err_xi, marker="o", linestyle="-", label=r"$|\xi|$")

    plt.xlabel("Clay content C [%]")
    plt.ylabel(r"Absolute deviation [pp for $\phi$, $C$, $S_b$; % for $\sigma_b^{-1}$, $\xi$]")
    plt.title(f"Absolute parameter recovery errors - {optimizer_label}")

    plt.legend()
    plt.grid(True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    config = load_yaml(CONFIG_PATH)
    constants = load_yaml(CONSTANTS_PATH)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        ROOT
        / config["experiment"]["output_dir"]
        / f"{config['experiment']['name']}_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    surrogate_config = load_yaml(ROOT / config["surrogate"]["config_path"])
    surrogate = SymbolicForwardSurrogate.from_config(
        config=surrogate_config,
        root=ROOT,
    )

    variable_order = config["variables"]["order"]
    bounds = config["bounds"]
    weights = config["objective"]["weights"]

    base_true_parameters = config["base_true_parameters"]
    clay_variable = config["clay_sweep"]["variable"]
    clay_values = config["clay_sweep"]["values"]

    noise_seed = config["synthetic_data"]["seed"]
    noise_config = config["synthetic_data"]["noise"]
    rng_noise = np.random.default_rng(noise_seed)

    optimizer_config = config["optimizer"]
    optimizer_name = "symbolic_least_squares"

    rows = []
    all_runs_by_experiment = {}

    total_start = perf_counter()

    for idx, C_true in enumerate(clay_values):
        x_true = dict(base_true_parameters)
        x_true[clay_variable] = C_true

        y_obs, y_true, _ = generate_synthetic_data_from_rng(
            x_true=x_true,
            constants=constants,
            noise_config=noise_config,
            rng=rng_noise,
        )

        start = perf_counter()

        best_symbolic_run, all_runs = run_symbolic_multistart_least_squares(
            surrogate=surrogate,
            y_obs=y_obs,
            bounds_dict=bounds,
            variable_order=variable_order,
            weights=weights,
            n_starts=optimizer_config["n_starts"],
            seed=int(optimizer_config["seed"]) + idx,
            max_nfev=optimizer_config["max_nfev"],
            ftol=optimizer_config["ftol"],
            xtol=optimizer_config["xtol"],
            gtol=optimizer_config["gtol"],
        )

        best_run, reranked_runs = rerank_symbolic_candidates_by_exact_loss(
            all_runs=all_runs,
            surrogate=surrogate,
            y_obs=y_obs,
            constants=constants,
            weights=weights,
        )

        runtime = perf_counter() - start

        x_before_refinement = best_run["x_hat"]

        symbolic_loss_before_refinement = float(best_run["symbolic_scalar_loss"])
        exact_loss_before_refinement = float(best_run["exact_loss_at_candidate"])

        refinement_config = config.get("exact_refinement", {"enabled": False})

        if refinement_config.get("enabled", False):
            refinement_result = run_exact_refinement_from_x(
                x_start=x_before_refinement,
                y_obs=y_obs,
                bounds_dict=bounds,
                variable_order=variable_order,
                weights=weights,
                constants=constants,
                exact_forward_model=forward_model,
                max_nfev=refinement_config["max_nfev"],
                ftol=refinement_config["ftol"],
                xtol=refinement_config["xtol"],
                gtol=refinement_config["gtol"],
            )

            x_hat = refinement_result["x_hat"]
        else:
            refinement_result = {
                "success": False,
                "status": 0,
                "message": "Exact refinement disabled.",
                "nfev": 0,
                "cost": None,
            }

            x_hat = x_before_refinement

        y_hat_symbolic = surrogate.predict_one(x_hat)
        y_hat_exact = forward_model(x_hat, constants)

        symbolic_loss = float(
            abs(y_hat_symbolic["Vp"] - y_obs["Vp"])
            + weights["W1"] * abs(y_hat_symbolic["Vs"] - y_obs["Vs"])
            + weights["W2"] * abs((1.0 / y_hat_symbolic["sigma"]) - (1.0 / y_obs["sigma"]))
        )

        exact_loss_at_recovered_x = objective(
            x=x_hat,
            y_obs=y_obs,
            constants=constants,
            weights=weights,
        )
        
        exact_loss_at_true_x = objective(
            x=x_true,
            y_obs=y_obs,
            constants=constants,
            weights=weights,
        )

        errors = compute_parameter_errors(
            x_hat=x_hat,
            x_true=x_true,
        )
        flat_errors = flatten_errors(errors)

        row = {
            "experiment_index": idx,
            "C_true": float(C_true),
            "optimizer": optimizer_name,
            "run_time_seconds": float(runtime),
            "best_start_index": int(best_run["start_index"]),
            "nfev_best": int(best_run["nfev"]),
            "success": bool(best_run["success"]),
            "symbolic_loss_at_recovered_x": symbolic_loss,
            "exact_loss_at_recovered_x": float(exact_loss_at_recovered_x),
            "exact_loss_at_true_x": float(exact_loss_at_true_x),
            "Vp_true": float(y_true["Vp"]),
            "Vs_true": float(y_true["Vs"]),
            "sigma_true": float(y_true["sigma"]),
            "Vp_obs": float(y_obs["Vp"]),
            "Vs_obs": float(y_obs["Vs"]),
            "sigma_obs": float(y_obs["sigma"]),
            "Vp_hat_symbolic": float(y_hat_symbolic["Vp"]),
            "Vs_hat_symbolic": float(y_hat_symbolic["Vs"]),
            "sigma_hat_symbolic": float(y_hat_symbolic["sigma"]),
            "Vp_hat_exact": float(y_hat_exact["Vp"]),
            "Vs_hat_exact": float(y_hat_exact["Vs"]),
            "sigma_hat_exact": float(y_hat_exact["sigma"]),
            "selection_rule": "best_exact_loss_among_symbolic_candidates",
            "best_symbolic_start_index": int(best_symbolic_run["start_index"]),
            "best_exact_start_index": int(best_run["start_index"]),
            "best_symbolic_loss": float(best_symbolic_run["symbolic_scalar_loss"]),
            "exact_loss_of_best_symbolic_candidate": float(
                objective(
                    x=best_symbolic_run["x_hat"],
                    y_obs=y_obs,
                    constants=constants,
                    weights=weights,
                )
            ),
            "exact_refinement_enabled": bool(refinement_config.get("enabled", False)),
            "exact_refinement_success": bool(refinement_result["success"]),
            "exact_refinement_nfev": int(refinement_result["nfev"]),
            "symbolic_loss_before_refinement": float(symbolic_loss_before_refinement),
            "exact_loss_before_refinement": float(exact_loss_before_refinement),
        }

        for name in variable_order:
            row[f"{name}_true"] = float(x_true[name])
            row[f"{name}_hat"] = float(x_hat[name])

        row.update(flat_errors)
        rows.append(row)

        all_runs_by_experiment[str(idx)] = reranked_runs

        print(
            f"[{idx + 1}/{len(clay_values)}] "
            f"C_true={C_true:.1f}, "
            f"exact_before={exact_loss_before_refinement:.6g}, "
            f"exact_after={exact_loss_at_recovered_x:.6g}, "
            f"true_loss={exact_loss_at_true_x:.6g}, "
            f"C_hat={x_hat['C_percent']:.3f}, "
            f"S_b_hat={x_hat['S_b_percent']:.3f}"
        )

    total_runtime = perf_counter() - total_start

    summary = {
        "experiment_name": config["experiment"]["name"],
        "optimizer": optimizer_name,
        "total_run_time_seconds": float(total_runtime),
        "rows": rows,
    }

    pd.DataFrame(rows).to_csv(run_dir / "summary_results.csv", index=False)

    with open(run_dir / "summary_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(run_dir / "all_optimizer_runs.json", "w") as f:
        json.dump(all_runs_by_experiment, f, indent=2)

    with open(run_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    save_signed_error_plot(
        rows=rows,
        path=run_dir / "all_parameter_signed_errors.png",
        optimizer_name=optimizer_name,
    )

    save_abs_error_plot(
        rows=rows,
        path=run_dir / "all_parameter_abs_errors.png",
        optimizer_name=optimizer_name,
    )

    print("\nSymbolic inverse completed.")
    print(f"Total runtime: {total_runtime:.3f} seconds")
    print("\nSaved to:")
    print(f"  {run_dir}")


if __name__ == "__main__":
    main()