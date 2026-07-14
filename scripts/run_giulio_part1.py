# scripts/experiment_giulio_part1.py

from pathlib import Path
import sys
from time import perf_counter
import csv
import json
from jupyterlab_server import config
import numpy as np

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config_utils import load_yaml
from src.synthetic_data import generate_synthetic_data_from_rng
from src.objective import objective
from src.forward_model import forward_model
from src.parameter_utils import (
    bounds_dict_to_list,
    x_dict_to_vector,
    x_vector_to_dict,
)
from src.optimizers.common import make_vector_objective
from src.optimizers.runner import run_optimizer
from src.experiment_utils import compute_parameter_errors
from src.report_utils import (
    create_run_directory,
    save_yaml,
    save_json,
)


CONSTANTS_PATH = ROOT / "config" / "constants.yaml"
EXPERIMENT_PATH = ROOT / "config" / "giulio_part1.yaml"


def save_summary_csv(rows, path):
    """
    Save Giulio part 1 summary rows as CSV.
    """
    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_clay_error_plot(rows, path):
    """
    Save a simple plot of C recovery error against true clay content.
    """
    C_true = [row["C_true"] for row in rows]
    C_error = [row["C_percent_error_percentage_points"] for row in rows]

    plt.figure()
    plt.plot(C_true, C_error, marker="o")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("True clay content C [%]")
    plt.ylabel("C recovery error [percentage points]")
    plt.title("Clay recovery error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def save_all_parameter_abs_error_plot(rows, path, optimizer_name):
    """
    Save a plot with absolute recovery errors for all five model parameters.

    x-axis:
        true clay content C [%]

    y-axis:
        absolute deviation:
            - percentage points for phi, C, S_b
            - relative percent for sigma_b_inv, xi
    """
    C_true = [row["C_true"] for row in rows]

    metrics = [
        (
            "phi_percent_abs_error_percentage_points",
            r"$|\phi|$",
            "blue",
        ),
        (
            "C_percent_abs_error_percentage_points",
            r"$|C|$",
            "orange",
        ),
        (
            "S_b_percent_abs_error_percentage_points",
            r"$|S_b|$",
            "green",
        ),
        (
            "sigma_b_inv_abs_error_relative_percent",
            r"$|\sigma_b^{-1}|$",
            "red",
        ),
        (
            "xi_abs_error_relative_percent",
            r"$|\xi|$",
            "purple",
        ),
    ]

    plt.figure(figsize=(8, 5))

    for key, label, color in metrics:
        values = [row[key] for row in rows]
        plt.plot(
            C_true,
            values,
            marker="o",
            label=label,
            color=color,
        )

    plt.xlabel("Clay content C [%]")
    plt.ylabel("Absolute deviation [pp for φ, C, S_b; % for σ_b⁻¹, ξ]")
    plt.title(f"Absolute parameter recovery errors - {optimizer_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def save_all_parameter_signed_error_plot(rows, path, optimizer_name):
    import matplotlib.pyplot as plt

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

def flatten_errors(errors):
    """
    Convert nested error dictionary into flat columns.
    """
    flat = {}

    for variable_name, error_info in errors.items():
        unit = error_info["unit"]

        if unit == "percentage_points":
            suffix = "percentage_points"
        elif unit == "relative_percent":
            suffix = "relative_percent"
        else:
            suffix = unit

        flat[f"{variable_name}_error_{suffix}"] = error_info["error"]
        flat[f"{variable_name}_abs_error_{suffix}"] = error_info["abs_error"]

    return flat


def main():
    constants = load_yaml(CONSTANTS_PATH)
    config = load_yaml(EXPERIMENT_PATH)

    variable_order = config["variables"]["order"]
    bounds = bounds_dict_to_list(config["bounds"], variable_order)

    optimizer_name = config["optimizer"]["name"]

    report_config = config["report"]

    run_dir = create_run_directory(
        base_dir=ROOT / report_config["base_dir"],
        category=f"giulio_part1/{optimizer_name}",
        run_name=report_config["run_name"],
    )

    full_config = {
        "constants_path": str(CONSTANTS_PATH),
        "experiment_path": str(EXPERIMENT_PATH),
        "experiment_config": config,
    }

    if report_config["save_config"]:
        save_yaml(full_config, run_dir / "run_config.yaml")

    base_true_parameters = config["base_true_parameters"]
    clay_variable = config["clay_sweep"]["variable"]
    clay_values = config["clay_sweep"]["values"]

    noise_seed = config["synthetic_data"]["seed"]
    noise_config = config["synthetic_data"]["noise"]
    weights = config["objective"]["weights"]

    rows = []

    total_start_time = perf_counter()

    rng_noise = np.random.default_rng(noise_seed)
    for idx, C_true in enumerate(clay_values):
        x_true = dict(base_true_parameters)
        x_true[clay_variable] = C_true

        
        y_obs, y_true, _ = generate_synthetic_data_from_rng(
           x_true=x_true,
            constants=constants,
            noise_config=noise_config,
            rng=rng_noise,
        )

        vector_objective = make_vector_objective(
            y_obs=y_obs,
            constants=constants,
            weights=weights,
            variable_order=variable_order,
        )

        x_true_vector = x_dict_to_vector(x_true, variable_order)
        loss_at_true_x = vector_objective(x_true_vector)

        start_time = perf_counter()

        optimizer_config = dict(config["optimizers"][optimizer_name])

        if optimizer_name in ["simulated_annealing_giulio", "pso_giulio", "cmaes_giulio"]:
            seed_base = optimizer_config.pop("seed_base")
            seed_add_clay_offset = optimizer_config.pop("seed_add_clay_offset", False)

            if seed_add_clay_offset:
                clay_offset = int(C_true - 10.0)
                optimizer_config["seed"] = int(seed_base + clay_offset)
            else:
                optimizer_config["seed"] = int(seed_base)

        optimizer_result = run_optimizer(
            optimizer_name=optimizer_name,
            optimizer_config=optimizer_config,
            vector_objective=vector_objective,
            bounds=bounds,
        )

        run_time_seconds = perf_counter() - start_time

        x_hat = x_vector_to_dict(
            optimizer_result["best_x_vector"],
            variable_order,
        )

        y_hat = forward_model(x_hat, constants)

        residuals = {
            key: float(y_hat[key] - y_obs[key])
            for key in ["Vp", "Vs", "sigma"]
        }

        errors = compute_parameter_errors(
            x_hat=x_hat,
            x_true=x_true,
        )

        flat_errors = flatten_errors(errors)

        row = {
            "experiment_index": idx,
            "C_true": C_true,
            "optimizer": optimizer_name,
            "run_time_seconds": run_time_seconds,
            "loss_at_true_x": float(loss_at_true_x),
            "loss_at_recovered_x": float(optimizer_result["best_loss"]),
            "Vp_true": float(y_true["Vp"]),
            "Vs_true": float(y_true["Vs"]),
            "sigma_true": float(y_true["sigma"]),
            "Vp_obs": float(y_obs["Vp"]),
            "Vs_obs": float(y_obs["Vs"]),
            "sigma_obs": float(y_obs["sigma"]),
            "Vp_hat": float(y_hat["Vp"]),
            "Vs_hat": float(y_hat["Vs"]),
            "sigma_hat": float(y_hat["sigma"]),
            "Vp_residual": residuals["Vp"],
            "Vs_residual": residuals["Vs"],
            "sigma_residual": residuals["sigma"],
        }

        for name in variable_order:
            row[f"{name}_true"] = float(x_true[name])
            row[f"{name}_hat"] = float(x_hat[name])

        row.update(flat_errors)

        rows.append(row)

        print(
            f"[{idx + 1}/{len(clay_values)}] "
            f"C_true={C_true:.1f}, "
            f"loss_true={loss_at_true_x:.6g}, "
            f"loss_hat={optimizer_result['best_loss']:.6g}, "
            f"C_hat={x_hat['C_percent']:.3f}"
        )

    total_run_time_seconds = perf_counter() - total_start_time

    summary = {
        "experiment_name": config["experiment"]["name"],
        "optimizer": optimizer_name,
        "n_experiments": len(rows),
        "total_run_time_seconds": total_run_time_seconds,
        "rows": rows,
    }

    save_summary_csv(rows, run_dir / "summary_results.csv")
    save_json(summary, run_dir / "summary_results.json")

    if report_config["save_plot"]:
        save_clay_error_plot(
            rows,
            run_dir / "clay_error_plot.png",
        )

        save_all_parameter_signed_error_plot(
            rows=rows,
            path=run_dir / "all_parameter_signed_errors.png",
            optimizer_name=optimizer_name,
        )

        save_all_parameter_abs_error_plot(
            rows=rows,
            path=run_dir / "all_parameter_abs_errors.png",
            optimizer_name=optimizer_name,
        )

    print("\nGiulio part 1 completed.")
    print(f"Optimizer: {optimizer_name}")
    print(f"Total run time: {total_run_time_seconds:.3f} seconds")
    print("\nReport saved to:")
    print(f"  {run_dir}")


if __name__ == "__main__":
    main()