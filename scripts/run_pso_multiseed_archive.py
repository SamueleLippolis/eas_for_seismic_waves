from pathlib import Path
import sys
from time import perf_counter
import csv
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config_utils import load_yaml
from src.synthetic_data import generate_synthetic_data_from_rng
from src.forward_model import forward_model
from src.objective import objective
from src.parameter_utils import (
    bounds_dict_to_list,
    x_dict_to_vector,
    x_vector_to_dict,
)
from src.optimizers.common import make_vector_objective
from src.optimizers.runner import run_optimizer
from src.experiment_utils import compute_parameter_errors
from src.report_utils import create_run_directory, save_yaml, save_json


CONSTANTS_PATH = ROOT / "config" / "constants.yaml"
EXPERIMENT_PATH = ROOT / "config" / "pso_multiseed.yaml"


def save_csv(rows, path):
    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_errors(errors):
    flat = {}

    for variable_name, error_info in errors.items():
        unit = error_info["unit"]

        if unit == "percentage_points":
            suffix = "percentage_points"
        elif unit == "relative_percent":
            suffix = "relative_percent"
        else:
            suffix = unit

        flat[f"{variable_name}_error_{suffix}"] = float(error_info["error"])
        flat[f"{variable_name}_abs_error_{suffix}"] = float(error_info["abs_error"])

    return flat


def oracle_score_from_flat_errors(flat_errors):
    """
    Non-fair diagnostic score: it uses x_true.
    Lower is better.

    This is only for analysis, not for real selection.
    """
    phi = flat_errors["phi_percent_abs_error_percentage_points"] / 70.0
    C = flat_errors["C_percent_abs_error_percentage_points"] / 100.0
    Sb = flat_errors["S_b_percent_abs_error_percentage_points"] / 20.0
    sig = flat_errors["sigma_b_inv_abs_error_relative_percent"] / 100.0
    xi = flat_errors["xi_abs_error_relative_percent"] / 100.0

    return float(np.mean([phi, C, Sb, sig, xi]))


def build_solution_row(
    *,
    experiment_index,
    C_true,
    seed_index,
    seed,
    source,
    source_index,
    loss_hat,
    x_vector,
    variable_order,
    constants,
    y_obs,
    y_true,
    x_true,
    loss_at_true_x,
    run_time_seconds=None,
    stop_iter=None,
    stop_reason=None,
):
    x_hat = x_vector_to_dict(x_vector, variable_order)
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
        "experiment_index": int(experiment_index),
        "C_true": float(C_true),
        "seed_index": int(seed_index),
        "seed": int(seed),
        "source": source,
        "source_index": source_index,
        "run_time_seconds": None if run_time_seconds is None else float(run_time_seconds),
        "stop_iter": stop_iter,
        "stop_reason": stop_reason,
        "loss_at_true_x": float(loss_at_true_x),
        "loss_hat": float(loss_hat),
        "oracle_score": oracle_score_from_flat_errors(flat_errors),
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

    return row


def summarize_best_rows(best_rows):
    df = pd.DataFrame(best_rows)

    summary_rows = []

    for C_true, group in df.groupby("C_true"):
        best_loss_row = group.loc[group["loss_hat"].idxmin()]
        best_oracle_row = group.loc[group["oracle_score"].idxmin()]

        row = {
            "C_true": float(C_true),
            "n_seeds": int(len(group)),
            "min_loss": float(group["loss_hat"].min()),
            "median_loss": float(group["loss_hat"].median()),
            "max_loss": float(group["loss_hat"].max()),
            "best_loss_seed_index": int(best_loss_row["seed_index"]),
            "best_loss_seed": int(best_loss_row["seed"]),
            "best_loss_C_hat": float(best_loss_row["C_percent_hat"]),
            "best_loss_phi_hat": float(best_loss_row["phi_percent_hat"]),
            "best_loss_xi_hat": float(best_loss_row["xi_hat"]),
            "best_loss_oracle_score": float(best_loss_row["oracle_score"]),
            "best_oracle_seed_index": int(best_oracle_row["seed_index"]),
            "best_oracle_seed": int(best_oracle_row["seed"]),
            "best_oracle_loss": float(best_oracle_row["loss_hat"]),
            "best_oracle_C_hat": float(best_oracle_row["C_percent_hat"]),
            "best_oracle_phi_hat": float(best_oracle_row["phi_percent_hat"]),
            "best_oracle_xi_hat": float(best_oracle_row["xi_hat"]),
            "best_oracle_score": float(best_oracle_row["oracle_score"]),
        }

        for param in [
            "phi_percent_hat",
            "C_percent_hat",
            "S_b_percent_hat",
            "sigma_b_inv_hat",
            "xi_hat",
        ]:
            row[f"{param}_mean"] = float(group[param].mean())
            row[f"{param}_std"] = float(group[param].std(ddof=0))
            row[f"{param}_min"] = float(group[param].min())
            row[f"{param}_max"] = float(group[param].max())

        summary_rows.append(row)

    return summary_rows


def main():
    constants = load_yaml(CONSTANTS_PATH)
    config = load_yaml(EXPERIMENT_PATH)

    variable_order = config["variables"]["order"]
    bounds = bounds_dict_to_list(config["bounds"], variable_order)

    optimizer_name = config["optimizer"]["name"]
    if optimizer_name != "pso_giulio":
        raise ValueError("This script is intended for optimizer.name = pso_giulio")

    report_config = config["report"]

    run_dir = create_run_directory(
        base_dir=ROOT / report_config["base_dir"],
        category="pso_multiseed",
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

    multiseed_config = config["multiseed"]
    n_seeds = int(multiseed_config.get("n_seeds", 30))
    seed_base = int(multiseed_config.get("seed_base", 369))
    seed_stride = int(multiseed_config.get("seed_stride", 10000))
    add_clay_offset = bool(multiseed_config.get("add_clay_offset", True))

    all_best_rows = []
    all_archive_rows = []

    total_start_time = perf_counter()

    rng_noise = np.random.default_rng(noise_seed)

    print("\nRunning PSO multi-seed archive")
    print(f"Clay values: {clay_values}")
    print(f"n_seeds: {n_seeds}")
    print(f"run_dir: {run_dir}\n")

    for idx, C_true in enumerate(clay_values):
        x_true = dict(base_true_parameters)
        x_true[clay_variable] = C_true

        # Important: y_obs is generated once per clay, not once per seed.
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

        clay_offset = int(C_true - 10.0) if add_clay_offset else 0

        clay_dir = run_dir / f"C_{int(round(C_true)):03d}"
        clay_dir.mkdir(parents=True, exist_ok=True)

        clay_best_rows = []
        clay_archive_rows = []

        print(f"--- C={C_true:.1f} ---")

        for seed_index in range(n_seeds):
            seed = seed_base + clay_offset + seed_index * seed_stride

            optimizer_config = dict(config["optimizers"][optimizer_name])

            # Remove Giulio single-seed fields if present.
            optimizer_config.pop("seed_base", None)
            optimizer_config.pop("seed_add_clay_offset", None)

            optimizer_config["seed"] = int(seed)

            start_time = perf_counter()

            optimizer_result = run_optimizer(
                optimizer_name=optimizer_name,
                optimizer_config=optimizer_config,
                vector_objective=vector_objective,
                bounds=bounds,
            )

            run_time_seconds = perf_counter() - start_time

            best_row = build_solution_row(
                experiment_index=idx,
                C_true=C_true,
                seed_index=seed_index,
                seed=seed,
                source="seed_best",
                source_index=None,
                loss_hat=optimizer_result["best_loss"],
                x_vector=optimizer_result["best_x_vector"],
                variable_order=variable_order,
                constants=constants,
                y_obs=y_obs,
                y_true=y_true,
                x_true=x_true,
                loss_at_true_x=loss_at_true_x,
                run_time_seconds=run_time_seconds,
                stop_iter=optimizer_result.get("stop_iter"),
                stop_reason=optimizer_result.get("stop_reason"),
            )

            clay_best_rows.append(best_row)
            all_best_rows.append(best_row)

            archive_candidates = optimizer_result.get("archive_candidates", [])

            for candidate in archive_candidates:
                archive_row = build_solution_row(
                    experiment_index=idx,
                    C_true=C_true,
                    seed_index=seed_index,
                    seed=seed,
                    source=candidate.get("source", "archive"),
                    source_index=candidate.get("source_index"),
                    loss_hat=candidate["loss"],
                    x_vector=candidate["x_vector"],
                    variable_order=variable_order,
                    constants=constants,
                    y_obs=y_obs,
                    y_true=y_true,
                    x_true=x_true,
                    loss_at_true_x=loss_at_true_x,
                    run_time_seconds=run_time_seconds,
                    stop_iter=optimizer_result.get("stop_iter"),
                    stop_reason=optimizer_result.get("stop_reason"),
                )

                archive_row["candidate_index"] = int(candidate.get("candidate_index", -1))
                archive_row["archive_threshold"] = float(candidate.get("archive_threshold", np.nan))
                archive_row["best_loss_reference"] = float(candidate.get("best_loss_reference", np.nan))

                clay_archive_rows.append(archive_row)
                all_archive_rows.append(archive_row)

            print(
                f"[seed {seed_index + 1:02d}/{n_seeds:02d}] "
                f"seed={seed} | "
                f"loss={optimizer_result['best_loss']:.3e} | "
                f"C_hat={best_row['C_percent_hat']:.2f} | "
                f"phi_hat={best_row['phi_percent_hat']:.2f} | "
                f"xi_hat={best_row['xi_hat']:.4f} | "
                f"time={run_time_seconds:.2f}s"
            )

        save_csv(clay_best_rows, clay_dir / "pso_multiseed_best.csv")
        save_csv(clay_archive_rows, clay_dir / "pso_multiseed_archive_candidates.csv")

        best_loss = min(clay_best_rows, key=lambda row: row["loss_hat"])
        best_oracle = min(clay_best_rows, key=lambda row: row["oracle_score"])

        print(
            f"C={C_true:.1f} done | "
            f"best_loss={best_loss['loss_hat']:.3e}, "
            f"C_hat_loss={best_loss['C_percent_hat']:.2f} | "
            f"best_oracle_loss={best_oracle['loss_hat']:.3e}, "
            f"C_hat_oracle={best_oracle['C_percent_hat']:.2f}\n"
        )

    total_run_time_seconds = perf_counter() - total_start_time

    save_csv(all_best_rows, run_dir / "pso_multiseed_best_all.csv")
    save_csv(all_archive_rows, run_dir / "pso_multiseed_archive_all.csv")

    summary_rows = summarize_best_rows(all_best_rows)
    save_csv(summary_rows, run_dir / "pso_multiseed_summary_by_clay.csv")

    summary = {
        "experiment_name": config["experiment"]["name"],
        "optimizer": optimizer_name,
        "n_clay_values": len(clay_values),
        "n_seeds": n_seeds,
        "total_run_time_seconds": total_run_time_seconds,
        "run_dir": str(run_dir),
    }

    save_json(summary, run_dir / "pso_multiseed_summary.json")

    print("\nPSO multi-seed archive completed.")
    print(f"Total run time: {total_run_time_seconds:.3f} seconds")
    print("\nReport saved to:")
    print(f"  {run_dir}")


if __name__ == "__main__":
    main()