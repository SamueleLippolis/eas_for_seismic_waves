# scripts/run_single_experiment.py

from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.optimizers.cmaes import run_cmaes
from src.forward_model import forward_model
from src.config_utils import load_yaml
from src.synthetic_data import generate_synthetic_data
from src.parameter_utils import (
    bounds_dict_to_list,
    x_vector_to_dict,
    x_dict_to_vector,
)
from src.optimizers.common import make_vector_objective
from src.optimizers.simulated_annealing import run_simulated_annealing
from src.optimizers.pso import run_pso
from src.experiment_utils import compute_parameter_errors
from src.report_utils import (
    create_run_directory,
    save_run_report,
)


CONSTANTS_PATH = ROOT / "config" / "constants.yaml"
EXPERIMENT_PATH = ROOT / "config" / "single_experiment.yaml"


def run_optimizer(optimizer_config, vector_objective, bounds):
    """
    Dispatch the optimizer based on the config name.
    """
    optimizer_name = optimizer_config["name"]

    if optimizer_name == "simulated_annealing":
        return run_simulated_annealing(
            objective_function=vector_objective,
            bounds=bounds,
            seed=optimizer_config["seed"],
            max_iter=optimizer_config["max_iter"],
            initial_temperature=optimizer_config["initial_temperature"],
            final_temperature=optimizer_config["final_temperature"],
            step_scale=optimizer_config["step_scale"],
        )

    if optimizer_name == "pso":
        return run_pso(
            objective_function=vector_objective,
            bounds=bounds,
            seed=optimizer_config["seed"],
            n_particles=optimizer_config["n_particles"],
            max_iter=optimizer_config["max_iter"],
            inertia_weight=optimizer_config["inertia_weight"],
            cognitive_weight=optimizer_config["cognitive_weight"],
            social_weight=optimizer_config["social_weight"],
            velocity_scale=optimizer_config["velocity_scale"],
        )
    
    if optimizer_name == "cmaes":
        return run_cmaes(
            objective_function=vector_objective,
            bounds=bounds,
            seed=optimizer_config["seed"],
            population_size=optimizer_config["population_size"],
            max_iter=optimizer_config["max_iter"],
            sigma0=optimizer_config["sigma0"],
        )

    raise ValueError(f"Unknown optimizer: {optimizer_name}")

def main():
    constants = load_yaml(CONSTANTS_PATH)
    config = load_yaml(EXPERIMENT_PATH)

    variable_order = config["variables"]["order"]
    x_true = config["true_parameters"]
    bounds = bounds_dict_to_list(config["bounds"], variable_order)

    seed = config["synthetic_data"]["seed"]
    noise_config = config["synthetic_data"]["noise"]
    weights = config["objective"]["weights"]
    optimizer_name = config["optimizer"]["name"]
    optimizer_config = config["optimizers"][optimizer_name]
    optimizer_config["name"] = optimizer_name
    report_config = config["report"]

    run_dir = create_run_directory(
        base_dir=ROOT / report_config["base_dir"],
        category=optimizer_name,
        run_name=report_config["run_name"],
    )

    y_obs, y_true, _ = generate_synthetic_data(
        x_true=x_true,
        constants=constants,
        noise_config=noise_config,
        seed=seed,
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

    optimizer_result = run_optimizer(
        optimizer_config=optimizer_config,
        vector_objective=vector_objective,
        bounds=bounds,
    )

    end_time = perf_counter()
    run_time_seconds = end_time - start_time

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

    result = {
        "experiment_name": config["experiment"]["name"],
        "optimizer": optimizer_config["name"],
        "run_time_seconds": run_time_seconds,
        "x_true": x_true,
        "x_hat": x_hat,
        "y_true": {
            key: float(y_true[key])
            for key in ["Vp", "Vs", "sigma"]
        },
        "y_obs": {
            key: float(y_obs[key])
            for key in ["Vp", "Vs", "sigma"]
        },
        "best_loss": optimizer_result["best_loss"],
        "errors": errors,
        "loss_at_true_x": float(loss_at_true_x),
        "loss_at_recovered_x": optimizer_result["best_loss"],
        "y_hat": {
         key: float(y_hat[key])
             for key in ["Vp", "Vs", "sigma"]
          },
        "residuals": residuals,
    }

    full_config = {
        "constants_path": str(CONSTANTS_PATH),
        "experiment_path": str(EXPERIMENT_PATH),
        "experiment_config": config,
    }

    save_run_report(
        run_dir=run_dir,
        full_config=full_config,
        result=result,
        history=optimizer_result["history"],
        save_config=report_config["save_config"],
        save_history=report_config["save_history"],
        save_plot=report_config["save_plot"],
    )

    print("Single experiment completed.")
    print(f"Optimizer: {optimizer_config['name']}")
    print(f"Run time: {run_time_seconds:.3f} seconds")

    print("\nTrue parameters:")
    for key, value in x_true.items():
        print(f"  {key}: {value}")

    print("\nRecovered parameters:")
    for key, value in x_hat.items():
        print(f"  {key}: {value:.6g}")

    print("\nLoss comparison:")
    print(f"  loss at true x:      {loss_at_true_x:.6g}")
    print(f"  loss at recovered x: {optimizer_result['best_loss']:.6g}")

    print("\nErrors:")
    for key, value in errors.items():
        print(f"  {key}: {value['error']:.6g} {value['unit']}")

    print("\nObservation fit:")
    for key in ["Vp", "Vs", "sigma"]:
        print(
            f"  {key}: obs={y_obs[key]:.6g}, "
            f"pred={y_hat[key]:.6g}, "
            f"residual={residuals[key]:.6g}"
        )

    print("\nReport saved to:")
    print(f"  {run_dir}")


if __name__ == "__main__":
    main()