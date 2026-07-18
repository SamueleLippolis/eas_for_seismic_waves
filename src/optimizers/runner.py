# src/optimizers/runner.py

from src.optimizers.cmaes_giulio import run_cmaes_giulio
from src.optimizers.pso_giulio import run_pso_giulio
from src.optimizers.simulated_annealing_giulio import run_simulated_annealing_giulio
from src.optimizers.simulated_annealing_basic import run_simulated_annealing
from src.optimizers.pso_basic import run_pso
from src.optimizers.cmaes_basic import run_cmaes


def run_optimizer(optimizer_name, optimizer_config, vector_objective, bounds):
    """
    Dispatch optimizer by name.
    """

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
            **optimizer_config,
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
    
    if optimizer_name == "simulated_annealing_giulio":
        best_x_vector, best_loss, history = run_simulated_annealing_giulio(
            objective_function=vector_objective,
            bounds=bounds,
            **optimizer_config,
        )

        return {
            "best_x_vector": best_x_vector,
            "best_loss": best_loss,
            "history": history,
        }
    
    if optimizer_name == "pso_giulio":
        best_x_vector, best_loss, history, stop_iter, stop_reason, archive_candidates = run_pso_giulio(
            objective_function=vector_objective,
            bounds=bounds,
            **optimizer_config,
        )

        return {
            "best_x_vector": best_x_vector,
            "best_loss": best_loss,
            "history": history,
            "stop_iter": stop_iter,
            "stop_reason": stop_reason,
            "archive_candidates": archive_candidates,
        }

    if optimizer_name == "cmaes_giulio":
        best_x_vector, best_loss, history, evaluations = run_cmaes_giulio(
            objective_function=vector_objective,
            bounds=bounds,
            **optimizer_config,
        )

        return {
            "best_x_vector": best_x_vector,
            "best_loss": best_loss,
            "history": history,
            "evaluations": evaluations,
        }

    raise ValueError(f"Unknown optimizer: {optimizer_name}")