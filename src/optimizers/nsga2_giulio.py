from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from src.forward_model import forward_model


def bounds_to_arrays(bounds_dict: dict, variable_order: list[str]):
    lo = np.array([bounds_dict[name][0] for name in variable_order], dtype=float)
    hi = np.array([bounds_dict[name][1] for name in variable_order], dtype=float)
    return lo, hi


def x_array_to_dict(x_array: np.ndarray, variable_order: list[str]) -> dict:
    return {name: float(value) for name, value in zip(variable_order, x_array)}


def safe_inverse_sigma(sigma: float, epsilon: float = 1.0e-12) -> float:
    sigma_safe = max(float(sigma), float(epsilon))
    return 1.0 / sigma_safe


def evaluate_giulio_multiobjective(
    x_array: np.ndarray,
    y_obs: dict,
    constants: dict,
    variable_order: list[str],
    W1: float = 1.0,
    W2: float = 100.0,
    epsilon_sigma: float = 1.0e-12,
):
    x_dict = x_array_to_dict(x_array, variable_order)
    y_pred = forward_model(x_dict, constants)

    e_vp = abs(float(y_pred["Vp"]) - float(y_obs["Vp"]))
    e_vs = abs(float(y_pred["Vs"]) - float(y_obs["Vs"]))

    inv_sigma_pred = safe_inverse_sigma(y_pred["sigma"], epsilon=epsilon_sigma)
    inv_sigma_obs = safe_inverse_sigma(y_obs["sigma"], epsilon=epsilon_sigma)
    e_inv_sigma = abs(inv_sigma_pred - inv_sigma_obs)

    f1 = e_vp
    f2 = float(W1) * e_vs
    f3 = float(W2) * e_inv_sigma

    scalar_loss = f1 + f2 + f3

    return {
        "x_dict": x_dict,
        "y_pred": y_pred,
        "f1_vp": float(f1),
        "f2_vs": float(f2),
        "f3_inv_sigma": float(f3),
        "e_vp_unscaled": float(e_vp),
        "e_vs_unscaled": float(e_vs),
        "e_inv_sigma_unscaled": float(e_inv_sigma),
        "giulio_scalar_loss": float(scalar_loss),
    }


class GiulioNSGA2Problem(ElementwiseProblem):
    def __init__(
        self,
        y_obs: dict,
        constants: dict,
        bounds_dict: dict,
        variable_order: list[str],
        W1: float = 1.0,
        W2: float = 100.0,
        epsilon_sigma: float = 1.0e-12,
    ):
        self.y_obs = y_obs
        self.constants = constants
        self.variable_order = variable_order
        self.W1 = float(W1)
        self.W2 = float(W2)
        self.epsilon_sigma = float(epsilon_sigma)

        lo, hi = bounds_to_arrays(bounds_dict, variable_order)

        super().__init__(
            n_var=len(variable_order),
            n_obj=3,
            n_constr=0,
            xl=lo,
            xu=hi,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        values = evaluate_giulio_multiobjective(
            x_array=np.asarray(x, dtype=float),
            y_obs=self.y_obs,
            constants=self.constants,
            variable_order=self.variable_order,
            W1=self.W1,
            W2=self.W2,
            epsilon_sigma=self.epsilon_sigma,
        )

        out["F"] = np.array(
            [
                values["f1_vp"],
                values["f2_vs"],
                values["f3_inv_sigma"],
            ],
            dtype=float,
        )


def population_to_dataframe(
    X: np.ndarray,
    y_obs: dict,
    constants: dict,
    variable_order: list[str],
    W1: float,
    W2: float,
    epsilon_sigma: float,
):
    rows = []

    X = np.atleast_2d(X)

    for idx, x_array in enumerate(X):
        values = evaluate_giulio_multiobjective(
            x_array=x_array,
            y_obs=y_obs,
            constants=constants,
            variable_order=variable_order,
            W1=W1,
            W2=W2,
            epsilon_sigma=epsilon_sigma,
        )

        row = {"solution_index": int(idx)}

        for name in variable_order:
            row[f"{name}_hat"] = values["x_dict"][name]

        row.update(
            {
                "Vp_hat": float(values["y_pred"]["Vp"]),
                "Vs_hat": float(values["y_pred"]["Vs"]),
                "sigma_hat": float(values["y_pred"]["sigma"]),
                "f1_vp": values["f1_vp"],
                "f2_vs": values["f2_vs"],
                "f3_inv_sigma": values["f3_inv_sigma"],
                "e_vp_unscaled": values["e_vp_unscaled"],
                "e_vs_unscaled": values["e_vs_unscaled"],
                "e_inv_sigma_unscaled": values["e_inv_sigma_unscaled"],
                "giulio_scalar_loss": values["giulio_scalar_loss"],
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def select_solution_by_giulio_loss(pareto_df: pd.DataFrame) -> pd.Series:
    best_idx = pareto_df["giulio_scalar_loss"].idxmin()
    return pareto_df.loc[best_idx].copy()


def run_nsga2_giulio(
    y_obs: dict,
    constants: dict,
    bounds_dict: dict,
    variable_order: list[str],
    objectives_config: dict,
    optimizer_config: dict,
):
    W1 = float(objectives_config.get("W1", 1.0))
    W2 = float(objectives_config.get("W2", 100.0))
    epsilon_sigma = float(objectives_config.get("epsilon_sigma", 1.0e-12))

    problem = GiulioNSGA2Problem(
        y_obs=y_obs,
        constants=constants,
        bounds_dict=bounds_dict,
        variable_order=variable_order,
        W1=W1,
        W2=W2,
        epsilon_sigma=epsilon_sigma,
    )

    algorithm = NSGA2(
        pop_size=int(optimizer_config.get("population_size", 200)),
        eliminate_duplicates=bool(optimizer_config.get("eliminate_duplicates", True)),
    )

    start = perf_counter()

    result = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=("n_gen", int(optimizer_config.get("generations", 300))),
        seed=int(optimizer_config.get("seed", 123)),
        verbose=bool(optimizer_config.get("verbose", False)),
    )

    runtime_seconds = perf_counter() - start

    # Approximate non-dominated / optimum set returned by pymoo.
    pareto_X = result.X
    if pareto_X is None:
        pareto_X = result.pop.get("X")

    pareto_df = population_to_dataframe(
        X=pareto_X,
        y_obs=y_obs,
        constants=constants,
        variable_order=variable_order,
        W1=W1,
        W2=W2,
        epsilon_sigma=epsilon_sigma,
    )

    final_population_X = result.pop.get("X")
    final_population_df = population_to_dataframe(
        X=final_population_X,
        y_obs=y_obs,
        constants=constants,
        variable_order=variable_order,
        W1=W1,
        W2=W2,
        epsilon_sigma=epsilon_sigma,
    )

    selected_solution = select_solution_by_giulio_loss(pareto_df)

    n_eval = None
    try:
        n_eval = int(result.algorithm.evaluator.n_eval)
    except Exception:
        pass

    return {
        "result": result,
        "pareto_df": pareto_df,
        "final_population_df": final_population_df,
        "selected_solution": selected_solution,
        "runtime_seconds": float(runtime_seconds),
        "n_eval": n_eval,
    }