# src/optimizers/cmaes_giulio.py

import numpy as np
import cma


def run_cmaes_giulio(
    objective_function,
    bounds,
    seed,
    max_evals=60000,
    popsize=64,
    init_sigma_y=0.5,
    init_mode=1,
    tol=1.0e-12,
):
    """
    Giulio-style CMA-ES in normalized space y in [0, 1]^d.

    This reproduces Giulio's Phase 2 CMA-ES logic:
      - normalized search space
      - y0 = midpoint if init_mode == 1
      - external cma package
      - bounds [0, 1]
      - exact max_evals budget
      - remaining evaluations handled manually
    """

    dim = len(bounds)

    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    span = hi - lo

    def y_to_x(y):
        y = np.asarray(y, dtype=np.float64)
        return lo + y * span

    def f_y(y):
        y = np.asarray(y, dtype=np.float64)

        if np.any(y < 0.0) or np.any(y > 1.0) or not np.all(np.isfinite(y)):
            return 1.0e300

        x = y_to_x(y)
        value = objective_function(x)

        if not np.isfinite(value):
            return 1.0e300

        return float(value)

    rng = np.random.default_rng(int(seed))

    if init_mode == 1:
        y0 = np.full(dim, 0.5, dtype=np.float64)
    else:
        y0 = rng.random(dim)

    opts = {
        "bounds": [0.0, 1.0],
        "seed": int(seed),
        "verb_disp": 0,
        "verbose": -9,
        "maxfevals": int(max_evals),
        "popsize": int(popsize),

        # Same idea as Giulio: avoid early stopping criteria and use budget.
        "tolfun": 0.0,
        "tolx": 0.0,
        "tolfunhist": 0.0,
        "tolstagnation": int(10**9),
        "maxiter": int(10**9),
    }

    es = cma.CMAEvolutionStrategy(y0.tolist(), float(init_sigma_y), opts)

    history = []

    # Use only entire populations first.
    while es.countevals + es.sp.popsize <= max_evals:
        Y = es.ask()
        F = [f_y(y) for y in Y]
        es.tell(Y, F)

        history.append(
            {
                "iteration": int(es.countiter),
                "evaluations": int(es.countevals),
                "loss_best": float(es.result.fbest),
            }
        )

    # Use remaining evaluations, if max_evals is not multiple of popsize.
    remaining = int(max_evals - es.countevals)

    y_best_extra = None
    f_best_extra = 1.0e300

    if remaining > 0:
        Y_extra = es.ask()[:remaining]
        F_extra = [f_y(y) for y in Y_extra]

        for y, f in zip(Y_extra, F_extra):
            if f < f_best_extra:
                f_best_extra = float(f)
                y_best_extra = np.array(y, dtype=np.float64)

    y_best = np.clip(np.array(es.result.xbest, dtype=np.float64), 0.0, 1.0)
    f_best = float(es.result.fbest)

    if y_best_extra is not None and f_best_extra < f_best:
        y_best = np.clip(y_best_extra, 0.0, 1.0)
        f_best = float(f_best_extra)

    x_best = y_to_x(y_best)

    history.append(
        {
            "iteration": int(es.countiter),
            "evaluations": int(max_evals),
            "loss_best": float(f_best),
        }
    )

    return x_best.tolist(), float(f_best), history, int(max_evals)