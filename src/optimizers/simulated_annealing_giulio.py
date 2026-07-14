# src/optimizers/simulated_annealing_giulio.py

import math
import numpy as np


def randn_box_muller():
    """
    Same Gaussian generation logic used in Giulio's notebook.
    """
    u1 = np.random.random()
    u2 = np.random.random()

    if u1 < 1e-16:
        u1 = 1e-16

    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def geometric_cooling_stepwise(T0, iteration, alpha, cooling_interval):
    """
    Giulio-style stepwise geometric cooling.

    T = T0 * alpha^k
    k = (iteration - 1) // cooling_interval
    """
    k = (iteration - 1) // cooling_interval
    return T0 * (alpha ** k)


def propose_neighbor(x, T, T0, lo, hi, step_frac):
    """
    Giulio-style proposal.

    dx_j ~ Normal(0, step_frac * span_j * sqrt(T / T0))
    followed by clipping inside bounds.
    """
    if T0 <= 0.0 or T <= 0.0:
        return x.copy()

    scale = math.sqrt(T / T0)
    y = x.copy()

    for j in range(len(x)):
        span = hi[j] - lo[j]
        dx = randn_box_muller() * (step_frac * span * scale)

        v = y[j] + dx

        if v < lo[j]:
            v = lo[j]
        elif v > hi[j]:
            v = hi[j]

        y[j] = v

    return y


def midpoint_start(lo, hi):
    return 0.5 * (lo + hi)


def run_simulated_annealing_giulio(
    objective_function,
    bounds,
    seed,
    n_iters=1_000_000,
    T0=1.0e4,
    step_frac=0.1,
    alpha=0.85,
    cooling_interval=2000,
    tol=1.0e-9,
    start="midpoint",
    history_interval=1000,
):
    """
    Giulio-style Simulated Annealing.

    This is designed to reproduce the dynamics of Giulio's notebook:
      - midpoint start
      - np.random.seed(seed)
      - Box-Muller Gaussian proposal
      - proposal scale proportional to sqrt(T/T0)
      - stepwise cooling
      - clipping to bounds
      - invalid losses mapped to 1e300

    Parameters
    ----------
    objective_function:
        Function taking a vector x and returning a scalar loss.
    bounds:
        List of (lower, upper) tuples.
    seed:
        Random seed.
    n_iters:
        Number of SA iterations.
    T0:
        Initial temperature.
    step_frac:
        Fraction of each parameter range used in the proposal.
    alpha:
        Cooling multiplier.
    cooling_interval:
        Number of iterations before each temperature drop.
    tol:
        Stop when best loss <= tol.
    start:
        Currently supports "midpoint".
    history_interval:
        Save history every this many iterations.
    """

    np.random.seed(int(seed))

    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)

    if start != "midpoint":
        raise ValueError(
            "Giulio-style SA currently supports only start='midpoint'."
        )

    x = midpoint_start(lo, hi)

    # Safety clipping, same spirit as Giulio's notebook.
    x = np.minimum(np.maximum(x, lo), hi)

    f = objective_function(x)
    if not np.isfinite(f):
        f = 1.0e300

    best = x.copy()
    f_best = float(f)

    history = [
        {
            "iteration": 0,
            "temperature": float(T0),
            "loss_current": float(f),
            "loss_best": float(f_best),
        }
    ]

    for i in range(1, int(n_iters) + 1):
        T = geometric_cooling_stepwise(
            T0=T0,
            iteration=i,
            alpha=alpha,
            cooling_interval=cooling_interval,
        )

        candidate = propose_neighbor(
            x=x,
            T=T,
            T0=T0,
            lo=lo,
            hi=hi,
            step_frac=step_frac,
        )

        f_candidate = objective_function(candidate)
        if not np.isfinite(f_candidate):
            f_candidate = 1.0e300

        accept = False

        if f_candidate < f:
            accept = True
        else:
            df = f_candidate - f

            if np.isfinite(df) and T > 0.0:
                p = math.exp(-df / T)

                if np.random.random() < p:
                    accept = True

        if accept:
            x = candidate
            f = float(f_candidate)

            if f < f_best:
                best = x.copy()
                f_best = float(f)

                if f_best <= tol:
                    history.append(
                        {
                            "iteration": i,
                            "temperature": float(T),
                            "loss_current": float(f),
                            "loss_best": float(f_best),
                        }
                    )
                    break

        if history_interval is not None and i % int(history_interval) == 0:
            history.append(
                {
                    "iteration": i,
                    "temperature": float(T),
                    "loss_current": float(f),
                    "loss_best": float(f_best),
                }
            )

    return best.tolist(), float(f_best), history