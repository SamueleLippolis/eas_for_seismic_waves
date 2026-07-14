# src/optimizers/pso_giulio.py

import math
import numpy as np


STOP_REASON_LABELS = {
    0: "none",
    1: "tol",
    2: "patience",
    3: "max_iters",
}


def randn_box_muller():
    u1 = np.random.random()
    u2 = np.random.random()

    if u1 < 1e-16:
        u1 = 1e-16

    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def clamp_velocity_component(v, vmax, vmin):
    av = abs(v)

    if av > vmax:
        return math.copysign(vmax, v)

    if vmin > 0.0 and av < vmin and av > 0.0:
        return math.copysign(vmin, v)

    return v


def reflect_01(y, v):
    while y < 0.0 or y > 1.0:
        if y < 0.0:
            y = -y
            v = -v
        elif y > 1.0:
            y = 2.0 - y
            v = -v

    if y < 0.0:
        y = 0.0
    elif y > 1.0:
        y = 1.0

    return y, v


def map_y_to_x(y, lo, hi):
    return lo + y * (hi - lo)


def outside_01(y_row):
    for value in y_row:
        if value < 0.0 or value > 1.0:
            return True

    return False


def run_pso_giulio(
    objective_function,
    bounds,
    seed,
    n_particles=1996,
    n_iters=500,
    tol=1.0e-9,
    w_start=0.2,
    w_end=0.02,
    c_soc=1.49445,
    c_cog=1.49445,
    vmax_abs=0.02,
    vmin_abs=0.0,
    init_mode=1,
    init_sigma_y=0.02,
    zero_vel=0,
    eps_improve=1.0e-6,
    patience=1000,
    history_interval=10,
):
    """
    Giulio-style PSO in normalized space y in [0, 1]^d.

    This reproduces Giulio's Phase 2 PSO logic:
      - normalized search space
      - midpoint Gaussian initialization
      - one particle exactly at midpoint
      - reflection boundary handling
      - velocity clamping
      - linear inertia decay
      - patience stopping criterion
    """

    np.random.seed(int(seed))

    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)

    dim = len(bounds)
    center = 0.5

    Y = np.empty((n_particles, dim), dtype=np.float64)
    V = np.empty((n_particles, dim), dtype=np.float64)

    PbestY = np.empty((n_particles, dim), dtype=np.float64)
    Fpbest = np.empty(n_particles, dtype=np.float64)

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
    for i in range(n_particles):
        for j in range(dim):
            if init_mode == 1:
                if i == 0:
                    y = center
                else:
                    y = center + init_sigma_y * randn_box_muller()
                    y, _ = reflect_01(y, 0.0)
            else:
                y = np.random.random()

            Y[i, j] = y

            if zero_vel == 1:
                V[i, j] = 0.0
            else:
                V[i, j] = (2.0 * np.random.random() - 1.0) * vmax_abs

        Fpbest[i] = 1.0e300

    # --------------------------------------------------
    # Initial evaluation
    # --------------------------------------------------
    gbestY = np.empty(dim, dtype=np.float64)
    f_gbest = 1.0e300

    for i in range(n_particles):
        if outside_01(Y[i]):
            f = 1.0e300
        else:
            x = map_y_to_x(Y[i], lo, hi)
            f = objective_function(x)

            if not np.isfinite(f):
                f = 1.0e300

        Fpbest[i] = f
        PbestY[i, :] = Y[i, :]

        if f < f_gbest:
            f_gbest = float(f)
            gbestY[:] = Y[i, :]

    stop_reason = 0
    stop_iter = 0

    history = [
        {
            "iteration": 0,
            "loss_best": float(f_gbest),
            "stop_reason": STOP_REASON_LABELS[stop_reason],
        }
    ]

    if f_gbest <= tol:
        stop_reason = 1
        stop_iter = 0
        best_x = map_y_to_x(gbestY, lo, hi)
        return best_x.tolist(), float(f_gbest), history, stop_iter, stop_reason

    best_seen = float(f_gbest)
    no_improve = 0

    # --------------------------------------------------
    # Main PSO loop
    # --------------------------------------------------
    for it in range(1, int(n_iters) + 1):
        t = (it - 1) / max(1, (n_iters - 1))
        w = w_start + (w_end - w_start) * t

        for i in range(n_particles):
            for j in range(dim):
                r1 = np.random.random()
                r2 = np.random.random()

                v_new = (
                    w * V[i, j]
                    + c_soc * r1 * (gbestY[j] - Y[i, j])
                    + c_cog * r2 * (PbestY[i, j] - Y[i, j])
                )

                v_new = clamp_velocity_component(v_new, vmax_abs, vmin_abs)

                y_new = Y[i, j] + v_new
                y_new, v_new = reflect_01(y_new, v_new)

                Y[i, j] = y_new
                V[i, j] = v_new

            if outside_01(Y[i]):
                f = 1.0e300
            else:
                x = map_y_to_x(Y[i], lo, hi)
                f = objective_function(x)

                if not np.isfinite(f):
                    f = 1.0e300

            if f < Fpbest[i]:
                Fpbest[i] = float(f)
                PbestY[i, :] = Y[i, :]

                if f < f_gbest:
                    f_gbest = float(f)
                    gbestY[:] = Y[i, :]

        if history_interval is not None and it % int(history_interval) == 0:
            history.append(
                {
                    "iteration": it,
                    "loss_best": float(f_gbest),
                    "stop_reason": STOP_REASON_LABELS[stop_reason],
                }
            )

        if f_gbest <= tol:
            stop_reason = 1
            stop_iter = it
            break

        threshold = eps_improve * max(1.0, best_seen)

        if best_seen - f_gbest > threshold:
            best_seen = float(f_gbest)
            no_improve = 0
        else:
            no_improve += 1

            if no_improve >= patience:
                stop_reason = 2
                stop_iter = it
                break

    if stop_reason == 0:
        stop_reason = 3
        stop_iter = int(n_iters)

    history.append(
        {
            "iteration": stop_iter,
            "loss_best": float(f_gbest),
            "stop_reason": STOP_REASON_LABELS[stop_reason],
        }
    )

    best_x = map_y_to_x(gbestY, lo, hi)

    return best_x.tolist(), float(f_gbest), history, stop_iter, stop_reason