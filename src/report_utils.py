# src/report_utils.py

from pathlib import Path
from datetime import datetime
import json
import csv
import yaml

import matplotlib.pyplot as plt
import numpy as np


def create_run_directory(base_dir, category, run_name):
    """
    Create a unique report directory.

    Structure:
        base_dir/category/run_name_timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{run_name}_{timestamp}"

    run_dir = Path(base_dir) / category / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir


def make_json_serializable(obj):
    """
    Convert numpy objects into JSON-serializable Python objects.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_yaml(data, path):
    """
    Save a dictionary as YAML.
    """
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(data, path):
    """
    Save a dictionary as JSON.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4, default=make_json_serializable)


def save_history_csv(history, path):
    """
    Save optimizer history as CSV.

    history is expected to be a list of dictionaries.
    """
    if history is None or len(history) == 0:
        return

    fieldnames = list(history[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_convergence_plot(history, path):
    """
    Save a simple convergence plot from optimizer history.

    The history must contain a 'loss_best' column.
    """
    if history is None or len(history) == 0:
        return

    if "loss_best" not in history[0]:
        return

    iterations = [row["iteration"] for row in history]
    losses = [row["loss_best"] for row in history]

    plt.figure()
    plt.plot(iterations, losses)
    plt.xlabel("Iteration")
    plt.ylabel("Best loss")
    plt.title("Optimization convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_run_report(
    run_dir,
    full_config,
    result,
    history=None,
    save_config=True,
    save_history=True,
    save_plot=True,
):
    """
    Save all standard outputs for one optimization run.
    """
    if save_config:
        save_yaml(full_config, run_dir / "run_config.yaml")

    save_json(result, run_dir / "result.json")

    if save_history:
        save_history_csv(history, run_dir / "history.csv")

    if save_plot:
        save_convergence_plot(history, run_dir / "convergence.png")