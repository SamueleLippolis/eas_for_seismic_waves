# scripts/compare_giulio_part1.py

from pathlib import Path
import sys
import csv
from datetime import datetime

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


REPORTS_ROOT = ROOT / "reports" / "giulio_part1"
OPTIMIZERS = ["pso", "cmaes", "simulated_annealing"]


def get_latest_run_dir(optimizer_name):
    optimizer_dir = REPORTS_ROOT / optimizer_name

    if not optimizer_dir.exists():
        raise FileNotFoundError(f"No report directory found for {optimizer_name}")

    run_dirs = [
        path for path in optimizer_dir.iterdir()
        if path.is_dir()
    ]

    if len(run_dirs) == 0:
        raise FileNotFoundError(f"No runs found for {optimizer_name}")

    latest_run_dir = max(run_dirs, key=lambda path: path.stat().st_mtime)

    return latest_run_dir


def read_summary_csv(path, optimizer_name):
    rows = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            row = dict(row)
            row["optimizer"] = optimizer_name

            # Convert numeric fields
            for key, value in row.items():
                if key == "optimizer":
                    continue

                try:
                    row[key] = float(value)
                except ValueError:
                    pass

            rows.append(row)

    return rows


def save_combined_csv(rows, path):
    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows_by_optimizer, metric_key, ylabel, title, output_path):
    plt.figure()

    for optimizer_name, rows in rows_by_optimizer.items():
        C_true = [row["C_true"] for row in rows]
        values = [row[metric_key] for row in rows]

        plt.plot(C_true, values, marker="o", label=optimizer_name)

    plt.axhline(0.0, linestyle="--")
    plt.xlabel("True clay content C [%]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_loss(rows_by_optimizer, output_path):
    plt.figure()

    for optimizer_name, rows in rows_by_optimizer.items():
        C_true = [row["C_true"] for row in rows]
        losses = [row["loss_at_recovered_x"] for row in rows]

        plt.plot(C_true, losses, marker="o", label=optimizer_name)

    plt.xlabel("True clay content C [%]")
    plt.ylabel("Recovered loss")
    plt.title("Recovered objective loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    comparison_dir = REPORTS_ROOT / "comparison" / f"comparison_{timestamp}"
    comparison_dir.mkdir(parents=True, exist_ok=False)

    rows_by_optimizer = {}
    all_rows = []

    for optimizer_name in OPTIMIZERS:
        latest_run_dir = get_latest_run_dir(optimizer_name)
        summary_path = latest_run_dir / "summary_results.csv"

        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary_results.csv in {latest_run_dir}")

        rows = read_summary_csv(summary_path, optimizer_name)

        rows_by_optimizer[optimizer_name] = rows
        all_rows.extend(rows)

        print(f"{optimizer_name}:")
        print(f"  loaded from {summary_path}")

    save_combined_csv(
        rows=all_rows,
        path=comparison_dir / "comparison_summary.csv",
    )

    plot_metric(
        rows_by_optimizer=rows_by_optimizer,
        metric_key="C_percent_error_percentage_points",
        ylabel="C error [percentage points]",
        title="Clay recovery error",
        output_path=comparison_dir / "clay_error_comparison.png",
    )

    plot_metric(
        rows_by_optimizer=rows_by_optimizer,
        metric_key="C_percent_abs_error_percentage_points",
        ylabel="Absolute C error [percentage points]",
        title="Absolute clay recovery error",
        output_path=comparison_dir / "clay_abs_error_comparison.png",
    )

    plot_loss(
        rows_by_optimizer=rows_by_optimizer,
        output_path=comparison_dir / "loss_comparison.png",
    )

    print("\nComparison completed.")
    print("Saved to:")
    print(f"  {comparison_dir}")


if __name__ == "__main__":
    main()