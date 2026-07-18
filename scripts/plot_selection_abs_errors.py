from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


ERROR_SERIES = [
    {
        "column": "phi_percent_abs_error_percentage_points",
        "label": "phi",
    },
    {
        "column": "C_percent_abs_error_percentage_points",
        "label": "C",
    },
    {
        "column": "S_b_percent_abs_error_percentage_points",
        "label": "S_b",
    },
    {
        "column": "sigma_b_inv_abs_error_relative_percent",
        "label": "sigma_b_inv",
    },
    {
        "column": "xi_abs_error_relative_percent",
        "label": "xi",
    },
]


def plot_errors_for_selection(selection_df, selection_name, output_dir):
    df = selection_df[selection_df["selection"] == selection_name].copy()
    df = df.sort_values("C_true")

    plt.figure(figsize=(10, 6))

    for series in ERROR_SERIES:
        plt.plot(
            df["C_true"],
            df[series["column"]],
            marker="o",
            label=series["label"],
        )

    plt.xlabel("Clay percentage")
    plt.ylabel("Absolute parameter recovery error")
    plt.title(f"Absolute parameter recovery errors - {selection_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / f"abs_parameter_errors_{selection_name}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def plot_mean_errors(summary_df, output_dir):
    mean_cols = [
        "mean_phi_percent_abs_error_percentage_points",
        "mean_C_percent_abs_error_percentage_points",
        "mean_S_b_percent_abs_error_percentage_points",
        "mean_sigma_b_inv_abs_error_relative_percent",
        "mean_xi_abs_error_relative_percent",
    ]

    labels = [
        "phi",
        "C",
        "S_b",
        "sigma_b_inv",
        "xi",
    ]

    long_rows = []

    for _, row in summary_df.iterrows():
        for col, label in zip(mean_cols, labels):
            long_rows.append(
                {
                    "selection": row["selection"],
                    "parameter": label,
                    "mean_abs_error": float(row[col]),
                }
            )

    long_df = pd.DataFrame(long_rows)

    selections = list(summary_df["selection"])
    parameters = labels

    x = range(len(selections))
    width = 0.15

    plt.figure(figsize=(12, 6))

    for idx, parameter in enumerate(parameters):
        values = []

        for selection in selections:
            value = long_df[
                (long_df["selection"] == selection)
                & (long_df["parameter"] == parameter)
            ]["mean_abs_error"].iloc[0]
            values.append(value)

        shifted_x = [value + (idx - 2) * width for value in x]

        plt.bar(
            shifted_x,
            values,
            width=width,
            label=parameter,
        )

    plt.xticks(list(x), selections, rotation=30, ha="right")
    plt.ylabel("Mean absolute parameter recovery error")
    plt.title("Mean absolute parameter recovery errors by selection rule")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "mean_abs_errors_by_selection.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--selection-dir",
        required=True,
        help="Directory containing selection_rules_by_clay.csv and selection_rules_summary.csv.",
    )

    args = parser.parse_args()

    selection_dir = Path(args.selection_dir)

    selection_path = selection_dir / "selection_rules_by_clay.csv"
    summary_path = selection_dir / "selection_rules_summary.csv"

    if not selection_path.exists():
        raise FileNotFoundError(f"Cannot find {selection_path}")

    if not summary_path.exists():
        raise FileNotFoundError(f"Cannot find {summary_path}")

    selection_df = pd.read_csv(selection_path)
    summary_df = pd.read_csv(summary_path)

    plot_dir = selection_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    created_paths = []

    for selection_name in selection_df["selection"].unique():
        created_paths.append(
            plot_errors_for_selection(
                selection_df=selection_df,
                selection_name=selection_name,
                output_dir=plot_dir,
            )
        )

    created_paths.append(
        plot_mean_errors(
            summary_df=summary_df,
            output_dir=plot_dir,
        )
    )

    print("\nPlots completed.")
    print(f"Output directory: {plot_dir}")
    print("\nCreated:")
    for path in created_paths:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()