from pathlib import Path
import argparse
import itertools

import pandas as pd
import numpy as np


PARAMS = [
    "phi_percent_hat",
    "C_percent_hat",
    "S_b_percent_hat",
    "sigma_b_inv_hat",
    "xi_hat",
]

ERROR_COLS = [
    "phi_percent_abs_error_percentage_points",
    "C_percent_abs_error_percentage_points",
    "S_b_percent_abs_error_percentage_points",
    "sigma_b_inv_abs_error_relative_percent",
    "xi_abs_error_relative_percent",
]


def summarize_distribution(df, loss_epsilon):
    rows = []

    for C_true, group in df.groupby("C_true"):
        min_loss = group["loss_hat"].min()
        feasible = group[group["loss_hat"] <= min_loss + loss_epsilon]

        best_loss = group.loc[group["loss_hat"].idxmin()]
        best_oracle = group.loc[group["oracle_score"].idxmin()]

        row = {
            "C_true": float(C_true),
            "n_seeds": int(len(group)),
            "loss_epsilon": float(loss_epsilon),
            "min_loss": float(min_loss),
            "median_loss": float(group["loss_hat"].median()),
            "max_loss": float(group["loss_hat"].max()),
            "n_feasible": int(len(feasible)),

            "best_loss_seed_index": int(best_loss["seed_index"]),
            "best_loss_seed": int(best_loss["seed"]),
            "best_loss_oracle_score": float(best_loss["oracle_score"]),

            "best_oracle_seed_index": int(best_oracle["seed_index"]),
            "best_oracle_seed": int(best_oracle["seed"]),
            "best_oracle_loss": float(best_oracle["loss_hat"]),
            "best_oracle_score": float(best_oracle["oracle_score"]),
        }

        for param in PARAMS:
            row[f"{param}_mean"] = float(group[param].mean())
            row[f"{param}_std"] = float(group[param].std(ddof=0))
            row[f"{param}_min"] = float(group[param].min())
            row[f"{param}_max"] = float(group[param].max())
            row[f"{param}_range"] = float(group[param].max() - group[param].min())

            row[f"feasible_{param}_mean"] = float(feasible[param].mean())
            row[f"feasible_{param}_std"] = float(feasible[param].std(ddof=0))
            row[f"feasible_{param}_min"] = float(feasible[param].min())
            row[f"feasible_{param}_max"] = float(feasible[param].max())
            row[f"feasible_{param}_range"] = float(feasible[param].max() - feasible[param].min())

            row[f"best_loss_{param}"] = float(best_loss[param])
            row[f"best_oracle_{param}"] = float(best_oracle[param])

        for error_col in ERROR_COLS:
            row[f"best_loss_{error_col}"] = float(best_loss[error_col])
            row[f"best_oracle_{error_col}"] = float(best_oracle[error_col])

        rows.append(row)

    return pd.DataFrame(rows)


def compute_correlations(df):
    rows = []

    for C_true, group in df.groupby("C_true"):
        for a, b in itertools.combinations(PARAMS, 2):
            if group[a].std(ddof=0) == 0 or group[b].std(ddof=0) == 0:
                corr = np.nan
            else:
                corr = group[[a, b]].corr().iloc[0, 1]

            rows.append(
                {
                    "C_true": float(C_true),
                    "param_a": a,
                    "param_b": b,
                    "correlation": None if pd.isna(corr) else float(corr),
                }
            )

    return pd.DataFrame(rows)


def build_selection_comparison(df, loss_epsilon):
    rows = []

    for C_true, group in df.groupby("C_true"):
        min_loss = group["loss_hat"].min()
        feasible = group[group["loss_hat"] <= min_loss + loss_epsilon]

        best_loss = group.loc[group["loss_hat"].idxmin()]
        best_oracle_all = group.loc[group["oracle_score"].idxmin()]
        best_oracle_feasible = feasible.loc[feasible["oracle_score"].idxmin()]

        for selection_name, selected in [
            ("best_loss", best_loss),
            ("best_oracle_all_nonfair", best_oracle_all),
            ("best_oracle_feasible_nonfair", best_oracle_feasible),
        ]:
            row = {
                "C_true": float(C_true),
                "selection": selection_name,
                "seed_index": int(selected["seed_index"]),
                "seed": int(selected["seed"]),
                "loss_hat": float(selected["loss_hat"]),
                "oracle_score": float(selected["oracle_score"]),
            }

            for param in PARAMS:
                row[param] = float(selected[param])

            for error_col in ERROR_COLS:
                row[error_col] = float(selected[error_col])

            rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the PSO multi-seed report directory.",
    )
    parser.add_argument(
        "--loss-epsilon",
        type=float,
        default=1.0e-6,
        help="Candidates with loss <= min_loss + epsilon are considered feasible.",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    input_path = run_dir / "pso_multiseed_best_all.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find {input_path}")

    df = pd.read_csv(input_path)

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    distribution_df = summarize_distribution(df, args.loss_epsilon)
    correlation_df = compute_correlations(df)
    selection_df = build_selection_comparison(df, args.loss_epsilon)

    distribution_df.to_csv(analysis_dir / "distribution_by_clay.csv", index=False)
    correlation_df.to_csv(analysis_dir / "correlations_by_clay.csv", index=False)
    selection_df.to_csv(analysis_dir / "selection_comparison_by_clay.csv", index=False)

    print("\nAnalysis completed.")
    print(f"Input: {input_path}")
    print(f"Output directory: {analysis_dir}")
    print("\nCreated:")
    print("  distribution_by_clay.csv")
    print("  correlations_by_clay.csv")
    print("  selection_comparison_by_clay.csv")


if __name__ == "__main__":
    main()