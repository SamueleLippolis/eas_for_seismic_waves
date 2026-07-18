from pathlib import Path
import argparse

import numpy as np
import pandas as pd


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


def safe_scale(values, min_scale=1.0e-12):
    """
    Robust scale used for normalized distance.

    Prefer std. If std is almost zero, use range.
    If range is also almost zero, return 1.0 to avoid division by zero.
    """
    std = float(values.std(ddof=0))

    if np.isfinite(std) and std > min_scale:
        return std

    value_range = float(values.max() - values.min())

    if np.isfinite(value_range) and value_range > min_scale:
        return value_range

    return 1.0


def add_central_score(feasible):
    """
    Add score measuring distance from the median candidate cloud.

    Lower score means the candidate is more central among feasible PSO solutions.
    """
    feasible = feasible.copy()

    medians = {
        param: float(feasible[param].median())
        for param in PARAMS
    }

    scales = {
        param: safe_scale(feasible[param])
        for param in PARAMS
    }

    score = np.zeros(len(feasible), dtype=float)

    for param in PARAMS:
        score += np.abs(feasible[param] - medians[param]) / scales[param]

    feasible["central_score"] = score

    for param in PARAMS:
        feasible[f"{param}_median_feasible"] = medians[param]
        feasible[f"{param}_scale_feasible"] = scales[param]

    return feasible


def select_best_loss(group):
    return group.loc[group["loss_hat"].idxmin()].copy()


def select_oracle_nonfair(group):
    """
    Non-fair diagnostic selection.
    Uses x_true through oracle_score.
    """
    return group.loc[group["oracle_score"].idxmin()].copy()


def select_central_candidate(group, loss_epsilon):
    """
    Fair rule.

    1. Keep candidates with loss <= min_loss + epsilon.
    2. Among them, choose the candidate closest to the median of the feasible cloud.
    """
    min_loss = float(group["loss_hat"].min())
    feasible = group[group["loss_hat"] <= min_loss + loss_epsilon].copy()

    if len(feasible) == 0:
        feasible = group.copy()

    feasible = add_central_score(feasible)

    return feasible.loc[feasible["central_score"].idxmin()].copy()


def build_selection_row(C_true, selection_name, selected, n_candidates, n_feasible):
    row = {
        "C_true": float(C_true),
        "selection": selection_name,
        "n_candidates": int(n_candidates),
        "n_feasible": int(n_feasible),
        "seed_index": int(selected["seed_index"]),
        "seed": int(selected["seed"]),
        "loss_hat": float(selected["loss_hat"]),
        "oracle_score": float(selected["oracle_score"]),
    }

    if "central_score" in selected:
        row["central_score"] = float(selected["central_score"])
    else:
        row["central_score"] = np.nan

    for param in PARAMS:
        row[param] = float(selected[param])

    for error_col in ERROR_COLS:
        row[error_col] = float(selected[error_col])

    return row


def compare_selection_rules(df, loss_epsilon):
    rows = []

    for C_true, group in df.groupby("C_true"):
        min_loss = float(group["loss_hat"].min())
        feasible = group[group["loss_hat"] <= min_loss + loss_epsilon]

        best_loss = select_best_loss(group)
        central_candidate = select_central_candidate(group, loss_epsilon)
        oracle_nonfair = select_oracle_nonfair(group)

        selections = [
            ("best_loss", best_loss),
            ("central_candidate", central_candidate),
            ("oracle_nonfair", oracle_nonfair),
        ]

        for selection_name, selected in selections:
            row = build_selection_row(
                C_true=C_true,
                selection_name=selection_name,
                selected=selected,
                n_candidates=len(group),
                n_feasible=len(feasible),
            )
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_selection_results(selection_df):
    rows = []

    for selection_name, group in selection_df.groupby("selection"):
        row = {
            "selection": selection_name,
            "n_cases": int(len(group)),
            "mean_loss_hat": float(group["loss_hat"].mean()),
            "median_loss_hat": float(group["loss_hat"].median()),
            "mean_oracle_score": float(group["oracle_score"].mean()),
            "median_oracle_score": float(group["oracle_score"].median()),
        }

        for error_col in ERROR_COLS:
            row[f"mean_{error_col}"] = float(group[error_col].mean())
            row[f"median_{error_col}"] = float(group[error_col].median())
            row[f"max_{error_col}"] = float(group[error_col].max())

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    selection_order = {
        "best_loss": 0,
        "central_candidate": 1,
        "oracle_nonfair": 2,
    }

    summary_df["selection_order"] = summary_df["selection"].map(selection_order)
    summary_df = summary_df.sort_values("selection_order").drop(columns=["selection_order"])

    return summary_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to PSO multi-seed report directory.",
    )

    parser.add_argument(
        "--loss-epsilon",
        type=float,
        default=1.0e-6,
        help="Candidates with loss <= min_loss + epsilon are feasible.",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    input_path = run_dir / "pso_multiseed_best_all.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input file: {input_path}")

    df = pd.read_csv(input_path)

    output_dir = run_dir / "selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_df = compare_selection_rules(
        df=df,
        loss_epsilon=args.loss_epsilon,
    )

    summary_df = summarize_selection_results(selection_df)

    selection_path = output_dir / "selection_rules_by_clay.csv"
    summary_path = output_dir / "selection_rules_summary.csv"

    selection_df.to_csv(selection_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nSelection completed.")
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")

    print("\nCreated:")
    print(f"  {selection_path.name}")
    print(f"  {summary_path.name}")

    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()