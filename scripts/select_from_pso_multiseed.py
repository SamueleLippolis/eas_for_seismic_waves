from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config_utils import load_yaml


ERROR_COLS = [
    "phi_percent_abs_error_percentage_points",
    "C_percent_abs_error_percentage_points",
    "S_b_percent_abs_error_percentage_points",
    "sigma_b_inv_abs_error_relative_percent",
    "xi_abs_error_relative_percent",
]


def safe_scale(values, min_scale=1.0e-12):
    std = float(values.std(ddof=0))
    if np.isfinite(std) and std > min_scale:
        return std

    value_range = float(values.max() - values.min())
    if np.isfinite(value_range) and value_range > min_scale:
        return value_range

    return 1.0


def central_score(feasible, params, weights=None):
    score = np.zeros(len(feasible), dtype=float)

    if weights is None:
        weights = {}

    for param in params:
        median = float(feasible[param].median())
        scale = safe_scale(feasible[param])
        weight = float(weights.get(param, 1.0))

        score += weight * np.abs(feasible[param] - median) / scale

    return score


def avoid_bounds_score(feasible, bounds, margin_fraction):
    """
    Penalize candidates close to parameter bounds.

    distance_to_bound is normalized in [0, 0.5].
    If distance is larger than margin_fraction, penalty is zero.
    """
    score = np.zeros(len(feasible), dtype=float)

    for param, bound_pair in bounds.items():
        lo, hi = float(bound_pair[0]), float(bound_pair[1])
        width = hi - lo

        if width <= 0:
            raise ValueError(f"Invalid bounds for {param}: {bound_pair}")

        x = feasible[param].astype(float)

        dist_low = (x - lo) / width
        dist_high = (hi - x) / width
        dist_to_nearest_bound = np.minimum(dist_low, dist_high)

        penalty = np.maximum(
            0.0,
            float(margin_fraction) - dist_to_nearest_bound,
        ) / float(margin_fraction)

        score += penalty

    return score


def compute_rule_score(feasible, rule):
    rule_type = rule["type"]

    if rule_type == "central":
        return central_score(
            feasible=feasible,
            params=rule["params"],
            weights=rule.get("weights"),
    )

    if rule_type == "avoid_bounds":
        return avoid_bounds_score(
            feasible=feasible,
            bounds=rule["bounds"],
            margin_fraction=float(rule.get("margin_fraction", 0.15)),
        )

    if rule_type == "combined":
        total_score = np.zeros(len(feasible), dtype=float)

        for component in rule["components"]:
            weight = float(component.get("weight", 1.0))
            component_score = compute_rule_score(feasible, component)
            total_score += weight * component_score

        return total_score

    raise ValueError(f"Cannot compute score for rule type: {rule_type}")


def select_candidate(group, rule, loss_epsilon):
    rule_type = rule["type"]

    if rule_type == "best_loss":
        selected = group.loc[group["loss_hat"].idxmin()].copy()
        selected["selection_score"] = 0.0
        return selected

    if rule_type == "oracle_nonfair":
        selected = group.loc[group["oracle_score"].idxmin()].copy()
        selected["selection_score"] = 0.0
        return selected

    min_loss = float(group["loss_hat"].min())
    feasible = group[group["loss_hat"] <= min_loss + loss_epsilon].copy()

    if len(feasible) == 0:
        feasible = group.copy()

    feasible["selection_score"] = compute_rule_score(feasible, rule)

    # Tie-breaker: if two candidates have identical selection score,
    # prefer the lower observed loss.
    feasible = feasible.sort_values(
        by=["selection_score", "loss_hat"],
        ascending=[True, True],
    )

    return feasible.iloc[0].copy()


def build_selection_row(C_true, rule_name, selected, n_candidates, n_feasible):
    row = {
        "C_true": float(C_true),
        "selection": rule_name,
        "n_candidates": int(n_candidates),
        "n_feasible": int(n_feasible),
        "seed_index": int(selected["seed_index"]),
        "seed": int(selected["seed"]),
        "loss_hat": float(selected["loss_hat"]),
        "oracle_score": float(selected["oracle_score"]),
        "selection_score": float(selected.get("selection_score", np.nan)),
    }

    param_cols = [
        col for col in selected.index
        if col.endswith("_hat")
        and col not in ["Vp_hat", "Vs_hat", "sigma_hat"]
    ]

    for col in param_cols:
        row[col] = float(selected[col])

    for error_col in ERROR_COLS:
        row[error_col] = float(selected[error_col])

    return row


def compare_selection_rules(df, rules, loss_epsilon):
    rows = []

    for C_true, group in df.groupby("C_true"):
        min_loss = float(group["loss_hat"].min())
        feasible = group[group["loss_hat"] <= min_loss + loss_epsilon]

        for rule in rules:
            selected = select_candidate(
                group=group,
                rule=rule,
                loss_epsilon=loss_epsilon,
            )

            row = build_selection_row(
                C_true=C_true,
                rule_name=rule["name"],
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

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config/selection_rules.yaml",
        help="Path to selection config.",
    )

    args = parser.parse_args()

    config = load_yaml(ROOT / args.config)

    run_dir = ROOT / config["input"]["run_dir"]
    loss_epsilon = float(config["input"].get("loss_epsilon", 1.0e-6))
    output_subdir = config["output"].get("subdir", "selection")
    rules = config["rules"]

    input_path = run_dir / "pso_multiseed_best_all.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input file: {input_path}")

    df = pd.read_csv(input_path)

    output_dir = run_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_df = compare_selection_rules(
        df=df,
        rules=rules,
        loss_epsilon=loss_epsilon,
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