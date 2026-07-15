# src/dataset_builder.py

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.forward_model import forward_model


def sample_random_uniform(bounds, variable_order, n_samples, seed):
    rng = np.random.default_rng(seed)

    X = {}

    for name in variable_order:
        low, high = bounds[name]
        X[name] = rng.uniform(low, high, size=n_samples)

    return X


def assign_split(n_samples, train_fraction, shuffle_before_split, seed):
    indices = np.arange(n_samples)

    if shuffle_before_split:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    n_train = int(train_fraction * n_samples)

    split = np.empty(n_samples, dtype=object)
    split[indices[:n_train]] = "train"
    split[indices[n_train:]] = "test"

    return split


def build_forward_dataset(config, constants):
    dataset_config = config["dataset"]
    variable_order = config["variables"]["order"]
    bounds = config["bounds"]
    sampling = config["sampling"]
    targets = config["targets"]
    options = config.get("options", {})

    method = sampling["method"]
    n_samples = int(sampling["n_samples"])
    seed = int(sampling["seed"])

    if method != "random_uniform":
        raise ValueError(
            f"Unsupported sampling method: {method}. "
            "Currently supported: random_uniform."
        )

    X = sample_random_uniform(
        bounds=bounds,
        variable_order=variable_order,
        n_samples=n_samples,
        seed=seed,
    )

    split = assign_split(
        n_samples=n_samples,
        train_fraction=float(sampling["train_fraction"]),
        shuffle_before_split=bool(sampling["shuffle_before_split"]),
        seed=seed + 1,
    )

    rows = []
    skip_invalid_outputs = bool(options.get("skip_invalid_outputs", True))

    for i in range(n_samples):
        x = {name: float(X[name][i]) for name in variable_order}
        y = forward_model(x, constants)

        row_is_valid = all(np.isfinite(y[target]) for target in targets)

        if skip_invalid_outputs and not row_is_valid:
            continue

        row = {
            "sample_id": i,
            "split": split[i],
        }

        for name in variable_order:
            row[name] = x[name]

        for target in targets:
            row[target] = float(y[target])

        rows.append(row)

    df = pd.DataFrame(rows)

    output_dir = Path(dataset_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = dataset_config["name"]
    output_format = dataset_config.get("output_format", "parquet")

    if output_format == "parquet":
        dataset_path = output_dir / f"{dataset_name}.parquet"
        df.to_parquet(dataset_path, index=False)
    elif output_format == "csv":
        dataset_path = output_dir / f"{dataset_name}.csv"
        df.to_csv(dataset_path, index=False)
    else:
        raise ValueError(
            f"Unsupported output format: {output_format}. "
            "Currently supported: parquet, csv."
        )

    metadata_path = output_dir / f"{dataset_name}_metadata.json"

    metadata = {
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "output_format": output_format,
        "n_requested_samples": n_samples,
        "n_saved_samples": int(len(df)),
        "n_train": int((df["split"] == "train").sum()),
        "n_test": int((df["split"] == "test").sum()),
        "variable_order": variable_order,
        "bounds": bounds,
        "targets": targets,
        "sampling": sampling,
        "options": options,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return df, dataset_path, metadata_path