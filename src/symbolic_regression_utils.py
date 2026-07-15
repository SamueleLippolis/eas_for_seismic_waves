# src/symbolic_regression_utils.py

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from requests.packages import target
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pysr import PySRRegressor


def load_symbolic_dataset(dataset_path):
    dataset_path = Path(dataset_path)

    if dataset_path.suffix == ".parquet":
        return pd.read_parquet(dataset_path)

    if dataset_path.suffix == ".csv":
        return pd.read_csv(dataset_path)

    raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")

def apply_target_transform(values, transform):
    values = np.asarray(values, dtype=float)

    if transform is None or transform == "identity":
        return values

    if transform == "sqrt":
        return np.sqrt(values)

    if transform == "log":
        return np.log(values)

    raise ValueError(f"Unsupported target transform: {transform}")

def prepare_xy(df, feature_source_columns, feature_model_names, target_config):
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    X_train = train_df[feature_source_columns].copy()
    X_test = test_df[feature_source_columns].copy()

    X_train.columns = feature_model_names
    X_test.columns = feature_model_names

    if isinstance(target_config, str):
        target_name = target_config
        source_column = target_config
        transform = "identity"
    else:
        target_name = target_config["name"]
        source_column = target_config["source_column"]
        transform = target_config.get("transform", "identity")

    y_train_raw = train_df[source_column].to_numpy(dtype=float)
    y_test_raw = test_df[source_column].to_numpy(dtype=float)

    y_train = apply_target_transform(y_train_raw, transform)
    y_test = apply_target_transform(y_test_raw, transform)

    return X_train, y_train, X_test, y_test, target_name, source_column, transform


def build_pysr_model(pysr_config):
    return PySRRegressor(
        niterations=int(pysr_config["niterations"]),
        populations=int(pysr_config["populations"]),
        population_size=int(pysr_config["population_size"]),
        maxsize=int(pysr_config["maxsize"]),
        model_selection=pysr_config["model_selection"],
        random_state=int(pysr_config["random_state"]),
        binary_operators=pysr_config["binary_operators"],
        unary_operators=pysr_config["unary_operators"],
        verbosity=1,
    )


def compute_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def save_true_vs_pred_plot(y_true, y_pred, target, path):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)

    min_value = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_value = max(float(np.max(y_true)), float(np.max(y_pred)))

    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    plt.xlabel(f"True {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"Symbolic regression: true vs predicted {target}")
    plt.grid(True)

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def train_symbolic_model_for_target(
    df,
    target_config,
    feature_source_columns,
    feature_model_names,
    pysr_config,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test, target, source_column, transform = prepare_xy(
        df=df,
        feature_source_columns=feature_source_columns,
        feature_model_names=feature_model_names,
        target_config=target_config,
    )

    target_dir = output_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)

    model = build_pysr_model(pysr_config)

    print(f"\nTraining symbolic model for target: {target}")
    print(f"Source column: {source_column}")
    print(f"Transform: {transform}")

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = compute_regression_metrics(y_train, y_train_pred)
    test_metrics = compute_regression_metrics(y_test, y_test_pred)

    metrics = {
        "target": target,
        "source_column": source_column,
        "transform": transform,
        "train": train_metrics,
        "test": test_metrics,
    }

    equations_path = target_dir / "equations.csv"
    model_path = target_dir / "model.pkl"
    metrics_path = target_dir / "metrics.json"
    predictions_path = target_dir / "test_predictions.parquet"
    plot_path = target_dir / "true_vs_pred.png"

    model.equations_.to_csv(equations_path, index=False)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    predictions_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_test_pred,
            "residual": y_test_pred - y_test,
        }
    )
    predictions_df.to_parquet(predictions_path, index=False)

    save_true_vs_pred_plot(
        y_true=y_test,
        y_pred=y_test_pred,
        target=target,
        path=plot_path,
    )

    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))

    print("\nSaved:")
    print(f"  equations:   {equations_path}")
    print(f"  model:       {model_path}")
    print(f"  metrics:     {metrics_path}")
    print(f"  predictions: {predictions_path}")
    print(f"  plot:        {plot_path}")

    return metrics