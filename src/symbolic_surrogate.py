# src/symbolic_surrogate.py

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


class SymbolicForwardSurrogate:
    def __init__(self, models, feature_source_columns, feature_model_names):
        self.models = models
        self.feature_source_columns = feature_source_columns
        self.feature_model_names = feature_model_names

    @staticmethod
    def load_model(path):
        path = Path(path)

        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def from_config(cls, config, root):
        feature_source_columns = config["features"]["source_columns"]
        feature_model_names = config["features"]["model_names"]

        models = {}

        for model_name, model_config in config["models"].items():
            model_path = root / model_config["path"]
            models[model_name] = {
                "model": cls.load_model(model_path),
                "output_transform": model_config.get("output_transform", "identity"),
            }

        return cls(
            models=models,
            feature_source_columns=feature_source_columns,
            feature_model_names=feature_model_names,
        )

    def prepare_X(self, df_or_dict):
        if isinstance(df_or_dict, dict):
            df = pd.DataFrame([df_or_dict])
        else:
            df = df_or_dict.copy()

        X = df[self.feature_source_columns].copy()
        X.columns = self.feature_model_names

        return X

    @staticmethod
    def apply_output_transform(values, transform):
        values = np.asarray(values, dtype=float)

        if transform == "identity":
            return values

        if transform == "square":
            return np.maximum(values, 1e-12) ** 2

        raise ValueError(f"Unsupported output transform: {transform}")

    def predict_dataframe(self, df):
        X = self.prepare_X(df)

        output = {}

        for model_name, model_info in self.models.items():
            raw_pred = model_info["model"].predict(X)
            pred = self.apply_output_transform(
                raw_pred,
                model_info["output_transform"],
            )

            if model_name == "sqrt_sigma":
                output["sigma"] = pred
            else:
                output[model_name] = pred

        return pd.DataFrame(output)

    def predict_one(self, x):
        pred_df = self.predict_dataframe(x)
        row = pred_df.iloc[0]

        return {
            "Vp": float(row["Vp"]),
            "Vs": float(row["Vs"]),
            "sigma": float(row["sigma"]),
        }