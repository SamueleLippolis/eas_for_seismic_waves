# scripts/train_symbolic_forward.py

from pathlib import Path
import sys
from datetime import datetime
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config_utils import load_yaml
from src.symbolic_regression_utils import (
    load_symbolic_dataset,
    train_symbolic_model_for_target,
)


CONFIG_PATH = ROOT / "config" / "symbolic_regression.yaml"


def main():
    config = load_yaml(CONFIG_PATH)

    experiment_name = config["experiment"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = (
        ROOT
        / config["experiment"]["output_dir"]
        / f"{experiment_name}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = ROOT / config["dataset"]["path"]
    df = load_symbolic_dataset(dataset_path)

    feature_source_columns = config["features"]["source_columns"]
    feature_model_names = config["features"]["model_names"]
    targets = config["targets"]
    pysr_config = config["pysr"]

    with open(output_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_metrics = {}

    for target_config in targets:
        metrics = train_symbolic_model_for_target(
            df=df,
            target_config=target_config,
            feature_source_columns=feature_source_columns,
            feature_model_names=feature_model_names,
            pysr_config=pysr_config,
            output_dir=output_dir,
        )
    
        all_metrics[metrics["target"]] = metrics

    with open(output_dir / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\nSymbolic regression training completed.")
    print("Report saved to:")
    print(f"  {output_dir}")


if __name__ == "__main__":
    main()