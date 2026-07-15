# scripts/build_dataset.py

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config_utils import load_yaml
from src.dataset_builder import build_forward_dataset


CONSTANTS_PATH = ROOT / "config" / "constants.yaml"
DATASET_CONFIG_PATH = ROOT / "config" / "datasets.yaml" 


def main():
    constants = load_yaml(CONSTANTS_PATH)
    config = load_yaml(DATASET_CONFIG_PATH)

    df, dataset_path, metadata_path = build_forward_dataset(
        config=config,
        constants=constants,
    )

    print("\nDataset built successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    print("\nSaved dataset to:")
    print(f"  {dataset_path}")

    print("\nSaved metadata to:")
    print(f"  {metadata_path}")

    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()