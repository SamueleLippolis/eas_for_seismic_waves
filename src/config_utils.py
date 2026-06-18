# src/config_utils.py

import yaml


def load_yaml(path):
    """
    Load a YAML file.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)