import json


def load_config(config_path):
    """Loads a JSON configuration file."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_path}")
        raise
