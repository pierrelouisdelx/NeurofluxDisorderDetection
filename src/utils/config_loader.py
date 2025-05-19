import json


class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """Loads a JSON configuration file."""
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {self.config_path}")
            raise
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.config_path}")
        raise
