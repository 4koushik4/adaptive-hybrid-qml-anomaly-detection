"""
Configuration Utility

Loads YAML configuration files.
"""

import yaml
import os


class ConfigLoader:

    def __init__(
        self,
        config_path="configs/config.yaml"
    ):

        self.config_path = config_path

        self.config = None

    # ─────────────────────────────────────────────────────

    def load(self):

        if not os.path.exists(
            self.config_path
        ):

            raise FileNotFoundError(
                f"Config file not found: {self.config_path}"
            )

        with open(
            self.config_path,
            "r"
        ) as file:

            self.config = yaml.safe_load(
                file
            )

        return self.config

    # ─────────────────────────────────────────────────────

    def get(
        self,
        key,
        default=None
    ):

        if self.config is None:

            self.load()

        return self.config.get(
            key,
            default
        )