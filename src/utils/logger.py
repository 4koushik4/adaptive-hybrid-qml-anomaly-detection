"""
Logger Utility

Handles:
- Console logging
- File logging
- Stream logging
- Error logging
"""

import logging
import os
from datetime import datetime


class QMLLogger:

    def __init__(
        self,
        log_dir="logs",
        log_name="qml_system"
    ):

        os.makedirs(
            log_dir,
            exist_ok=True
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        log_file = os.path.join(
            log_dir,
            f"{log_name}_{timestamp}.log"
        )

        self.logger = logging.getLogger(
            log_name
        )

        self.logger.setLevel(
            logging.INFO
        )

        # Prevent duplicate handlers
        if not self.logger.handlers:

            formatter = logging.Formatter(

                "[%(asctime)s] "
                "[%(levelname)s] "
                "%(message)s"
            )

            # File Handler
            file_handler = logging.FileHandler(
                log_file
            )

            file_handler.setFormatter(
                formatter
            )

            # Console Handler
            console_handler = logging.StreamHandler()

            console_handler.setFormatter(
                formatter
            )

            self.logger.addHandler(
                file_handler
            )

            self.logger.addHandler(
                console_handler
            )

    # ─────────────────────────────────────────────────────

    def info(
        self,
        message
    ):

        self.logger.info(message)

    # ─────────────────────────────────────────────────────

    def warning(
        self,
        message
    ):

        self.logger.warning(message)

    # ─────────────────────────────────────────────────────

    def error(
        self,
        message
    ):

        self.logger.error(message)

    # ─────────────────────────────────────────────────────

    def debug(
        self,
        message
    ):

        self.logger.debug(message)