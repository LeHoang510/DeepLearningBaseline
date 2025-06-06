import os
import os.path as osp
import logging
from datetime import datetime
from pathlib import Path

class ColoredFormatter(logging.Formatter):

    COLORS = {
        "DEBUG": "\033[0;36m",  # Cyan
        "INFO": "\033[0;32m",  # Green
        "WARNING": "\033[0;33m",  # Yellow
        "ERROR": "\033[0;31m",  # Red
        "CRITICAL": "\033[0;37m\033[41m",  # White on Red BG
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        msg = (
            self.COLORS.get(record.levelname, self.COLORS["RESET"]) 
            + super().format(record) 
            + self.COLORS["RESET"]
        )
        return msg


class Logger:
    _instance = None  # Singleton

    def __new__(cls, input_name: Path|str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_logger(input_name)
        return cls._instance

    def _init_logger(self, input_name: Path|str = None):
        self._has_error = False
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        base_folder = "logs"

        self.log_dir = Path(base_folder) / (input_name or "")

        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = osp.join(self.log_dir, f"{timestamp}.log")


        self.logger = logging.getLogger("SystemLogger")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        file_formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%H:%M:%S"
        )

        console_formatter = ColoredFormatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%H:%M:%S"
        )

        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)

        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def info(self, msg: str):
        self.logger.info(msg, stacklevel=2)

    def debug(self, msg: str):
        self.logger.debug(msg, stacklevel=2)

    def warning(self, msg: str):
        self.logger.warning(msg, stacklevel=2)

    def error(self, msg: str):
        self._has_error = True
        self.logger.error(msg, stacklevel=2)

    def critical(self, msg: str):
        self.logger.critical(msg, stacklevel=2)

    def get_log_path(self) -> Path:
        return self.log_file

    def has_errors(self) -> bool:
        return self._has_error