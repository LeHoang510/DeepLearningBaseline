import os
import logging
from datetime import datetime

class Logger:
    _instance = None  # Singleton

    def __new__(cls, input_name=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_logger(input_name)
        return cls._instance

    def _init_logger(self, input_name):
        timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
        base_folder = "logs"

        if input_name:
            self.log_dir = os.path.join(base_folder, input_name, timestamp)
        else:
            self.log_dir = os.path.join(base_folder, timestamp)

        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "log.txt")

        self.logger = logging.getLogger("SystemLogger")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%H:%M:%S"
        )

        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def info(self, msg):
        self.logger.info(msg, stacklevel=2)

    def debug(self, msg):
        self.logger.debug(msg, stacklevel=2)

    def warning(self, msg):
        self.logger.warning(msg, stacklevel=2)

    def error(self, msg):
        self.logger.error(msg, stacklevel=2)

    def get_log_path(self):
        return self.log_file
