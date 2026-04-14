import logging
import sys
import os
from datetime import datetime


def setup_logger(name: str = "trading_analyzer") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Console handler (always works)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (only if logs folder can be created)
        try:
            os.makedirs("logs", exist_ok=True)
            file_handler = logging.FileHandler(
                f"logs/{name}_{datetime.now().strftime('%Y%m%d')}.log"
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (OSError, PermissionError):
            # Streamlit Cloud / read-only filesystem — skip file logging
            pass

    return logger


logger = setup_logger()