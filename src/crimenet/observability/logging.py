# common/logging.py

import logging

from pyspark.logger import PySparkLogger


def get_logger(name: str) -> PySparkLogger:
    logger = PySparkLogger.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger