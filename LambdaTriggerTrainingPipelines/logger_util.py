import logging

FORMATTER = logging.Formatter(
    "%(asctime)s — %(levelname)s — "
    "%(funcName)s:%(lineno)d — %(message)s")

def get_console_logging_handler() -> logging.StreamHandler:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FORMATTER)
    console_handler.setLevel(logging.INFO)

    return console_handler

logger = logging.getLogger(__name__)
logger.addHandler(get_console_logging_handler())
logger.setLevel(logging.DEBUG)