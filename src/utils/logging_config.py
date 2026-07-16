import logging

_LOG_FORMAT = "%(asctime)s [%(name)s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(tag: str) -> logging.Logger:
    """
    Returns a logger tagged with `tag`, formatted as "HH:MM:SS [TAG] message".
    Safe to call repeatedly for the same tag (e.g. once per module import) --
    handlers are only attached once per logger name.
    """
    logger = logging.getLogger(tag)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
