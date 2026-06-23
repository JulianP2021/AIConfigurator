import logging


# Module-level logger used by debug_print
logger = logging.getLogger(name="configurator")
# logging.basicConfig(filename='example.log', encoding='utf-8', format='%(asctime)s - %(levelname)s - %(message)s')


def set_debug(enabled: bool) -> None:
    """Enable or disable debug logging globally."""
    level = logging.DEBUG if enabled else logging.WARNING
    logger.setLevel(level)


def is_debug() -> bool:
    """Return whether debug logging is enabled."""
    return logger.isEnabledFor(logging.DEBUG)


def debug_print(msg: str) -> None:
    """Log a debug message via the module logger."""
    logger.debug(msg)
