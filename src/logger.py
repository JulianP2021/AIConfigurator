import logging

# Module-level logger used by debug_print
logger = logging.getLogger("configurator")


def set_debug(enabled: bool) -> None:
    """Enable or disable debug logging globally."""
    level = logging.DEBUG if enabled else logging.WARNING
    logger.setLevel(level)
    # Also set level on root handler if present
    for handler in logging.root.handlers:
        handler.setLevel(level)
    if not logging.root.handlers:
        logging.basicConfig(level=level)


def is_debug() -> bool:
    """Return whether debug logging is enabled."""
    return logger.isEnabledFor(logging.DEBUG)


def debug_print(msg: str) -> None:
    """Log a debug message via the module logger."""
    logger.debug(msg)
