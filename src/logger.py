import logging
import os


# Component bit masks for selective logging.
# LOG_MASK is an integer built by OR-ing the desired components.
# Examples:
#   0  (0000) : nothing
#   1  (0001) : cache only
#   2  (0010) : instances only
#   4  (0100) : router only
#   8  (1000) : simulation only
#   15 (1111) : everything
LOG_NONE = 0
LOG_CACHE = 1 << 0
LOG_INSTANCE = 1 << 1
LOG_ROUTER = 1 << 2
LOG_SIMULATION = 1 << 3
LOG_ALL = LOG_CACHE | LOG_INSTANCE | LOG_ROUTER | LOG_SIMULATION

_COMPONENT_NAMES: dict[int, str] = {
    LOG_CACHE: "CACHE",
    LOG_INSTANCE: "INSTANCE",
    LOG_ROUTER: "ROUTER",
    LOG_SIMULATION: "SIMULATION",
}


# Module-level logger used by log()
logger = logging.getLogger(name="configurator")
logging.basicConfig(
    filename="example.log",
    encoding="utf-8",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Active mask and minimum severity. These can be changed at runtime.
_log_mask: int = LOG_ALL
_min_level: int = logging.DEBUG


def _mask_from_env() -> int:
    """Read LOG_MASK from the environment as an integer."""
    raw = os.environ.get("LOG_MASK", "")
    if not raw:
        return LOG_ALL
    try:
        # Support both decimal and hex strings (e.g. 15 or 0xF)
        return int(raw, 0)
    except ValueError:
        return LOG_ALL


def set_log_mask(mask: int) -> None:
    """Set the active component bitmask."""
    global _log_mask
    _log_mask = mask
    # Ensure the underlying logger is at DEBUG so our filter controls output.
    logger.setLevel(logging.DEBUG)


def set_min_level(level: int) -> None:
    """Set the minimum logging level (e.g. logging.DEBUG or logging.INFO)."""
    global _min_level
    _min_level = level
    logger.setLevel(level)


def set_debug(enabled: bool) -> None:
    """Legacy helper: enable all debug logging when True."""
    if enabled:
        set_log_mask(LOG_ALL)
        set_min_level(logging.DEBUG)
    else:
        set_log_mask(LOG_NONE)
        set_min_level(logging.WARNING)


def get_log_mask() -> int:
    """Return the current component bitmask."""
    return _log_mask


def is_debug() -> bool:
    """Return whether debug logging is enabled for all components."""
    return _min_level <= logging.DEBUG and _log_mask == LOG_ALL


def log(component: int, msg: str, level: int = logging.DEBUG) -> None:
    """Log a message if its component is enabled and the level is sufficient.

    Args:
        component: one of the LOG_* bit masks.
        msg: message to log.
        level: Python logging level (default DEBUG).
    """
    if component & _log_mask and level >= _min_level:
        name = _COMPONENT_NAMES.get(component, "UNKNOWN")
        logger.log(level, f"[{name}] {msg}")


def debug_print(msg: str) -> None:
    """Legacy debug print that logs under the INSTANCE component."""
    log(LOG_INSTANCE, msg)


# Initialise from environment so `.env` / shell exports work out of the box.
set_log_mask(_mask_from_env())
