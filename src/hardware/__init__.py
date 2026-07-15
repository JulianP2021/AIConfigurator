"""Hardware definitions and local JSON-backed GPU / machine database."""

from .hardware import Hardware, HardwareSpec
from .legacy.vast_scraper import refresh_file, refresh_machines_file
from .scraper import (
    fetch_machine_hardware,
    lookup,
    lookup_machine,
)


__all__ = [
    "Hardware",
    "HardwareSpec",
    "fetch_machine_hardware",
    "lookup",
    "lookup_machine",
    "refresh_file",
    "refresh_machines_file",
]
