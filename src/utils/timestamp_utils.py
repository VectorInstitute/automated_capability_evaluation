"""Timestamp utility functions for the pipeline."""

from datetime import datetime


def timestamp_tag() -> str:
    """Return a timestamp tag in `_YYYYMMDD_HHMMSS` format."""
    return f"_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"


def iso_timestamp() -> str:
    """Return an ISO 8601 formatted timestamp with UTC timezone."""
    return datetime.utcnow().isoformat() + "Z"
