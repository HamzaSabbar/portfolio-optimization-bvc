"""Shared utility helpers for the BVC portfolio optimization project."""

from pathlib import Path


def is_valid_file(path: str | Path) -> bool:
    """Return True only when path points to an existing non-empty file."""
    try:
        file_path = Path(path)
    except TypeError:
        return False

    try:
        return file_path.is_file() and file_path.stat().st_size > 0
    except OSError:
        return False
