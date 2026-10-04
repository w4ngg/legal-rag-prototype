from __future__ import annotations

"""Predictable loading of local process environment files."""

from pathlib import Path

from dotenv import load_dotenv


def load_environment_file(path: str | Path = ".env") -> bool:
    """Load an existing dotenv file without overriding process variables."""

    selected = Path(path).expanduser()
    if not selected.is_file():
        return False
    return bool(load_dotenv(dotenv_path=selected, override=False))
