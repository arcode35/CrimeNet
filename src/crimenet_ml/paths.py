from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Return the nearest parent directory containing pyproject.toml or .git."""

    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for directory in (current, *current.parents):
        if (directory / "pyproject.toml").is_file() or (directory / ".git").exists():
            return directory

    raise FileNotFoundError(
        "Could not find the CrimeNet project root. "
        "Expected a parent containing pyproject.toml or .git."
    )


PROJECT_ROOT = find_project_root()
