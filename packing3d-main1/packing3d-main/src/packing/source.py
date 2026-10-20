"""Source-checkout identity; no distribution or release version is maintained."""

from pathlib import Path
import subprocess

SOURCE_LABEL = "source"


def source_revision() -> str | None:
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True, timeout=3,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
