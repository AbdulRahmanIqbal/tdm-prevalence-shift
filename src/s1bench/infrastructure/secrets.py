"""Minimal .env reader (KEY=VALUE lines). Values are returned, never logged or written anywhere."""

from __future__ import annotations

import os
from pathlib import Path


def load_env(path: Path = Path(".env")) -> dict[str, str]:
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return {**values, **{k: v for k, v in os.environ.items() if k in values}}
