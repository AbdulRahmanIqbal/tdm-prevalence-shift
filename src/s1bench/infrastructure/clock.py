from __future__ import annotations

from datetime import datetime, timezone


class UtcClock:
    def now_iso(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
