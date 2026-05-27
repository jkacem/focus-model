from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np

def _parse_ts(ts: str) -> datetime:
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(tz=timezone.utc)

@dataclass
class SessionTracker:
    session_id: str
    _start_ts: datetime | None = None
    _last_ts: datetime | None = None

    _scores: list[dict] = field(default_factory=list)

    def add_frame(self, frame: dict):
        ts = _parse_ts(str(frame.get("timestamp", "")))
        scores = frame.get("scores", {})

        if self._start_ts is None:
            self._start_ts = ts
            self._last_ts = ts

        self._last_ts = ts
        
        # Guard clause in case a frame has no valid scores yet
        if "focus_global" in scores:
            self._scores.append(scores)

    def get_session_metrics(self) -> dict:
        if not self._scores:
            return {
                "concentration": 0.0,
                "posture": 0.0,
                "fatigue": 0.0,
                "distraction": 0.0,
                "focus_global": 0.0
            }

        concentration = np.mean([s.get("concentration", 0.0) for s in self._scores])
        
        posture_vals = [s.get("posture") for s in self._scores if s.get("posture") is not None]
        posture = np.mean(posture_vals) if posture_vals else None
        
        fatigue = np.mean([s.get("fatigue", 0.0) for s in self._scores])
        distraction = np.mean([s.get("distraction", 0.0) for s in self._scores])
        focus_global = np.mean([s.get("focus_global", 0.0) for s in self._scores])

        if fatigue >= 99.0:
            focus_global = min(focus_global, 50.0)
        if distraction >= 99.0:
            focus_global = min(focus_global, 10.0)

        return {
            "concentration": round(float(concentration), 1),
            "posture": round(float(posture), 1) if posture is not None else None,
            "fatigue": round(float(fatigue), 1),
            "distraction": round(float(distraction), 1),
            "focus_global": round(float(focus_global), 1)
        }

    # Old interface for backward compatibility with `main_cv.py` if it explicitly calls finalize()
    def finalize(self) -> dict:
        return self.get_session_metrics()
