"""
CV Monitor — Backend State Poller
==================================
Polls the SmartFocus FastAPI backend every POLL_INTERVAL_SECONDS seconds
and puts CV state updates into a shared queue for the AlertManager.

Runs in a daemon background thread — isolated from TTS/STT/wake-word threads.
Not coupled to any provider: it only knows about config values and HTTP.

Queue message schema
--------------------
  {"type": "snapshot", "data": {
      "attention_score": float,
      "posture_score":   float,
      "vigilance_score": float,
      "global_focus_score": float,
      ...   (any extra backend fields are passed through unchanged)
  }}

  {"type": "critical_event", "data": {
      "description": str,
      "event_type":  str,
      "level":       "critical",
  }}
"""

from __future__ import annotations

import json
import logging
import queue
import time
from typing import Any, Optional

import requests

from voice_assistant import config

logger = logging.getLogger("smartfocus.voice")


class CVMonitor:
    """Polls the backend and forwards CV state to the AlertManager via a queue."""

    # Back-off limits for unreachable backend
    _INITIAL_RETRY_DELAY: float = config.POLL_INTERVAL_SECONDS
    _MAX_RETRY_DELAY: float = 60.0

    def __init__(
        self,
        session_id: str,
        backend_url: str,
        state_queue: "queue.Queue[dict]",
    ) -> None:
        self.session_id = session_id
        self.backend_url = backend_url.rstrip("/")
        self.state_queue = state_queue
        self.running = False

        # Track which event IDs we've already reported to avoid duplicates
        self._seen_event_ids: set[str] = set()
        self._retry_delay = self._INITIAL_RETRY_DELAY

        # Headers for authenticated requests
        headers: dict[str, str] = {"Accept": "application/json"}
        if config.PI_API_KEY:
            headers["X-API-Key"] = config.PI_API_KEY
        self._headers = headers

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Main polling loop — call in a daemon thread."""
        self.running = True
        logger.info(
            "[CVMonitor] Started. Session=%s  Poll=%.0fs",
            self.session_id,
            config.POLL_INTERVAL_SECONDS,
        )

        while self.running:
            try:
                self._tick()
                self._retry_delay = self._INITIAL_RETRY_DELAY  # reset on success
            except _PollError as exc:
                logger.debug("[CVMonitor] Poll failed: %s. Retry in %.0fs.", exc, self._retry_delay)
                self._retry_delay = min(self._retry_delay * 2, self._MAX_RETRY_DELAY)
            except Exception as exc:
                logger.error("[CVMonitor] Unexpected error: %s", exc, exc_info=True)

            time.sleep(self._retry_delay)

    def stop(self) -> None:
        """Signal the run loop to stop."""
        self.running = False
        logger.info("[CVMonitor] Stopping.")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        """One polling iteration: fetch snapshot + critical events."""
        self._poll_snapshot()
        self._poll_critical_events()

    def _poll_snapshot(self) -> None:
        """Read latest snapshot from local .jsonl file (100% local mode)."""
        data = self._read_local_snapshot()
        if data is None:
            # No active session yet — skip silently, no backoff
            return
        self._put({"type": "snapshot", "data": data})
        logger.info(
            "[CVMonitor] Snapshot → focus=%.1f posture=%.1f vigilance=%.1f",
            data.get("global_focus_score", -1),
            data.get("posture_score", -1),
            data.get("vigilance_score", -1),
        )

    # Number of recent frames to average for alert snapshots (~15s at 2fps)
    _ALERT_WINDOW_FRAMES: int = 30

    def _read_local_snapshot(self) -> Optional[dict]:
        """Return a sliding-window average of the last N frames from the active
        session JSONL. More stable than a single frame, more reactive than the
        full-session average.
        """
        from datetime import datetime, timezone

        if not config.SESSIONS_DIR.exists():
            return None

        all_files = [
            f for f in config.SESSIONS_DIR.glob("*.jsonl")
            if not f.stem.endswith("_summary")
        ]
        if not all_files:
            return None

        # Pick the file with the most recent last-frame timestamp
        best_file = None
        best_ts = ""
        for f in all_files:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    last = None
                    for line in fh:
                        s = line.strip()
                        if s:
                            last = s
                if not last:
                    continue
                entry = json.loads(last)
                ts = entry.get("timestamp", "")
                if ts > best_ts and entry.get("scores"):
                    best_ts = ts
                    best_file = f
            except (OSError, json.JSONDecodeError):
                continue

        if best_file is None:
            return None

        # Reject stale data — session has ended
        try:
            frame_ts = datetime.strptime(best_ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - frame_ts).total_seconds() > 30:
                return None
        except Exception:
            pass

        # Read last N frames and average their scores
        try:
            lines = []
            with open(best_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    s = line.strip()
                    if s:
                        lines.append(s)
            window = lines[-self._ALERT_WINDOW_FRAMES:]
            entries = [json.loads(l) for l in window]
            scores_list = [e["scores"] for e in entries if e.get("scores")]
            if not scores_list:
                return None

            def avg(key):
                vals = [s.get(key) for s in scores_list if s.get(key) is not None]
                return round(sum(vals) / len(vals), 1) if vals else None

            fatigue_pct = float(avg("fatigue") or 0.0)
            return {
                "attention_score":    avg("concentration"),
                "posture_score":      avg("posture"),
                "fatigue_score":      round(fatigue_pct, 1),
                "vigilance_score":    max(0.0, 100.0 - fatigue_pct),
                "global_focus_score": avg("focus_global"),
            }
        except (OSError, json.JSONDecodeError):
            return None

    def _poll_critical_events(self) -> None:
        """No-op in local mode — critical events come from JSONL alerts field."""
        pass

    def _get(self, url: str) -> Optional[dict]:
        """GET with error handling. Returns None for 404, raises _PollError otherwise."""
        try:
            resp = requests.get(url, headers=self._headers, timeout=5.0)
        except requests.exceptions.ConnectionError as exc:
            raise _PollError(f"Backend unreachable: {exc}") from exc
        except requests.exceptions.Timeout:
            raise _PollError(f"Request timed out: {url}")
        except requests.exceptions.RequestException as exc:
            raise _PollError(f"Request error: {exc}") from exc

        if resp.status_code == 404:
            return None  # Session/snapshot not yet available
        if resp.status_code != 200:
            raise _PollError(f"HTTP {resp.status_code} from {url}")

        try:
            return resp.json()
        except ValueError as exc:
            raise _PollError(f"Invalid JSON from {url}: {exc}") from exc

    def _put(self, message: dict) -> None:
        """Non-blocking queue put — silently drop if queue is full."""
        try:
            self.state_queue.put_nowait(message)
        except queue.Full:
            logger.debug("[CVMonitor] State queue full — dropping message.")


class _PollError(Exception):
    """Raised on transient polling failures to trigger back-off retry."""
