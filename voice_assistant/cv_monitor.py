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
                logger.warning("[CVMonitor] Poll failed: %s. Retry in %.0fs.", exc, self._retry_delay)
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
        """GET /api/v1/sessions/{id}/latest → push snapshot to queue.
        Falls back to local .jsonl file when backend is unreachable.
        """
        url = f"{self.backend_url}/api/v1/sessions/{self.session_id}/latest"
        data = None

        try:
            data = self._get(url)
        except _PollError:
            data = self._read_local_snapshot()
            if data is None:
                raise  # both failed — trigger backoff warning

        if data is None:
            return

        self._put({"type": "snapshot", "data": data})
        logger.debug(
            "[CVMonitor] Snapshot → focus=%.0f posture=%.0f vigilance=%.0f",
            data.get("global_focus_score", -1),
            data.get("posture_score", -1),
            data.get("vigilance_score", -1),
        )

    def _read_local_snapshot(self) -> Optional[dict]:
        """Read the latest snapshot from the most recently active session file.

        Sorts by the timestamp embedded in the last JSON line of each file,
        so the file with the freshest data is always used regardless of mtime.
        """
        if not config.SESSIONS_DIR.exists():
            return None

        all_files = [
            f for f in config.SESSIONS_DIR.glob("*.jsonl")
            if not f.stem.endswith("_summary")
        ]
        if not all_files:
            return None

        scored = []
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
                if not entry.get("scores"):
                    continue
                scored.append((entry.get("timestamp", ""), f, entry))
            except (OSError, json.JSONDecodeError):
                continue

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        _, _, raw = scored[0]

        scores = raw["scores"]
        fatigue_ratio = float(scores.get("fatigue", 0.0))
        vigilance = max(0.0, (1.0 - min(fatigue_ratio, 1.0)) * 100.0)
        return {
            "attention_score": scores.get("concentration"),
            "posture_score": scores.get("posture"),
            "vigilance_score": vigilance,
            "global_focus_score": scores.get("focus_global"),
            "state": raw.get("state"),
        }

    def _poll_critical_events(self) -> None:
        """GET /api/v1/vision/events?... → push critical events to queue.

        This endpoint is best-effort: if it returns 404 or is not implemented
        by the backend, we silently skip it. The snapshot scores cover the
        majority of alert cases; this is only for explicit "critical" events.
        """
        url = (
            f"{self.backend_url}/api/v1/vision/events"
            f"?session_id={self.session_id}&level=critical&limit=5"
        )
        try:
            resp = requests.get(url, headers=self._headers, timeout=4.0)
        except requests.exceptions.RequestException:
            return  # Silently skip — events endpoint is optional

        if resp.status_code in (404, 405, 501):
            return  # Endpoint not implemented — skip

        if resp.status_code != 200:
            return

        try:
            events: Any = resp.json()
        except ValueError:
            return

        if not isinstance(events, list):
            events = events.get("events") or events.get("items") or []

        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("level") != "critical":
                continue

            event_id = str(event.get("id") or event.get("timestamp") or id(event))
            if event_id in self._seen_event_ids:
                continue

            self._seen_event_ids.add(event_id)
            # Prune seen-set to avoid unbounded growth
            if len(self._seen_event_ids) > 200:
                self._seen_event_ids = set(list(self._seen_event_ids)[-100:])

            self._put({
                "type": "critical_event",
                "data": {
                    "description": event.get("description", "Alerte critique"),
                    "event_type": event.get("event_type", "unknown"),
                    "level": "critical",
                },
            })
            logger.info("[CVMonitor] Critical event: %s", event.get("description", ""))

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
