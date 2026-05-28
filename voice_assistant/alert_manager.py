"""
Alert Manager — Proactive Voice Alerts
=======================================
Monitors the CVMonitor queue and speaks French alerts via TTS when
CV metrics cross their configured thresholds.

Design principles:
  - Each alert type has its own independent cooldown timer.
  - Posture and "good focus" alerts require SUSTAINED conditions
    (posture bad > 2 min, good focus > 20 min) before speaking.
  - All speaking is coordinated through `speak_lock` so alerts never
    interrupt wake-word conversations.
  - The user can silence all alerts for ALERT_MUTE_DURATION seconds
    by saying "silence" / "tais-toi" (handled via mute() method).
  - Break timer: start_break_timer(N) schedules a "break over" alert.

Not coupled to any provider — uses BaseTTS interface only.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

from voice_assistant import config
from voice_assistant.providers.base import BaseTTS

logger = logging.getLogger("smartfocus.voice")


# ── French alert messages ─────────────────────────────────────────────────────

_MSG_FOCUS_LOW = (
    "Attention, ton niveau de concentration a chuté. "
    "Essaie de te refocaliser sur ta tâche."
)

_MSG_POSTURE_BAD = (
    "Redresse-toi ! Ta posture n'est pas bonne depuis quelques minutes. "
    "Pense à ton dos."
)

_MSG_FATIGUE = (
    "Tu sembles fatigué. Une pause de 5 minutes t'aidera à être bien "
    "plus efficace après."
)

_MSG_GOOD_FOCUS = (
    "Excellent travail ! Tu es concentré depuis 20 minutes. "
    "Pense à faire une courte pause bientôt."
)

_MSG_BREAK_OVER = (
    "Ta pause de 5 minutes est terminée. "
    "C'est le moment de reprendre le travail, tu vas y arriver !"
)

_MSG_MUTED = (
    "Compris. Je ne te dérangerai plus pendant 15 minutes."
)

_MSG_BREAK_CONFIRM = (
    "D'accord. Profite bien de ta pause. Je t'avertirai dans 5 minutes."
)


class AlertManager:
    """Reads CV state from a queue and speaks proactive alerts via TTS.

    Args:
        cv_queue:   Queue fed by CVMonitor (snapshots + critical events).
        tts:        TTS engine implementing BaseTTS.
        speak_lock: Shared mutex ensuring only one speaker at a time.
    """

    def __init__(
        self,
        cv_queue: "queue.Queue[dict]",
        tts: BaseTTS,
        speak_lock: threading.Lock,
    ) -> None:
        self.cv_queue = cv_queue
        self.tts = tts
        self.speak_lock = speak_lock

        self.running = False

        # Per-type cooldown: alert_type → timestamp of last firing
        self._last_fired: dict[str, float] = {}

        # Sustained-condition tracking
        self._posture_bad_since: Optional[float] = None
        self._good_focus_since: Optional[float] = None

        # Mute state
        self._muted_until: float = 0.0

        # Latest snapshot (for STATS intent in assistant.py)
        self._latest_snapshot: dict = {}

        # Break timer handle
        self._break_timer: Optional[threading.Timer] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Main processing loop — call in a daemon thread."""
        self.running = True
        logger.info("[Alerts] Alert manager started.")

        while self.running:
            try:
                item = self.cv_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            except Exception as exc:
                logger.error("[Alerts] Queue read error: %s", exc)
                continue

            try:
                self._process_item(item)
            except Exception as exc:
                logger.error("[Alerts] Error processing item: %s", exc, exc_info=True)

    def stop(self) -> None:
        self.running = False
        if self._break_timer is not None:
            self._break_timer.cancel()

    def mute(self) -> None:
        """Silence all proactive alerts for ALERT_MUTE_DURATION seconds."""
        self._muted_until = time.monotonic() + config.ALERT_MUTE_DURATION
        logger.info("[Alerts] Muted for %d seconds.", config.ALERT_MUTE_DURATION)
        self._speak_alert(_MSG_MUTED)

    def unmute(self) -> None:
        """Unmute alerts immediately."""
        self._muted_until = 0.0
        logger.info("[Alerts] Unmuted.")

    def start_break_timer(self, duration_s: int = config.BREAK_DURATION_SECONDS) -> None:
        """Schedule a 'break over' alert after *duration_s* seconds."""
        if self._break_timer is not None:
            self._break_timer.cancel()

        self._speak_alert(_MSG_BREAK_CONFIRM)
        self._break_timer = threading.Timer(
            interval=float(duration_s),
            function=self._on_break_over,
        )
        self._break_timer.daemon = True
        self._break_timer.start()
        logger.info("[Alerts] Break timer set for %d seconds.", duration_s)

    def get_latest_snapshot(self) -> dict:
        """Return the most recent CV snapshot dict (may be empty on startup)."""
        return dict(self._latest_snapshot)

    # ── Internal processing ───────────────────────────────────────────────────

    def _process_item(self, item: dict) -> None:
        item_type = item.get("type")

        if item_type == "snapshot":
            state: dict = item.get("data", {})
            self._latest_snapshot = state
            self._evaluate_snapshot(state)

        elif item_type == "critical_event":
            event_data: dict = item.get("data", {})
            description = event_data.get("description", "Alerte critique du système.")
            logger.warning("[Alerts] Critical event: %s", description)
            if self._can_fire("critical_event", config.COOLDOWN_CRITICAL_EVENT):
                self._speak_alert(f"Alerte : {description}")
                self._mark_fired("critical_event")

        else:
            logger.debug("[Alerts] Unknown item type: %s", item_type)

    def _evaluate_snapshot(self, state: dict) -> None:
        """Check all alert conditions against the latest snapshot."""
        if self._is_muted():
            return

        now = time.monotonic()

        focus = float(state.get("global_focus_score", 100.0))
        posture = float(state.get("posture_score", 100.0))
        vigilance = float(state.get("vigilance_score", 100.0))

        # ── Low focus alert ───────────────────────────────────────────────
        if focus < config.FOCUS_ALERT_THRESHOLD:
            if self._can_fire("focus", config.COOLDOWN_FOCUS_ALERT):
                self._speak_alert(_MSG_FOCUS_LOW)
                self._mark_fired("focus")

        # ── Bad posture alert (sustained > 2 min) ─────────────────────────
        if posture < config.POSTURE_ALERT_THRESHOLD:
            if self._posture_bad_since is None:
                self._posture_bad_since = now
                logger.debug("[Alerts] Posture bad — starting sustain timer.")
            elif (now - self._posture_bad_since) >= config.POSTURE_SUSTAINED_SECONDS:
                if self._can_fire("posture", config.COOLDOWN_POSTURE_ALERT):
                    self._speak_alert(_MSG_POSTURE_BAD)
                    self._mark_fired("posture")
                    # Reset sustain timer after firing so we track the *next* episode
                    self._posture_bad_since = now
        else:
            if self._posture_bad_since is not None:
                logger.debug("[Alerts] Posture recovered — resetting timer.")
            self._posture_bad_since = None

        # ── Fatigue / vigilance alert ─────────────────────────────────────
        if vigilance < config.FATIGUE_THRESHOLD:
            if self._can_fire("fatigue", config.COOLDOWN_FATIGUE_ALERT):
                self._speak_alert(_MSG_FATIGUE)
                self._mark_fired("fatigue")

        # ── Good focus reinforcement (sustained > 20 min) ─────────────────
        if focus >= config.GOOD_FOCUS_THRESHOLD:
            if self._good_focus_since is None:
                self._good_focus_since = now
            elif (now - self._good_focus_since) >= config.GOOD_FOCUS_DURATION:
                if self._can_fire("good_focus", config.COOLDOWN_GOOD_FOCUS):
                    self._speak_alert(_MSG_GOOD_FOCUS)
                    self._mark_fired("good_focus")
                    # Reset so the next 20-min window can trigger again
                    self._good_focus_since = now
        else:
            if self._good_focus_since is not None:
                logger.debug("[Alerts] Focus dropped below good threshold — resetting.")
            self._good_focus_since = None

    def _on_break_over(self) -> None:
        """Callback fired when the break timer expires."""
        logger.info("[Alerts] Break timer expired.")
        self._break_timer = None
        self._speak_alert(_MSG_BREAK_OVER)

    # ── Cooldown helpers ──────────────────────────────────────────────────────

    def _can_fire(self, alert_type: str, cooldown_s: int) -> bool:
        last = self._last_fired.get(alert_type, 0.0)
        return (time.monotonic() - last) >= cooldown_s

    def _mark_fired(self, alert_type: str) -> None:
        self._last_fired[alert_type] = time.monotonic()

    def _is_muted(self) -> bool:
        return time.monotonic() < self._muted_until

    # ── TTS dispatch ──────────────────────────────────────────────────────────

    def _speak_alert(self, text: str) -> None:
        """Speak an alert — non-blocking, tries to acquire the speak lock.

        If the lock is already held (a conversation or another alert is in
        progress), the alert is dropped rather than queued.  Cooldowns prevent
        the same alert from being missed permanently.
        """
        acquired = self.speak_lock.acquire(blocking=False)
        if not acquired:
            logger.debug("[Alerts] speak_lock busy — dropping alert: %s", text[:50])
            return

        try:
            logger.info("[Alerts] Speaking: %s", text[:80])
            self.tts.speak(text)
        except Exception as exc:
            logger.error("[Alerts] TTS error during alert: %s", exc)
        finally:
            self.speak_lock.release()
