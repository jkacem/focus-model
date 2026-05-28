"""
Voice Assistant Orchestrator
=============================
Wires together all components and manages the main interaction loop.

Thread model:
  Main thread    → orchestrator (monitors shutdown signal)
  Thread: cv     → CVMonitor (polls backend every 5s)
  Thread: alerts → AlertManager (processes CV queue, speaks alerts)
  Thread: wake   → WakeWordListener (always listening)
  Thread: wakeword-callback → spawned per detection (conversation flow)

All TTS calls share a single threading.Lock (speak_lock) to ensure only
one voice is active at a time. The mic_busy threading.Event signals the
wake word listener to pause while STT is recording.

Provider switching:
  Changing VOICE_PROVIDER in config.py is the only change needed.
  "local"       → PiperTTS + WhisperSTT + LocalAIBrain
  "gemini_live" → GeminiLiveSession (all three in one object)
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

from voice_assistant import config
from voice_assistant.alert_manager import AlertManager
from voice_assistant.cv_monitor import CVMonitor
from voice_assistant.providers.base import BaseAIBrain, BaseSTT, BaseTTS
from voice_assistant.wake_word import WakeWordListener

logger = logging.getLogger("smartfocus.voice")

# ── Intent keywords ───────────────────────────────────────────────────────────
# Keyword matching in French — no ML required.

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "STATS": [
        "concentration", "score", "comment je vais", "progression",
        "focus", "statistiques", "résultats", "niveau",
    ],
    "PLANNING": [
        "planning", "aujourd'hui", "emploi du temps", "sessions",
        "programme", "calendrier", "horaire",
    ],
    "BREAK": [
        "pause", "repos", "fatigué", "j'ai besoin",
        "reposer", "souffler", "arrêter",
    ],
    "MUTE": [
        "silence", "tais-toi", "arrête", "chut",
        "stop", "plus d'alertes", "arrête de parler",
    ],
}


class VoiceAssistant:
    """Main orchestrator for the SmartFocus voice assistant."""

    def __init__(
        self,
        session_id: str,
        backend_url: str = config.BACKEND_URL,
        provider: str = config.VOICE_PROVIDER,
    ) -> None:
        self.session_id = session_id
        self.backend_url = backend_url
        self.provider = provider

        # Shared synchronization primitives
        self._speak_lock = threading.Lock()
        self._mic_busy = threading.Event()  # Set → wake word pauses mic

        # State queue: CVMonitor → AlertManager
        self._cv_queue: queue.Queue[dict] = queue.Queue(maxsize=100)

        # Load provider
        self._tts: BaseTTS
        self._stt: BaseSTT
        self._brain: BaseAIBrain
        self._load_provider(provider)

        # Core components
        self._cv_monitor = CVMonitor(session_id, backend_url, self._cv_queue)
        self._alert_manager = AlertManager(self._cv_queue, self._tts, self._speak_lock)
        self._wake_word = WakeWordListener(
            on_wake=self._on_wake_word_detected,
            mic_busy=self._mic_busy,
        )

        self._shutdown_event = threading.Event()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start all threads and block until shutdown signal (Ctrl-C)."""
        self._start_threads()
        self._announce_ready()

        try:
            while not self._shutdown_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Gracefully stop all components and say goodbye."""
        logger.info("[Assistant] Shutting down…")
        self._shutdown_event.set()
        self._wake_word.stop()
        self._cv_monitor.stop()
        self._alert_manager.stop()

        # Final goodbye
        try:
            with self._speak_lock:
                self._tts.speak("Au revoir !")
        except Exception:
            pass

        logger.info("[Assistant] Shutdown complete.")

    # ── Thread management ─────────────────────────────────────────────────────

    def _start_threads(self) -> None:
        threads = [
            threading.Thread(
                target=self._cv_monitor.run,
                name="cv-monitor",
                daemon=True,
            ),
            threading.Thread(
                target=self._alert_manager.run,
                name="alert-manager",
                daemon=True,
            ),
            threading.Thread(
                target=self._wake_word.run,
                name="wake-word",
                daemon=True,
            ),
        ]
        for t in threads:
            t.start()
            logger.debug("[Assistant] Started thread: %s", t.name)

    def _announce_ready(self) -> None:
        """Speak startup confirmation after all threads are running."""
        time.sleep(0.5)  # Let threads stabilize
        if self._tts.is_ready():
            with self._speak_lock:
                self._tts.speak(
                    "SmartFocus est prêt. Dis 'hey Jarvis' pour me parler."
                )

    # ── Wake word callback ────────────────────────────────────────────────────

    def _on_wake_word_detected(self) -> None:
        """Called by WakeWordListener when the wake word is heard.

        Tries to acquire speak_lock non-blocking: if already speaking
        (alert or previous conversation), silently ignore this detection.
        """
        if self._shutdown_event.is_set():
            return

        acquired = self._speak_lock.acquire(blocking=False)
        if not acquired:
            logger.debug("[Assistant] Wake word detected but speak_lock busy — ignoring.")
            return

        try:
            self._handle_conversation()
        except Exception as exc:
            logger.error("[Assistant] Conversation error: %s", exc, exc_info=True)
        finally:
            self._speak_lock.release()

    def _handle_conversation(self) -> None:
        """Full wake-word → listen → intent → respond flow.

        This runs with speak_lock held throughout, so:
          - No alerts can interrupt mid-conversation.
          - The wake word listener is implicitly paused (mic_busy set below).
        """
        # 1. Acknowledge
        logger.info("[Assistant] Conversation started.")
        self._tts.speak("Je t'écoute.")

        # 2. Signal mic in use so wake word stops reading the mic
        self._mic_busy.set()
        try:
            # 3. Record + transcribe
            if not self._stt.is_ready():
                self._tts.speak(
                    "Désolé, le microphone n'est pas disponible pour l'instant."
                )
                return

            transcription = self._stt.listen_and_transcribe()

            if not transcription:
                self._tts.speak(
                    "Je n'ai pas compris, répète s'il te plaît."
                )
                return

            logger.info("[Assistant] Transcription: %r", transcription)

            # 4. Classify intent
            intent = self._classify_intent(transcription)
            logger.info("[Assistant] Intent: %s", intent)

            # 5. Handle intent and get response
            response = self._handle_intent(intent, transcription)

            # 6. Speak response
            if response:
                logger.info("[Assistant] >>> Jarvis dit : %s", response)
                self._tts.speak(response)

        finally:
            self._mic_busy.clear()
            logger.info("[Assistant] Conversation ended.")

    # ── Intent classification ─────────────────────────────────────────────────

    @staticmethod
    def _classify_intent(text: str) -> str:
        """Keyword-based intent detection (no ML). Returns one of:
        STATS, PLANNING, BREAK, MUTE, CHATBOT.
        """
        text_lower = text.lower()
        for intent, keywords in _INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return intent
        return "CHATBOT"

    # ── Intent handlers ───────────────────────────────────────────────────────

    def _handle_intent(self, intent: str, raw_text: str) -> Optional[str]:
        """Dispatch to the appropriate handler and return a French TTS string."""
        handlers = {
            "STATS":    self._handle_stats,
            "PLANNING": self._handle_planning,
            "BREAK":    self._handle_break,
            "MUTE":     self._handle_mute,
            "CHATBOT":  self._handle_chatbot,
        }
        handler = handlers.get(intent, self._handle_chatbot)
        try:
            return handler(raw_text)
        except Exception as exc:
            logger.error("[Assistant] Handler %s raised: %s", intent, exc)
            return "Désolé, une erreur s'est produite. Réessaie dans un moment."

    def _handle_stats(self, _: str) -> str:
        """Fetch and speak current CV scores."""
        return self._brain.get_latest_stats(self.session_id)

    def _handle_planning(self, _: str) -> str:
        """Fetch and speak today's study planning."""
        return self._brain.get_planning_today()

    def _handle_break(self, _: str) -> str:
        """Start a 5-minute break countdown."""
        # Start timer in alert_manager (it holds the TTS reference)
        threading.Thread(
            target=self._alert_manager.start_break_timer,
            kwargs={"duration_s": config.BREAK_DURATION_SECONDS},
            daemon=True,
        ).start()
        # Return empty so the conversation doesn't double-speak the confirmation
        # (alert_manager._MSG_BREAK_CONFIRM is spoken inside start_break_timer)
        return ""

    def _handle_mute(self, _: str) -> str:
        """Mute all proactive alerts."""
        # Speak confirmation BEFORE muting (otherwise mute suppresses this too)
        self._alert_manager.mute()
        return ""  # _MSG_MUTED is spoken inside mute()

    def _handle_chatbot(self, raw_text: str) -> str:
        """Forward raw transcription to the backend chatbot."""
        return self._brain.ask_chatbot(raw_text)

    # ── Provider loading ──────────────────────────────────────────────────────

    def _load_provider(self, provider: str) -> None:
        if provider == "local":
            self._load_local_provider()
        elif provider == "gemini_live":
            self._load_gemini_provider()
        else:
            raise ValueError(
                f"Unknown VOICE_PROVIDER: {provider!r}. "
                "Valid values: 'local', 'gemini_live'."
            )

    def _load_local_provider(self) -> None:
        logger.info("[Assistant] Loading local provider (Piper + Whisper + HTTP)…")

        from voice_assistant.providers.local.tts_engine import PiperTTS  # noqa: PLC0415
        from voice_assistant.providers.local.stt_engine import WhisperSTT  # noqa: PLC0415
        from voice_assistant.providers.local.ai_brain import LocalAIBrain  # noqa: PLC0415

        self._tts = PiperTTS()
        self._stt = WhisperSTT()
        self._brain = LocalAIBrain(self.backend_url, self.session_id)

        if not self._tts.is_ready():
            logger.warning(
                "[Assistant] TTS not ready — alerts and responses will be silent. "
                "Check piper binary and voice model."
            )
        if not self._stt.is_ready():
            logger.warning(
                "[Assistant] STT not ready — wake word conversations disabled. "
                "Check microphone and Whisper model."
            )

    def _load_gemini_provider(self) -> None:
        logger.info("[Assistant] Loading Gemini Live provider…")

        from voice_assistant.providers.gemini_live.session import GeminiLiveSession  # noqa: PLC0415

        # GeminiLiveSession implements all three interfaces
        session = GeminiLiveSession()  # Will raise NotImplementedError until implemented
        self._tts = session  # type: ignore[assignment]
        self._stt = session  # type: ignore[assignment]
        self._brain = session  # type: ignore[assignment]
