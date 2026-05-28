"""
Abstract base classes for the SmartFocus voice provider system.

Every provider (local, gemini_live, …) MUST implement these three interfaces.
The rest of the codebase only imports these base classes — never concrete ones.

Switching providers in config.py is the only change needed to swap the stack.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger("smartfocus.voice")


# ─────────────────────────────────────────────────────────────────────────────
# BaseTTS
# ─────────────────────────────────────────────────────────────────────────────

class BaseTTS(ABC):
    """Text-to-Speech interface.

    Concrete implementations must synthesize French audio and block until
    playback is complete (i.e. speak() is synchronous from the caller's view).
    Thread-safety of speak() is the caller's responsibility.
    """

    @abstractmethod
    def speak(self, text: str) -> None:
        """Synthesize *text* and play it through the default audio output.

        Args:
            text: French text to speak. May contain punctuation but should
                  not contain SSML tags unless the provider explicitly supports them.

        Raises:
            RuntimeError: If TTS engine is not initialized or playback fails.
        """
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if the TTS engine is initialized and ready to speak.

        This is used by assistant.py to gracefully disable TTS on startup
        failure without crashing the whole assistant.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# BaseSTT
# ─────────────────────────────────────────────────────────────────────────────

class BaseSTT(ABC):
    """Speech-to-Text interface.

    Concrete implementations must record audio from the microphone,
    detect silence automatically, and return a transcription.
    The language is always French (fr) as defined in config.LANGUAGE.
    """

    @abstractmethod
    def listen_and_transcribe(self) -> Optional[str]:
        """Record until silence detected (or max duration), then transcribe.

        Returns:
            Transcribed French text, stripped of leading/trailing whitespace.
            Returns None if nothing was heard or transcription failed.

        Raises:
            RuntimeError: If microphone is unavailable.
        """
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if STT engine is initialized and microphone is available."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# BaseAIBrain
# ─────────────────────────────────────────────────────────────────────────────

class BaseAIBrain(ABC):
    """AI reasoning + backend data retrieval interface.

    Provides the three backend call types needed by the assistant:
      1. General chatbot Q&A (backed by user documents via RAG)
      2. Today's study planning
      3. Latest CV session statistics

    All methods return already-formatted French strings ready for TTS.
    """

    @abstractmethod
    def ask_chatbot(self, question: str, user_id: int = 1) -> str:
        """POST /api/v1/chatbot/ask — free-form Q&A.

        Args:
            question: User's question in French (raw transcription).
            user_id:  Backend user ID. Default 1 for single-user Pi setup.

        Returns:
            French answer string suitable for TTS playback.
        """
        ...

    @abstractmethod
    def get_planning_today(self) -> str:
        """GET /api/v1/planning/today — today's study schedule.

        Returns:
            French summary of today's planning sessions, ready for TTS.
            Returns a polite "nothing planned" message if no sessions exist.
        """
        ...

    @abstractmethod
    def get_latest_stats(self, session_id: str) -> str:
        """GET /api/v1/sessions/{session_id}/latest — live CV scores.

        Args:
            session_id: Active CV session UUID.

        Returns:
            French summary of current attention/posture/fatigue scores.
            Returns a "no data yet" message if the session has no snapshots.
        """
        ...
