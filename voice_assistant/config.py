"""
SmartFocus Voice Assistant — Configuration
==========================================
All tunable knobs live here.

To switch the entire voice stack from local to Gemini Live, change ONE line:
    VOICE_PROVIDER = "local"        # Piper + Whisper + HTTP (current)
    VOICE_PROVIDER = "gemini_live"  # Google Gemini Live streaming (future)

All other code (cv_monitor, alert_manager, wake_word, assistant) is
provider-agnostic and never needs to change.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

# ── Provider selection ────────────────────────────────────────────────────────
# Change this ONE value to swap the full TTS/STT/AI stack.
VOICE_PROVIDER: str = "local"  # "local" | "gemini_live"

# ── Backend connection ────────────────────────────────────────────────────────
BACKEND_URL: str = "http://localhost:8000"
PI_API_KEY: str = os.environ.get("PI_API_KEY", "")
SESSION_ID: str = ""  # Injected at startup via --session-id CLI argument

# ── Language ──────────────────────────────────────────────────────────────────
LANGUAGE: str = "fr"  # ISO 639-1: all speech input/output is French

# ── STT — OpenAI Whisper (local, offline) ─────────────────────────────────────
# "small" gives the best accuracy/speed trade-off for Pi 5 + Windows dev.
# Other options: "tiny" (fastest), "base", "medium" (most accurate, slower).
WHISPER_MODEL: str = "tiny"    # "small" too slow on Pi CPU — use "small" on PC

# ── TTS — Piper (local, offline) ──────────────────────────────────────────────
PIPER_VOICE_MODEL: str = "fr_FR-upmc-medium"

# Voice model files are stored relative to the voice_assistant/ package directory.
# The _RUNTIME_DIR is resolved at import time so relative paths work regardless
# of where the process is launched from.
_RUNTIME_DIR: Path = Path(__file__).resolve().parent
PIPER_MODELS_DIR: Path = _RUNTIME_DIR / "piper_models"

# ── Local session data (written by main_cv.py) ────────────────────────────────
SESSIONS_DIR: Path = _RUNTIME_DIR.parent / "output" / "sessions"
SESSION_SUMMARY_PATH: Path = _RUNTIME_DIR.parent.parent / "session_summary.json"

# Piper binary — platform/arch auto-detected.
# The binary is downloaded automatically on first run by tts_engine.py.
_system = platform.system()
_arch = platform.machine().lower()

if _system == "Windows":
    PIPER_BINARY_PATH: Path = _RUNTIME_DIR / "piper_bin" / "piper.exe"
    PIPER_DOWNLOAD_URL: str = (
        "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/"
        "piper_windows_amd64.zip"
    )
    PIPER_ARCHIVE_TYPE: str = "zip"
elif _system == "Linux" and ("aarch64" in _arch or "arm64" in _arch):
    # Raspberry Pi 5 (ARM64)
    PIPER_BINARY_PATH = _RUNTIME_DIR / "piper_bin" / "piper"
    PIPER_DOWNLOAD_URL = (
        "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/"
        "piper_linux_aarch64.tar.gz"
    )
    PIPER_ARCHIVE_TYPE = "tar.gz"
else:
    # Linux x86_64 (dev machine)
    PIPER_BINARY_PATH = _RUNTIME_DIR / "piper_bin" / "piper"
    PIPER_DOWNLOAD_URL = (
        "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/"
        "piper_linux_x86_64.tar.gz"
    )
    PIPER_ARCHIVE_TYPE = "tar.gz"

# Voice model download URLs (Hugging Face, rhasspy/piper-voices)
PIPER_VOICE_ONNX_URL: str = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx"
)
PIPER_VOICE_JSON_URL: str = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx.json"
)

# ── Wake word — OpenWakeWord (local, offline) ──────────────────────────────────
# "hey_jarvis" is the closest available model to "Hey Focus".
# Sensitivity 0.5 balances false positives vs missed detections.
WAKE_WORD_MODEL: str = "hey_jarvis"
WAKE_WORD_SENSITIVITY: float = 0.5

# ── CV alert thresholds ───────────────────────────────────────────────────────
FOCUS_ALERT_THRESHOLD: int = 40       # % — alert when global_focus_score < this
CRITICAL_FOCUS_THRESHOLD: int = 20   # % — critical alert when focus is extremely low
POSTURE_ALERT_THRESHOLD: int = 35     # % — alert when posture_score < this
FATIGUE_THRESHOLD: int = 30           # % — alert when fatigue_score > this (0-100)
GOOD_FOCUS_THRESHOLD: int = 80        # % — positive reinforcement above this
GOOD_FOCUS_DURATION: int = 1200       # seconds (20 min) sustained good focus — set to 30 for testing
POSTURE_SUSTAINED_SECONDS: float = 120.0  # 2 min bad posture before alerting — set to 10.0 for testing

# ── Polling ───────────────────────────────────────────────────────────────────
POLL_INTERVAL_SECONDS: float = 5.0   # CV snapshot poll interval

# ── Mute ──────────────────────────────────────────────────────────────────────
ALERT_MUTE_DURATION: int = 900        # seconds (15 min) — silences all alerts

# ── Audio recording ───────────────────────────────────────────────────────────
MAX_RECORDING_SECONDS: float = 8.0
SILENCE_TIMEOUT_SECONDS: float = 2.0
SILENCE_RMS_THRESHOLD: float = 0.01  # RMS energy below = silence
AUDIO_SAMPLE_RATE: int = 16000
AUDIO_CHANNELS: int = 1

# ── Break timer ───────────────────────────────────────────────────────────────
BREAK_DURATION_SECONDS: int = 300     # 5 minutes

# ── Per-alert-type cooldowns (seconds) ───────────────────────────────────────
COOLDOWN_FOCUS_ALERT: int = 180       # 3 min
COOLDOWN_CRITICAL_FOCUS: int = 120    # 2 min — fires more often than regular focus
COOLDOWN_POSTURE_ALERT: int = 240     # 4 min
COOLDOWN_FATIGUE_ALERT: int = 300     # 5 min
COOLDOWN_GOOD_FOCUS: int = 1500       # 25 min
COOLDOWN_CRITICAL_EVENT: int = 60     # 1 min

# ── Gemini Live (future provider) ─────────────────────────────────────────────
# These are used only when VOICE_PROVIDER = "gemini_live"
GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_LIVE_MODEL: str = "gemini-2.0-flash-live-001"
GEMINI_VOICE_NAME: str = "Charon"     # French-compatible voice
