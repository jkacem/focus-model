"""
Wake Word Listener — OpenWakeWord (Local, Offline)
===================================================
Continuously listens on the microphone for the configured wake word
("hey_jarvis") and fires a callback when detected.

Runs in a dedicated daemon thread.
Automatically pauses when `mic_busy` event is set (STT recording or TTS
feedback is in progress) to avoid microphone conflicts.

OpenWakeWord expects:
  - 16 kHz mono int16 audio
  - Chunks of exactly 1280 frames (80 ms per chunk)

Detection threshold: WAKE_WORD_SENSITIVITY (0.5 by default).
After a detection, a 1-second cooldown prevents double-firing.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from voice_assistant import config

logger = logging.getLogger("smartfocus.voice")

# OpenWakeWord requires exactly 1280 frames at 16 kHz (80 ms)
_CHUNK_FRAMES = 1280
_SAMPLE_RATE = 16_000  # Hz — fixed by OpenWakeWord

# Minimum seconds between consecutive detections (prevents double-firing)
_DETECTION_COOLDOWN_S = 1.0


class WakeWordListener:
    """Listens for the wake word and fires an async callback.

    Args:
        on_wake:    Called (in the listener thread) when wake word detected.
                    Must be thread-safe and return quickly; heavy work should
                    be dispatched to another thread.
        mic_busy:   A threading.Event. When set, the listener pauses mic reads
                    to avoid conflicts with STT recording. The wake_word loop
                    polls this every 100 ms.
    """

    def __init__(
        self,
        on_wake: Callable[[], None],
        mic_busy: threading.Event,
    ) -> None:
        self.on_wake = on_wake
        self.mic_busy = mic_busy

        self.running = False
        self._last_detection: float = 0.0
        self._oww_model = None
        self._model_ready = False
        self._mic_available = True

        self._load_model()

    # ── Public API ────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._model_ready and self._mic_available

    def run(self) -> None:
        """Main listener loop — call in a daemon thread."""
        self.running = True

        if not self._model_ready:
            logger.error("[WakeWord] Model not loaded — listener disabled.")
            return

        logger.info(
            "[WakeWord] Listening for '%s' (sensitivity=%.2f)…",
            config.WAKE_WORD_MODEL,
            config.WAKE_WORD_SENSITIVITY,
        )

        try:
            self._listen_loop()
        except Exception as exc:
            logger.error("[WakeWord] Fatal error in listener loop: %s", exc, exc_info=True)
        finally:
            self.running = False
            logger.info("[WakeWord] Listener stopped.")

    def stop(self) -> None:
        """Signal the run loop to exit."""
        self.running = False

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        try:
            from openwakeword.model import Model  # noqa: PLC0415
            from openwakeword.utils import download_models  # noqa: PLC0415

            logger.info("[WakeWord] Ensuring model '%s' is downloaded…", config.WAKE_WORD_MODEL)
            # download_models() is a no-op if the file already exists.
            # Must be called before Model() — openwakeword does NOT auto-download.
            download_models(model_names=[config.WAKE_WORD_MODEL])

            logger.info("[WakeWord] Loading model '%s'…", config.WAKE_WORD_MODEL)
            # inference_framework="onnx" works on both x86_64 and aarch64
            self._oww_model = Model(
                wakeword_models=[config.WAKE_WORD_MODEL],
                inference_framework="onnx",
            )
            self._model_ready = True
            logger.info("[WakeWord] Model '%s' loaded.", config.WAKE_WORD_MODEL)

        except ImportError:
            logger.error(
                "[WakeWord] openwakeword not installed. "
                "Run: pip install openwakeword"
            )
        except Exception as exc:
            logger.error("[WakeWord] Model load error: %s", exc)

    # ── Listening loop ────────────────────────────────────────────────────────

    def _listen_loop(self) -> None:
        """Open microphone and continuously feed chunks to OpenWakeWord."""
        try:
            stream = sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=_CHUNK_FRAMES,
            )
        except sd.PortAudioError as exc:
            self._mic_available = False
            logger.error("[WakeWord] Cannot open microphone: %s", exc)
            return

        with stream:
            while self.running:
                # Pause when mic is in use by STT
                if self.mic_busy.is_set():
                    time.sleep(0.1)
                    continue

                try:
                    audio_chunk, _overflowed = stream.read(_CHUNK_FRAMES)
                except sd.PortAudioError as exc:
                    logger.error("[WakeWord] PortAudio read error: %s", exc)
                    self._mic_available = False
                    break

                # openwakeword expects a flat numpy int16 array
                audio_flat = audio_chunk.flatten().astype(np.int16)

                # Run inference
                try:
                    prediction: dict = self._oww_model.predict(audio_flat)
                except Exception as exc:
                    logger.warning("[WakeWord] Prediction error: %s", exc)
                    continue

                score = float(prediction.get(config.WAKE_WORD_MODEL, 0.0))

                if score >= config.WAKE_WORD_SENSITIVITY:
                    now = time.monotonic()
                    if now - self._last_detection >= _DETECTION_COOLDOWN_S:
                        self._last_detection = now
                        logger.info(
                            "[WakeWord] Detected '%s' (score=%.3f)",
                            config.WAKE_WORD_MODEL,
                            score,
                        )
                        self._fire_callback()

    def _fire_callback(self) -> None:
        """Call on_wake in a separate thread so the listener loop stays live."""
        threading.Thread(
            target=self._safe_callback,
            name="wakeword-callback",
            daemon=True,
        ).start()

    def _safe_callback(self) -> None:
        try:
            self.on_wake()
        except Exception as exc:
            logger.error("[WakeWord] on_wake() callback raised: %s", exc, exc_info=True)
