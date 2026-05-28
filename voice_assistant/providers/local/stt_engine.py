"""
Whisper STT Engine — Local, Offline, Cross-Platform
=====================================================
Records French speech from the microphone and transcribes it using
OpenAI Whisper running entirely on-device (no network required).

Recording strategy:
  - Chunk-based streaming via sounddevice InputStream.
  - Silence detection: RMS energy below SILENCE_RMS_THRESHOLD for
    SILENCE_TIMEOUT_SECONDS → stop recording.
  - Hard cap at MAX_RECORDING_SECONDS to prevent runaway recordings.
  - Audio saved to a temporary WAV file (closed before Whisper reads it,
    Windows-safe) then deleted immediately after transcription.

Whisper model is loaded ONCE at instantiation, not per call.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from voice_assistant import config
from voice_assistant.providers.base import BaseSTT

logger = logging.getLogger("smartfocus.voice")

# Chunk size for recording loop (0.25 s windows for responsive silence detection)
_CHUNK_DURATION_S: float = 0.25


class WhisperSTT(BaseSTT):
    """OpenAI Whisper speech-to-text engine — French, fully local."""

    def __init__(self) -> None:
        self._ready = False
        self._model = None
        self._mic_available = True

        try:
            self._load_model()
            self._check_microphone()
            self._ready = True
            logger.info("[STT] Whisper '%s' model loaded and mic available.", config.WHISPER_MODEL)
        except Exception as exc:
            logger.error("[STT] Initialization failed: %s", exc)

    # ── Public interface ──────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._ready and self._mic_available

    def listen_and_transcribe(self) -> Optional[str]:
        """Record until silence, transcribe, return French text or None."""
        if not self._ready:
            logger.warning("[STT] Not ready — skipping listen.")
            return None
        if not self._mic_available:
            logger.warning("[STT] Microphone unavailable — skipping listen.")
            return None

        logger.debug("[STT] Starting recording…")
        audio_data = self._record_until_silence()

        if audio_data is None or len(audio_data) == 0:
            logger.debug("[STT] No audio captured.")
            return None

        return self._transcribe(audio_data)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        import whisper  # noqa: PLC0415 (lazy import — heavy)

        logger.info("[STT] Loading Whisper model '%s'…", config.WHISPER_MODEL)
        self._model = whisper.load_model(config.WHISPER_MODEL)
        logger.info("[STT] Whisper model loaded.")

    def _check_microphone(self) -> None:
        """Try to open the default microphone briefly to verify it exists."""
        try:
            with sd.InputStream(
                samplerate=config.AUDIO_SAMPLE_RATE,
                channels=config.AUDIO_CHANNELS,
                dtype="float32",
                blocksize=int(config.AUDIO_SAMPLE_RATE * 0.1),
            ):
                pass
        except sd.PortAudioError as exc:
            self._mic_available = False
            raise RuntimeError(f"Microphone not found: {exc}") from exc

    # ── Audio recording ───────────────────────────────────────────────────────

    def _record_until_silence(self) -> Optional[np.ndarray]:
        """Stream microphone until silence detected or max duration reached.

        Returns:
            1-D float32 numpy array at AUDIO_SAMPLE_RATE, or None on error.
        """
        chunk_frames = int(config.AUDIO_SAMPLE_RATE * _CHUNK_DURATION_S)
        chunks: list[np.ndarray] = []
        silence_started_at: Optional[float] = None
        recording_started_at = time.monotonic()

        try:
            with sd.InputStream(
                samplerate=config.AUDIO_SAMPLE_RATE,
                channels=config.AUDIO_CHANNELS,
                dtype="float32",
                blocksize=chunk_frames,
            ) as stream:
                logger.debug("[STT] Microphone open, listening…")

                while True:
                    chunk, _overflowed = stream.read(chunk_frames)
                    chunk = chunk.copy()

                    # Flatten to 1-D (mono)
                    audio_flat = chunk[:, 0] if chunk.ndim == 2 else chunk.flatten()
                    chunks.append(audio_flat)

                    # RMS silence detection
                    rms = float(np.sqrt(np.mean(audio_flat ** 2)))
                    now = time.monotonic()

                    if rms < config.SILENCE_RMS_THRESHOLD:
                        if silence_started_at is None:
                            silence_started_at = now
                            logger.debug("[STT] Silence detected, waiting %.1fs…", config.SILENCE_TIMEOUT_SECONDS)
                        elif now - silence_started_at >= config.SILENCE_TIMEOUT_SECONDS:
                            logger.debug("[STT] Silence timeout reached — stopping.")
                            break
                    else:
                        silence_started_at = None  # Reset silence timer on speech

                    # Hard cap
                    if now - recording_started_at >= config.MAX_RECORDING_SECONDS:
                        logger.debug("[STT] Max recording duration reached — stopping.")
                        break

        except sd.PortAudioError as exc:
            logger.error("[STT] PortAudio recording error: %s", exc)
            self._mic_available = False
            return None
        except Exception as exc:
            logger.error("[STT] Recording error: %s", exc)
            return None

        if not chunks:
            return None

        audio = np.concatenate(chunks, axis=0)
        duration = len(audio) / config.AUDIO_SAMPLE_RATE
        logger.debug("[STT] Recorded %.2f s of audio.", duration)
        return audio

    # ── Transcription ─────────────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray) -> Optional[str]:
        """Write audio to temp WAV, call Whisper, clean up, return text."""
        # Write to temp file — close handle BEFORE reading on Windows
        tmp_fd, tmp_name = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        wav_path = Path(tmp_name)

        try:
            sf.write(str(wav_path), audio, config.AUDIO_SAMPLE_RATE)

            logger.debug("[STT] Transcribing with Whisper…")
            result = self._model.transcribe(
                str(wav_path),
                language=config.LANGUAGE,
                fp16=False,  # fp16 unsupported on CPU
                task="transcribe",
            )

            text: str = result.get("text", "").strip()
            logger.info("[STT] Transcription: %r", text)
            return text if text else None

        except Exception as exc:
            logger.error("[STT] Transcription error: %s", exc)
            return None
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except OSError:
                pass
