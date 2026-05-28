"""
Piper TTS Engine — Local, Offline, Cross-Platform
===================================================
Synthesizes French speech using the Piper neural TTS binary.

Flow:
  1. On first instantiation, download Piper binary + voice model if missing.
  2. speak(text) → subprocess Piper (text via stdin) → WAV → sounddevice playback.
  3. Temporary WAV files are always deleted after playback (Windows-safe).

Cross-platform notes:
  - Never uses aplay / paplay / sox — only sounddevice + soundfile.
  - Binary URL is selected per OS + architecture in config.py.
  - All paths use pathlib.Path — no hardcoded slashes.
"""

from __future__ import annotations

import io
import logging
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
import tarfile
from pathlib import Path
from typing import Optional

import sounddevice as sd
import soundfile as sf

from voice_assistant import config
from voice_assistant.providers.base import BaseTTS

logger = logging.getLogger("smartfocus.voice")


class PiperTTS(BaseTTS):
    """Piper neural TTS engine — French voice, fully local."""

    def __init__(self) -> None:
        self._ready = False
        self._binary: Optional[Path] = None
        self._model_path: Optional[Path] = None

        try:
            self._setup()
            self._ready = True
            logger.info(
                "[TTS] Piper ready | binary=%s | model=%s",
                self._binary,
                self._model_path,
            )
        except Exception as exc:
            logger.error("[TTS] Piper initialization failed: %s", exc)

    # ── Public interface ──────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._ready

    def speak(self, text: str) -> None:
        """Synthesize *text* and play it. Blocks until playback finishes."""
        if not self._ready:
            logger.warning("[TTS] Not ready, skipping speech: %s", text[:60])
            return

        text = text.strip()
        if not text:
            return

        wav_path: Optional[Path] = None
        try:
            wav_path = self._synthesize(text)
            self._play(wav_path)
        except Exception as exc:
            logger.error("[TTS] speak() error: %s", exc)
        finally:
            if wav_path is not None:
                try:
                    wav_path.unlink(missing_ok=True)
                except OSError:
                    pass

    # ── Setup & download ──────────────────────────────────────────────────────

    def _setup(self) -> None:
        """Ensure Piper binary and voice model are present; download if needed."""
        self._ensure_binary()
        self._ensure_model()
        self._binary = config.PIPER_BINARY_PATH.resolve()
        onnx = config.PIPER_MODELS_DIR / f"{config.PIPER_VOICE_MODEL}.onnx"
        self._model_path = onnx.resolve()

    def _ensure_binary(self) -> None:
        binary = config.PIPER_BINARY_PATH
        if binary.exists():
            return
        logger.info("[TTS] Piper binary not found — downloading…")
        self._download_binary(binary)

    def _download_binary(self, target: Path) -> None:
        """Download and extract the Piper binary for the current platform."""
        import urllib.request

        target.parent.mkdir(parents=True, exist_ok=True)
        url = config.PIPER_DOWNLOAD_URL
        archive_type = config.PIPER_ARCHIVE_TYPE

        logger.info("[TTS] Downloading Piper from %s", url)

        # Stream-download to a temp file to avoid memory issues on Pi
        with tempfile.NamedTemporaryFile(
            suffix=f".{archive_type}", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)

        try:
            urllib.request.urlretrieve(url, tmp_path)

            dest_dir = target.parent
            dest_dir.mkdir(parents=True, exist_ok=True)

            if archive_type == "zip":
                with zipfile.ZipFile(tmp_path, "r") as zf:
                    # Extract only the piper executable
                    for member in zf.namelist():
                        if member.endswith("piper.exe") or member.endswith("/piper"):
                            zf.extract(member, dest_dir)
                            extracted = dest_dir / member
                            # Flatten nested directories: move binary to dest_dir/piper.exe
                            final = dest_dir / Path(member).name
                            if extracted != final:
                                shutil.move(str(extracted), str(final))
            else:
                with tarfile.open(tmp_path, "r:gz") as tf:
                    for member in tf.getmembers():
                        if member.name.endswith("piper") and member.isfile():
                            member.name = Path(member.name).name  # strip dirs
                            tf.extract(member, dest_dir)

            # Ensure executable bit on Linux
            if platform.system() != "Windows":
                binary_file = dest_dir / target.name
                binary_file.chmod(
                    binary_file.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
                )

        finally:
            tmp_path.unlink(missing_ok=True)

        if not target.exists():
            raise FileNotFoundError(
                f"Piper binary not found at {target} after extraction.\n"
                f"Please download manually from: {url}\n"
                f"And place the binary at: {target}"
            )
        logger.info("[TTS] Piper binary ready at %s", target)

    def _ensure_model(self) -> None:
        """Download voice model .onnx and .onnx.json if not present."""
        import urllib.request

        models_dir = config.PIPER_MODELS_DIR
        models_dir.mkdir(parents=True, exist_ok=True)

        onnx_path = models_dir / f"{config.PIPER_VOICE_MODEL}.onnx"
        json_path = models_dir / f"{config.PIPER_VOICE_MODEL}.onnx.json"

        for path, url in [
            (onnx_path, config.PIPER_VOICE_ONNX_URL),
            (json_path, config.PIPER_VOICE_JSON_URL),
        ]:
            if path.exists():
                continue
            logger.info("[TTS] Downloading voice model %s …", path.name)
            try:
                urllib.request.urlretrieve(url, path)
                logger.info("[TTS] Downloaded %s (%d KB)", path.name, path.stat().st_size // 1024)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download Piper voice model from {url}.\n"
                    f"Error: {exc}\n"
                    f"Download manually and place at: {path}"
                ) from exc

    # ── Synthesis & playback ──────────────────────────────────────────────────

    def _synthesize(self, text: str) -> Path:
        """Run Piper subprocess → write WAV → return path to WAV file."""
        # Create temp file first, close it so Piper can write to it on Windows
        tmp_fd, tmp_name = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        wav_path = Path(tmp_name)

        cmd = [
            str(self._binary),
            "--model",
            str(self._model_path),
            "--output_file",
            str(wav_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("[TTS] Piper synthesis timed out after 30s")
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"[TTS] Piper binary not found at {self._binary}. "
                "Run install_voice.sh / install_voice_windows.bat first."
            ) from exc

        if result.returncode != 0:
            stderr_msg = result.stderr.decode("utf-8", errors="replace")[:200]
            raise RuntimeError(f"[TTS] Piper exited with code {result.returncode}: {stderr_msg}")

        if not wav_path.exists() or wav_path.stat().st_size == 0:
            raise RuntimeError("[TTS] Piper produced no output WAV file")

        return wav_path

    def _play(self, wav_path: Path) -> None:
        """Play a WAV file using sounddevice (cross-platform, no shell commands)."""
        data, sample_rate = sf.read(str(wav_path), dtype="float32")
        sd.play(data, samplerate=sample_rate)
        sd.wait()
