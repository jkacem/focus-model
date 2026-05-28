"""Local provider — fully offline, no cloud dependencies.

Components:
  PiperTTS     → Piper binary via subprocess + sounddevice playback
  WhisperSTT   → OpenAI Whisper local model + sounddevice recording
  LocalAIBrain → HTTP calls to the SmartFocus FastAPI backend
"""
