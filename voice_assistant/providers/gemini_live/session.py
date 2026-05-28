"""
Gemini Live Provider — Stub for Future Migration
=================================================

## Migration guide: switching from "local" to "gemini_live"

### SDK to use
Use `google-genai` (the new unified SDK), NOT `google-generativeai` (deprecated):
    pip install google-genai

### Model
    gemini-2.0-flash-live-001   ← low-latency live streaming, supports audio I/O

### Architecture overview
Gemini Live uses a persistent **WebSocket** session instead of separate TTS/STT calls.
The session handles both input (microphone audio) and output (synthesized voice audio)
simultaneously. This eliminates the wake-word → record → transcribe → speak waterfall
and enables natural back-and-forth conversation.

### Session lifecycle
```python
from google import genai
from google.genai import types

client = genai.Client(api_key=GOOGLE_API_KEY)

config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
        )
    ),
    system_instruction="Tu es un assistant d'étude intelligent nommé SmartFocus...",
)

async with client.aio.live.connect(model="gemini-2.0-flash-live-001", config=config) as session:
    # Send audio chunks from microphone → session.send(audio_chunk)
    # Receive audio chunks from Gemini  → async for response in session.receive(): ...
    # Session handles VAD (voice activity detection) server-side — no wake word needed
```

### Config values needed
    GOOGLE_API_KEY  (config.py, read from env var GOOGLE_API_KEY)
    GEMINI_LIVE_MODEL = "gemini-2.0-flash-live-001"
    GEMINI_VOICE_NAME = "Charon"   # or "Puck", "Aoede", "Fenrir", "Kore"

### Key differences from local stack
    | Feature         | local                  | gemini_live                    |
    |-----------------|------------------------|--------------------------------|
    | Wake word       | OpenWakeWord (offline) | Server-side VAD (no wake word) |
    | STT             | Whisper (offline)      | Gemini server-side             |
    | TTS             | Piper (offline)        | Gemini server-side             |
    | Latency         | ~2-4 s                 | ~200-500 ms                    |
    | Privacy         | Fully local            | Audio sent to Google           |
    | Internet needed | No                     | Yes                            |

### assistant.py changes needed
When VOICE_PROVIDER == "gemini_live", assistant.py should:
    1. Disable the OpenWakeWord thread (Gemini handles VAD)
    2. Disable the separate STT recording thread
    3. Open a single GeminiLiveSession and run the async send/receive loop
    4. Pipe CV context (scores, events) into the Gemini system prompt at session start
    5. Keep cv_monitor and alert_manager as-is (provider-agnostic)

### Files to add/install
    pip install google-genai>=1.0.0
    Add GOOGLE_API_KEY to .env

---
STUB IMPLEMENTATION BELOW
All methods raise NotImplementedError until the migration is completed.
"""

from __future__ import annotations

import logging
from typing import Optional

from voice_assistant.providers.base import BaseAIBrain, BaseSTT, BaseTTS

logger = logging.getLogger("smartfocus.voice")


class GeminiLiveSession(BaseTTS, BaseSTT, BaseAIBrain):
    """Gemini Live unified voice session.

    Implements BaseTTS + BaseSTT + BaseAIBrain in a single class because
    Gemini Live uses one persistent WebSocket for all audio I/O.

    All methods are stubs — implement them following the migration guide
    in the module docstring above.
    """

    def __init__(self) -> None:
        # TODO: Initialize google-genai client
        #   from google import genai
        #   from voice_assistant import config
        #   self._client = genai.Client(api_key=config.GOOGLE_API_KEY)
        #   self._session = None   # opened lazily or in connect()
        raise NotImplementedError(
            "GeminiLiveSession.__init__: Install google-genai and implement "
            "the async session lifecycle. See module docstring for details."
        )

    # ── BaseTTS ───────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        # TODO: Return True only when the WebSocket session is open and healthy.
        #   return self._session is not None and not self._session.closed
        raise NotImplementedError(
            "GeminiLiveSession.is_ready: Check WebSocket connection status."
        )

    def speak(self, text: str) -> None:
        # TODO: Send text to Gemini and receive + play the audio response.
        #
        # Option A — text-only turn (simpler, extra latency):
        #   await self._session.send(input=text, end_of_turn=True)
        #   async for response in self._session.receive():
        #       if response.data:          # raw PCM audio bytes
        #           play_pcm(response.data, sample_rate=24000)
        #
        # Option B — use Gemini's audio output directly in the bidirectional loop
        # (preferred: no extra round-trip, audio arrives while listening).
        raise NotImplementedError(
            "GeminiLiveSession.speak: Send text to Gemini, receive PCM audio, "
            "play with sounddevice. See module docstring option A/B."
        )

    # ── BaseSTT ───────────────────────────────────────────────────────────────

    def listen_and_transcribe(self) -> Optional[str]:
        # TODO: With Gemini Live, STT is server-side.
        # Stream microphone PCM to the WebSocket and Gemini returns text.
        #
        #   chunk_size = 1024   # bytes of 16-bit PCM at 16kHz
        #   while recording:
        #       chunk = mic.read(chunk_size)
        #       await self._session.send(input={"data": chunk, "mime_type": "audio/pcm"})
        #
        # Gemini fires a server_content event with input_transcription when done.
        # You can intercept that in the receive() loop and return the text.
        #
        # NOTE: With Gemini Live you likely want to remove the explicit
        # listen_and_transcribe() call from assistant.py and instead let the
        # bidirectional streaming loop handle the full conversation turn.
        raise NotImplementedError(
            "GeminiLiveSession.listen_and_transcribe: Stream PCM from mic to "
            "Gemini WebSocket and await the input_transcription response."
        )

    # ── BaseAIBrain ───────────────────────────────────────────────────────────

    def ask_chatbot(self, question: str, user_id: int = 1) -> str:
        # TODO: Inject the chatbot question into the Gemini session turn.
        # The system prompt should already include CV context so Gemini can
        # answer questions about focus, posture, fatigue, and study materials.
        #
        #   await self._session.send(input=question, end_of_turn=True)
        #   # collect text response from receive() loop
        #
        # Alternatively, keep the HTTP call to /chatbot/ask (LocalAIBrain)
        # as a tool_call that Gemini can invoke natively via function calling.
        raise NotImplementedError(
            "GeminiLiveSession.ask_chatbot: Either route the question through "
            "the live session turn or call the backend via LocalAIBrain and "
            "speak the result with self.speak()."
        )

    def get_planning_today(self) -> str:
        # TODO: Same approach as ask_chatbot — fetch from backend and inject
        # as context into the Gemini session, or use native function calling.
        #
        # Suggested: reuse LocalAIBrain.get_planning_today() for the HTTP call
        # and feed the result as text input to the Gemini session.
        raise NotImplementedError(
            "GeminiLiveSession.get_planning_today: Fetch from backend and "
            "inject into Gemini session turn."
        )

    def get_latest_stats(self, session_id: str) -> str:
        # TODO: Same approach as get_planning_today.
        # The CV snapshot can be injected at session start as system context
        # so Gemini is always aware of current focus/posture/fatigue state.
        raise NotImplementedError(
            "GeminiLiveSession.get_latest_stats: Fetch from backend and "
            "inject into Gemini session turn or system prompt."
        )

    # ── Session lifecycle (to implement) ─────────────────────────────────────

    async def connect(self) -> None:
        # TODO: Open the Gemini Live WebSocket session.
        #
        #   from google import genai
        #   from google.genai import types
        #   from voice_assistant import config
        #
        #   system_prompt = (
        #       "Tu es SmartFocus, un assistant d'étude intelligent. "
        #       "Tu parles uniquement en français. "
        #       "Tu aides l'utilisateur à rester concentré, à surveiller sa posture "
        #       "et à gérer sa fatigue pendant ses sessions d'étude."
        #   )
        #
        #   live_config = types.LiveConnectConfig(
        #       response_modalities=["AUDIO"],
        #       speech_config=types.SpeechConfig(
        #           voice_config=types.VoiceConfig(
        #               prebuilt_voice_config=types.PrebuiltVoiceConfig(
        #                   voice_name=config.GEMINI_VOICE_NAME
        #               )
        #           )
        #       ),
        #       system_instruction=system_prompt,
        #   )
        #
        #   self._session = await self._client.aio.live.connect(
        #       model=config.GEMINI_LIVE_MODEL,
        #       config=live_config,
        #   ).__aenter__()
        raise NotImplementedError("GeminiLiveSession.connect: Open WebSocket session.")

    async def disconnect(self) -> None:
        # TODO: Close the WebSocket session gracefully.
        #   if self._session:
        #       await self._session.__aexit__(None, None, None)
        #       self._session = None
        raise NotImplementedError("GeminiLiveSession.disconnect: Close WebSocket session.")
