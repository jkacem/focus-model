"""
Local AI Brain — HTTP Backend Calls
=====================================
Implements BaseAIBrain by talking to the SmartFocus FastAPI backend.

Endpoints used:
  POST /api/v1/chatbot/ask         → AI answer from user's documents (Gemini/Groq)
  GET  /api/v1/planning/today      → Today's study sessions
  GET  /api/v1/sessions/{id}/latest → Live CV snapshot scores

All public methods return ready-to-speak French strings.
Connection errors are caught and converted to graceful French error messages.
Retries with exponential back-off are applied on transient errors.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import requests

from voice_assistant import config
from voice_assistant.providers.base import BaseAIBrain

logger = logging.getLogger("smartfocus.voice")

# Request timeout (seconds)
_TIMEOUT = 8.0
# Max retries for transient errors
_MAX_RETRIES = 2


class LocalAIBrain(BaseAIBrain):
    """HTTP client for the SmartFocus FastAPI backend."""

    def __init__(self, backend_url: str, session_id: str) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.session_id = session_id

        headers = {"Content-Type": "application/json"}
        if config.PI_API_KEY:
            headers["X-API-Key"] = config.PI_API_KEY
        self._headers = headers

    # ── BasAIBrain interface ──────────────────────────────────────────────────

    def ask_chatbot(self, question: str, user_id: int = 1) -> str:
        """Answer a free-form question. Tries Gemini directly, falls back to backend."""
        if config.GOOGLE_API_KEY:
            try:
                return self._call_gemini(question)
            except Exception as exc:
                logger.warning("[Brain] Gemini direct call failed: %s", exc)

        url = f"{self.backend_url}/chatbot/chat"
        payload = {"question": question, "user_id": user_id, "document_ids": []}
        try:
            data = self._post(url, payload)
            if isinstance(data, dict):
                answer = (
                    data.get("answer")
                    or data.get("response")
                    or data.get("message")
                    or str(data)
                )
            else:
                answer = str(data)
            return answer.strip() or "Je n'ai pas trouvé de réponse à ta question."
        except _BackendError as exc:
            logger.warning("[Brain] chatbot/ask failed: %s", exc)
            return "Désolé, je ne peux pas répondre à cette question pour l'instant. Vérifie ta clé GOOGLE_API_KEY dans le fichier .env."

    def get_planning_today(self) -> str:
        """GET /api/v1/planning/today — summary of today's study sessions."""
        url = f"{self.backend_url}/api/v1/planning/today"

        try:
            data = self._get(url)
        except _BackendError as exc:
            logger.warning("[Brain] planning/today failed: %s", exc)
            return "Je ne peux pas accéder au planning pour le moment."

        return self._format_planning(data)

    def get_latest_stats(self, session_id: str) -> str:
        """Return current focus/posture/fatigue scores. Reads local file first."""
        local_data = self._read_local_stats(session_id)
        if local_data:
            return self._format_stats(local_data)

        url = f"{self.backend_url}/api/v1/sessions/{session_id}/latest"
        try:
            data = self._get(url)
        except _BackendError as exc:
            logger.warning("[Brain] sessions/latest failed: %s", exc)
            return "Impossible de récupérer tes statistiques pour le moment."

        return self._format_stats(data)

    def _read_local_stats(self, session_id: str) -> Optional[dict]:
        """Read stats from session_summary.json (exact averaged values from main_cv.py).

        Falls back to the most recent per-frame .jsonl entry if no summary exists.
        """
        # Primary: active session — a .jsonl file with NO matching _summary.json yet,
        # AND modified within the last 2 hours (excludes old orphaned files).
        import time as _time
        _TWO_HOURS = 2 * 3600
        if config.SESSIONS_DIR.exists():
            now = _time.time()
            jsonl_files = list(config.SESSIONS_DIR.glob("*.jsonl"))
            active = [
                f for f in jsonl_files
                if not (config.SESSIONS_DIR / f"{f.stem}_summary.json").exists()
                and (now - f.stat().st_mtime) < _TWO_HOURS
            ]
            # Pick the most recently modified active session file
            active.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            for f in active:
                try:
                    # Read last 60 frames and average them for a stable score
                    lines = []
                    with open(f, "r", encoding="utf-8") as fh:
                        for line in fh:
                            s = line.strip()
                            if s:
                                lines.append(s)
                    recent = lines[-60:] if len(lines) >= 60 else lines
                    if not recent:
                        continue
                    entries = [json.loads(l) for l in recent]
                    scores_list = [e.get("scores", {}) for e in entries if e.get("scores")]
                    if not scores_list:
                        continue
                    def avg(key):
                        vals = [s.get(key) for s in scores_list if s.get(key) is not None]
                        return round(sum(vals) / len(vals), 1) if vals else None
                    return {
                        "attention_score": avg("concentration"),
                        "posture_score": avg("posture"),
                        "fatigue_score": avg("fatigue"),
                        "distraction_score": avg("distraction"),
                        "global_focus_score": avg("focus_global"),
                    }
                except (OSError, json.JSONDecodeError):
                    continue

        # Fallback: session_summary.json — last completed session's averaged data.
        summary_path = config.SESSION_SUMMARY_PATH
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("focus_global") is not None:
                    return {
                        "attention_score": data.get("concentration"),
                        "posture_score": data.get("posture"),
                        "fatigue_score": data.get("fatigue"),
                        "distraction_score": data.get("distraction"),
                        "global_focus_score": data.get("focus_global"),
                    }
            except (OSError, json.JSONDecodeError):
                pass

        # Fallback: last frame from the most recent .jsonl file
        if not config.SESSIONS_DIR.exists():
            return None
        all_files = [
            f for f in config.SESSIONS_DIR.glob("*.jsonl")
            if not f.stem.endswith("_summary")
        ]
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
                scored.append((entry.get("timestamp", ""), entry))
            except (OSError, json.JSONDecodeError):
                continue
        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        raw = scored[0][1]["scores"]
        return {
            "attention_score": raw.get("concentration"),
            "posture_score": raw.get("posture"),
            "fatigue_score": None,
            "distraction_score": raw.get("distraction"),
            "global_focus_score": raw.get("focus_global"),
        }

    def _call_gemini(self, question: str) -> str:
        """Call Gemini API directly for free-form questions."""
        try:
            from google import genai  # google-genai>=1.0.0
            client = genai.Client(api_key=config.GOOGLE_API_KEY)
            system = (
                "Tu es SmartFocus, un assistant vocal qui aide les étudiants à améliorer "
                "leur concentration. Réponds en français, de façon concise et encourageante. "
                "Tes réponses seront lues à voix haute — pas de markdown, pas de listes."
            )
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"{system}\n\nQuestion: {question}",
            )
            return response.text.strip()
        except ImportError:
            pass

        # Fallback: try older google-generativeai SDK
        import google.generativeai as genai2
        genai2.configure(api_key=config.GOOGLE_API_KEY)
        model = genai2.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(question)
        return response.text.strip()

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _get(self, url: str) -> Any:
        return self._request("GET", url)

    def _post(self, url: str, payload: dict) -> Any:
        return self._request("POST", url, json=payload)

    def _request(self, method: str, url: str, **kwargs) -> Any:
        """Perform an HTTP request with retries and consistent error handling."""
        last_exc: Optional[Exception] = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = requests.request(
                    method,
                    url,
                    headers=self._headers,
                    timeout=_TIMEOUT,
                    **kwargs,
                )
                if resp.status_code == 404:
                    raise _BackendError(f"404 Not Found: {url}")
                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.ConnectionError as exc:
                last_exc = _BackendError(f"Backend unreachable ({url}): {exc}")
            except requests.exceptions.Timeout:
                last_exc = _BackendError(f"Request timed out ({url})")
            except requests.exceptions.HTTPError as exc:
                # 4xx client errors — don't retry
                raise _BackendError(f"HTTP error {exc.response.status_code}: {exc}") from exc
            except ValueError as exc:
                raise _BackendError(f"Invalid JSON response from {url}: {exc}") from exc

            if attempt < _MAX_RETRIES:
                wait = 2 ** attempt  # 1s, 2s
                logger.debug("[Brain] Retry %d/%d in %ds…", attempt + 1, _MAX_RETRIES, wait)
                time.sleep(wait)

        raise last_exc  # type: ignore[misc]

    # ── Response formatters ───────────────────────────────────────────────────

    @staticmethod
    def _format_stats(data: Any) -> str:
        """Convert raw snapshot JSON → readable French score summary."""
        if not isinstance(data, dict):
            return "Aucune donnée de session disponible pour l'instant."

        focus = data.get("global_focus_score")
        attention = data.get("attention_score")
        posture = data.get("posture_score")
        fatigue = data.get("fatigue_score")
        distraction = data.get("distraction_score")

        parts: list[str] = []

        if focus is not None:
            level = _score_label(focus)
            parts.append(f"Ton score de focus global est de {int(focus)} pour cent, ce qui est {level}.")

        if attention is not None:
            parts.append(f"Ta concentration est à {int(attention)} pour cent.")

        if posture is not None:
            posture_comment = "bonne" if posture >= 60 else ("correcte" if posture >= 35 else "mauvaise")
            parts.append(f"Ta posture est {posture_comment} avec un score de {int(posture)} pour cent.")

        if fatigue is not None:
            fatigue_comment = "très fatigué" if fatigue >= 2.0 else ("un peu fatigué" if fatigue >= 0.8 else "en forme")
            parts.append(f"Pour la fatigue, tu sembles {fatigue_comment}.")

        if distraction is not None:
            distraction_comment = "élevée" if distraction >= 20 else ("modérée" if distraction >= 10 else "faible")
            parts.append(f"Ton niveau de distraction est {distraction_comment} à {round(distraction, 1)} pour cent.")

        if not parts:
            return "Pas encore de données de session disponibles."

        return " ".join(parts)

    @staticmethod
    def _format_planning(data: Any) -> str:
        """Convert raw planning JSON → readable French schedule summary."""
        if not data:
            return "Tu n'as aucune session d'étude planifiée pour aujourd'hui. Profites-en pour t'organiser !"

        # data may be a list of sessions or a dict with a sessions key
        sessions: list = []
        if isinstance(data, list):
            sessions = data
        elif isinstance(data, dict):
            sessions = data.get("sessions") or data.get("items") or data.get("data") or []

        if not sessions:
            return "Aucune session planifiée pour aujourd'hui."

        lines = [f"Tu as {len(sessions)} session{'s' if len(sessions) > 1 else ''} prévue{'s' if len(sessions) > 1 else ''} aujourd'hui."]

        for i, session in enumerate(sessions[:5], 1):  # Speak at most 5
            if isinstance(session, dict):
                name = session.get("name") or session.get("title") or session.get("subject") or f"Session {i}"
                start = session.get("start_time") or session.get("start") or ""
                end = session.get("end_time") or session.get("end") or ""

                if start and end:
                    lines.append(f"Session {i} : {name}, de {start} à {end}.")
                elif start:
                    lines.append(f"Session {i} : {name}, à partir de {start}.")
                else:
                    lines.append(f"Session {i} : {name}.")

        return " ".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

class _BackendError(Exception):
    """Wraps all backend communication errors."""


def _score_label(score: float) -> str:
    """Return a French adjective matching the score range."""
    if score >= 80:
        return "excellent"
    if score >= 60:
        return "bon"
    if score >= 40:
        return "moyen"
    return "faible"
