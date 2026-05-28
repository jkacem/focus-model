"""
SmartFocus Voice Assistant — Entry Point
=========================================
Usage:
    python run_voice_assistant.py --session-id <UUID>

Options:
    --session-id   UUID of the active CV session (required)
    --backend-url  FastAPI backend base URL (default: http://localhost:8000)
    --provider     Override VOICE_PROVIDER from config.py (local | gemini_live)
    --debug        Enable verbose DEBUG logging

Examples:
    python run_voice_assistant.py --session-id abc123-...
    python run_voice_assistant.py --session-id abc123-... --debug
    python run_voice_assistant.py --session-id abc123-... --provider gemini_live
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import shutil
import sys
from pathlib import Path

# ── Make sure the pi_client root is on sys.path ──────────────────────────────
# This allows `from voice_assistant import ...` regardless of where the script
# is launched from.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── Load .env file (PI_API_KEY, GOOGLE_API_KEY, BACKEND_URL) ─────────────────
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_HERE / ".env")
except ImportError:
    pass

# ── Ensure ffmpeg is on PATH (needed by Whisper's audio decoder) ─────────────
# Git Bash and some launchers don't inherit the Windows user PATH where winget
# installs ffmpeg. Search known locations and inject if not already found.
if shutil.which("ffmpeg") is None and platform.system() == "Windows":
    _ffmpeg_candidates = [
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Microsoft/WinGet/Packages"
    ]
    for _base in _ffmpeg_candidates:
        if _base.exists():
            for _p in _base.rglob("ffmpeg.exe"):
                os.environ["PATH"] = str(_p.parent) + os.pathsep + os.environ["PATH"]
                break


def find_latest_session() -> str | None:
    """Return the UUID of the most recently modified session file, or None."""
    sessions_dir = _HERE / "output" / "sessions"
    if not sessions_dir.exists():
        return None
    jsonl_files = [f for f in sessions_dir.glob("*.jsonl") if not f.stem.endswith("_summary")]
    if not jsonl_files:
        return None
    latest = max(jsonl_files, key=lambda f: f.stat().st_mtime)
    return latest.stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SmartFocus AI Voice Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--session-id",
        required=False,
        default=None,
        metavar="UUID",
        help="CV session UUID. If omitted, the most recent session is used automatically.",
    )
    parser.add_argument(
        "--backend-url",
        default=None,
        metavar="URL",
        help="FastAPI backend URL (default: value from config.py)",
    )
    parser.add_argument(
        "--provider",
        choices=["local", "gemini_live"],
        default=None,
        help="Voice provider override (default: value from config.py)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduce noise from noisy third-party loggers
    for noisy in ("urllib3", "httpx", "httpcore", "absl"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def print_banner(session_id: str, backend_url: str, provider: str) -> None:
    sep = "═" * 56
    arch = platform.machine()
    os_name = f"{platform.system()} {platform.release()} ({arch})"

    print(f"\n{sep}")
    print("   🎙  SmartFocus Voice Assistant")
    print(sep)
    print(f"   Platform   : {os_name}")
    print(f"   Provider   : {provider}")
    print(f"   Backend    : {backend_url}")
    print(f"   Session ID : {session_id}")
    print(sep)
    print("   Say 'Hey Jarvis' to start a conversation.")
    print("   Press Ctrl-C to quit.")
    print(f"{sep}\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(args.debug)
    logger = logging.getLogger("smartfocus.voice")

    # ── Import config and apply CLI overrides ─────────────────────────────────
    from voice_assistant import config  # noqa: PLC0415

    session_id: str = args.session_id or find_latest_session() or ""
    if not session_id:
        logger.warning(
            "No session found. Start main_cv.py first, or pass --session-id. "
            "Stats and alerts will be unavailable."
        )
        import uuid as _uuid
        session_id = str(_uuid.uuid4())
    backend_url: str = args.backend_url or config.BACKEND_URL
    provider: str = args.provider or config.VOICE_PROVIDER

    # Inject session_id globally so config.SESSION_ID is always set
    config.SESSION_ID = session_id

    print_banner(session_id, backend_url, provider)

    # ── Build and run assistant ───────────────────────────────────────────────
    try:
        from voice_assistant.assistant import VoiceAssistant  # noqa: PLC0415
    except ImportError as exc:
        logger.critical(
            "Failed to import VoiceAssistant: %s\n"
            "Run install_voice.sh (Linux/Pi) or install_voice_windows.bat (Windows) first.",
            exc,
        )
        sys.exit(1)

    try:
        assistant = VoiceAssistant(
            session_id=session_id,
            backend_url=backend_url,
            provider=provider,
        )
        assistant.run()

    except KeyboardInterrupt:
        print("\n[SmartFocus] Interrupted by user.")
    except Exception as exc:
        logger.critical("Assistant crashed: %s", exc, exc_info=args.debug)
        sys.exit(1)


if __name__ == "__main__":
    main()
