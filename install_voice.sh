#!/usr/bin/env bash
# =============================================================================
# SmartFocus Voice Assistant — Linux / Raspberry Pi 5 Install Script
# =============================================================================
# Supports:
#   Raspberry Pi 5  (aarch64, Raspberry Pi OS Bookworm)
#   Linux x86_64    (Ubuntu / Debian)
#
# Run:
#   chmod +x install_voice.sh
#   bash install_voice.sh
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC}  $*"; }
info() { echo -e "${CYAN}[--]${NC}  $*"; }
warn() { echo -e "${YELLOW}[!!]${NC}  $*"; }
err()  { echo -e "${RED}[ERR]${NC} $*" >&2; }

# ── Detect platform ───────────────────────────────────────────────────────────
ARCH=$(uname -m)
OS=$(uname -s)

if [[ "$OS" != "Linux" ]]; then
    err "This script is for Linux/Raspberry Pi only. Use install_voice_windows.bat on Windows."
    exit 1
fi

echo -e "\n${BOLD}══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}   SmartFocus Voice Assistant — Linux/Pi Installer${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════════${NC}"
info "Architecture : $ARCH"
info "OS           : $OS $(lsb_release -rs 2>/dev/null || echo '')"
echo ""

# ── Script directory ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VA_DIR="$SCRIPT_DIR/voice_assistant"

# ── Step 1: System dependencies ───────────────────────────────────────────────
echo -e "${BOLD}[1/7] Installing system packages…${NC}"
sudo apt-get update -qq
sudo apt-get install -y \
    portaudio19-dev \
    ffmpeg \
    espeak-ng \
    libsndfile1 \
    libsndfile1-dev \
    libasound2-dev \
    python3-dev \
    libatlas-base-dev \
    curl \
    wget \
    unzip \
    tar \
    python3-pip \
    python3-venv
ok "System packages installed."

# ── Step 2: Python dependencies ───────────────────────────────────────────────
echo -e "\n${BOLD}[2/7] Installing Python packages…${NC}"
pip install --upgrade pip

# PyTorch CPU (ARM64) — must be installed before openai-whisper
info "Installing PyTorch CPU wheel for ARM64…"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install pyaudio

# Use Pi-specific requirements (no piper-tts package, no httpx)
if [[ -f "$SCRIPT_DIR/requirements_pi_voice.txt" ]]; then
    pip install -r "$SCRIPT_DIR/requirements_pi_voice.txt"
else
    pip install -r "$SCRIPT_DIR/requirements_voice.txt"
fi
ok "Python packages installed."

# ── Step 3: Piper TTS binary ──────────────────────────────────────────────────
echo -e "\n${BOLD}[3/7] Downloading Piper TTS binary…${NC}"
PIPER_BIN_DIR="$VA_DIR/piper_bin"
mkdir -p "$PIPER_BIN_DIR"

if [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "arm64" ]]; then
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz"
    info "Detected ARM64 (Raspberry Pi 5) → downloading aarch64 binary"
else
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"
    info "Detected x86_64 → downloading x86_64 binary"
fi

PIPER_BINARY="$PIPER_BIN_DIR/piper"
if [[ -f "$PIPER_BINARY" ]]; then
    warn "Piper binary already exists at $PIPER_BINARY — skipping download."
else
    TMP_ARCHIVE=$(mktemp --suffix=".tar.gz")
    info "Downloading from: $PIPER_URL"
    curl -L --progress-bar "$PIPER_URL" -o "$TMP_ARCHIVE"

    info "Extracting…"
    tar -xzf "$TMP_ARCHIVE" -C "$PIPER_BIN_DIR" --strip-components=1

    rm -f "$TMP_ARCHIVE"

    if [[ -f "$PIPER_BINARY" ]]; then
        chmod +x "$PIPER_BINARY"
        ok "Piper binary ready at $PIPER_BINARY"
    else
        err "Piper binary not found after extraction. Check the URL or extract manually."
        err "URL: $PIPER_URL"
        exit 1
    fi
fi

# ── Step 4: Piper voice model (fr_FR-upmc-medium) ─────────────────────────────
echo -e "\n${BOLD}[4/7] Downloading Piper French voice model…${NC}"
MODELS_DIR="$VA_DIR/piper_models"
mkdir -p "$MODELS_DIR"

VOICE_MODEL="fr_FR-upmc-medium"
ONNX_FILE="$MODELS_DIR/${VOICE_MODEL}.onnx"
JSON_FILE="$MODELS_DIR/${VOICE_MODEL}.onnx.json"
HF_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium"

for FILE_PATH in "$ONNX_FILE" "$JSON_FILE"; do
    FILENAME=$(basename "$FILE_PATH")
    if [[ -f "$FILE_PATH" ]]; then
        warn "$FILENAME already exists — skipping."
    else
        info "Downloading $FILENAME…"
        curl -L --progress-bar "$HF_BASE/$FILENAME" -o "$FILE_PATH"
        ok "$FILENAME downloaded ($(du -sh "$FILE_PATH" | cut -f1))"
    fi
done

# ── Step 5: OpenWakeWord models ───────────────────────────────────────────────
echo -e "\n${BOLD}[5/7] Downloading OpenWakeWord models…${NC}"
info "OpenWakeWord downloads 'hey_jarvis' automatically on first use."
info "Pre-downloading now to avoid first-run delay…"
python3 -c "
from openwakeword.utils import download_models
print('  Downloading hey_jarvis model…')
download_models(model_names=['hey_jarvis'])
print('  Model ready.')
" && ok "OpenWakeWord models ready." || warn "Could not pre-download OpenWakeWord models — they will download on first run."

# ── Step 6: Verify Whisper ────────────────────────────────────────────────────
echo -e "\n${BOLD}[6/7] Verifying Whisper…${NC}"
python3 -c "
import whisper
print('  Loading Whisper tiny model (optimised for Pi CPU)…')
m = whisper.load_model('tiny')
print('  Whisper ready.')
" && ok "Whisper model verified." || warn "Whisper verification failed — will retry on first run."

# ── Step 7: Final check ───────────────────────────────────────────────────────
echo -e "\n${BOLD}[7/7] Final verification…${NC}"

CHECKS_PASSED=0
CHECKS_TOTAL=4

check_item() {
    local DESC="$1"; local CMD="$2"
    if eval "$CMD" &>/dev/null; then
        ok "$DESC"
        ((CHECKS_PASSED++)) || true
    else
        warn "$DESC — FAILED"
    fi
}

check_item "Piper binary executable"    "[[ -x '$PIPER_BINARY' ]]"
check_item "Voice model .onnx present"  "[[ -f '$ONNX_FILE' ]]"
check_item "Voice model .json present"  "[[ -f '$JSON_FILE' ]]"
check_item "sounddevice importable"     "python3 -c 'import sounddevice'"

echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════${NC}"
if [[ "$CHECKS_PASSED" -eq "$CHECKS_TOTAL" ]]; then
    echo -e "${GREEN}${BOLD}   Installation complete! ($CHECKS_PASSED/$CHECKS_TOTAL checks passed)${NC}"
else
    echo -e "${YELLOW}${BOLD}   Installation done with warnings ($CHECKS_PASSED/$CHECKS_TOTAL checks passed)${NC}"
fi
echo -e "${BOLD}══════════════════════════════════════════════════════${NC}"
echo ""
echo "  Quick start (run both in parallel):"
echo "    Terminal 1:  python main_cv.py"
echo "    Terminal 2:  python run_voice_assistant.py"
echo ""
echo "  With debug logging:"
echo "    python run_voice_assistant.py --debug"
echo ""
