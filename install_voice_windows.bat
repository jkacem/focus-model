@echo off
setlocal

:: =============================================================================
:: SmartFocus Voice Assistant - Windows Install Script
:: Installs all voice packages into the project venv.
:: Run from CMD or PowerShell (no admin rights needed):
::   .\install_voice_windows.bat
:: =============================================================================

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "VENV_DIR=%SCRIPT_DIR%\venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "VA_DIR=%SCRIPT_DIR%\voice_assistant"
set "PIPER_BIN_DIR=%VA_DIR%\piper_bin"
set "MODELS_DIR=%VA_DIR%\piper_models"
set "VOICE_MODEL=fr_FR-upmc-medium"

echo.
echo ============================================================
echo    SmartFocus Voice Assistant - Windows Installer
echo ============================================================
echo    Project : %SCRIPT_DIR%
echo    Venv    : %VENV_DIR%
echo ============================================================
echo.

:: ------------------------------------------------------------
:: STEP 1 - Check system Python
:: ------------------------------------------------------------
echo [1/9] Checking system Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERR] Python not found in PATH.
    echo       Install Python 3.9+ from https://python.org
    echo       Tick "Add Python to PATH" during install.
    goto :fail
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo       %%v
echo [OK]  Python found.

:: ------------------------------------------------------------
:: STEP 2 - Create or reuse venv
:: ------------------------------------------------------------
echo.
echo [2/9] Setting up virtual environment...
if exist "%VENV_PY%" (
    echo [OK]  Existing venv found - reusing it.
) else (
    echo       Creating venv at %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERR] Failed to create venv.
        goto :fail
    )
    echo [OK]  Venv created.
)
for /f "tokens=*" %%v in ('"%VENV_PY%" --version 2^>^&1') do echo       Venv uses %%v
echo.
echo       All packages go into: %VENV_DIR%

:: ------------------------------------------------------------
:: STEP 3 - Check ffmpeg
:: ------------------------------------------------------------
echo.
echo [3/9] Checking ffmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARN] ffmpeg NOT found. Whisper needs it for audio decoding.
    echo        Install: winget install --id Gyan.FFmpeg -e --source winget
    echo        Or: https://ffmpeg.org/download.html  then add to PATH.
    echo        Continuing - install ffmpeg before using the assistant.
) else (
    echo [OK]  ffmpeg found.
)

:: ------------------------------------------------------------
:: STEP 4 - Upgrade pip inside venv
:: ------------------------------------------------------------
echo.
echo [4/9] Upgrading pip in venv...
"%VENV_PY%" -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [!!]  pip upgrade failed - continuing with current version.
) else (
    echo [OK]  pip upgraded.
)

:: ------------------------------------------------------------
:: STEP 5 - PyAudio
:: Note: pipwin is broken on Python 3.12. We try pip directly.
:: sounddevice is the primary audio library; pyaudio is optional.
:: ------------------------------------------------------------
echo.
echo [5/9] Installing PyAudio (optional - sounddevice is the primary driver)...
"%VENV_PY%" -m pip install pyaudio --quiet
if errorlevel 1 (
    echo [WARN] PyAudio install failed. This is OK - sounddevice handles all audio.
    echo        If you need pyaudio later: pip install pipwin then pipwin install pyaudio
) else (
    echo [OK]  PyAudio installed.
)

:: ------------------------------------------------------------
:: STEP 6 - Voice requirements
:: ------------------------------------------------------------
echo.
echo [6/9] Installing voice assistant packages...
"%VENV_PY%" -m pip install -r "%SCRIPT_DIR%\requirements_voice.txt"
if errorlevel 1 (
    echo [ERR] pip install failed. Check requirements_voice.txt and network.
    goto :fail
)
echo [OK]  All voice packages installed into venv.

:: ------------------------------------------------------------
:: STEP 7 - Piper binary
:: ------------------------------------------------------------
echo.
echo [7/9] Downloading Piper TTS binary (Windows amd64)...
if not exist "%PIPER_BIN_DIR%" mkdir "%PIPER_BIN_DIR%"

set "PIPER_EXE=%PIPER_BIN_DIR%\piper.exe"
set "PIPER_URL=https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"
set "PIPER_ZIP=%TEMP%\piper_windows_amd64.zip"

if exist "%PIPER_EXE%" (
    echo [OK]  piper.exe already present - skipping download.
    goto :piper_done
)

echo       Downloading from: %PIPER_URL%
powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PIPER_URL%' -OutFile '%PIPER_ZIP%' -UseBasicParsing"
if errorlevel 1 (
    echo [ERR] Download failed. Check internet connection.
    goto :fail
)

echo       Extracting...
powershell -NoProfile -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem; [IO.Compression.ZipFile]::ExtractToDirectory('%PIPER_ZIP%', '%PIPER_BIN_DIR%')"
del /f /q "%PIPER_ZIP%" >nul 2>&1

:: Piper zip may nest files inside a piper\ subfolder - flatten it
if exist "%PIPER_BIN_DIR%\piper\piper.exe" (
    move /y "%PIPER_BIN_DIR%\piper\piper.exe" "%PIPER_BIN_DIR%\piper.exe" >nul
    for %%f in ("%PIPER_BIN_DIR%\piper\*") do move /y "%%f" "%PIPER_BIN_DIR%\" >nul 2>&1
    rmdir /s /q "%PIPER_BIN_DIR%\piper" >nul 2>&1
)

if exist "%PIPER_EXE%" (
    echo [OK]  piper.exe ready.
) else (
    echo [ERR] piper.exe not found after extraction.
    echo       Download manually from: %PIPER_URL%
    echo       Extract piper.exe + all .dll files into: %PIPER_BIN_DIR%\
    goto :fail
)
:piper_done

:: ------------------------------------------------------------
:: STEP 8 - French voice model
:: ------------------------------------------------------------
echo.
echo [8/9] Downloading Piper French voice model (%VOICE_MODEL%)...
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"

set "HF_BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium"
set "ONNX_FILE=%MODELS_DIR%\%VOICE_MODEL%.onnx"
set "JSON_FILE=%MODELS_DIR%\%VOICE_MODEL%.onnx.json"

if exist "%ONNX_FILE%" (
    echo [OK]  %VOICE_MODEL%.onnx already present - skipping.
) else (
    echo       Downloading %VOICE_MODEL%.onnx - approx 63 MB...
    powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%HF_BASE%/%VOICE_MODEL%.onnx' -OutFile '%ONNX_FILE%' -UseBasicParsing"
    if errorlevel 1 (
        echo [ERR] .onnx download failed.
        goto :fail
    )
    echo [OK]  %VOICE_MODEL%.onnx downloaded.
)

if exist "%JSON_FILE%" (
    echo [OK]  %VOICE_MODEL%.onnx.json already present - skipping.
) else (
    echo       Downloading %VOICE_MODEL%.onnx.json...
    powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%HF_BASE%/%VOICE_MODEL%.onnx.json' -OutFile '%JSON_FILE%' -UseBasicParsing"
    if errorlevel 1 (
        echo [ERR] .json download failed.
        goto :fail
    )
    echo [OK]  %VOICE_MODEL%.onnx.json downloaded.
)

:: ------------------------------------------------------------
:: STEP 9 - OpenWakeWord models
:: ------------------------------------------------------------
echo.
echo [9/9] Pre-downloading OpenWakeWord hey_jarvis model...
"%VENV_PY%" -c "from openwakeword.utils import download_models; print('  Downloading hey_jarvis...'); download_models(model_names=['hey_jarvis']); print('  Done.')"
if errorlevel 1 (
    echo [WARN] Pre-download failed - model will download on first run.
) else (
    echo [OK]  OpenWakeWord model cached.
)

:: ------------------------------------------------------------
:: SUCCESS
:: ------------------------------------------------------------
echo.
echo ============================================================
echo    Installation complete!
echo ============================================================
echo.
echo    HOW TO RUN (from pi_client directory):
echo.
echo    Option A - activate venv first (recommended):
echo      venv\Scripts\activate
echo      python run_voice_assistant.py --session-id YOUR-UUID
echo.
echo    Option B - call venv Python directly:
echo      venv\Scripts\python run_voice_assistant.py --session-id YOUR-UUID
echo.
echo    WORKFLOW:
echo      1. Start main_cv.py  (gives you a session UUID in its output)
echo      2. Start run_voice_assistant.py with that UUID
echo      3. Say "Hey Jarvis" to talk to the assistant
echo.
echo ============================================================
echo.
pause
exit /b 0

:: ------------------------------------------------------------
:fail
echo.
echo [ERR] Installation did not complete. See messages above.
echo.
pause
exit /b 1
