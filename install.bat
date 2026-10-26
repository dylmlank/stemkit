@echo off
REM One-shot StemKit install into a local venv.
REM Uncomment the CUDA line below first if you have an NVIDIA GPU — separation
REM goes from ~15 minutes a song to ~1.
cd /d "%~dp0"

if not exist venv (
    echo Creating venv...
    python -m venv venv
)

echo Installing build tooling...
venv\Scripts\python.exe -m pip install --upgrade pip wheel

REM venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121

echo Installing StemKit dependencies...
venv\Scripts\python.exe -m pip install -r requirements.txt || goto :failed

echo Installing basic-pitch (no-deps — see requirements.txt for why)...
venv\Scripts\python.exe -m pip install --no-deps basic-pitch || goto :failed

echo.
venv\Scripts\python.exe -c "from core import Workspace; import core.tools as T; print(T.check_setup(Workspace()))"
echo.
echo Done. Run start_stemkit.bat to use the agent.
goto :eof

:failed
echo.
echo Install failed. ffmpeg/ffprobe must also be on PATH — they are not pip packages.
exit /b 1
