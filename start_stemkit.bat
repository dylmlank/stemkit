@echo off
REM Launch the StemKit agent in a terminal.
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
python agent.py %*
pause
