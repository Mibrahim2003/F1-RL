@echo off
REM ============================================================================
REM  F1-RL quick start
REM  Launches the two halves of the app in their own windows:
REM    1. Backend  - FastAPI on uvicorn  (http://127.0.0.1:8000)
REM    2. Frontend - Vite dev server     (http://localhost:5173, proxies -> 8000)
REM  Then opens the app in your browser. Close either window to stop that half.
REM ============================================================================
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [F1-RL] venv Python not found at:
  echo         "%PY%"
  echo         Create the venv, then:  "%PY%" -m pip install -e ".[dev]"
  pause
  exit /b 1
)

echo [F1-RL] starting backend  (FastAPI/uvicorn on :8000) ...
REM /d sets the working dir (handles the space in the path); PYTHONPATH=src lets
REM f1rl import without an editable install. --reload restarts on code changes.
start "F1-RL backend" /d "%ROOT%" cmd /k "set PYTHONPATH=src&& .venv\Scripts\python.exe -m uvicorn f1rl.server.app:app --reload"

echo [F1-RL] starting frontend (Vite dev server on :5173) ...
REM First run installs node_modules; later runs skip straight to the dev server.
REM if/else (no trailing operator): cmd discards a '&&' placed after a parenthesized
REM if-body, so the old one-liner never reached 'npm run dev' when node_modules existed.
REM 'call' so npm.cmd returns control and the install->dev chain actually continues.
start "F1-RL frontend" /d "%ROOT%web" cmd /k "if exist node_modules (call npm run dev) else (call npm install && call npm run dev)"

REM Give Vite a moment to boot, then open the app.
timeout /t 5 /nobreak >nul
start "" http://localhost:5173

echo [F1-RL] backend + frontend launched in separate windows.
endlocal
