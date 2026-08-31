@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"
if not exist logs mkdir logs

set "PY_EXE="

if exist "%~dp0venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0venv\Scripts\python.exe"
    goto :found_python
)
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0.venv\Scripts\python.exe"
    goto :found_python
)
if exist "%USERPROFILE%\miniconda3\envs\py312\python.exe" (
    set "PY_EXE=%USERPROFILE%\miniconda3\envs\py312\python.exe"
    goto :found_python
)
if exist "%USERPROFILE%\anaconda3\envs\py312\python.exe" (
    set "PY_EXE=%USERPROFILE%\anaconda3\envs\py312\python.exe"
    goto :found_python
)
if exist "C:\ProgramData\miniconda3\envs\py312\python.exe" (
    set "PY_EXE=C:\ProgramData\miniconda3\envs\py312\python.exe"
    goto :found_python
)
if exist "C:\ProgramData\anaconda3\envs\py312\python.exe" (
    set "PY_EXE=C:\ProgramData\anaconda3\envs\py312\python.exe"
    goto :found_python
)

for /f "delims=" %%i in ('where python 2^>nul') do (
    if not defined PY_EXE (
        set "PY_EXE=%%i"
        goto :found_python
    )
)

:found_python
if not defined PY_EXE (
    echo [ERROR] Python not found.
    echo Please install Conda py312 environment or create venv in project root.
    pause
    exit /b 1
)

echo Python interpreter: !PY_EXE!
start "sentry-backend" cmd /c ""!PY_EXE!" backend/main_multi.py > logs/main_multi.log 2>&1"
echo Backend service started: http://localhost:8111
echo Monitor page:            http://localhost:8111/monitor
echo Log file:                logs/main_multi.log
