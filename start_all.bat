@echo off
chcp 65001 >nul
REM Sentry startup script for Windows

setlocal enabledelayedexpansion

REM Locate project directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Activate Conda environment py312
call conda activate py312
if errorlevel 1 (
    echo [ERROR] Failed to activate Conda environment py312
    pause
    exit /b 1
)

REM Load environment variables if .env exists
if exist ".env" (
    for /f "usebackq tokens=*" %%a in (".env") do (
        set "line=%%a"
        if not "!line:~0,1!"=="#" (
            if not "!line!"=="" (
                for /f "tokens=1,2 delims==" %%b in ("!line!") do (
                    set "%%b=%%c"
                )
            )
        )
    )
)

REM Set Python path
set "PYTHONPATH=%SCRIPT_DIR%;%SCRIPT_DIR%\backend;%PYTHONPATH%"

REM Ensure logs directory exists
if not exist "logs" mkdir logs

echo ==========================================
echo  Sentry service startup
echo ==========================================

REM Check if port 8111 is in use
netstat -ano | findstr ":8111" | findstr "LISTENING" >nul
if %errorlevel% == 0 (
    echo [WARN] Port 8111 is already in use, skipping safety detection service
) else (
    echo [1/1] Starting safety detection service -^> http://0.0.0.0:8111
    python backend\main_multi.py > logs\main_multi.log 2>&1
)

echo ==========================================
echo  Safety detection: http://localhost:8111
echo ==========================================
echo  Logs: logs\main_multi.log
echo ==========================================
pause
