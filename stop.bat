@echo off
setlocal enabledelayedexpansion
set "PID="
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do set "PID=%%a"
if not defined PID (
    echo Port 8000: no process found
    exit /b 1
)
taskkill /PID %PID% /F > nul 2>&1
if %errorlevel% EQU 0 (echo Stopped PID %PID%) else (echo Failed to stop PID %PID%)
