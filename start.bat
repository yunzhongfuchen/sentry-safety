@echo off
if not exist logs mkdir logs
start "sentry-backend" cmd /c "conda run -n base python backend/main_multi.py > logs/main_multi.log 2>&1"
echo Backend started: http://localhost:8000
echo Frontend:        http://localhost:8000/monitor
