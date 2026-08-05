@echo off
if not exist logs mkdir logs
start "sentry-backend" cmd /c "C:\Users\12800\miniconda3\envs\py312\python.exe backend/main_multi.py > logs/main_multi.log 2>&1"
echo Backend started: http://localhost:8000
echo Frontend:        http://localhost:8000/monitor
