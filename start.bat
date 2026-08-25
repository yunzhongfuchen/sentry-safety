@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

cd /d "%~dp0"
if not exist logs mkdir logs

REM 1. 查找 Python 路径
set "PY_EXE="

if exist "%~dp0venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0.venv\Scripts\python.exe"
) else if exist "%USERPROFILE%\miniconda3\envs\py312\python.exe" (
    set "PY_EXE=%USERPROFILE%\miniconda3\envs\py312\python.exe"
) else if exist "%USERPROFILE%\anaconda3\envs\py312\python.exe" (
    set "PY_EXE=%USERPROFILE%\anaconda3\envs\py312\python.exe"
) else if exist "C:\ProgramData\miniconda3\envs\py312\python.exe" (
    set "PY_EXE=C:\ProgramData\miniconda3\envs\py312\python.exe"
) else if exist "C:\ProgramData\anaconda3\envs\py312\python.exe" (
    set "PY_EXE=C:\ProgramData\anaconda3\envs\py312\python.exe"
) else (
    where python >nul 2>&1
    if !errorlevel! == 0 (
        for /f "delims=" %%i in ('where python') do (
            if not defined PY_EXE set "PY_EXE=%%i"
        )
    )
)

if not defined PY_EXE (
    echo [错误] 未找到 Python 运行环境！
    echo 请先安装 Conda py312 环境或在项目根目录创建 venv 虚拟环境。
    echo 参考部署文档: DEPLOY_WINDOWS.md
    pause
    exit /b 1
)

echo 使用 Python: !PY_EXE!
start "sentry-backend" cmd /c ""!PY_EXE!" backend/main_multi.py > logs/main_multi.log 2>&1"
echo 后端服务已启动: http://localhost:8000
echo 实时监控页面:   http://localhost:8000/monitor
echo 日志文件路径:   logs/main_multi.log
