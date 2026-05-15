#!/bin/bash
# Sentry 双服务启动脚本（安全检测 + 有限空间）
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 激活虚拟环境
PYTHON_BIN="/home/yangrunfu/miniconda3/envs/py310/bin/python"
if [[ -d "venv" ]]; then
    source venv/bin/activate
    PYTHON_BIN="python"
elif [[ -d ".venv" ]]; then
    source .venv/bin/activate
    PYTHON_BIN="python"
fi

# 加载环境变量
if [[ -f ".env" ]]; then
    set -a
    source .env
    set +a
fi

# 设置 Python 路径
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/backend:${PYTHONPATH}"

# 确保日志目录存在
mkdir -p logs

echo "=========================================="
echo " Sentry 双服务启动"
echo "=========================================="

# 启动安全检测服务（端口 8000）
if lsof -i :8000 >/dev/null 2>&1; then
    echo "[WARN] 端口 8000 已被占用，跳过安全检测服务"
else
    echo "[1/2] 启动安全检测服务 -> http://0.0.0.0:8000"
    nohup "$PYTHON_BIN" backend/main_multi.py > logs/main_multi.log 2>&1 &
    echo $! > /tmp/sentry_main.pid
fi

# 启动有限空间服务（端口 8001）
if lsof -i :8001 >/dev/null 2>&1; then
    echo "[WARN] 端口 8001 已被占用，跳过有限空间服务"
else
    echo "[2/2] 启动有限空间服务 -> http://0.0.0.0:8001"
    nohup "$PYTHON_BIN" -m uvicorn backend/main_confined:app --host 0.0.0.0 --port 8001 > logs/main_confined.log 2>&1 &
    echo $! > /tmp/sentry_confined.pid
fi

echo "=========================================="
echo " 安全检测: http://localhost:8000"
echo " 有限空间: http://localhost:8001"
echo "=========================================="
echo " 日志: logs/main_multi.log"
echo " 日志: logs/main_confined.log"
echo "=========================================="
echo " 停止命令: ./stop_all.sh"
