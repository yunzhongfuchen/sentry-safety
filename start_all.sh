#!/bin/bash
# Sentry 启动脚本（安全检测）
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 激活虚拟环境
PYTHON_BIN="/home/user/miniconda3/envs/py312/bin/python"
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
echo " Sentry 服务启动"
echo "=========================================="

# 启动安全检测服务（端口 8111）
if lsof -i :8111 >/dev/null 2>&1; then
    echo "[WARN] 端口 8111 已被占用，跳过安全检测服务"
else
    echo "[1/1] 启动安全检测服务 -> http://0.0.0.0:8111"
    nohup "$PYTHON_BIN" backend/main_multi.py > logs/main_multi.log 2>&1 &
    echo $! > /tmp/sentry_main.pid
fi

echo "=========================================="
echo " 安全检测: http://localhost:8111"
echo "=========================================="
echo " 日志: logs/main_multi.log"
echo "=========================================="
echo " 停止命令: ./stop_all.sh"
