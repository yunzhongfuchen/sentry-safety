#!/bin/bash
# Sentry 启动脚本 (RK3588)
set -e

# 定位项目目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 激活虚拟环境
if [[ -d "venv" ]]; then
    source venv/bin/activate
elif [[ -d ".venv" ]]; then
    source .venv/bin/activate
fi

# 加载环境变量
if [[ -f ".env" ]]; then
    set -a
    source .env
    set +a
fi

# 设置 Python 路径
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/backend:${PYTHONPATH}"

# 检查 RKNN 模型 (NPU 模式)
if [[ "$DETECTION_DEVICE" == "npu" ]]; then
    RKNN_PATH="${RKNN_MODEL:-models/yolov8n.rknn}"
    if [[ ! -f "$RKNN_PATH" ]]; then
        echo "[WARN] RKNN 模型不存在: $RKNN_PATH，回退到 CPU 模式"
        export DETECTION_DEVICE=cpu
    fi
fi

# 确保数据目录存在
mkdir -p data/frames logs

# 选择启动模式
MODE="${SENTRY_MODE:-multi}"
HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8111}"

echo "=========================================="
echo " Sentry 受限空间安全哨兵"
echo "=========================================="
echo " 模式: $MODE"
echo " 检测: ${DETECTION_DEVICE:-cpu}"
echo " 地址: http://${HOST}:${PORT}"
echo "=========================================="

if [[ "$MODE" == "single" ]]; then
    exec python backend/main.py
else
    exec python backend/main_multi.py
fi
