#!/bin/bash
# Sentry GPU 动态调度器启动脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# GPU 调度器配置（可修改以下参数）
export USE_GPU_SCHEDULER=true
export GPU_SCHEDULER_NUM_QUEUES=0      # 0 = 自动（一模型一队列）
export GPU_SCHEDULER_INTERVAL=0.5      # 调度周期（秒）
export GPU_SCHEDULER_HALF=false        # 是否启用 FP16 半精度推理

# 加载其他 .env 配置（如果存在）
if [[ -f ".env" ]]; then
    set -a
    source .env
    set +a
fi

echo "=========================================="
echo " Sentry GPU 调度模式启动"
echo "=========================================="
echo " USE_GPU_SCHEDULER = $USE_GPU_SCHEDULER"
echo " GPU_SCHEDULER_NUM_QUEUES = $GPU_SCHEDULER_NUM_QUEUES"
echo " GPU_SCHEDULER_INTERVAL = $GPU_SCHEDULER_INTERVAL"
echo " GPU_SCHEDULER_HALF = $GPU_SCHEDULER_HALF"
echo "=========================================="

# 调用主启动脚本
exec "$SCRIPT_DIR/start_all.sh"
