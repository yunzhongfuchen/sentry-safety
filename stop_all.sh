#!/bin/bash
# Sentry 双服务停止脚本

echo "=========================================="
echo " 停止 Sentry 服务"
echo "=========================================="

# 停止安全检测服务：先读 pid 文件，再按端口兜底
STOPPED=false

if [[ -f /tmp/sentry_main.pid ]]; then
    PID=$(cat /tmp/sentry_main.pid)
    if kill "$PID" 2>/dev/null; then
        echo "[OK] 安全检测服务已停止 (PID: $PID)"
        STOPPED=true
    fi
    rm -f /tmp/sentry_main.pid
fi

# 按端口 8111 查找并强制清理残留进程
PIDS=$(lsof -t -i :8111 2>/dev/null)
if [[ -n "$PIDS" ]]; then
    echo "[INFO] 发现端口 8111 残留进程: $PIDS"
    kill -9 $PIDS 2>/dev/null
    echo "[OK] 已强制清理端口 8111 残留进程"
    STOPPED=true
fi

if [[ "$STOPPED" == false ]]; then
    echo "[WARN] 未找到运行中的安全检测服务"
fi

echo "=========================================="
