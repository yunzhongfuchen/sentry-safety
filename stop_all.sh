#!/bin/bash
# Sentry 双服务停止脚本

echo "=========================================="
echo " 停止 Sentry 服务"
echo "=========================================="

# 停止安全检测服务
if [[ -f /tmp/sentry_main.pid ]]; then
    PID=$(cat /tmp/sentry_main.pid)
    if kill "$PID" 2>/dev/null; then
        echo "[OK] 安全检测服务已停止 (PID: $PID)"
    else
        echo "[WARN] 安全检测服务未运行"
    fi
    rm -f /tmp/sentry_main.pid
else
    echo "[WARN] 未找到安全检测服务 PID"
fi

echo "=========================================="
