#!/bin/bash
# Sentry 有限空间服务启动脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 自动解析 Python 解释器 ──
# 优先级:
#   1) 显式设置的 PYTHON_BIN 环境变量(最高优先,适合 systemd / 自定义部署)
#   2) 项目内的 venv / .venv (会自动 activate)
#   3) 已激活的 conda / virtualenv ($VIRTUAL_ENV / $CONDA_PREFIX)
#   4) 自动探测常见 conda 安装下的项目专用环境 (py310 / sentry)
#   5) 系统 python3 / python
PYTHON_BIN_RESOLVED=""
resolve_python() {
    if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN:-}" ]]; then
        PYTHON_BIN_RESOLVED="$PYTHON_BIN"
        return 0
    fi
    if [[ -x "$SCRIPT_DIR/venv/bin/python" ]]; then
        # shellcheck disable=SC1091
        source "$SCRIPT_DIR/venv/bin/activate"
        PYTHON_BIN_RESOLVED="$(command -v python)"
        return 0
    fi
    if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
        # shellcheck disable=SC1091
        source "$SCRIPT_DIR/.venv/bin/activate"
        PYTHON_BIN_RESOLVED="$(command -v python)"
        return 0
    fi
    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV:-}/bin/python" ]]; then
        PYTHON_BIN_RESOLVED="$VIRTUAL_ENV/bin/python"
        return 0
    fi
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX:-}/bin/python" ]]; then
        PYTHON_BIN_RESOLVED="$CONDA_PREFIX/bin/python"
        return 0
    fi
    local base env
    for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" \
                "/opt/miniconda3" "/opt/conda" "/opt/anaconda3"; do
        for env in py310 sentry; do
            if [[ -x "$base/envs/$env/bin/python" ]]; then
                PYTHON_BIN_RESOLVED="$base/envs/$env/bin/python"
                return 0
            fi
        done
    done
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN_RESOLVED="$(command -v python3)"
        return 0
    fi
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN_RESOLVED="$(command -v python)"
        return 0
    fi
    return 1
}

if ! resolve_python; then
    echo "[ERROR] 找不到 Python 解释器" >&2
    echo "  - 设置环境变量 PYTHON_BIN 指向 python 可执行文件, 或" >&2
    echo "  - 激活 conda/venv 环境后再运行, 或" >&2
    echo "  - 在项目目录创建 venv/.venv" >&2
    exit 1
fi
PYTHON_BIN="$PYTHON_BIN_RESOLVED"
echo "[INFO] Python: $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"

# ── 加载 .env ──
if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# ── 设置 PYTHONPATH ──
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/backend${PYTHONPATH:+:$PYTHONPATH}"

# ── 确保日志目录 ──
mkdir -p logs

# ── 端口 / 主机可配置 ──
HOST="${CONFINED_HOST:-${API_HOST:-0.0.0.0}}"
PORT="${CONFINED_PORT:-8001}"

echo "=========================================="
echo " Sentry 有限空间监控服务"
echo "=========================================="
echo " 地址: http://${HOST}:${PORT}"
echo "=========================================="

exec "$PYTHON_BIN" -m uvicorn backend.main_confined:app --host "$HOST" --port "$PORT" --log-level warning
