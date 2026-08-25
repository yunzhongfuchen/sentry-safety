#!/bin/bash
# ============================================================
# Sentry RK3588 一键安装脚本
# 在 RK3588 设备上运行
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="/opt/sentry"
SERVICE_USER="sentry"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "=========================================="
echo " Sentry RK3588 安装程序"
echo "=========================================="
echo ""

# ── 检查权限 ──
if [[ $EUID -ne 0 ]]; then
    error "请使用 sudo 运行此脚本"
    echo "  sudo ./install.sh"
    exit 1
fi

# ── 检查架构 ──
ARCH=$(uname -m)
if [[ "$ARCH" != "aarch64" ]]; then
    warn "当前架构: $ARCH (非 aarch64)"
    read -p "此安装包为 RK3588 (aarch64) 设计，是否继续? [y/N] " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

# ── 检查 NPU 驱动 ──
NPU_AVAILABLE=false
if [[ -f "/usr/lib/librknnrt.so" ]] || [[ -f "/usr/lib/librknnmrt.so" ]]; then
    info "检测到 RKNN NPU 驱动 ✓"
    NPU_AVAILABLE=true
else
    warn "未检测到 RKNN NPU 驱动，将使用 CPU 模式"
    warn "如需 NPU 加速，请先安装 RKNPU2 驱动"
fi

# ── 步骤 1: 系统依赖 ──
echo ""
info "[1/6] 安装系统依赖..."
apt-get update -qq
apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-venv \
    python3-opencv \
    libopencv-dev \
    ffmpeg \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    fonts-noto-cjk \
    > /dev/null 2>&1

info "系统依赖安装完成 ✓"

# ── 步骤 2: 创建安装目录 ──
echo ""
info "[2/6] 部署文件到 ${INSTALL_DIR}..."

# 如果已存在，备份配置
if [[ -d "$INSTALL_DIR" ]]; then
    warn "检测到已有安装，备份配置..."
    [[ -f "$INSTALL_DIR/.env" ]] && cp "$INSTALL_DIR/.env" "/tmp/sentry_env_backup"
    [[ -f "$INSTALL_DIR/config/cameras.json" ]] && cp "$INSTALL_DIR/config/cameras.json" "/tmp/sentry_cameras_backup"
fi

mkdir -p "$INSTALL_DIR"

# 复制文件
cp -r "$SCRIPT_DIR/backend"   "$INSTALL_DIR/"
cp -r "$SCRIPT_DIR/frontend"  "$INSTALL_DIR/"
cp -r "$SCRIPT_DIR/config"    "$INSTALL_DIR/"
cp -r "$SCRIPT_DIR/models"    "$INSTALL_DIR/"
[[ -d "$SCRIPT_DIR/weights" ]] && cp -r "$SCRIPT_DIR/weights" "$INSTALL_DIR/"
[[ -d "$SCRIPT_DIR/fonts" ]] && cp -r "$SCRIPT_DIR/fonts" "$INSTALL_DIR/"
cp    "$SCRIPT_DIR/start.sh"  "$INSTALL_DIR/"
cp    "$SCRIPT_DIR/requirements.txt" "$INSTALL_DIR/"

mkdir -p "$INSTALL_DIR/data/frames"
mkdir -p "$INSTALL_DIR/logs"

# 恢复备份的配置
if [[ -f "/tmp/sentry_env_backup" ]]; then
    cp "/tmp/sentry_env_backup" "$INSTALL_DIR/.env"
    info "已恢复 .env 配置"
fi
if [[ -f "/tmp/sentry_cameras_backup" ]]; then
    cp "/tmp/sentry_cameras_backup" "$INSTALL_DIR/config/cameras.json"
    info "已恢复摄像头配置"
fi

# 如果没有 .env，从模板创建
if [[ ! -f "$INSTALL_DIR/.env" ]]; then
    cp "$SCRIPT_DIR/.env.default" "$INSTALL_DIR/.env"
    warn "已创建默认 .env，请编辑填入 API Key:"
    warn "  nano $INSTALL_DIR/.env"
fi

chmod +x "$INSTALL_DIR/start.sh"
info "文件部署完成 ✓"

# ── 步骤 3: Python 虚拟环境 ──
echo ""
info "[3/6] 创建 Python 虚拟环境..."

if [[ -d "$INSTALL_DIR/venv" ]]; then
    info "虚拟环境已存在，跳过创建"
else
    python3 -m venv "$INSTALL_DIR/venv"
fi

source "$INSTALL_DIR/venv/bin/activate"
pip install --upgrade pip setuptools wheel -q

info "虚拟环境就绪 ✓"

# ── 步骤 4: 安装 Python 依赖 ──
echo ""
info "[4/6] 安装 Python 依赖 (可能需要几分钟)..."
pip install -r "$INSTALL_DIR/requirements.txt" -q

info "Python 依赖安装完成 ✓"

# ── 步骤 5: 安装 RKNN Lite2 ──
echo ""
info "[5/6] 配置 RKNN 运行时..."

if $NPU_AVAILABLE; then
    # 查找本地 RKNN wheel
    RKNN_WHEEL=$(find "$SCRIPT_DIR" "$INSTALL_DIR" /opt -name "rknn_toolkit_lite2*aarch64.whl" 2>/dev/null | head -1)

    if [[ -n "$RKNN_WHEEL" ]]; then
        info "找到 RKNN Lite2: $RKNN_WHEEL"
        pip install "$RKNN_WHEEL" -q
        info "RKNN Lite2 安装完成 ✓"
    elif pip list 2>/dev/null | grep -q rknn; then
        info "RKNN Lite2 已安装 ✓"
    else
        warn "未找到 RKNN Lite2 wheel 包"
        warn "请手动安装: pip install rknn_toolkit_lite2-*-aarch64.whl"
        warn "下载地址: https://github.com/airockchip/rknn-toolkit2/tree/master/rknn_toolkit_lite2/packages"
    fi

    # 检查 RKNN 模型
    if [[ -f "$INSTALL_DIR/models/yolov8n.rknn" ]]; then
        info "RKNN 模型就绪 ✓"
    elif [[ -f "$INSTALL_DIR/models/yolov8n.onnx" ]]; then
        info "找到 ONNX 模型，尝试在线转换为 RKNN..."
        # 尝试用 rknn_toolkit_lite2 或 rknn_toolkit2 转换
        source "$INSTALL_DIR/venv/bin/activate"
        python3 - "$INSTALL_DIR/models/yolov8n.onnx" "$INSTALL_DIR/models/yolov8n.rknn" << 'CONVERTEOF' && {
import sys
onnx_path = sys.argv[1]
rknn_path = sys.argv[2]
try:
    from rknn.api import RKNN
except ImportError:
    try:
        from rknnlite.api import RKNNLite as RKNN
    except ImportError:
        print("SKIP: no rknn toolkit available")
        sys.exit(1)

rknn = RKNN(verbose=False)
rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], target_platform='rk3588')
if rknn.load_onnx(model=onnx_path) != 0:
    print("FAIL: load onnx")
    sys.exit(1)
if rknn.build(do_quantization=False) != 0:
    print("FAIL: build")
    sys.exit(1)
if rknn.export_rknn(rknn_path) != 0:
    print("FAIL: export")
    sys.exit(1)
rknn.release()
print("OK")
CONVERTEOF
            info "RKNN 模型转换成功 ✓"
            sed -i 's/DETECTION_DEVICE=cpu/DETECTION_DEVICE=npu/' "$INSTALL_DIR/.env"
        } || {
            warn "RKNN 转换失败（需要 rknn-toolkit2，通常只能在 x86 上运行）"
            warn "系统将使用 CPU 模式 (YOLOv8)"
            sed -i 's/DETECTION_DEVICE=npu/DETECTION_DEVICE=cpu/' "$INSTALL_DIR/.env"
        }
        deactivate 2>/dev/null || true
    else
        warn "未找到 RKNN 模型: $INSTALL_DIR/models/yolov8n.rknn"
        warn "请在 x86 主机上转换模型后复制到此目录"
        # 回退到 CPU 模式
        sed -i 's/DETECTION_DEVICE=npu/DETECTION_DEVICE=cpu/' "$INSTALL_DIR/.env"
        warn "已自动切换为 CPU 模式"
    fi
else
    info "NPU 不可用，使用 CPU 模式"
    sed -i 's/DETECTION_DEVICE=npu/DETECTION_DEVICE=cpu/' "$INSTALL_DIR/.env"
fi

deactivate

# ── 步骤 6: 安装 systemd 服务 ──
echo ""
info "[6/6] 配置系统服务..."

cat > /etc/systemd/system/sentry.service << SVCEOF
[Unit]
Description=Sentry 受限空间安全哨兵
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${INSTALL_DIR}
Environment=PYTHONPATH=${INSTALL_DIR}:${INSTALL_DIR}/backend
EnvironmentFile=-${INSTALL_DIR}/.env
ExecStart=${INSTALL_DIR}/start.sh
Restart=always
RestartSec=5
LimitNOFILE=65535

StandardOutput=journal
StandardError=journal
SyslogIdentifier=sentry

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
info "systemd 服务已配置 ✓"

# ── 完成 ──
echo ""
echo "=========================================="
echo " 安装完成!"
echo "=========================================="
echo ""
echo " 安装目录: $INSTALL_DIR"
echo " NPU 模式: $($NPU_AVAILABLE && echo '可用' || echo '不可用 (CPU 回退)')"
echo ""
echo " 下一步操作:"
echo ""
echo " 1. 编辑配置 (必须):"
echo "    nano $INSTALL_DIR/.env"
echo "    # 填入 ARK_API_KEY 和 VLM_ENDPOINT"
echo ""
echo " 2. 配置摄像头:"
echo "    nano $INSTALL_DIR/config/cameras.json"
echo ""
if ! $NPU_AVAILABLE || [[ ! -f "$INSTALL_DIR/models/yolov8n.rknn" ]]; then
echo " 3. (可选) 放置 RKNN 模型后启用 NPU:"
echo "    scp yolov8n.rknn user@this-device:$INSTALL_DIR/models/"
echo "    # 然后修改 .env 中 DETECTION_DEVICE=npu"
echo ""
fi
echo " 启动服务:"
echo "    sudo systemctl start sentry"
echo "    sudo systemctl enable sentry   # 开机自启"
echo ""
echo " 查看日志:"
echo "    journalctl -u sentry -f"
echo ""
echo " 手动运行 (调试):"
echo "    cd $INSTALL_DIR && ./start.sh"
echo ""

# 获取 IP
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [[ -n "$IP" ]]; then
    PORT=$(grep -oP 'API_PORT=\K\d+' "$INSTALL_DIR/.env" 2>/dev/null || echo "8000")
    echo " 访问地址: http://${IP}:${PORT}"
    echo ""
fi
