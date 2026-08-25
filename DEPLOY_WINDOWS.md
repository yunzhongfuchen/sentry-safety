# 目标机（Windows + NVIDIA GPU）部署指引

本机已清理干净，按下面步骤在另一台电脑上安装环境并运行。

---

## 1. 拷贝文件到目标机

从本机（开发机）复制以下内容到目标机（保持目录结构一致）：

```
sentry-safety/            ← 整个项目文件夹（代码）
├── backend/              ← 后端代码
├── frontend/             ← 前端页面
├── config/               ← 运行配置（algorithms.json 等 6 个，务必拷全）
├── weights/              ← 12 个模型（约 203M，必须带）
├── fonts/                ← CJK 字体（约 3.6M）
├── image/                ← 图标
├── scripts/
├── tests/  docs/
├── requirements.txt
├── start.bat  stop.bat
└── .env.example
```

> **不要拷贝**：`.git/`、`data/`、`uploads/`、`logs/`、`交大交付/`、`sophon-stream-master/`、`__pycache__/`、`.codegraph/`、`.playwright-mcp/`、`.claude/`、`.env`
> （这些是运行时产物或开发工具，目标机不需要；`data/`、`uploads/`、`logs/` 启动时会自动创建）

---

## 2. 安装 Python 环境（3.12）

**方式 A：安装 Miniconda（推荐，与本机一致）**
1. 下载安装 [Miniconda3](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)
2. 打开 "Anaconda Prompt"（或 CMD 后 `conda activate` 前先确认 conda 可用）
3. 创建环境：
   ```bash
   conda create -n py312 python=3.12 -y
   conda activate py312
   ```

**方式 B：仅用 Python 自带 venv**
```bash
python -m venv venv
venv\Scripts\activate
```
（需要已安装 Python 3.12）

---

## 3. 安装依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

### GPU 推理需要 CUDA 版 PyTorch

`requirements.txt` 里的 `ultralytics` 默认会拉 CPU 版 torch。**要 GPU 推理必须先装 CUDA 版 torch**：

```bash
# 推荐：先装 CUDA 版 torch（本机验证过 cu126），再装其余依赖
pip install torch==2.12.0+cu126 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

> 若目标机显卡较新（RTX 40/50 系），可考虑 `cu126` 以上版本；若不确定，先按 cu126 装，代码里 `torch.cuda.is_available()` 能自动检测。

### 可选：RKNN 相关（本机 Windows 不需要）

- `requirements.txt` 里的 `rknn` 相关未列出（Windows 下无 aarch64 wheel），代码已在 [inference_engine.py:19-24](backend/inference_engine.py#L19-L24) 做了 try/except 兜底，**不装也能跑**，自动走 GPU/CPU。

---

## 4. 配置

```bash
copy .env.example .env
```

编辑 `.env`：
- 项目会自动探测设备（GPU > NPU > CPU），Windows + NVIDIA 会自动用 GPU，`DETECTION_DEVICE` 无需强改
- 填好 VLM 配置（ARK_API_KEY / ARK_MODEL，若使用视频理解功能）
- `API_PORT` 按需修改（默认 8000）

---

## 5. 启动

**方式一（推荐）：双击 `start.bat`**

`start.bat` 里 `python.exe` 路径是开发机的，目标机需改为自己的 Python 路径：
```bat
start "sentry-backend" cmd /c "C:\Users\<用户名>\miniconda3\envs\py312\python.exe backend/main_multi.py > logs/main_multi.log 2>&1"
```
> 改法：把路径换成目标机实际的 `python.exe` 位置（`where python` 可查）；若用 venv 则是 `venv\Scripts\python.exe`。

**方式二：命令行直接启动**
```bash
conda activate py312
python backend/main_multi.py
```

启动后：
- 后端 API：http://localhost:8000
- 监控页面：http://localhost:8000/monitor

---

## 6. 验证

打开 http://localhost:8000/monitor，能看到页面即成功。若 GPU 生效，日志中应有 `CUDA GPU detected and verified, using GPU mode`。
