# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在本仓库中工作时提供指引。

## Karpathy 编码准则

减少 LLM 常见编码错误的行为准则，源自 [Andrej Karpathy 的观察](https://x.com/karpathy/status/2015883857489522876)。

**取舍：** 这套准则倾向谨慎而非速度。对琐碎任务请自行判断。

### 1. 先思考，再写代码

**不要假设。不要隐藏困惑。把权衡摆到台面上。**

动手之前：
- 显式说明你的假设。不确定时——问。
- 如果存在多种解读，全部列出——不要替用户默默选一种。
- 如果有更简单的方案，说出来。该顶时就顶。
- 如果有不清楚的地方，停下。把困惑命名出来。问。

### 2. 简单优先

**用解决问题的最少代码。不要任何投机性内容。**

- 不写未被要求的功能。
- 不为只用一次的代码做抽象。
- 不引入未被要求的"灵活性"或"可配置性"。
- 不为不可能发生的场景写错误处理。
- 如果你写了 200 行而本可以 50 行——重写。

问自己："资深工程师会觉得这过度复杂吗？" 如果是，简化。

### 3. 外科手术式改动

**只动该动的地方。只清理你自己造成的混乱。**

修改现有代码时：
- 不要"顺手优化"周围的代码、注释或格式。
- 不要重构没坏的东西。
- 即便你会用别的写法，也要匹配现有风格。
- 如果你看到无关的死代码，告诉用户——不要自己删。

当你的改动产生了孤儿代码：
- 删除被你的改动弄成无用的 import / 变量 / 函数。
- 不要删除原本就存在的死代码，除非用户要求。

判定标准：每一行改动都应能直接追溯到用户的请求。

### 4. 目标驱动执行

**定义成功标准。循环到可验证为止。**

把任务变成可验证的目标：
- "加个校验" → "为非法输入写测试，然后让它们通过"
- "修这个 bug" → "写一个能复现 bug 的测试，然后让它通过"
- "重构 X" → "改前改后测试都能通过"

多步任务先列简短计划：

```
1. [步骤] → 验证：[检查方式]
2. [步骤] → 验证：[检查方式]
3. [步骤] → 验证：[检查方式]
```

强成功标准让你能独立循环到位。弱标准（"让它能用"）只会让你反复追问澄清。

## 项目概述

Sentry 是一套面向 RK3588 SoC 的边缘 AI 安全监控系统，由两个独立服务组成。后端使用 Python/FastAPI；前端为纯 HTML + Vue，作为静态文件直接提供（无构建步骤）。所有 UI 文案为简体中文（zh-CN）；代码与注释中英文混合。

## 常用命令

```bash
# 同时运行两个服务（开发模式）—— 安全检测 8000，有限空间 8001
./start_all.sh
./stop_all.sh

# 单一服务（入口由 .env 中的 SENTRY_MODE 决定：默认 "multi"，"single" 为旧版单摄像头）
./start.sh
./start_confined.sh        # 通过 uvicorn 单独启动有限空间服务

# RK3588 上的生产部署（部署到 /opt/sentry，安装 systemd 服务）
sudo ./install.sh
sudo systemctl start sentry
journalctl -u sentry -f

# 把内嵌 base64 的旧版 records.json 迁移为：元数据 + data/frames/*.jpg
python backend/migrate_records.py

# 一次性的检测器烟雾测试脚本
python backend/test_sleep.py
python backend/test_sleep_cam1.py

# 手动运行指定入口（PYTHONPATH 必须包含 backend/）
PYTHONPATH=$(pwd):$(pwd)/backend python backend/main_multi.py
PYTHONPATH=$(pwd):$(pwd)/backend python -m uvicorn backend.main_confined:app --port 8001
```

仓库中没有测试套件、Lint 配置或构建流水线。两个 `test_sleep*.py` 脚本是临时的烟雾测试，并非 unittest/pytest 用例。

## 架构

### 两个独立服务，两个不同端口

`backend/main_multi.py`（端口 8000）—— **安全检测**：明火（fire）、烟雾（smoke）、工服（uniform）、口罩（mask）、香烟（cigarette）、睡岗（sleep）。它会同时挂载 `safety_detection` 路由和（可选的）`confined_space` 路由——所以如果只想跑一个进程，这个服务也能托管有限空间的前端。前端入口为 `/multi`（`frontend/safety_detection/multi.html`）。

`backend/main_confined.py`（端口 8001）—— 仅 **有限空间监控**：基于区域（zone）的人员计数 + VLM 复核。前端入口为 `/monitor`（`frontend/confined_space/monitor.html`）。

`backend/main.py` 是为了 `/legacy` 保留的旧版单摄像头入口，不要在它之上扩展。

### 检测流水线（safety_detection）

1. **CameraManager**（`camera_manager.py`）为每路摄像头维护一条读取线程（RTSP / USB / 本地视频，带自动重连）。每个 `CameraState` 内部有一个滚动 `frame_history` 队列，触发告警时可以保存触发前 5 秒的帧。RTSP 通过 `OPENCV_FFMPEG_CAPTURE_OPTIONS` 强制使用 TCP 传输，避免 WSL2 / UDP 丢包。
2. **SafetyDetector**（`inference_engine.py`）是一个支持懒加载的多模型注册表。CPU/GPU 模型为全局单例；NPU 模型则**每个 NPU 核心实例化一次**（RK3588 共 3 个核心，掩码为 `RKNNLite.NPU_CORE_0/1/2`）。`detect_best_device()` 按 GPU > NPU > CPU 的优先级选择设备。
3. **MultiDetector**（`safety_detection/detector_core.py`）以 (摄像头, 检测类型) 为粒度维护 `TypeSchedule`，通过 `DetectionStrategy` 调度运行：
   - `CorePinnedStrategy`：每个 NPU 核心一条 worker 线程，摄像头按取模分配到各核心。NPU 核心数 ≥ 2 时启用。
   - `SerialStrategy`：单线程轮询所有摄像头，CPU 模式下的回退方案。
4. **每种类型的规则**（写在 `TypeSchedule` 中）：
   - `fire` / `smoke`（P0）—— 经 VLM 复核后告警，冷却时间较短。
   - `mask` / `cigarette` / `uniform` / `sleep`（P1）—— 必须经 VLM 确认后才告警。
   - `uniform` 使用 `compliance_window_seconds`（默认 30 秒）—— 在该时间窗口内任意时刻检测到反光背心，则抑制告警。
   - `sleep` 仅 CPU（YOLOv8-pose），通过 `safety_detection/sleep_detect.py` 基于 17 个关键点判断姿态。
   - `mask` / `cigarette` 使用 **DamoYOLO（来自 ModelScope）**，而不是 Ultralytics。`inference_engine._sync_weights_to_cache()` 会把 `weights/` 下更新过的 `.pt` 文件复制到 `~/.cache/modelscope/...`。
5. **VLMQueue**（`vlm_queue.py`）—— 双优先级队列（P0 上限 50，P1 上限 100），用 `Semaphore(max_concurrent)` 限制并发（默认 3）。调用 `understander.analyze_multi()`，后者通过 `arkitect` 与火山引擎 Ark 通信。
6. **VLMInspector**（`vlm_inspector.py`）—— 每 30 秒挑选若干摄像头，用一个多类型「巡检 prompt」让 VLM 整体扫描一遍，弥补小模型漏检；命中后把"伪检测"注入回 `MultiDetector`（同时与活跃告警、待 VLM、冷却中、睡岗状态机做四重去重）。
7. **触发回调** 在 `main_multi.py` 中（`on_trigger`）：创建告警记录，把带框快照与窗口帧保存为 JPEG 文件，并通过 `_records_dirty` 事件触发持久化。

### 有限空间流水线

`backend/confined_space/zone_counter.py` 在一条约 2 fps 的循环中运行（`main_multi.py` 中的 `_confined_space_loop`，或 `main_confined.py` 中的同名函数）。每个区域支持两种模式：

- **direct（直视）**：YOLO 直接在 ROI 内数人；连续帧防抖；人数变化时可选 VLM 复核。
- **entrance（入口）**：YOLO 检测「入口处有人」→ 状态机在窗口内积累帧 → 把窗口提交给 VLM，由 VLM 把窗口分类为 `enter` / `leave` / `stay` 并据此更新人数。适用于摄像头只能看到入口、看不到内部的场景。

区域配置持久化到 `config/confined_cameras.json`（每个摄像头条目包含一个 `zones` 数组）。`confined_space/api.py` 挂载在 `/api` 前缀下。

### 帧渲染与检测解耦

一条独立的**渲染线程**（`main_multi.py` 中的 `_overlay_loop`）从 `CameraManager` 取帧，从 `multi_detector._latest_results` 拿到最新的检测结果缓存，把原始帧和带框帧分别推送到 **MJPEGStreamServer**（`video_stream.py`）。流服务器为每个摄像头维护双缓冲（原始 + 标注），所以前端可以通过 `/cameras/{id}/stream?raw=1` 拿到不带框的原始流。这样渲染帧率就和检测器延迟解耦了。

`overlay_types` 是 `config/global.json` 中的全局开关，决定哪些检测类型会被画框——它独立于摄像头的 `detection_types`，必须**全局开关 + 该摄像头启用了对应类型**两个条件同时满足，才会在该摄像头上画框。这条交集逻辑写在 `_overlay_loop` 中。

### 持久化布局

- `data/records.json` —— 安全检测告警记录的元数据（不含图片）。
- `data/frames/{record_id}_snapshot.jpg`、`{record_id}_frame_{NNN}.jpg` —— 图片文件。
- `data/confined_records.json` —— 有限空间事件记录。
- `data/prompt.txt` —— 运行时可编辑的 VLM 提示词（旧版单摄像头流程使用）。格式为 `prompt\n\n---\n\nquestion`，文件为空或缺失时使用 `config.py` 中的内置默认值。
- 仓库中存在两个存储模块：`backend/storage.py`（旧版，单摄像头用）和 `backend/performance_storage.py`（当前在用，含 LRU 图片缓存、分页、清理循环）。新代码请使用 `performance_storage`。
- `backend/confined_space/storage.py` 是有限空间专用的存储，与安全检测的存储独立。

### 配置文件

`config/` 目录里全是**运行时可编辑的 JSON**，不是固化在版本里的配置：

- `cameras.json` —— 安全检测的摄像头列表，含每路摄像头的 `detection_types`。
- `camera_globals.json` —— 新增安全检测摄像头时的默认值（`backend/config.py` 中的 `apply_camera_globals` 会把它合并进去）。
- `global.json` —— 系统级参数（VLM 并发、冷却时长、存储上限、`overlay_types` 等）。
- `confined_cameras.json` —— 有限空间的摄像头与区域配置。
- `confined_global.json` —— 有限空间的全局设置。

`backend/config.py` 是这些文件加载/保存的唯一权威入口。它在加载摄像头配置时会把 `camera_globals` 深度合并进每条摄像头条目，并对旧格式做迁移（例如：缺失 `detection_types` 时根据旧版的 `detection_enabled` 字段补齐）。

### 重要：导入依赖 `PYTHONPATH=backend/`

`backend/` 下的模块使用裸名导入（`import config`、`from camera_manager import ...`）。启动脚本会设置 `PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/backend"`。子包（`safety_detection/`、`confined_space/`）内部则用 `import config as app_config`——这里的 `config` 指的是 **本项目的 `backend/config.py`**，并非 Python 标准库。如果你要在 `backend/` 下新增模块，请沿用裸名导入；如果直接执行脚本，记得先设置 `PYTHONPATH`，否则会报 `ImportError`。

## 环境与部署

`.env`（首次安装时由 `.env.default` 模板生成）由 `start*.sh` 加载，并被 `backend/config.py` 读取。关键变量：

- `ARK_API_KEY`、`VLM_ENDPOINT` —— 火山引擎 Ark 凭据。VLM 阶段必须，缺失时系统会退化到带告警的 "mock 模式"。
- `DETECTION_DEVICE` —— 取值 `npu` | `cpu` | `cuda`。`install.sh` 在缺少 `librknnrt.so` 或 `.rknn` 模型时会自动降级为 `cpu`；手动切换只需一条 `sed` 命令。
- `RKNN_MODEL`、`YOLO_MODEL` —— 模型路径。
- `CAMERAS=cam_01:0:cam_02:rtsp://...` —— 通过环境变量传入摄像头列表（替代 `cameras.json` 的另一种方式）。

`install.sh` 仅适用于 RK3588（会校验 `aarch64` 架构）；它把项目部署到 `/opt/sentry`，使用 `sentry` 用户运行，安装 `sentry.service`，并尝试在线把本地 ONNX 模型转换为 RKNN（只有装了 `rknn-toolkit2` 才会成功——通常只能在 x86 上跑）。开发环境直接在项目目录里跑 `start_all.sh` 即可。

## 修改代码时需要注意

- 安全检测前端（`frontend/safety_detection/multi.html`）和有限空间前端（`frontend/confined_space/monitor.html`）都是单文件大型 HTML，里面内联了 Vue 代码，没有模块打包工具。
- 新增一种检测类型需要联动改动多个文件：`inference_engine.py`（模型加载 + `_detect_<type>` 方法）、`safety_detection/detector_core.py`（`TypeSchedule` 调度逻辑）、`config.py` 中的 `DEFAULT_TYPE_CONFIG`、`understander.py` 中的 VLM 提示词模板，以及前端的类型列表。
- 修改摄像头需要在三处保持同步：`CameraManager` 内存状态、`MultiDetector` 的调度（`update_camera_config`）、以及 `cameras.json` 的持久化。可参考现有的 `/cameras/{camera_id}/config` 端点的写法。
- 超过 `max_records`（默认 100 条）的记录会被裁剪；`storage.delete_record_images()` 会一并删除 JPEG 文件。新写代码时不要无界地把所有记录加载进内存。
- `main_multi.py` 的启动钩子按特定顺序构造各组件（CameraManager → SafetyDetector → VLMQueue → MultiDetector → VLMInspector → ZoneCounter → 启动各线程）。组件构造函数会拿其它组件作为依赖，所以这个顺序不能随意打乱。
