# 有限空间独立服务设计方案

## 1. 目标

将有限空间监控从安全检测中彻底拆出，成为可独立启动的完整服务。

- **安全检测服务**（`main_multi.py`，端口 8000）：现有代码完全不动
- **有限空间服务**（`main_confined.py`，端口 8001）：全部新建
- **约束**：同时只启动一个服务，节省资源

## 2. 目录结构

```
backend/
├── main_multi.py              # 安全检测入口（不动）
├── main_confined.py           # 有限空间入口（新建）
├── camera_manager.py          # 平台层：摄像头管理（共享逻辑，独立实例）
├── inference_engine.py        # 平台层：模型推理（共享逻辑，独立实例）
├── vlm_queue.py               # 平台层：VLM队列（共享逻辑，独立实例）
├── understander.py            # 平台层：VLM提示词（扩展模板）
├── video_stream.py            # 平台层：视频流服务（共享逻辑，独立实例）
├── performance_storage.py     # 平台层：记录存储（共享逻辑，独立数据文件）
├── config.py                  # 平台层：配置管理（共享逻辑，独立配置文件）
├── safety_detection/          # 安全检测业务（不动）
│   ├── api.py
│   ├── detector_core.py
│   └── sleep_detect.py
└── confined_space/            # 有限空间业务（扩展）
    ├── __init__.py
    ├── zone_counter.py        # 核心计数状态机（direct + entrance 双模式）
    ├── api.py                 # 基础区域管理API
    ├── monitor_api.py         # 监控/视频流API
    ├── records_api.py         # 记录查询API
    └── storage.py             # 独立记录存储封装

frontend/
├── safety_detection/          # 现有页面（不动）
│   ├── multi.html
│   ├── records.html
│   ├── settings.html
│   ├── style.css
│   └── ...
└── confined_space/            # 全部新建
    ├── monitor.html           # 一主多副监控页
    ├── records.html           # 事件记录页
    ├── settings.html          # 区域管理 + 全局设置
    └── style.css              # 复用dark主题风格

config/
├── cameras.json               # 安全检测摄像头配置
├── global.json                # 安全检测全局配置
├── confined_cameras.json      # 有限空间摄像头配置（新建）
└── confined_global.json       # 有限空间全局配置（新建）

data/
├── records.json               # 安全检测记录
└── confined_records.json      # 有限空间记录（新建）
```

## 3. 后端设计

### 3.1 入口文件 main_confined.py

```python
app = FastAPI(title="Sentry Confined Space Monitoring API")

# 静态文件：/static -> frontend/confined_space/
app.mount("/static", StaticFiles(directory="frontend/confined_space"), name="static")

# 挂载有限空间业务路由
app.include_router(confined_api.router, prefix="/api")
```

**初始化流程（init_components()）：**
1. 加载 `config/confined_global.json`
2. 加载 `config/confined_cameras.json`
3. 初始化 `CameraManager`（独立实例，只加载有限空间的摄像头）
4. 初始化 `InferenceEngine`（加载 person 模型）
5. 初始化 `VLMQueue` + `Understander`
6. 初始化 `ZoneCounter`
7. 注册所有区域（从配置加载或后续API动态创建）
8. 启动摄像头、启动处理线程

### 3.2 模块复用策略

| 平台层模块 | 复用方式 | 说明 |
|---|---|---|
| `camera_manager.py` | 直接 import | 独立实例，独立配置 |
| `inference_engine.py` | 直接 import | 加载 person 模型 |
| `vlm_queue.py` | 直接 import | 独立队列实例 |
| `understander.py` | 扩展 prompt | 新增 `confined_count_review` 和 `confined_window_review` |
| `video_stream.py` | 直接 import | 独立流服务实例 |
| `config.py` | 扩展函数 | 新增 `load_confined_settings()` / `save_confined_settings()` 等 |
| `performance_storage.py` | 扩展常量 | 新增 `CONFINED_RECORDS_FILE` |

### 3.3 API 路由设计

```
页面路由：
GET  /           -> monitor.html（监控主页）
GET  /monitor    -> monitor.html
GET  /records    -> records.html
GET  /settings   -> settings.html

业务 API：
GET  /api/zones                    列出所有区域
POST /api/zones                    创建/更新区域
GET  /api/zones/{id}               获取区域详情
DELETE /api/zones/{id}             删除区域
GET  /api/zones/{id}/status        区域实时状态（当前人数等）
GET  /api/zones/{id}/events        区域事件历史
POST /api/zones/{id}/calibrate     手动校准人数

GET  /api/cameras                  摄像头列表（有限空间自己的）
POST /api/cameras/{id}/config      摄像头配置
GET  /api/cameras/{id}/stream      视频流（复用 video_stream）

GET  /api/records                  全局记录查询（分页）
GET  /api/records/stats            记录统计

GET  /api/settings                 全局设置
POST /api/settings                 保存全局设置

POST /api/system/restart           重启服务
```

## 4. 前端设计

`frontend/confined_space/` 下新建 4 个文件，风格复用现有 dark 主题（颜色、字体、圆角、阴影等），页面结构独立。

### 4.1 monitor.html — 一主多副监控页

**布局（参考 multi.html）：**
- 顶部 Header：标题"有限空间监控" + 导航栏（监控 | 记录 | 设置）
- 上方状态栏：区域总数、当前告警数、在线摄像头数
- 主体区域：
  - 左侧/上方：主画面（大）— 选中区域的摄像头实时画面 + 当前人数大数字覆盖显示
  - 右侧/下方：副画面网格（小）— 所有区域的缩略图，点击切换主画面
- 每个画面叠加：区域名称、当前人数/上限、ROI 框（可选显示）
- 底部或侧边：实时事件滚动条（最近进出记录）

**关键差异（vs 安全检测）：**
- 画面标题显示"区域名 + 当前人数"，不是"摄像头名 + 检测类型"
- 主画面叠加一个大圆圈显示当前人数
- 超员时画面边框变红 + 闪烁提示

### 4.2 records.html — 事件记录页

**布局（参考 records.html）：**
- 顶部 Header + 导航栏
- 上方统计栏：今日进入人次、今日离开人次、当前告警数、超员次数
- 筛选栏：按区域筛选、按事件类型筛选（进入/离开/其他）、按时间范围筛选
- 记录表格：时间 | 区域 | 摄像头 | 事件类型 | 人数变化 | 描述 | 操作（查看截图）
- 分页控件
- 点击记录：弹出详情面板，显示当时的关键帧截图 + VLM 复核结果

**关键差异：**
- 记录类型是"进入/离开/其他"，不是"fire/smoke/mask"
- 详情面板显示人数变化（旧→新）和 VLM 返回的标签

### 4.3 settings.html — 设置页

**布局（参考 settings.html 的 tab 形式）：**
- 顶部 Header + 导航栏
- Tab 1：区域管理
  - 区域列表表格：区域ID、名称、摄像头、模式、人数上限、当前人数、操作
  - 添加区域按钮
  - 编辑弹窗：区域ID、名称、选择摄像头、监控模式、人数上限、ROI设置、连续确认帧数
- Tab 2：全局参数
  - 检测间隔、VLM复核开关、VLM并发数、事件保存天数、窗口统计间隔（默认10秒）
- Tab 3：系统信息
  - 模型状态、服务版本、重启按钮

### 4.4 style.css

**风格统一为明亮主题**，复用 `frontend/safety_detection/multi.html` 的 CSS 变量：

```css
:root {
    --bg: #f1f5f9;
    --surface: #ffffff;
    --surface-hover: #f8fafc;
    --border: #e2e8f0;
    --text: #0f172a;
    --text-secondary: #64748b;
    --text-muted: #94a3b8;
    --accent: #0ea5e9;
    --accent-light: #e0f2fe;
    --danger: #ef4444;
    --danger-light: #fee2e2;
    --warning: #f97316;
    --warning-light: #ffedd5;
    --success: #22c55e;
    --success-light: #dcfce7;
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --radius: 12px;
    --radius-sm: 8px;
}
```

在此基础上添加有限空间特有的组件样式（人数大圆圈、事件类型标签等）。

## 5. 数据模型与双模式状态机

### 5.1 两种监控模式

每个区域配置一个 `monitor_mode`：

| 模式 | 名称 | 适用场景 | 核心逻辑 |
|---|---|---|---|
| **direct** | 直接计数 | 能看到有限空间内部 | YOLO直接数ROI人数，VLM复核准确性 |
| **entrance** | 入口判断 | 只能看到入口（主要场景）| YOLO检测入口有人，周期性窗口提交VLM统计进入/离开人数 |

### 5.2 ZoneConfig（配置）

```python
zone_id: str              # 唯一标识
name: str                 # 显示名称
camera_id: str            # 关联摄像头
roi: List[int]            # [x1, y1, x2, y2]
max_personnel: int        # 人数上限
monitor_mode: str         # "direct" | "entrance"
consecutive_required: int # 连续帧确认（防抖）
enable_vlm_review: bool   # 是否启用VLM复核
window_interval: int      # entrance模式下VLM窗口统计间隔（秒，默认10）
```

### 5.3 ZoneState（运行时状态）

```python
config: ZoneConfig
current_count: int
observing_count: Optional[int]      # direct模式：正在观察的人数
consecutive_frames: int             # direct模式：连续计数帧数

# entrance模式专用
frame_buffer: List[Tuple[float, np.ndarray]]  # 帧缓存（时间戳, 帧）
last_vlm_submit_time: float         # 上次提交VLM的时间
is_active: bool                     # 入口是否有人

event_history: List[Event]
vlm_review_pending: bool
last_review_result: Optional[dict]
```

### 5.4 direct 模式状态机

```
每帧：
  1. YOLO 检测 ROI 内所有人
  2. 得到 detected_count
  3. 连续帧防抖（如连续5帧都是3人，才确认人数=3）
  4. 人数确认变化后：
     a. 更新 current_count
     b. 记录事件（old_count → new_count）
     c. 提交VLM复核："画面中有多少人在空间内？"
     d. VLM返回 {"count": N, "confidence": ...}
     e. 如VLM结果与小模型差异大，标记"待复核"
```

### 5.5 entrance 模式状态机（主要场景）

```
状态：IDLE（空闲）→ ACTIVE（活跃）→ IDLE

IDLE:
  小模型检测到入口ROI有人（连续2帧有人）
  → 进入 ACTIVE
  → 启动帧缓存（每秒存1帧）

ACTIVE:
  持续缓存帧（最多保留最近 window_interval 秒的帧）

  事件A：window_interval 定时器触发
    → 提交VLM（带缓存的所有帧）
    → VLM返回窗口统计结果
    → 更新 current_count
    → 记录事件
    → 清空已统计的帧（保留最新1帧作为上下文）
    → 重置定时器

  事件B：连续3帧无人（入口空了）
    → 提交VLM（带最后一段时间的帧）
    → VLM返回窗口统计结果
    → 更新 current_count
    → 记录事件
    → 清空缓存
    → 回到 IDLE
```

### 5.6 VLM Prompt 模板

**direct 模式 — 人数复核：**

```
你正在复核一个有限空间监控画面中的人数统计。
请仔细数一下画面中有多少人位于这个有限空间内部。
注意排除以下误判：
- 只露出部分身体但在空间外部的人
- 在入口处徘徊但未真正进入的人
- 画面中的倒影、海报等

请以 JSON 格式返回：
{"count": 整数, "confidence": 0.0-1.0, "reason": "判断理由"}
```

**entrance 模式 — 窗口统计（核心）：**

```
你正在分析一段有限空间入口的监控片段，共 {N} 张连续截图，时间跨度约 {N} 秒。

请判断这段时间内是否发生了以下情况：
1. 有人从外部进入了有限空间（entered）
2. 有人从有限空间离开了（left）
3. 有其他情况，如人员在入口附近徘徊但未进出（other）

注意：
- 已经站在空间内部的人不要重复统计
- 只是路过、在门口徘徊但没有跨过门槛的人算 other
- 如果同一个人进出多次，按实际次数统计
- 多种情况可以同时发生

请以 JSON 格式返回：
{
  "entered": true/false,
  "left": true/false,
  "other": true/false,
  "entered_count": 整数,
  "left_count": 整数,
  "confidence": 0.0-1.0,
  "reason": "判断理由"
}
```

### 5.7 Event（事件记录）

```python
event_id: str
timestamp: str              # ISO格式
zone_id: str
zone_name: str
camera_id: str
event_type: str             # "enter" | "leave" | "other"
monitor_mode: str           # "direct" | "entrance"
old_count: int
new_count: int
diff: int
vlm_result: dict            # VLM完整返回
  - entered: bool
  - left: bool
  - other: bool
  - entered_count: int
  - left_count: int
  - confidence: float
  - reason: str
description: str
alert: bool                 # 是否超员
```

**事件生成规则：**

每次提交VLM后，根据返回的标志位生成独立事件：
- `entered=true` → 生成 enter 事件
- `left=true` → 生成 leave 事件
- `other=true` → 生成 other 事件
- 三个标志位独立，可以同时为 true
- 每个事件都记录 old_count → new_count 的变化

### 5.8 校准机制

- **手动校准 API**：`POST /api/zones/{id}/calibrate {current_count: N}`
- **定时巡检（可选）**：每 N 分钟自动提交 VLM 任务，问"当前画面中有多少人在空间内部"
- 服务重启后默认从 0 开始，需人工校准或等待实际事件更新

## 6. 配置与存储

### 6.1 独立配置

`config.py` 扩展：

```python
CONFINED_GLOBAL_FILE = PROJECT_ROOT / "config" / "confined_global.json"
CONFINED_CAMERAS_FILE = PROJECT_ROOT / "config" / "confined_cameras.json"

def load_confined_settings() -> dict: ...
def save_confined_settings(settings: dict) -> None: ...
def load_confined_cameras() -> List[dict]: ...
def save_confined_cameras(cameras: List[dict]) -> None: ...
```

### 6.2 独立存储

`performance_storage.py` 扩展：

```python
CONFINED_RECORDS_FILE = PROJECT_ROOT / "data" / "confined_records.json"

def save_confined_records(records: List[dict]) -> None: ...
def load_confined_records() -> List[dict]: ...
def get_confined_records_paginated(page: int, size: int, filters: dict) -> dict: ...
```

## 7. 现有代码保护原则

- `main_multi.py`：**不修改任何业务逻辑**
- `safety_detection/`：**不动**
- `frontend/safety_detection/`：**不动**
- 所有改动只发生在：
  - 新建文件：`main_confined.py`
  - 扩展文件：`config.py`、`performance_storage.py`、`understander.py`
  - 新建目录：`confined_space/`（已有文件扩展）、`frontend/confined_space/`（全部新建）

## 8. 启动方式

```bash
# 安全检测服务
python3 -m uvicorn main_multi:app --host 0.0.0.0 --port 8000

# 有限空间服务
python3 -m uvicorn main_confined:app --host 0.0.0.0 --port 8001
```

**约束**：同时只启动一个服务。
