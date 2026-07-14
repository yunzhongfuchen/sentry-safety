# 检测类型注册表框架设计

> **目标**：将视频诊断系统的检测类型（fire/smoke/mask/cigarette/uniform/sleep）从硬编码改造为配置驱动的注册表框架，新增检测类型只需编辑 JSON 配置 + 放模型文件，不需要改代码。

## 1. 注册表数据格式

### 1.1 注册表文件：`config/detection_types.json`

单文件，定义所有检测类型。首次启动时从内置默认值自动生成，运行时以文件为准。

```json
{
  "fire": {
    "label": "明火",
    "color": "#ef4444",
    "icon": "flame",

    "model_path": "fire_smoke.pt",
    "npu_model_path": "fire_smoke.rknn",
    "post_process": "yolo_box",
    "classes": [0],
    "model_confidence": 0.5,

    "vlm_prompt_key": "fire_review",
    "inspection_label": "明火",

    "defaults": {
      "enabled": false,
      "interval": 1,
      "threshold": 0.6,
      "consecutive_required": 3,
      "cooldown": 60,
      "use_vlm": false,
      "min_box_count": 1,
      "max_box_count": null
    }
  },

  "smoke": {
    "label": "烟雾",
    "color": "#f97316",
    "icon": "cloud",

    "model_path": "fire_smoke.pt",
    "npu_model_path": "fire_smoke.rknn",
    "post_process": "yolo_box",
    "classes": [1],
    "model_confidence": 0.5,

    "vlm_prompt_key": "smoke_review",
    "inspection_label": "烟雾",

    "defaults": {
      "enabled": false,
      "interval": 1,
      "threshold": 0.55,
      "consecutive_required": 3,
      "cooldown": 60,
      "use_vlm": false,
      "min_box_count": 1,
      "max_box_count": null
    }
  },

  "uniform": {
    "label": "工服",
    "color": "#22c55e",
    "icon": "shirt",

    "model_path": "uniform.pt",
    "npu_model_path": "uniform.rknn",
    "post_process": "yolo_box",
    "classes": [1],
    "model_confidence": 0.5,

    "vlm_prompt_key": "uniform_review",
    "inspection_label": "未穿工服",

    "defaults": {
      "enabled": false,
      "interval": 1,
      "threshold": 0.5,
      "consecutive_required": 3,
      "cooldown": 60,
      "use_vlm": false,
      "min_box_count": 1,
      "max_box_count": null
    }
  },

  "mask": {
    "label": "口罩",
    "color": "#0ea5e9",
    "icon": "shield",

    "model_path": "mask.pt",
    "npu_model_path": "mask.rknn",
    "post_process": "yolo_box",
    "classes": [1],
    "model_confidence": 0.5,

    "vlm_prompt_key": "mask_review",
    "inspection_label": "未戴口罩",

    "defaults": {
      "enabled": false,
      "interval": 1,
      "threshold": 0.5,
      "consecutive_required": 3,
      "cooldown": 60,
      "use_vlm": false,
      "min_box_count": 1,
      "max_box_count": null
    }
  },

  "cigarette": {
    "label": "吸烟",
    "color": "#a855f7",
    "icon": "cigarette",

    "model_path": "cigarette.pt",
    "npu_model_path": "cigarette.rknn",
    "post_process": "yolo_box",
    "classes": [0],
    "model_confidence": 0.5,

    "vlm_prompt_key": "cigarette_review",
    "inspection_label": "吸烟",

    "defaults": {
      "enabled": false,
      "interval": 1,
      "threshold": 0.5,
      "consecutive_required": 3,
      "cooldown": 60,
      "use_vlm": false,
      "min_box_count": 1,
      "max_box_count": null
    }
  },

  "sleep": {
    "label": "睡岗",
    "color": "#eab308",
    "icon": "moon",

    "model_path": "yolov8n-pose.pt",
    "npu_model_path": null,
    "post_process": "yolo_pose",
    "classes": null,
    "model_confidence": 0.1,

    "vlm_prompt_key": "sleep_review",
    "inspection_label": "睡岗/打盹",

    "defaults": {
      "enabled": false,
      "interval": 60,
      "threshold": 0.7,
      "consecutive_required": 3,
      "cooldown": 60,
      "use_vlm": false,
      "min_box_count": 1,
      "max_box_count": null
    }
  }
}
```

### 1.2 字段说明

| 字段 | 类型 | 用途 |
|------|------|------|
| `label` | string | 前端显示名称 |
| `color` | string | hex 颜色，前端和后端画框共用（后端自动转 BGR） |
| `icon` | string | 前端图标名（预留） |
| `model_path` | string | 模型文件名，相对 `models/` 目录 |
| `npu_model_path` | string\|null | NPU 模型文件名，null 表示不支持 NPU |
| `post_process` | string | 后处理策略：`yolo_box` / `yolo_pose` |
| `classes` | list[int]\|null | 模型输出中要保留的 class ID，null 表示不过滤 |
| `model_confidence` | float | 模型推理时的 conf 参数（注意区别于报警阈值 threshold） |
| `vlm_prompt_key` | string | VLM 复核 prompt 模板名，对应 `vlm_prompts.json` 中的 key |
| `inspection_label` | string | VLM 巡检 prompt 中的中文描述 |
| `defaults` | object | 运行参数默认值，可被摄像头级配置覆盖 |

**defaults 字段说明**：

| 字段 | 类型 | 用途 |
|------|------|------|
| `enabled` | bool | 是否启用 |
| `interval` | float | 检测间隔（秒） |
| `threshold` | float | 报警置信度阈值 |
| `consecutive_required` | int | 连续检测到 N 次才触发 |
| `cooldown` | float | 触发后冷却时间（秒），冷却期内不检测 |
| `use_vlm` | bool | 是否启用 VLM 复核 |
| `min_box_count` | int\|null | 检测框数量 ≥ N 才算检测到，null 不限制 |
| `max_box_count` | int\|null | 检测框数量 ≤ N 才算检测到，null 不限制 |

### 1.3 摄像头级覆盖

`cameras.json` 中每个摄像头的 `detection_types` 只覆盖需要调整的运行参数，缺失字段继承注册表 `defaults`：

```json
{
  "camera_id": "cam-01",
  "detection_types": {
    "fire": {
      "enabled": true,
      "threshold": 0.8,
      "roi": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
      "roi_invert": false
    },
    "person_count": {
      "enabled": true,
      "max_box_count": 0,
      "roi": [[0.2, 0.2], [0.7, 0.2], [0.7, 0.8], [0.2, 0.8]],
      "roi_invert": true
    }
  }
}
```

- `roi`：归一化坐标（0-1）的多边形顶点数组，只出现在摄像头级
- `roi_invert`：`false`（默认）= 只检测区域内；`true` = 只检测区域外

## 2. 后处理策略模式

### 2.1 统一返回格式

所有策略的输出统一为：

```python
{
    "detected": bool,
    "boxes": [[x1, y1, x2, y2], ...],
    "scores": [float, ...],
    # yolo_pose 额外返回：
    "subjects": [{"sleeping": bool, "keypoints": [...], "bbox": [...]}, ...]
}
```

### 2.2 两种策略

**`yolo_box`（标准框检测）**

适用于：fire、smoke、uniform、mask、cigarette，以及未来新增的大多数类型。

流程：模型推理 → 按 `classes` 过滤 → 输出 boxes + scores。

共享模型的去重由 `detect()` 方法自动处理：`model_path` 相同的类型只推理一次，各自按自己的 `classes` 从同一个推理结果中过滤。例如 fire（classes=[0]）和 smoke（classes=[1]）共享 `fire_smoke.pt`，推理一次，按各自 classes 拆分结果。

**`yolo_pose`（姿态分析）**

适用于：sleep。

流程：pose 模型推理 → 提取 17 关键点 → `sleep_detect.process_frame` 判断是否睡岗 → 输出 boxes + scores + subjects（含 keypoints、sleeping 状态）。

`_annotate_frame` 绘制时根据 `post_process == "yolo_pose"` 走骨架绘制分支。

### 2.3 推理引擎分发逻辑

`inference_engine.py` 的 `detect()` 方法改为注册表驱动：

```python
def detect(self, frame, detection_types, core_id=0):
    results = {}
    processed_models = set()

    for dtype in detection_types:
        type_def = registry.get(dtype)
        model_key = type_def["model_path"]

        if model_key in processed_models:
            continue

        raw_output = self._run_model(model_key, frame, type_def, core_id)

        for related_dtype in registry.get_types_by_model(model_key):
            if related_dtype in detection_types:
                related_def = registry.get(related_dtype)
                processor = get_post_processor(related_def["post_process"])
                results[related_dtype] = processor(raw_output, related_def)

        processed_models.add(model_key)

    return results
```

### 2.4 扩展新策略

新增后处理策略需要写代码：
1. 在 `detection_registry.py` 中注册新策略名
2. 编写对应的后处理函数

但使用已有两种策略的类型不需要改代码。

## 3. 检测管道与报警条件

### 3.1 完整管道

```
1. 冷却检查 → 冷却期内直接跳过，不推理（节省资源）
   ↓
2. 模型推理 → 后处理策略
   ↓
3. ROI 过滤
   ↓
4. threshold 过滤：去掉 score < threshold 的框
   ↓
5. box_count 判断：
   - min_box_count 不为 null：框数量 >= N 才算检测到
   - max_box_count 不为 null：框数量 <= N 才算检测到
   ↓
6. consecutive_required：连续 N 次检测到才触发
   ↓
7. 触发告警 / VLM 复核，进入冷却
```

### 3.2 冷却检查前置

现有逻辑是推理完成后才判断冷却。改为在 `_get_due_types` 阶段就排除冷却中的类型，连模型推理都省掉。GPU scheduler 的 `_collect_due_frames` 同理。

### 3.3 ROI 过滤

在后处理策略之后、报警条件判断之前执行：

```python
def filter_by_roi(result, roi, roi_invert, frame_width, frame_height):
    if not roi:
        return result

    polygon = np.array([
        [int(x * frame_width), int(y * frame_height)]
        for x, y in roi
    ], dtype=np.int32)

    filtered_boxes, filtered_scores = [], []
    for box, score in zip(result["boxes"], result["scores"]):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        inside = cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0
        keep = inside if not roi_invert else not inside
        if keep:
            filtered_boxes.append(box)
            filtered_scores.append(score)

    return {
        **result,
        "boxes": filtered_boxes,
        "scores": filtered_scores,
        "detected": len(filtered_boxes) > 0,
    }
```

对于 `yolo_pose` 策略，`subjects` 列表按 bbox 中心点同步过滤。

### 3.4 box_count 场景

| 场景 | min_box_count | max_box_count | 含义 |
|------|---------------|---------------|------|
| 明火检测 | 1 | null | 有火就报警（默认） |
| 人数超限 | 5 | null | ≥ 5 人才报警 |
| 离岗检测 | null | 0 | 0 人就报警 |
| 人数区间 | 1 | 3 | 0 人或 ≥ 4 人报警 |

`max_box_count` 场景（如离岗检测）：模型没检测到目标也算一种结果，需要进入判断流程，不能像现有逻辑一样"没有框就跳过"。

## 4. 后端架构改造

### 4.1 新增文件

**`backend/detection_registry.py`** — 注册表核心类

```python
class DetectionTypeRegistry:
    def load(self)                                          # 加载/重载注册表
    def get(self, dtype) -> dict                            # 获取类型定义
    def all_types(self) -> list[str]                        # 所有类型 key
    def get_types_by_model(self, model_path) -> list[str]   # 共享模型的类型
    def get_color_bgr(self, dtype) -> tuple                 # hex → BGR
    def get_defaults(self, dtype) -> dict                   # 默认运行参数
    def merge_camera_config(self, dtype, overrides) -> dict # 合并摄像头级覆盖
    def validate(self) -> list[str]                         # 校验（模型文件是否存在等）
```

全局单例，启动时初始化，各模块通过 `from detection_registry import registry` 引用。

### 4.2 改造文件

**`backend/inference_engine.py`**

- 删除 6 个 `_load_xxx_model` 和 6 个 `_detect_xxx` 硬编码方法
- `ensure_models_loaded` 改为遍历注册表，按 `model_path` 去重加载
- `detect` 改为注册表驱动 + 策略分发
- 后处理函数独立为 `_process_yolo_box`、`_process_yolo_pose`
- 预计代码量从 ~900 行降到 ~400 行

**`backend/main_multi.py`**

- GPU scheduler 的 `model_configs`（原 436-452 行硬编码）改为从注册表生成
- `_resolve_model_path` 改为从注册表读文件名

**`backend/safety_detection/detector_core.py`**

- `_annotate_frame`：颜色和标签从注册表读取，骨架绘制通过 `post_process == "yolo_pose"` 判断
- `_get_due_types`：冷却检查前置，冷却中的类型不进入推理
- `_handle_standard_detection`：新增 ROI 过滤和 box_count 判断

**`backend/config.py`**

- `DEFAULT_TYPE_CONFIG` 从注册表 `defaults` 动态生成
- `DEFAULT_GLOBAL_SETTINGS.display_detection_types` 从注册表动态生成

**`backend/understander.py`**

- `_build_inspection_prompt` 的 `type_desc` 从注册表 `inspection_label` 读取

### 4.3 不改的文件

- `backend/gpu_scheduler.py` — 核心逻辑不变，只是入参从硬编码变成注册表驱动
- `backend/vlm_queue.py` — 只负责队列调度，不关心类型定义

## 5. 前端改造与 API 设计

### 5.1 新增 API

| 方法 | 路径 | 用途 | 第一期 |
|------|------|------|--------|
| `GET` | `/detector/types` | 获取所有类型定义 | 实现 |
| `GET` | `/detector/types/{dtype}` | 获取单个类型 | 实现 |
| `PUT` | `/detector/types/{dtype}` | 更新类型运行参数 | 实现 |
| `POST` | `/detector/types` | 新增类型 | 预留 |
| `DELETE` | `/detector/types/{dtype}` | 删除类型 | 预留 |
| `POST` | `/detector/types/{dtype}/model` | 上传模型文件 | 预留 |

第一期 `PUT` 只允许修改 `defaults` 里的运行参数，不允许修改 `model_path`、`post_process` 等结构性字段。

`GET /detector/types` 返回格式：

```json
{
  "types": [
    {
      "key": "fire",
      "label": "明火",
      "color": "#ef4444",
      "icon": "flame",
      "post_process": "yolo_box",
      "defaults": { ... }
    }
  ]
}
```

### 5.2 前端改造

**`shared.js`**

`DETECTION_TYPES` 从硬编码改为调用 `GET /detector/types` 动态加载。加 fallback：API 调不通时用内置的 6 个默认类型，保证页面不白屏。

`getTypeLabel()`、`getTypeColor()`、`defaultDetectionTypes()` 都改为从动态加载的 `DETECTION_TYPES` 读取。

**`settings.html`**

检测类型列表从 `DETECTION_TYPES` 动态渲染，每个类型的可配置项（enabled、threshold、interval、consecutive_required、cooldown、use_vlm、min_box_count、max_box_count）统一用循环生成表单。

**`monitor.html`**

检测类型筛选按钮从 `DETECTION_TYPES` 动态渲染。

**`records.html`**

类型筛选下拉框从 `DETECTION_TYPES` 动态生成。

### 5.3 ROI（第一期）

第一期 ROI 只通过 API 配置（随摄像头配置 `PUT /cameras/{id}` 保存），不做前端绘制交互。第二期再加前端画多边形的交互。

## 6. 迁移兼容性

### 6.1 首次启动自动生成

`DetectionTypeRegistry.load()` 检查 `config/detection_types.json`：
- 不存在 → 从内置 `DEFAULT_DETECTION_TYPE_REGISTRY` 生成文件
- 已存在 → 读取文件，缺失字段补全默认值，自动写回

与现有 `load_vlm_prompts()` 模式一致。

### 6.2 现有配置兼容

- `cameras.json`：格式不变，新增的 `roi`、`roi_invert`、`min_box_count`、`max_box_count` 都是可选字段，不存在就用默认值，老配置不需要修改
- `global.json`：`display_detection_types` 格式不变，启动时自动补全注册表中的新类型
- `vlm_prompts.json`：不动，注册表里的 `vlm_prompt_key` 只是引用名

### 6.3 API 兼容

现有 API 不变（`GET /detector/models`、`GET /detector/status`、`POST /cameras/{id}/test-alert`），新增的 `/detector/types` 是纯新增。

### 6.4 前端兼容

`shared.js` 改为动态加载后内置 fallback，API 调不通时用默认 6 个类型，页面不白屏。

## 7. 新增检测类型操作流程

以新增"安全帽检测"为例：

**步骤 1**：放模型文件到 `models/helmet.pt`

**步骤 2**：编辑 `config/detection_types.json`，新增：

```json
{
  "helmet": {
    "label": "安全帽",
    "color": "#f59e0b",
    "icon": "hard-hat",
    "model_path": "helmet.pt",
    "npu_model_path": null,
    "post_process": "yolo_box",
    "classes": [1],
    "model_confidence": 0.5,
    "vlm_prompt_key": "helmet_review",
    "inspection_label": "未戴安全帽",
    "defaults": {
      "enabled": false,
      "interval": 1,
      "threshold": 0.5,
      "consecutive_required": 3,
      "cooldown": 60,
      "use_vlm": false,
      "min_box_count": 1,
      "max_box_count": null
    }
  }
}
```

**步骤 3（可选）**：在 `config/vlm_prompts.json` 新增 `helmet_review` prompt 模板

**步骤 4**：重启服务

重启后前端自动出现"安全帽"类型的开关、参数配置、筛选按钮。全程不改一行 Python 或 JavaScript 代码。

**需要改代码的唯一情况**：新增一种全新的后处理策略（如语义分割），需要编写对应的后处理函数并在 `detection_registry.py` 中注册。
