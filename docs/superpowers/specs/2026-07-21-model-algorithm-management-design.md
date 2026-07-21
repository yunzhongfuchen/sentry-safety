# 模型管理 + 算法管理拆分设计

> 日期：2026-07-21
> 目标：将现有"类型管理"拆分为**模型管理**（只负责上传模型）与**算法管理**（基于模型创建不同参数版本），摄像头配置简化为"选算法 + 画区域 + 启停"。

## 1. 背景与决策

现有检测类型注册表（`config/detection_types.json`）把模型绑定、显示信息、运行参数全部揉在一个"类型"里，摄像头弹窗里还要平铺一堆参数，界面拥挤。

经确认的决策：

1. **算法即类型** —— 算法取代现有"检测类型"概念，告警、监控页筛选、记录页都显示算法名
2. **参数全归算法** —— 阈值/间隔/连续/冷却/VLM/人数条件全部绑在算法上；摄像头侧只选算法 + 画 ROI + 启停
3. **模型上传自动解析** —— 上传 `.pt` 时自动读出后处理类型与类别清单，创建算法时勾选而非手填类别 ID
4. **多算法即多版本** —— 一个模型下建多个算法（如"漏液-标准""漏液-高灵敏"），无额外版本号概念
5. **方案 A：双注册表 + 平滑迁移** —— 新建 `models.json` + `algorithms.json`，算法 key 沿用原类型 key，保留 `/detector/types` 兼容接口
6. **迁移从简** —— 系统未上线，摄像头均为测试数据，摄像头级参数覆盖直接丢弃，不做保留

## 2. 数据模型

### 2.1 `config/models.json` —— 模型注册表（新）

```json
{
  "fire_smoke": {
    "name": "火焰烟雾检测模型",
    "file": "fire_smoke.pt",
    "post_process": "yolo_box",
    "class_names": { "0": "fire", "1": "smoke" },
    "file_size": 6241478,
    "uploaded_at": "2026-07-21T15:00:00"
  }
}
```

| 字段 | 说明 |
|------|------|
| key | 自动生成：文件名去扩展名，冲突加后缀（`leak` → `leak_1`） |
| `name` | 显示名，默认取文件名 |
| `file` | 模型文件名，`.pt` 或 `.rknn` 统一用此字段（同一部署环境只会有一种格式） |
| `post_process` | `yolo_box` / `yolo_pose`；`.pt` 上传时自动解析，`.rknn` 手动选择 |
| `class_names` | `{ "0": "fire", ... }`；`.pt` 自动解析，`.rknn` 手动填，解析失败则为空 |
| `file_size` / `uploaded_at` | 文件大小（字节）/ 上传时间（ISO 8601） |

模型文件存放目录沿用现有 `weights/`。

### 2.2 `config/algorithms.json` —— 算法注册表（取代 detection_types.json）

```json
{
  "fire": {
    "label": "明火",
    "color": "#ef4444",
    "icon": "flame",
    "model_key": "fire_smoke",
    "classes": [0],
    "model_confidence": 0.5,
    "vlm_prompt_key": "fire_review",
    "inspection_label": "明火",
    "params": {
      "interval": 1,
      "threshold": 0.6,
      "consecutive_required": 3,
      "cooldown": 60,
      "use_vlm": false,
      "box_count_mode": "gte",
      "min_box_count": 1,
      "max_box_count": null
    }
  }
}
```

- 一个 `model_key` 可被多个算法引用（多算法 = 同一模型的不同参数版本）
- 原 `defaults` 改名 `params`，人数条件的 `box_count_mode`（`gte` / `lte` / `between` / `outside`）收编进 `params`
- 原类型注册表中的 `npu_model_path` 字段取消（模型侧只有单一 `file` 字段）
- key 规则沿用现有：新增时后端自动生成，编辑时不可改

### 2.3 `cameras.json` —— 摄像头配置（瘦身）

```json
{
  "camera_id": "cam-01",
  "algorithms": {
    "fire": {
      "enabled": true,
      "roi": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
      "roi_invert": false
    }
  }
}
```

- 原 `detection_types` 段改名 `algorithms`
- 每个算法只保留 `enabled` / `roi` / `roi_invert`，参数全部在算法上维护
- 修改算法参数后，所有引用该算法的摄像头一起生效

## 3. 迁移（启动时自动执行，一次性）

1. 存在 `config/detection_types.json` 且不存在新注册表文件时触发：
   - 按 `model_path` 去重生成 `models.json`（`class_names` 为空，下次重新上传模型时可自动补全）
   - 每个旧类型原样变成一个算法，**key 不变**，`defaults` → `params`
2. `cameras.json`：`detection_types` 段改名 `algorithms`，剔除参数字段只留 `enabled` / `roi` / `roi_invert`，参数覆盖直接丢弃
3. 旧 `detection_types.json` 重命名为 `detection_types.json.bak`，可人工回滚

## 4. 后端 API

### 模型管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/models` | 模型列表（含 class_names、文件大小等） |
| POST | `/models/upload` | 上传模型文件（`.pt` / `.rknn`），创建条目；`.pt` 自动解析元数据 |
| PUT | `/models/{key}` | 更新名称、post_process、class_names（主要用于 `.rknn` 手填） |
| DELETE | `/models/{key}` | 删除条目与模型文件；被算法引用时返回 409 |

### 算法管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/algorithms` | 算法列表（label/color/icon/model_key/classes/params） |
| POST | `/algorithms` | 新增算法 |
| PUT | `/algorithms/{key}` | 更新算法（key 不可改） |
| DELETE | `/algorithms/{key}` | 删除；被摄像头引用时返回 409 |

### 兼容接口（不改动）

`GET/PUT /detector/types`、`GET /detector/types/{dtype}` 内部转调算法注册表，输出格式与现状完全一致，监控页、记录页、`shared.js` 零改动。

### 推理侧适配

- `inference_engine` / `gpu_scheduler` 的模型加载与"同模型只推理一次"去重逻辑，从按 `model_path` 分组改为按 `model_key` 分组，行为不变
- 算法引用不存在的 `model_key`：保存时校验拒绝；运行中模型文件缺失时该算法标记"不可用"并跳过，不阻塞其他算法
- `camera_manager` 与 `PUT /cameras/{id}` 改为读写摄像头配置中的 `algorithms` 段（仅 enabled/roi/roi_invert），不再接受参数覆盖字段

## 5. 前端页面

### 5.1 模型管理（`/models.html`，新页面）

- 侧边导航"类型管理"替换为两个入口：**模型管理**、**算法管理**
- 布局：顶部标题 + "上传模型"按钮；主体为卡片网格
- 卡片内容：模型名称、`file`、post_process 标签、类别清单（`0:fire 1:smoke`）、文件大小、被引用算法数、操作（编辑 / 删除）
- 上传弹窗：选文件 + 名称（默认文件名）；`.pt` 提示"将自动解析类别"（解析中显示 loading），`.rknn` 展开手动填写 post_process 和类别

### 5.2 算法管理（`/algorithms.html`，由原 types.html 改造）

- 卡片网格展示算法：颜色条、label、引用模型名、classes、关键参数摘要（阈值/间隔/冷却）
- 新增/编辑弹窗：
  - 必填：label、color（调色板分配）、**模型（下拉，显示模型名）**、**类别过滤（按所选模型的 `class_names` 渲染为勾选项）**、params 参数组（阈值/间隔/连续/冷却/VLM/人数条件下拉）
  - 选填（折叠"高级设置"）：icon、model_confidence、vlm_prompt_key、inspection_label
  - 切换模型时按新模型的 class_names 重新渲染类别勾选项
- 删除：沿用引用检查，被摄像头引用返回 409 提示

### 5.3 摄像头配置弹窗（settings.html，简化）

- 检测类型区域改为算法列表，每行：`[颜色点] 算法名  [启用开关]  [展开▼]`
- 展开只保留 ROI 区（绘制按钮 / 顶点数 / 清除 / 区域内·外切换）
- 参数全部移除（间隔/阈值/连续/冷却/VLM/人数条件下移到算法管理页）

### 5.4 监控页 / 记录页 / shared.js

零改动 —— 通过 `/detector/types` 兼容接口拿到算法列表，显示的就是算法 label。

### 交互流程

```
上传模型（自动解析类别）→ 创建算法（选模型 → 勾选类别 → 设参数）→ 摄像头选算法 + 画 ROI → 完成
```

## 6. 边界与异常

| 场景 | 处理 |
|------|------|
| 上传模型重名 | 自动加后缀（`leak_1.pt`），不覆盖已有文件 |
| 上传非 `.pt` / `.rknn` | 前端限制 + 后端 400 |
| `.pt` 自动解析失败（损坏/非 YOLO） | 模型条目仍创建，`class_names` 为空，创建算法时手动填 |
| 删除被算法引用的模型 | 409，提示先删除引用它的算法 |
| 删除被摄像头引用的算法 | 409，提示先清理摄像头配置（沿用现有逻辑） |
| 算法 label 重复 | 400 提示 |
| 算法引用不存在的 model_key | 保存时校验拒绝 |
| 运行中模型文件缺失 | 该算法标记"不可用"并跳过，不阻塞其他算法 |
| 兼容接口异常 | `/detector/types` 内部转调，输出格式与现状一致 |

## 7. 测试要点

- 上传 `.pt` 后 `class_names` 自动解析正确
- 同一模型创建两个算法（不同 threshold），`GET /detector/types` 都返回
- 共享模型的两个算法挂同一摄像头，推理只跑一次（沿用现有去重逻辑）
- 旧 `detection_types.json` + `cameras.json` 启动迁移：算法 key 不变、ROI 保留、参数覆盖被丢弃、旧文件变 `.bak`
- 删除被引用模型 / 算法返回 409
- 监控页开关、告警 badge、记录页筛选显示算法 label（兼容接口无回归）
