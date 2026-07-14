# 检测类型注册表前端管理设计

> 日期：2026-07-14
> 目标：为检测类型注册表提供完整的前端管理能力，并补齐摄像头级别的 ROI 与人数条件配置。

## 1. 背景

当前系统已实现检测类型注册表（`backend/detection_registry.py`），支持通过 JSON 配置驱动检测类型。但前端存在以下缺口：

- 无独立页面管理注册表类型（新增、删除、上传模型）
- 摄像头配置弹窗缺少 ROI 绘制和人数条件配置
- 检测类型的高级参数（间隔、阈值等）在摄像头弹窗中平铺，界面拥挤

## 2. 目标

1. 新增 `/types.html` 独立页面，提供检测类型的增删改查与模型上传
2. 优化 `settings.html` 摄像头配置弹窗，检测类型区域支持手风琴展开
3. 摄像头级别支持 ROI 多边形绘制与人数条件配置
4. 后端扩展 API 支持类型管理与"区间外"人数条件

## 3. 页面与交互设计

### 3.1 检测类型管理页面（types.html）

**入口：** 侧边导航栏新增"类型管理"入口。

**布局：**
- 顶部：页面标题 + "新增类型"按钮
- 主体：卡片网格展示所有注册类型

**卡片内容：**
- 顶部颜色条（使用类型 color）
- `label`（显示名称）
- `model_path` / `npu_model_path`
- `post_process` 标签
- 操作按钮：编辑、删除、上传模型

**新增类型弹窗：**

必填项（默认显示）：
- `label`：显示名称，如"明火"
- `color`：颜色，带颜色选择器，默认从调色板分配
- `model_path`：CPU 模型路径，可手动输入，也可通过"上传模型"自动填入
- `post_process`：后处理策略，下拉选择 `yolo_box` / `yolo_pose`，默认 `yolo_box`

选填项（折叠"高级设置"）：
- `npu_model_path`：NPU 模型路径
- `classes`：类别过滤数组，如 `[0]`
- `model_confidence`：模型置信度阈值
- `icon`：图标名
- `vlm_prompt_key`：VLM 复核提示词键
- `inspection_label`：巡检显示名
- `defaults`：默认运行参数（enabled、interval、threshold、consecutive_required、cooldown、use_vlm、min_box_count、max_box_count）

**key 处理：**
- 新增时后端自动生成唯一 key（基于 label 拼音或 UUID 短码），不暴露给用户
- 编辑时 key 不可修改

**编辑类型弹窗：**
- 与新增弹窗相同，但 key 字段禁用

**删除类型：**
- 点击删除后检查是否有摄像头配置引用该类型
- 有引用则禁止删除，Toast 提示"请先清理摄像头配置中的该类型"
- 无引用则二次确认后删除

**上传模型：**
- 支持 `.pt`（CPU）和 `.rknn`（NPU）文件
- 根据扩展名自动填入 `model_path` 或 `npu_model_path`
- 文件保存到 `weights/` 目录
- 上传成功后刷新卡片信息

### 3.2 摄像头配置弹窗（settings.html）

**保持不变的部分：**
- 摄像头名称、视频源、启用开关、分辨率等基础配置

**检测类型配置区域（重新设计）：**

每行一个类型，默认只显示：
```
[颜色点] 类型名称    [启用开关]    [展开 ▼]
```

**手风琴模式：** 一次只展开一个类型的高级配置，展开新的自动收起旧的。

**展开后显示：**

第一行（运行参数）：
- 间隔（秒）
- 阈值（0~1）
- 连续（次）
- 冷却（秒）
- VLM（开关）

第二行（人数条件）：
- 模式下拉框：
  - `目标数 ≥ a`
  - `目标数 ≤ b`
  - `a ≤ 目标数 ≤ b`
  - `目标数 < a 或 > b`
- 根据模式动态显示 1~2 个数字输入框

第三行（检测区域）：
- "绘制 ROI" 按钮
- 已绘制时显示顶点数量和"清除"按钮

**ROI 绘制窗口：**
- 模态弹窗，显示摄像头快照（静态画面）
- 鼠标左键点击添加多边形顶点
- 双击完成绘制并闭合多边形
- 支持"区域内报警" / "区域外报警"切换（`roi_invert`）
- 坐标以归一化形式保存（0~1），适配不同分辨率

## 4. 后端 API 扩展

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/detector/types` | 获取类型列表（已有） |
| GET | `/detector/types/{dtype}` | 获取单个类型（已有） |
| PUT | `/detector/types/{dtype}` | 更新默认运行参数（已有） |
| POST | `/detector/types` | 新增检测类型 |
| DELETE | `/detector/types/{dtype}` | 删除检测类型（检查引用） |
| POST | `/detector/types/{dtype}/model` | 上传模型文件 |

**人数条件扩展：**

`TypeSchedule` 新增 `box_count_mode` 字段，取值：
- `gte`：目标数 ≥ a，映射 `min_box_count = a, max_box_count = null`
- `lte`：目标数 ≤ b，映射 `min_box_count = null, max_box_count = b`
- `between`：a ≤ x ≤ b，映射 `min_box_count = a, max_box_count = b`
- `outside`：x < a 或 x > b，需要后端 `check_box_count` 支持双区间判断

**check_box_count 扩展：**

`outside` 模式仍使用 `min_box_count` 和 `max_box_count` 存储区间两端：
- `min_box_count = a`（下界）
- `max_box_count = b`（上界）

```python
def check_box_count(result, min_box_count=None, max_box_count=None, box_count_mode=None):
    box_count = len(result.get("boxes", []))
    if box_count_mode == "outside":
        # 目标数 < a 或 > b 时报警
        if min_box_count is not None and box_count < min_box_count:
            return {**result, "detected": True}
        if max_box_count is not None and box_count > max_box_count:
            return {**result, "detected": True}
        return {**result, "detected": False}
    # 原有 gte / lte / between 逻辑...
```

## 5. 数据流

1. 前端页面加载时调用 `GET /detector/types` 获取类型列表
2. 类型管理页面通过 REST API 完成增删改查
3. 摄像头配置弹窗从 `cameraDialog.detection_types` 读取配置，保存时写回
4. ROI 绘制窗口通过 `GET /cameras/{id}/snapshot` 获取快照
5. 人数条件模式与阈值保存在摄像头配置的 `detection_types[dtype]` 中

## 6. 边界与异常

- 新增类型时 label 重复：后端返回 400，前端提示"类型名称已存在"
- 上传模型格式不支持：前端限制 `.pt` / `.rknn`，后端二次校验
- 删除被引用的类型：返回 409，前端 Toast 提示
- ROI 未闭合：双击自动闭合，少于 3 个顶点时提示"至少需要 3 个顶点"
- 快照获取失败：ROI 窗口显示"无法获取摄像头画面"，禁用绘制

## 7. 测试要点

- 新增类型后 `GET /detector/types` 能返回新类型
- 删除被引用类型返回 409
- 上传 `.pt` 后 `model_path` 自动填入
- 摄像头弹窗手风琴展开/收起正常
- ROI 绘制保存后坐标正确（归一化）
- 人数条件四种模式映射到正确的 `min_box_count` / `max_box_count`
