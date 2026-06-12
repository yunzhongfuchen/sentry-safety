# 告警记录统一级别与 VLM 复核流程设计

日期：2026-06-12  
主题：取消 P0/P1 区分，统一告警级别显示，引入人工确认状态

## 1. 背景与目标

当前系统把检测类型按 `P0` / `P1` 分级，并在后端根据级别走不同的告警/VLM 复核逻辑。实际业务希望：

- 所有检测类型统一对待，不再区分 P0/P1。
- 每类检测可独立配置是否启用 VLM 复核；不启用时仅走小模型检测。
- 记录列表里显示「小模型报警 / 大模型报警 / 大模型忽略」。
- 记录状态精简为「待确认 / 已确认 / 误报」，全部需要人工在详情页确认。
- VLM 复核结果只影响显示级别，不自动改变状态。

## 2. 数据模型

### 2.1 `level` 字段（告警来源/级别）

旧值：`P0` / `P1`  
新值：

| 值 | 含义 | 出现时机 |
|---|---|---|
| `small_model_alarm` | 小模型报警 | 小模型触发告警时 |
| `vlm_alarm` | 大模型报警 | VLM 复核后确认告警 |
| `vlm_ignore` | 大模型忽略 | VLM 复核后判定为误报 |

### 2.2 `status` 字段（人工确认状态）

旧值：`alerted` / `pending` / `confirmed` / `rejected` / `false_positive`  
新值：

| 值 | 含义 |
|---|---|
| `pending` | 待确认（默认值） |
| `confirmed` | 已确认（人工点击「确认报警」） |
| `false_positive` | 误报（人工点击「确认误报」） |

### 2.3 完整记录示例

```json
{
  "id": "cam01_fire_1718172800000",
  "camera_id": "cam01",
  "detection_type": "fire",
  "level": "vlm_alarm",
  "status": "pending",
  "time": "2025-06-12 14:32:18",
  "confidence": 0.87,
  "reason": "画面中可见明显火焰，复核通过",
  "small_model": {
    "detected": true,
    "confidence": 0.87,
    "boxes": [[100, 200, 300, 400]]
  },
  "vlm_review": {
    "confirmed": true,
    "confidence": 0.92,
    "reason": "画面中可见明显火焰，复核通过"
  },
  "source": "small_model",
  "frame_count": 12
}
```

## 3. 状态流转

```
小模型触发
    │
    ▼
level = small_model_alarm
status = pending
    │
    ├─ 该类型未启用 VLM ─────────────────┐
    │                                    ▼
    │                          等待人工确认
    │                                    │
    │                    ┌───────────────┴───────────────┐
    │                    ▼                               ▼
    │         点击「确认报警」                 点击「确认误报」
    │                    │                               │
    │                    ▼                               ▼
    │         status = confirmed          status = false_positive
    │
    └─ 该类型启用 VLM ─────────────────────┐
                                         ▼
                              提交 VLM 复核
                                         │
                          ┌──────────────┴──────────────┐
                          ▼                              ▼
                 VLM 确认                        VLM 判误报
                          │                              │
                          ▼                              ▼
           level = vlm_alarm              level = vlm_ignore
           status 保持 pending            status 保持 pending
                          │                              │
                          └──────────────┬──────────────┘
                                         ▼
                               等待人工确认
                                         │
                          ┌──────────────┴──────────────┐
                          ▼                              ▼
               点击「确认报警」                    点击「确认误报」
                          │                              │
                          ▼                              ▼
               status = confirmed         status = false_positive
```

关键规则：

- VLM 复核结果**只修改 `level` 和 `vlm_review`**，不修改 `status`。
- 所有新记录初始 `status = pending`。
- 只有人工操作能改变 `status`。

## 4. 后端变更

### 4.1 `backend/safety_detection/detector_core.py`

- `TypeSchedule` 移除 `level: str` 字段。
- `TypeSchedule` 新增 `cooldown: float` 字段（秒）。
- `MultiDetector.__init__` 移除 `p0_cooldown`、`p1_cooldown`、`use_vlm` 参数。
- `MultiDetector` 内部按类型读取 `schedule.use_vlm` 和 `schedule.cooldown`。
- `is_in_cooldown` 不再按 fire/smoke  vs 其他类型区分冷却，统一使用 `schedule.cooldown`。
- 触发逻辑统一：
  - 所有类型触发时 `result["level"] = "small_model_alarm"`。
  - 如果 `schedule.use_vlm` 为真，提交 VLM 复核/确认任务。
  - 如果不启用 VLM，直接回调 `trigger_callback`。
- VLM 回调统一更新 `level` 为 `vlm_alarm` 或 `vlm_ignore`，不更新 `status`。

### 4.2 `backend/main_multi.py`

- `on_trigger`：
  - 创建记录时 `level = "small_model_alarm"`，`status = "pending"`。
  - 移除 P0/P1 分支。
  - `vlm_review` 初始为 `None`。
- `on_vlm_result`：
  - 根据 VLM 结果更新 `level` 为 `vlm_alarm` / `vlm_ignore`。
  - 不修改 `status`。
  - 回填 `reason` 说明来源。
- `MultiDetector` 初始化：
  - 不再传 `use_vlm`、`p0_cooldown`、`p1_cooldown`。
  - 从全局设置读取每类检测的 `cooldown` 和 `use_vlm`。
- 新增 API：`POST /alerts/{record_id}/confirm`。
- 修改 API：`POST /alerts/{record_id}/ignore` 保持语义，只是把 `status` 改为 `false_positive`。
- `GET /alerts/stats` 返回：
  - `total`
  - `pending`
  - `confirmed`
  - `false_positive`
- 启动时清空历史记录（测试数据）。

### 4.3 `backend/performance_storage.py`

- `get_records_paginated` 的 `level` 参数支持新值。
- `get_record_summary` 的 `by_level` 统计移除 P0/P1，改为 `by_status` 统计。
- `by_level` 统计改为 `by_alarm_source`，统计三种 `level` 的分布（可选，供前端展示）。

## 5. API 变更

| 方法 | 路径 | 变更 |
|---|---|---|
| GET | `/alerts` | `level` 参数支持 `small_model_alarm` / `vlm_alarm` / `vlm_ignore`；`status` 参数支持 `pending` / `confirmed` / `false_positive` |
| GET | `/alerts/stats` | 返回 `total`, `pending`, `confirmed`, `false_positive` |
| POST | `/alerts/{id}/confirm` | 新增：将记录 `status` 设为 `confirmed` |
| POST | `/alerts/{id}/ignore` | 保留：将记录 `status` 设为 `false_positive` |
| GET | `/record/{id}` | 返回新 `level` 和 `status` |
| GET | `/settings` | 不再返回全局 `use_vlm`、`p0_alert_cooldown`、`p1_alert_cooldown` |
| POST | `/settings` | 同上，不再接收这些字段 |
| POST | `/cameras/{id}/config` | 检测类型配置中 `level` 字段移除，新增 `cooldown` |
| POST | `/cameras/batch-config` | 同上 |
| POST | `/cameras/reset-config` | 同上 |

## 6. 前端变更

### 6.1 `frontend/safety_detection/records.html`

- 统计卡片：
  - 移除 `P0 告警`、`P1 告警`、`已忽略`。
  - 改为 `总记录`、`待确认`、`已确认`、`误报`。
- 过滤器：
  - 「告警级别」选项改为：全部 / 小模型报警 / 大模型报警 / 大模型忽略。
  - 「状态」选项改为：全部 / 待确认 / 已确认 / 误报。
- 表格：
  - 级别列显示对应中文标签。
  - 状态列显示对应中文标签。
- 详情弹窗：
  - 新增/调整按钮：`确认报警`、`确认误报`、`关闭`。
  - 关闭按钮仅关闭弹窗，不修改状态。
  - 根据当前状态禁用已确认/已误报对应的按钮。
  - 告警级别显示新标签。

### 6.2 `frontend/safety_detection/settings.html`

- 全局配置：
  - 移除「启用 VLM 复核」开关。
  - 移除「P0 告警冷却」和「P1 告警冷却」。
- 检测类型默认值表格：
  - 移除「级别」列。
  - 新增「冷却(s)」列。
  - 保留「VLM」勾选列。
- 摄像头编辑弹窗、批量配置弹窗同步上述表格变更。
- 默认类型对象 `defaultTypes()` 移除 `level`，新增 `cooldown`。

### 6.3 `frontend/safety_detection/multi.html`（监控大屏）

- 移除 `p0` / `p1` 样式和逻辑。
- 告警条/徽章按新的三种级别显示。
- 统计摘要同步调整（如果需要）。

### 6.4 `frontend/safety_detection/hud.html`

- 如有 P0/P1 显示，同步替换为新级别。

## 7. 配置变更

### 7.1 全局设置 (`global.json`)

移除字段：

- `use_vlm`
- `p0_alert_cooldown`
- `p1_alert_cooldown`

保留字段（示例）：

- `vlm_max_concurrent`
- `vlm_inspection_interval`
- `max_records`
- `max_storage_mb`
- `memory_threshold_percent`
- `emergency_cleanup_ratio`
- `snapshot_quality`
- `frame_quality`

### 7.2 检测类型配置

旧结构：

```json
{
  "fire": {
    "enabled": true,
    "interval": 1,
    "threshold": 0.6,
    "consecutive_required": 2,
    "level": "P0",
    "use_vlm": true
  }
}
```

新结构：

```json
{
  "fire": {
    "enabled": true,
    "interval": 1,
    "threshold": 0.6,
    "consecutive_required": 2,
    "cooldown": 10,
    "use_vlm": true
  }
}
```

- `level` 字段移除。
- `cooldown` 字段新增，单位秒。
- 工服类型保留 `compliance_window_seconds`，不受影响。

## 8. 数据清理

按用户要求，旧记录均为测试数据，启动时全部清空。实现方式：

- 删除 `storage/records.json`。
- 删除 `storage/frames/` 下的所有快照和帧图片。

或：保留文件但加载后清空 `detection_records` 列表并在首次保存时覆盖。

## 9. 验收标准

- [ ] 创建新记录时 `level=small_model_alarm`、`status=pending`。
- [ ] 不启用 VLM 的类型触发后，列表里一直显示「小模型报警 / 待确认」，直到人工确认。
- [ ] 启用 VLM 的类型，VLM 确认后列表里显示「大模型报警 / 待确认」；VLM 判误报后显示「大模型忽略 / 待确认」。
- [ ] 详情页点击「确认报警」后 `status=confirmed`；点击「确认误报」后 `status=false_positive`；点击「关闭」不修改状态。
- [ ] 设置页可以按类型配置「是否启用 VLM」和「冷却时间」，没有 P0/P1 相关配置。
- [ ] 旧记录不再出现在列表和统计中。

## 10. 决策记录

| 决策 | 选择 | 原因 |
|---|---|---|
| 方案 | 方案一：直接复用 `level` 字段 | 改动最小，现有过滤/统计可直接复用 |
| VLM 是否自动改状态 | 否 | 用户明确要求人工确认 |
| VLM 开关层级 | 仅类型级 | 用户明确要求去掉全局开关 |
| 冷却时间配置 | 每类型独立 | 用户明确要求 |
| 历史记录 | 全部清空 | 用户说明都是测试数据 |
| 「关闭」按钮 | 仅关闭弹窗 | 用户确认 |

## 11. 风险与注意事项

- `level` 字段语义从「级别」变为「来源/复核结果」，需要在代码注释和文档中说明。
- `multi.html` 和 `hud.html` 中可能还有 P0/P1 相关显示，需要一并检查。
- 清空历史记录是破坏性操作，实施前确认无重要数据。
