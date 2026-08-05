# 算法管理：报警说明字段 + 类别选择文案/样式调整

## 背景
当前小模型触发告警时，记录的“说明”字段由 `detector_core.py` 固定生成：`检测到 {dtype}，置信度 {max_conf:.2f}`。用户希望算法可以配置自己的报警说明文本；未配置时回退到模板 `检测到 {算法显示名称} 异常`。

另外，算法编辑弹窗中的“类别过滤（不勾 = 不过滤）”文案和勾选框样式需要调整，以与“运行参数默认值”区域保持一致。

## 目标
1. 算法管理中新增“报警说明”字段：`alarm_description`。
2. 小模型直接触发告警时，优先使用该字段内容；未填写时使用模板 `检测到 {label} 异常`。
3. VLM 复核后的说明逻辑保持不变（继续由 `alarm_state.py` 覆盖为 `[VLM 确认] ...` / `[VLM 已排除] ...`）。
4. 编辑弹窗中“类别过滤”改名为“类别选择（全不勾选 = 全部选择）”。
5. 类别选择勾选框样式与“运行参数默认值”区域一致。

## 数据模型

### 算法注册表字段（`config/algorithms.json`）
每个算法条目新增可选字段：
```json
{
  "label": "明火",
  "alarm_description": "发现明火，请立即处理",
  "...": "..."
}
```
- `alarm_description` 为字符串，可选，空字符串视为未填写。

### API 响应字段
`/algorithms` 列表和详情中增加：
```json
{
  "key": "fire",
  "label": "明火",
  "alarm_description": "发现明火，请立即处理",
  "..."
}
```

## 后端变更

### `backend/detection_registry.py`
1. `_algo_to_response` 和 `to_api_list` 增加 `alarm_description` 字段返回。
2. `add_type` 在创建算法时写入 `alarm_description`（缺省为空字符串）。
3. `update_type` 允许更新 `alarm_description` 字段。

### `backend/safety_detection/detector_core.py`
在 `_handle_standard_detection` 触发告警分支中：
1. 通过 `registry.get(dtype)` 获取算法定义。
2. 如果 `alarm_description` 非空，则 `result["reason"] = alarm_description`。
3. 否则 `result["reason"] = f"检测到 {label} 异常"`。

注意：当前 `_handle_standard_detection` 内部没有直接访问 `registry`，需要通过导入全局 `registry` 实例或从 schedule/外部传入。由于 `MultiDetector` 已与 `camera_manager` 等协作，推荐直接导入 `backend.detection_registry.registry` 单例读取。

## 前端变更

### `frontend/safety_detection/algorithms.html`
1. 编辑弹窗“高级参数”区域增加：
   - 标签：报警说明
   - 输入框：`v-model="dialog.alarm_description"`，placeholder 提示“留空则使用默认模板：检测到xx异常”。
2. `openDialog` 中初始化/回填 `alarm_description` 字段。
3. `saveType` 的 payload 中增加 `alarm_description`。
4. 将“类别过滤（不勾 = 不过滤）”文案改为“类别选择（全不勾选 = 全部选择）”。
5. 将类别选择勾选框外层容器改为 `class="type-card-checkboxes"`，每个选项使用 `class="type-card-checkbox"`，与“运行参数默认值”区域一致。

### `backend/safety_detection/api.py`
1. `structural_fields` 集合增加 `"alarm_description"`。
2. `_algo_to_response` 返回 `alarm_description` 字段。

## 兼容性
- 旧算法没有 `alarm_description` 字段时，回退到默认模板。
- 不影响 VLM 复核、人工确认/误报等后续状态。
- 不影响摄像头配置中 `algorithms` 的存储结构。

## 测试要点
1. 创建/编辑算法时，`alarm_description` 可正常保存和回显。
2. 未填写报警说明时，小模型触发记录的 reason 为 `检测到 {label} 异常`。
3. 填写后，小模型触发记录使用该自定义文本。
4. VLM 复核后仍然覆盖 reason。
5. 类别选择 UI 文案和样式符合要求。

## 设计决策
- `alarm_description` 放在算法结构性字段中，而非 `defaults`，因为它是业务描述元数据，不是运行参数。
- 默认模板中的 `{label}` 使用算法显示名称（`label`），而不是 key，便于用户阅读。
