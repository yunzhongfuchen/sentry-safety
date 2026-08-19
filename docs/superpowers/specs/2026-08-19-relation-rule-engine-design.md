# 组合检测规则引擎设计（界面无代码配置）

- 日期：2026-08-19
- 状态：待评审
- 背景：交大交付的化工厂检测（13 种事件）由"单个 14 类 YOLO 模型 + 组合规则"构成（如 人∩焊接火花→动火作业）。现有框架只支持单类别过滤（yolo_box）和硬编码规则插件（yolo_pose 睡岗），无法界面化配置组合规则。本设计引入配置驱动的规则引擎。

## 1. 目标

- 算法管理页无代码配置组合检测算法：多模型、多类别、关系/数量条件、条件组（且/或）
- 将现有单类别检测、数量报警（离岗）统一到同一规则模型，老字段硬切换（项目未上线，不留兼容层）
- 一期：空间关系 + 数量条件；二期：跨算法状态条件（脱岗）、轨迹（攀爬）

## 2. 规则模型

```
算法 = 模型列表 + 触发规则

模型列表：负责"要跑哪些模型"
  [{model_key, model_confidence(底线过滤)}, ...]

触发规则：条件组之间为【或】，组内条件为【且】（DNF）
  条件组 = 条件1 且 条件2 且 ...
  条件 = 左侧类型 + 算子 + (右侧类型 + 关系参数 | 数值)
  类型 = {model_key, classes(多选), conf(可选, 缺省用模型级)}
```

### 2.1 算子表

| 分组 | 算子 op | 参数 | 粒度 |
|---|---|---|---|
| 关系类 | overlap 重叠 | iou | 逐对象 |
| 关系类 | contain 包含（右侧落入主体占比） | ratio | 逐对象 |
| 关系类 | above 在上方（主体中心高于右侧顶部且水平重叠） | iou | 逐对象 |
| 关系类 | not_overlap 无重叠 | iou | 逐对象（负向，报主体） |
| 关系类 | not_contain 不包含 | ratio | 逐对象（负向，报主体） |
| 全局类 | exists 存在 | 无 | 全局（底层 = count ≥ 1） |
| 全局类 | absent 不存在 | 无 | 全局（底层 = count = 0） |
| 全局类 | count 数量 | cmp: gt/ge/lt/le/eq/ne 或 outside + value/min/max | 全局 |

exists/absent 是数量算子的 UI 快捷项，底层统一走 count 判定（优化点 2）。

### 2.2 判定语义

1. **对象绑定**：组内**左侧相同**的逐对象条件绑定**同一个对象**；左侧不同的条件各自独立判定（存在性）。求值时按组内出现的不同左侧类型做对象赋值（回溯试探），存在一组赋值使全部条件成立 → 组命中。每类框数量小，开销可忽略。例：安全带规则两个条件左侧都是"人"→ 必须同一人既在脚手架上又不含安全带，防止"张三踩架子、李四没戴安全带"拼成误报；而"人∩火花 且 背心存在"中背心是全局条件，不参与绑定。
2. **纯全局组**：组内全是全局条件时无绑定对象，命中报全场（无框）。
3. **组间或**：任一条件组命中即 detected=true；报警框 = 命中赋值中所有逐对象条件的左侧对象（含负向条件的左侧）去重后的并集。
4. **综合置信度**：组内有关系条件时，取各匹配对 min(左conf, 右conf) 的最大值；纯全局条件命中时按 1.0 计（无框命中不能被告警阈值挡死）。
5. **ROI 语义**：摄像头级 ROI 只过滤逐对象条件的左右两侧和数量条件的计数对象（即"会被判定/报告的目标"），参照物若仅作为关系右侧且无需 ROI 限定则不过滤——简化实现：**ROI 统一过滤所有目标框**为一期行为，需要在参照物上豁免 ROI 的场景二期再议（交大"高处区域"类需求用 ROI 直接画在主体上即可满足）。
6. **静态过滤**：无框命中时跳过 static_filter。
7. **持续计时**：不新增原语，用 interval × consecutive_required 表达（如 5s × 6 = 持续 30 秒）。

### 2.3 推理与置信度过滤（优化点 1）

- 算法到期时，按模型列表逐个推理；同一帧同一模型只推理一次（去重键：model_path + camera_id + frame_seq，帧序号由 camera_manager 在解码线程维护）。
- 推理后的底线过滤取该模型被引用的**最低** conf（模型级与所有条件侧 conf 的最小值）；各条件侧判定时再按自己的 conf 二次过滤。无额外推理开销。

## 3. 配置 Schema（config/algorithms.json）

```json
"welding_no_helmet": {
  "label": "动火作业未戴安全帽",
  "color": "#ef4444",
  "post_process": "yolo_relation",
  "models": [
    {"model_key": "chemical14", "model_confidence": 0.3},
    {"model_key": "ppe_merged", "model_confidence": 0.5}
  ],
  "rule": {
    "groups": [{
      "conditions": [
        {"left": {"model_key": "chemical14", "classes": [1], "conf": 0.4},
         "op": "overlap",
         "right": {"model_key": "chemical14", "classes": [2], "conf": 0.3},
         "iou": 0.001},
        {"left": {"model_key": "chemical14", "classes": [1]},
         "op": "not_contain",
         "right": {"model_key": "ppe_merged", "classes": [0]},
         "ratio": 0.5}
      ]
    }]
  },
  "alarm_description": "检测到动火作业未戴安全帽"
}
```

单类别算法（迁移后的统一形态）：

```json
"smoke": {
  "label": "烟雾", "post_process": "yolo_relation",
  "models": [{"model_key": "fire_smoke", "model_confidence": 0.5}],
  "rule": {"groups": [{"conditions": [
    {"left": {"model_key": "fire_smoke", "classes": [1]}, "op": "exists"}
  ]}]}
}
```

离岗（数量条件）：

```json
{"left": {"model_key": "chemical14", "classes": [1]}, "op": "count", "cmp": "eq", "value": 0}
```

## 4. 后端改动

| 文件 | 改动 |
|---|---|
| backend/detection_registry.py | 新 Schema 校验（算子合法、类别存在于模型）；算子列表注册并下发前端；启动迁移：老单模型字段（model_key/classes/model_confidence）→ models + rule；**删除** min_box_count/max_box_count/box_count_mode 及 yolo_box 相关读取 |
| backend/inference_engine.py | detect() 收集到期算法的全部模型，去重推理（帧序号缓存），raw 按 model_key 分发；新增 POST_PROCESSORS["yolo_relation"]：DNF 判定，输出标准 result {detected, boxes, scores, max_confidence}；删除 _process_yolo_box |
| backend/safety_detection/detector_core.py | _handle_standard_detection 中 static_filter 增加无框跳过；删除 check_box_count 调用（被数量条件取代） |
| backend/camera_manager.py | 帧状态增加 frame_seq（解码线程递增），供推理缓存 |
| backend/safety_detection/api.py | 下发算子列表接口（前端渲染下拉） |
| backend/safety_detection/sleep_detect.py | 不动（yolo_pose 保留为硬编码插件） |

摄像头侧链路（interval / consecutive / cooldown / ROI / 排程 / VLM 复核 / 告警记录）零改动。

## 5. 前端改动（frontend/safety_detection/algorithms.html）

- 算法编辑弹窗重做：模型列表编辑器（添加模型 → 设置信度）+ 条件组编辑器
- 条件行：左侧类型选择器（模型▼ + 类别多选▼ + 可选conf，自由选择）+ 算子下拉（关系类/数量类分组）+ 按算子类型动态渲染右侧（类型选择器 + 参数 或 数值输入）
- 条件组卡片："添加条件"（组内且）、"添加条件组"（组间或）；组内左侧相同的条件自动绑定同一对象（界面给出提示文案）
- 老算法卡片打开时显示迁移后的新结构（只读提示一次）
- 摄像头弹窗不变

## 6. 分期

**一期**（本设计实现范围）：
- 规则引擎全部算子（关系 5 + 全局 3 类）
- 多模型推理去重 + 帧序号缓存
- 注册表迁移（硬切换）
- 算法弹窗 UI 重做
- 覆盖交大 13 事件中的 11 个（烟/火/异常烟/背心/动土/吊装直接检测；电焊/切割/高处作业/安全带/护栏空间关系）

**二期**（不在本设计）：
- 跨算法状态条件原语（"算法X激活中"）→ 监护人脱岗
- 轨迹条件原语（连续 N 帧位移方向）→ 违规攀爬
- 内置规则模板库（从模板新建算法）

## 7. 测试

- 单测（tests/）：
  - 各算子判定：overlap/contain/above/负向/count 各 cmp
  - 组内且、组间或、对象绑定（同左侧条件必须同一对象，不同对象不得拼合；不同左侧条件各自独立）
  - 综合置信度规则（关系对 min、纯全局 1.0、无框不被 threshold 挡）
  - ROI 一期行为：统一过滤所有目标框
  - 迁移：老 algorithms.json → 新 Schema 等价语义；老 min/max_box_count 配置 → 数量条件
  - 推理去重：两算法共用模型 + 同帧只推理一次（frame_seq 缓存命中）
- 现有测试套件全量回归（迁移后 278 项需同步更新）
- 前端手动验证：建组合算法 → 挂摄像头 → 触发告警全流程
