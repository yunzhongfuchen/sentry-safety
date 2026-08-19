# 组合检测规则引擎 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 算法管理页无代码配置组合检测算法（多模型 + 条件组规则），统一替换现有单类别 yolo_box 与 min/max_box_count 机制。

**Architecture:** 新增纯函数规则求值器 `relation_rules.py`（条件组 DNF + 同左侧对象绑定）；注册表算法 Schema 从 `{model_key, classes, model_confidence}` 硬切换为 `{models[], rule{groups[]}}`；inference_engine 按算法收集多模型去重推理（帧序号缓存）；detector_core 外层链路（ROI/连续帧/冷却/静态过滤/VLM）不变。

**Tech Stack:** Python 3.12（conda 环境 `py312`）、FastAPI、Vue3（algorithms.html 单页）、pytest。

**Spec:** docs/superpowers/specs/2026-08-19-relation-rule-engine-design.md

## Global Constraints

- 项目未上线：老字段（model_key/classes/model_confidence 顶层、min_box_count/max_box_count/box_count_mode）**硬删除**，不留兼容层；启动时迁移 algorithms.json。
- conda 环境：`py312`；运行测试用 `python -m pytest tests/ -x -q`。
- 遵守 CLAUDE.md：Surgical Changes——只改本计划列出的行；注释用中文、匹配现有风格。
- `yolo_pose`（睡岗）不进入规则模型，保留为硬编码 post_process，但其算法条目同样改用 `models[]` 字段。
- box 数据约定：raw box = `{"xyxy": [x1,y1,x2,y2], "class_id": int, "confidence": float}`（inference_engine 现有格式）。

---

### Task 1: 规则求值器 relation_rules.py

**Files:**
- Create: `backend/safety_detection/relation_rules.py`
- Test: `tests/test_relation_rules.py`

**Interfaces:**
- Produces（后续 Task 依赖这些名字）:
  - `evaluate_rule(raw_by_model: dict[str, list[dict]], rule: dict) -> dict`，返回 `{"detected": bool, "boxes": list[list], "scores": list[float], "max_confidence": float}`
  - `RELATION_OPS = ("overlap", "contain", "above", "not_overlap", "not_contain")`
  - `GLOBAL_OPS = ("exists", "absent", "count")`
  - `OPERATORS: list[dict]` — 前端下拉元数据（见 Step 3）
- raw_by_model 的 key 是 model_key（注册表 slug），value 是该模型的 raw box 列表。

- [ ] **Step 1: 写失败测试（几何与关系算子）**

```python
# tests/test_relation_rules.py
from backend.safety_detection.relation_rules import evaluate_rule

def _box(x1, y1, x2, y2, cls, conf):
    return {"xyxy": [x1, y1, x2, y2], "class_id": cls, "confidence": conf}

def _side(cls, conf=None):
    s = {"model_key": "m", "classes": [cls]}
    if conf is not None:
        s["conf"] = conf
    return s

def _rule(*conditions):
    return {"groups": [{"conditions": list(conditions)}]}

def test_overlap_hit():
    raw = {"m": [_box(0, 0, 100, 200, 1, 0.9), _box(50, 50, 80, 80, 2, 0.8)]}
    rule = _rule({"left": _side(1), "op": "overlap", "right": _side(2), "iou": 0.001})
    r = evaluate_rule(raw, rule)
    assert r["detected"] is True and len(r["boxes"]) == 1
    assert r["boxes"][0] == [0, 0, 100, 200]
    assert abs(r["max_confidence"] - 0.8) < 1e-6  # min(左, 右)

def test_overlap_miss():
    raw = {"m": [_box(0, 0, 10, 10, 1, 0.9), _box(500, 500, 520, 520, 2, 0.8)]}
    rule = _rule({"left": _side(1), "op": "overlap", "right": _side(2), "iou": 0.3})
    assert evaluate_rule(raw, rule)["detected"] is False

def test_above():
    # 人中心 y=50 高于脚手架顶部 y=100，且水平重叠
    raw = {"m": [_box(100, 0, 160, 100, 1, 0.9), _box(90, 100, 300, 300, 4, 0.8)]}
    rule = _rule({"left": _side(1), "op": "above", "right": _side(4), "iou": 0.001})
    assert evaluate_rule(raw, rule)["detected"] is True

def test_contain_and_not_contain():
    raw = {"m": [_box(0, 0, 100, 200, 1, 0.9), _box(20, 20, 60, 60, 5, 0.8)]}
    rule = _rule({"left": _side(1), "op": "contain", "right": _side(5), "ratio": 0.8})
    assert evaluate_rule(raw, rule)["detected"] is True
    rule2 = _rule({"left": _side(1), "op": "not_contain", "right": _side(5), "ratio": 0.8})
    assert evaluate_rule(raw, rule2)["detected"] is False

def test_exists_absent_count():
    raw = {"m": [_box(0, 0, 10, 10, 8, 0.9), _box(20, 0, 30, 10, 8, 0.8)]}
    assert evaluate_rule(raw, _rule({"left": _side(8), "op": "exists"}))["detected"] is True
    assert evaluate_rule(raw, _rule({"left": _side(7), "op": "absent"}))["detected"] is True
    assert evaluate_rule(raw, _rule({"left": _side(8), "op": "absent"}))["detected"] is False
    assert evaluate_rule(raw, _rule({"left": _side(8), "op": "count", "cmp": "ge", "value": 2}))["detected"] is True
    assert evaluate_rule(raw, _rule({"left": _side(8), "op": "count", "cmp": "lt", "value": 2}))["detected"] is False
    assert evaluate_rule(raw, _rule({"left": _side(8), "op": "count", "cmp": "outside", "min": 1, "max": 5}))["detected"] is False
    assert evaluate_rule(raw, _rule({"left": _side(8), "op": "count", "cmp": "outside", "min": 3, "max": 5}))["detected"] is True

def test_pure_global_hit_conf_is_1():
    r = evaluate_rule({"m": []}, _rule({"left": _side(8), "op": "absent"}))
    assert r["detected"] is True and r["boxes"] == [] and r["max_confidence"] == 1.0
```

- [ ] **Step 2: 写失败测试（绑定与条件组语义）**

```python
def test_same_left_binds_same_object():
    # 张三(在脚手架上,含安全带) + 李四(不在脚手架上,不含安全带) → 不得拼合命中
    zhang = _box(100, 0, 160, 100, 1, 0.9)     # 在脚手架上
    li = _box(400, 300, 460, 500, 1, 0.9)      # 在地面上
    scaffold = _box(90, 100, 300, 300, 4, 0.8)
    harness = _box(110, 20, 150, 60, 5, 0.8)   # 张三的安全带
    raw = {"m": [zhang, li, scaffold, harness]}
    rule = _rule(
        {"left": _side(1), "op": "above", "right": _side(4), "iou": 0.001},
        {"left": _side(1), "op": "not_contain", "right": _side(5), "ratio": 0.5},
    )
    assert evaluate_rule(raw, rule)["detected"] is False

def test_same_left_binds_same_object_hit():
    # 李四在脚手架上且不含安全带 → 命中，报李四
    li = _box(100, 0, 160, 100, 1, 0.9)
    zhang = _box(400, 300, 460, 500, 1, 0.9)
    scaffold = _box(90, 100, 300, 300, 4, 0.8)
    harness = _box(410, 320, 450, 360, 5, 0.8)  # 张三(地面)的安全带
    raw = {"m": [li, zhang, scaffold, harness]}
    rule = _rule(
        {"left": _side(1), "op": "above", "right": _side(4), "iou": 0.001},
        {"left": _side(1), "op": "not_contain", "right": _side(5), "ratio": 0.5},
    )
    r = evaluate_rule(raw, rule)
    assert r["detected"] is True and r["boxes"] == [[100, 0, 160, 100]]

def test_groups_or():
    raw = {"m": [_box(0, 0, 10, 10, 2, 0.9)]}
    rule = {"groups": [
        {"conditions": [{"left": _side(1), "op": "exists"}]},
        {"conditions": [{"left": _side(2), "op": "exists"}]},
    ]}
    assert evaluate_rule(raw, rule)["detected"] is True

def test_side_conf_filter():
    # 条件侧 conf=0.95 过滤掉 0.9 的框
    raw = {"m": [_box(0, 0, 10, 10, 8, 0.9)]}
    assert evaluate_rule(raw, _rule({"left": _side(8, conf=0.95), "op": "exists"}))["detected"] is False

def test_empty_rule_not_detected():
    assert evaluate_rule({"m": []}, {"groups": []})["detected"] is False
```

- [ ] **Step 3: 运行测试确认失败，然后实现**

Run: `python -m pytest tests/test_relation_rules.py -x -q` → FAIL（模块不存在）

实现 `backend/safety_detection/relation_rules.py`：

```python
"""组合规则求值器 — 纯函数，无外部状态

规则结构（见 docs/superpowers/specs/2026-08-19-relation-rule-engine-design.md）：
  条件组之间为或，组内条件为且；组内左侧相同的逐对象条件绑定同一个对象。
raw box 格式：{"xyxy": [x1,y1,x2,y2], "class_id": int, "confidence": float}
"""

import logging

logger = logging.getLogger(__name__)

RELATION_OPS = ("overlap", "contain", "above", "not_overlap", "not_contain")
GLOBAL_OPS = ("exists", "absent", "count")
ALL_OPS = RELATION_OPS + GLOBAL_OPS

# 前端算子下拉元数据（GET /algorithms/operators 下发）
OPERATORS = [
    {"op": "overlap", "label": "重叠", "group": "relation", "param": "iou", "default": 0.3},
    {"op": "contain", "label": "包含", "group": "relation", "param": "ratio", "default": 0.5},
    {"op": "above", "label": "在上方", "group": "relation", "param": "iou", "default": 0.001},
    {"op": "not_overlap", "label": "无重叠", "group": "relation", "param": "iou", "default": 0.3},
    {"op": "not_contain", "label": "不包含", "group": "relation", "param": "ratio", "default": 0.5},
    {"op": "exists", "label": "存在", "group": "global"},
    {"op": "absent", "label": "不存在", "group": "global"},
    {"op": "count", "label": "数量", "group": "global"},
]
COUNT_CMPS = [
    {"cmp": "gt", "label": "数量 >"}, {"cmp": "ge", "label": "数量 ≥"},
    {"cmp": "lt", "label": "数量 <"}, {"cmp": "le", "label": "数量 ≤"},
    {"cmp": "eq", "label": "数量 ="}, {"cmp": "ne", "label": "数量 ≠"},
    {"cmp": "outside", "label": "超出区间[min,max]"},
]


def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _inter(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a, b):
    u = _area(a) + _area(b) - _inter(a, b)
    return _inter(a, b) / u if u > 0 else 0.0


def contain_ratio(outer, inner):
    """inner 落入 outer 的面积占比"""
    a = _area(inner)
    return _inter(outer, inner) / a if a > 0 else 0.0


def _above(l, r, overlap_thr):
    """l 中心高于 r 顶部，且 l 与 r 有重叠（按 l 自 IoU 阈值）"""
    if _area(l) <= 0 or _inter(l, r) / _area(l) < overlap_thr:
        return False
    return (l[1] + l[3]) / 2 < r[1]


def _count_hit(n, cond):
    cmp_ = cond.get("cmp")
    if cmp_ == "outside":
        lo, hi = cond.get("min"), cond.get("max")
        return (lo is not None and n < lo) or (hi is not None and n > hi)
    v = cond.get("value", 0)
    return {"gt": n > v, "ge": n >= v, "lt": n < v,
            "le": n <= v, "eq": n == v, "ne": n != v}.get(cmp_, False)


def _side_key(side):
    return (side.get("model_key"), tuple(sorted(side.get("classes") or [])))


def _side_boxes(side, raw_by_model):
    """取一侧类型的框：按 classes + conf 过滤"""
    classes = side.get("classes")
    conf = side.get("conf")
    out = []
    for b in raw_by_model.get(side.get("model_key"), []):
        if classes is not None and b.get("class_id") not in classes:
            continue
        if conf is not None and b.get("confidence", 0) < conf:
            continue
        out.append(b)
    return out


def _rel_match(op, lbox, rbox, cond):
    l, r = lbox["xyxy"], rbox["xyxy"]
    if op == "overlap":
        return iou(l, r) >= cond.get("iou", 0.3)
    if op == "contain":
        return contain_ratio(l, r) >= cond.get("ratio", 0.5)
    if op == "above":
        return _above(l, r, cond.get("iou", 0.001))
    return False


def _check_obj(cond, lobj, raw_by_model):
    """单个逐对象条件对指定左侧对象判定；命中返回分数，未命中返回 None"""
    lc = cond["left"].get("conf")
    if lc is not None and lobj.get("confidence", 0) < lc:
        return None
    rights = _side_boxes(cond["right"], raw_by_model)
    op = cond["op"]
    if op in ("not_overlap", "not_contain"):
        pos_op = "overlap" if op == "not_overlap" else "contain"
        for r in rights:
            if _rel_match(pos_op, lobj, r, cond):
                return None  # 找到匹配 → 负向条件失败
        return lobj.get("confidence", 1.0)
    best = None
    for r in rights:
        if _rel_match(op, lobj, r, cond):
            s = min(lobj.get("confidence", 1.0), r.get("confidence", 1.0))
            best = s if best is None else max(best, s)
    return best


_MAX_SOLUTIONS = 50  # 多左侧组回溯上限，防笛卡尔积爆炸


def _eval_group(conditions, raw_by_model):
    """命中返回 [(left_box, score), ...]，未命中返回 None；纯全局命中返回 []"""
    obj_conds = []
    for c in conditions:
        if c["op"] in GLOBAL_OPS:
            n = len(_side_boxes(c["left"], raw_by_model))
            op = c["op"]
            hit = (n >= 1) if op == "exists" else (n == 0) if op == "absent" else _count_hit(n, c)
            if not hit:
                return None
        else:
            obj_conds.append(c)
    if not obj_conds:
        return []

    left_keys = []
    for c in obj_conds:
        k = _side_key(c["left"])
        if k not in left_keys:
            left_keys.append(k)
    conds_by_key = {k: [c for c in obj_conds if _side_key(c["left"]) == k] for k in left_keys}

    found = {}  # id(box) -> (box, score)
    solutions = [0]

    def bt(i, assignment):
        if solutions[0] >= _MAX_SOLUTIONS:
            return
        if i == len(left_keys):
            solutions[0] += 1
            for k, (box, score) in assignment.items():
                bid = id(box)
                if bid not in found or found[bid][1] < score:
                    found[bid] = (box, score)
            return
        key = left_keys[i]
        for cand in _side_boxes({"model_key": key[0], "classes": list(key[1]) or None}, raw_by_model):
            score = None
            for c in conds_by_key[key]:
                s = _check_obj(c, cand, raw_by_model)
                if s is None:
                    score = None
                    break
                score = s if score is None else min(score, s)
            if score is not None:
                assignment[key] = (cand, score)
                bt(i + 1, assignment)
                assignment.pop(key)

    bt(0, {})
    if solutions[0] == 0:
        return None
    return list(found.values())


def evaluate_rule(raw_by_model, rule):
    """规则入口：任一条件组命中即 detected"""
    boxes, scores = [], []
    seen = {}
    any_hit = False
    for group in rule.get("groups", []):
        res = _eval_group(group.get("conditions", []), raw_by_model)
        if res is None:
            continue
        any_hit = True
        for box, score in res:
            bid = id(box)
            if bid not in seen:
                seen[bid] = len(boxes)
                boxes.append(box["xyxy"])
                scores.append(score)
            else:
                scores[seen[bid]] = max(scores[seen[bid]], score)
    if not any_hit:
        return {"detected": False, "boxes": [], "scores": [], "max_confidence": 0.0}
    return {
        "detected": True,
        "boxes": boxes,
        "scores": scores,
        "max_confidence": max(scores) if scores else 1.0,  # 纯全局命中按 1.0
    }
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_relation_rules.py -x -q` → 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add backend/safety_detection/relation_rules.py tests/test_relation_rules.py
git commit -m "feat: 组合规则求值器（条件组DNF + 同左侧对象绑定）"
```

---

### Task 2: 注册表 Schema v2 + 启动迁移

**Files:**
- Modify: `backend/detection_registry.py`
- Test: `tests/test_detection_registry.py`

**Interfaces:**
- Consumes: Task 1 的 `ALL_OPS`（校验算子用，从 `backend.safety_detection.relation_rules` 导入）。
- Produces:
  - 算法条目新结构：`{label, color, post_process, models: [{model_key, model_confidence}], rule: {groups: [...]}|None, vlm_prompt, inspection_label, alarm_description, defaults}`
  - `registry.get(dtype)` 返回条目中 `models` 每项额外注入 `model_path`（模型文件名）；`post_process` 为 `yolo_relation` 或 `yolo_pose`
  - `registry.validate_rule(rule) -> list[str]`（错误列表，空=合法）
  - defaults 不再含 `min_box_count/max_box_count/box_count_mode`

- [ ] **Step 1: 改默认值与内置类型**

- `UNIVERSAL_DEFAULTS` 删除 `min_box_count/max_box_count/box_count_mode` 三个 key。
- `DEFAULT_DETECTION_TYPE_REGISTRY` 中 fire/smoke/uniform/mask/cigarette 改为新结构，例如 fire：

```python
    "fire": {
        "label": "明火",
        "color": "#ef4444",
        "post_process": "yolo_relation",
        "models": [{"model_key": "fire_smoke", "model_confidence": 0.5}],
        "rule": {"groups": [{"conditions": [
            {"left": {"model_key": "fire_smoke", "classes": [0]}, "op": "exists"}
        ]}]},
        "vlm_prompt": FIRE_REVIEW_PROMPT,
        "inspection_label": "明火",
        "defaults": { ... 删除三个 box_count 键，其余不变 ... },
    },
```

- sleep 条目：`"post_process": "yolo_pose"`，`"models": [{"model_key": "yolov8n-pose", "model_confidence": 0.1}]`，`"rule": None`，删除 `classes`。
- 内置条目引用的 model_key（fire_smoke / uniform / mask / cigarette / yolov8n-pose）由迁移逻辑按 model_path 生成（见 Step 2，key = 文件名 stem，与 `_migrate_type_dicts` 现有行为一致）。

- [ ] **Step 2: 写迁移函数**

在 `migrate_legacy_registry` 之后新增 `_migrate_algorithms_v2()`，并在 `load()` 中 `migrate_legacy_registry()` 之后调用：

```python
def _migrate_algorithms_v2() -> None:
    """algorithms.json 老结构(顶层 model_key/classes/model_confidence) → models[]+rule 新结构。

    老 defaults 中的 box_count 设置折算为数量条件拼进第一组；迁移后写回文件。
    """
    if not ALGORITHMS_FILE.exists():
        return
    with open(ALGORITHMS_FILE, "r", encoding="utf-8") as f:
        stored = json.load(f)
    if not stored or "models" in next(iter(stored.values())):
        return  # 已是新结构
    migrated = {}
    for key, td in stored.items():
        td = dict(td)
        mkey = td.pop("model_key", None)
        classes = td.pop("classes", None)
        conf = td.pop("model_confidence", 0.5)
        defaults = td.get("defaults", {})
        mn = defaults.pop("min_box_count", None)
        mx = defaults.pop("max_box_count", None)
        mode = defaults.pop("box_count_mode", None)
        conditions = []
        if td.get("post_process") != "yolo_pose":
            conditions.append({"left": {"model_key": mkey, "classes": classes}, "op": "exists"})
            # 老 box_count → 数量条件（默认 min=1 已被 exists 覆盖，跳过）
            side = {"model_key": mkey, "classes": classes}
            if mode == "outside":
                conditions.append({"left": side, "op": "count", "cmp": "outside", "min": mn, "max": mx})
            else:
                if mn is not None and mn > 1:
                    conditions.append({"left": side, "op": "count", "cmp": "ge", "value": mn})
                if mx is not None:
                    conditions.append({"left": side, "op": "count", "cmp": "le", "value": mx})
        td["models"] = [{"model_key": mkey, "model_confidence": conf}] if mkey else []
        td["rule"] = {"groups": [{"conditions": conditions}]} if conditions else None
        if td.get("post_process") == "yolo_box":
            td["post_process"] = "yolo_relation"
        migrated[key] = td
    with open(ALGORITHMS_FILE, "w", encoding="utf-8") as f:
        json.dump(migrated, f, ensure_ascii=False, indent=2)
    logger.info(f"Migrated {len(migrated)} algorithms to v2 (models+rule)")
```

注意：mode 为 None 时老语义是 `min_box_count=1`（至少1框），等价于 exists，已被第一条件覆盖；`between`（mn 与 mx 同时存在且 mode 非 outside）折算为 ge+le 两条。

- [ ] **Step 3: 改 load/get/to_api_list/add_type/update_type/validate**

- `load()`：删掉"动态注入 model_path"段，改为给 `models` 每项注入 `model_path`：

```python
        for td in self._types.values():
            td.pop("icon", None)
            td.pop("vlm_prompt_key", None)
            for m in td.get("models", []):
                model = model_registry.get(m.get("model_key") or "")
                m["model_path"] = model["file"] if model else None
```

- `get()`：`result = copy.deepcopy(td)` 后同样给 models 注入 model_path（深拷贝避免污染缓存）。
- `get_types_by_model(model_key)`：改为匹配 `models` 数组内任一 entry：

```python
        return [dt for dt, td in self._types.items()
                if any(m.get("model_key") == model_key for m in td.get("models", []))]
```

- `get_model_keys_in_use()` / `get_model_usage_counts()`：同样遍历 `models` 数组。
- 新增校验：

```python
    def validate_rule(self, rule: dict) -> list[str]:
        """校验规则结构，返回错误列表（空=合法）"""
        from backend.safety_detection.relation_rules import ALL_OPS, RELATION_OPS
        errors = []
        groups = (rule or {}).get("groups")
        if not groups:
            errors.append("rule.groups 不能为空")
            return errors
        for gi, g in enumerate(groups):
            for ci, c in enumerate(g.get("conditions", [])):
                where = f"组{gi+1}条件{ci+1}"
                op = c.get("op")
                if op not in ALL_OPS:
                    errors.append(f"{where}: 未知算子 {op}")
                    continue
                for side_name in ("left", "right"):
                    side = c.get(side_name)
                    if side_name == "right" and op not in RELATION_OPS:
                        continue
                    if not side or not side.get("model_key"):
                        errors.append(f"{where}: {side_name} 缺少 model_key")
                        continue
                    if model_registry.get(side["model_key"]) is None:
                        errors.append(f"{where}: 未知模型 {side['model_key']}")
                        continue
                    names = model_registry.get(side["model_key"]).get("class_names") or {}
                    for cls in side.get("classes") or []:
                        if names and str(cls) not in names:
                            errors.append(f"{where}: 类别 {cls} 不在模型类别表中")
                if op == "count" and c.get("cmp") not in ("gt", "ge", "lt", "le", "eq", "ne", "outside"):
                    errors.append(f"{where}: 未知数量比较符 {c.get('cmp')}")
        return errors
```

- `add_type(type_def)`：新结构——`models`（必填非空，每个 model_key 必须存在）、`rule`（yolo_relation 必填，调 `validate_rule` 有错则 raise ValueError）、post_process 由模型决定（任一模型是 yolo_pose 则 yolo_pose，否则 yolo_relation；pose 只允许单模型）。删除 classes/model_key/model_confidence 顶层写入。
- `update_type()`：允许更新 `models`、`rule`（更新 rule 时同样 validate_rule）；删除对 `classes`/`model_confidence` 顶层字段的处理。`update_defaults` 的 allowed 集合自然不再含 box_count 三键（因为 defaults 里已没有）。
- `validate()`：遍历 models 检查 `model_registry.file_exists`；post_process 合法集改为 `("yolo_relation", "yolo_pose")`；有 rule 的条目跑 validate_rule 并入 warnings。
- `to_api_list()`：输出 `models`（含每项 model_path/model_name/classes 清单供前端渲染类别多选：`model_registry.get(key)` 的 `class_names`）、`rule`、`post_process`、`label/color/vlm_prompt/inspection_label/alarm_description/defaults`；删除 `model_key/classes/model_confidence` 顶层输出。

- [ ] **Step 4: 更新测试**

`tests/test_detection_registry.py` 现有断言全部基于老结构，逐条改写：
- 断言 `registry.get("fire")["models"][0]["model_key"] == "fire_smoke"` 且 `["rule"]["groups"][0]["conditions"][0]["op"] == "exists"`
- add_type 老调用（传 model_key/classes）改为传 models+rule；断言缺 models 报 ValueError、rule 中未知算子报 ValueError
- 迁移测试：构造老结构临时 algorithms.json（含 min_box_count/max_box_count），调用 `_migrate_algorithms_v2` 后断言 models/rule 结构及数量条件折算正确
- 先跑 `python -m pytest tests/test_detection_registry.py -x -q` 看失败，再改到全绿

- [ ] **Step 5: Commit**

```bash
git add backend/detection_registry.py tests/test_detection_registry.py
git commit -m "feat: 算法注册表切换 models+rule 新结构（启动自动迁移）"
```

---

### Task 3: 帧序号 frame_seq

**Files:**
- Modify: `backend/camera_manager.py`（CameraState 约 line 68）
- Modify: `backend/decode_scheduler.py:232`
- Test: `tests/test_camera_config_api.py`（或新增小测试文件 `tests/test_frame_seq.py`）

**Interfaces:**
- Produces:
  - `CameraState.frame_seq: int = 0`
  - `CameraManager.get_frame_seq(camera_id: str) -> int`（无此摄像头返回 -1）
- Task 4/5 消费：detect() 缓存键、detector_core 传参。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_frame_seq.py
from backend.camera_manager import CameraManager

def test_frame_seq_starts_zero_and_getter():
    cm = CameraManager()
    assert cm.get_frame_seq("nonexistent") == -1
```

注：CameraManager 构造若依赖资源，参考现有 camera 测试的构造方式（tests/ 下已有用法照搬）。

- [ ] **Step 2: 实现**

- `camera_manager.py` CameraState 增加字段 `frame_seq: int = 0`
- 增加方法：

```python
    def get_frame_seq(self, camera_id: str) -> int:
        """当前帧序号（解码线程每写一帧 +1），无此摄像头返回 -1"""
        with self._lock:
            state = self._cameras.get(camera_id)
        if state is None:
            return -1
        with state.lock:
            return state.frame_seq
```

- `decode_scheduler.py:232` `state.current_frame = frame` 处紧接着加 `state.frame_seq += 1`（在同一把 state.lock 内）。

- [ ] **Step 3: 测试通过 + Commit**

Run: `python -m pytest tests/test_frame_seq.py -q` → PASS

```bash
git add backend/camera_manager.py backend/decode_scheduler.py tests/test_frame_seq.py
git commit -m "feat: 摄像头帧增加 frame_seq 序号"
```

---

### Task 4: inference_engine 多模型推理 + 缓存 + yolo_relation 接线

**Files:**
- Modify: `backend/inference_engine.py`（`detect` 419-456、`_run_model` 377-417、`_process_yolo_box` 156-175、`_process_yolo_pose` 172-197、`POST_PROCESSORS` 200-203）
- Test: `tests/test_relation_engine.py`（新建）

**Interfaces:**
- Consumes: Task 1 `evaluate_rule`；Task 2 新结构（`models[]`/`rule`）；Task 3 frame_seq。
- Produces:
  - `SafetyDetector.detect(frame, detection_types, core_id=0, camera_id=None, frame_seq=None, roi_map=None) -> dict`
  - `roi_map`: `{dtype: (roi, roi_invert)}`，提供时对该 dtype 的所有 raw 框先按 ROI 过滤再判定，且结果带 `"roi_applied": True`

- [ ] **Step 1: 写失败测试**

用假模型注册表 + monkeypatch `_run_model` 验证编排逻辑（不加载真实 YOLO）：

```python
# tests/test_relation_engine.py
import numpy as np
from backend.inference_engine import SafetyDetector
from backend.detection_registry import registry

class _FakeRegistry:
    def __init__(self, types):
        self._types = types
    def get(self, dtype):
        return self._types.get(dtype)

def _make_detector(monkeypatch, types, raw_map, calls):
    det = SafetyDetector(npu_cores=0, device="cpu")
    monkeypatch.setattr("backend.inference_engine.registry", _FakeRegistry(types))
    def fake_run(model_path, frame, conf, is_pose, core_id=0):
        calls.append(model_path)
        return raw_map[model_path]
    monkeypatch.setattr(det, "_run_model", fake_run)
    return det

def test_shared_model_inferred_once(monkeypatch):
    types = {
        "algo_weld": {"post_process": "yolo_relation",
                      "models": [{"model_key": "chem", "model_path": "chem.pt", "model_confidence": 0.3}],
                      "rule": {"groups": [{"conditions": [
                          {"left": {"model_key": "chem", "classes": [1]}, "op": "overlap",
                           "right": {"model_key": "chem", "classes": [2]}, "iou": 0.001}]}]}},
        "algo_smoke": {"post_process": "yolo_relation",
                       "models": [{"model_key": "chem", "model_path": "chem.pt", "model_confidence": 0.5}],
                       "rule": {"groups": [{"conditions": [
                           {"left": {"model_key": "chem", "classes": [8]}, "op": "exists"}]}]}},
    }
    raw = [{"xyxy": [0, 0, 100, 200], "class_id": 1, "confidence": 0.9},
           {"xyxy": [50, 50, 80, 80], "class_id": 2, "confidence": 0.8}]
    calls = []
    det = _make_detector(monkeypatch, types, {"chem.pt": raw}, calls)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = det.detect(frame, ["algo_weld", "algo_smoke"], camera_id="c1", frame_seq=1)
    assert calls == ["chem.pt"]  # 同帧同模型只推理一次
    assert results["algo_weld"]["detected"] is True
    assert results["algo_smoke"]["detected"] is False

def test_frame_seq_cache(monkeypatch):
    types = {"algo_smoke": {"post_process": "yolo_relation",
                            "models": [{"model_key": "chem", "model_path": "chem.pt", "model_confidence": 0.5}],
                            "rule": {"groups": [{"conditions": [
                                {"left": {"model_key": "chem", "classes": [8]}, "op": "exists"}]}]}}}
    calls = []
    det = _make_detector(monkeypatch, types, {"chem.pt": []}, calls)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det.detect(frame, ["algo_smoke"], camera_id="c1", frame_seq=7)
    det.detect(frame, ["algo_smoke"], camera_id="c1", frame_seq=7)  # 同帧 → 缓存命中
    det.detect(frame, ["algo_smoke"], camera_id="c1", frame_seq=8)  # 新帧 → 重新推理
    assert calls == ["chem.pt", "chem.pt"]

def test_roi_prefilter(monkeypatch):
    types = {"algo_smoke": {"post_process": "yolo_relation",
                            "models": [{"model_key": "chem", "model_path": "chem.pt", "model_confidence": 0.5}],
                            "rule": {"groups": [{"conditions": [
                                {"left": {"model_key": "chem", "classes": [8]}, "op": "exists"}]}]}}}
    raw = [{"xyxy": [590, 430, 630, 470], "class_id": 8, "confidence": 0.9}]  # 右下角，ROI 外
    calls = []
    det = _make_detector(monkeypatch, types, {"chem.pt": raw}, calls)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    roi = [[(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]]  # 左上四分之一
    results = det.detect(frame, ["algo_smoke"], camera_id="c1", frame_seq=1,
                         roi_map={"algo_smoke": (roi, False)})
    assert results["algo_smoke"]["detected"] is False
    assert results["algo_smoke"].get("roi_applied") is True
```

注意 `_FakeRegistry` 需要 monkeypatch 到 `backend.inference_engine.registry`；`model_registry` 依赖已通过 type_def 里直接注入的 `model_path` 规避（detect 实现中 model_path 取自 models entry，不再查 model_registry）。

- [ ] **Step 2: 运行确认失败** → AttributeError/签名不符

- [ ] **Step 3: 实现**

3a. 删除 `_process_yolo_box` 函数与 `POST_PROCESSORS` 中的 `"yolo_box"` 项，改为：

```python
from backend.safety_detection.relation_rules import evaluate_rule

POST_PROCESSORS = {
    "yolo_relation": None,   # 占位：走 detect() 内的专用分支
    "yolo_pose": _process_yolo_pose,
}
```

3b. `_process_yolo_pose` 中 `type_def.get("model_confidence", 0.1)` 改为：

```python
        conf = (type_def.get("models") or [{}])[0].get("model_confidence", 0.1)
        subjects = process_frame(model, frame, conf=conf)
```

3c. `_run_model` 改签名：

```python
    def _run_model(self, model_path: str, frame: np.ndarray,
                   conf: float, is_pose: bool, core_id: int = 0):
        """执行模型推理，返回原始检测结果（pose 返回模型实例）"""
        model = None
        use_npu = False
        with self._model_lock:
            if model_path in self._npu_models and core_id in self._npu_models[model_path]:
                model = self._npu_models[model_path][core_id]
                use_npu = True
            elif model_path in self._cpu_models:
                model = self._cpu_models[model_path]

        if model is None:
            logger.warning(f"Model {model_path} not loaded")
            return None if is_pose else []
        if is_pose:
            return model
        try:
            if use_npu:
                input_frame = self._preprocess(frame)
                outputs = model.inference(inputs=[input_frame])
                return self._postprocess_rknn(outputs, frame.shape[:2], conf_threshold=conf)
            pred = model.predict(frame, conf=conf, verbose=False)
            boxes = []
            if pred and pred[0].boxes is not None:
                for b in pred[0].boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    boxes.append({"xyxy": [x1, y1, x2, y2],
                                  "class_id": int(b.cls[0]),
                                  "confidence": float(b.conf[0])})
            return boxes
        except Exception as e:
            logger.error(f"Model {model_path} inference error: {e}")
            return []
```

3d. `detect` 重写（两遍结构：先推理去重，再逐算法判定；缓存键 = `(camera_id, frame_seq)`，帧未变则整组 raw 复用）：

```python
    def detect(self, frame: np.ndarray, detection_types: List[str],
               core_id: int = 0, camera_id: str = None,
               frame_seq: int = None, roi_map: dict = None) -> Dict[str, dict]:
        """对单帧执行多类型检测（注册表驱动）

        共享模型文件的算法只推理一次；camera_id+frame_seq 提供时按帧缓存 raw 结果。
        roi_map: {dtype: (roi, roi_invert)}，relation 算法在判定前按 ROI 过滤目标框。
        """
        results: Dict[str, dict] = {}
        cache = getattr(self, "_raw_cache", None)
        if camera_id is not None and frame_seq is not None \
                and cache is not None and cache["key"] == (camera_id, frame_seq):
            raw_by_path: Dict[str, list] = dict(cache["value"])  # 帧未变 → 复用上轮 raw
        else:
            raw_by_path = {}
        new_infer = False

        def _conf_floor(model_path: str) -> float:
            """该模型被引用的最低 conf：模型级与所有条件侧 conf 的最小值"""
            floor = 1.0
            for dt in detection_types:
                td = registry.get(dt)
                if td is None:
                    continue
                path_by_key = {m.get("model_key"): m.get("model_path")
                               for m in td.get("models", [])}
                for m in td.get("models", []):
                    if m.get("model_path") == model_path:
                        floor = min(floor, m.get("model_confidence", 0.5))
                for g in (td.get("rule") or {}).get("groups", []):
                    for c in g.get("conditions", []):
                        for sn in ("left", "right"):
                            s = c.get(sn)
                            if s and s.get("conf") is not None \
                                    and path_by_key.get(s.get("model_key")) == model_path:
                                floor = min(floor, s["conf"])
            return floor

        # 第一遍：收集所有到期算法引用的模型，逐模型推理（同帧去重）
        for dtype in detection_types:
            type_def = registry.get(dtype)
            if type_def is None:
                logger.warning(f"Unknown detection type: {dtype}")
                continue
            is_pose = type_def.get("post_process") == "yolo_pose"
            for m in type_def.get("models", []):
                mpath = m.get("model_path")
                if not mpath or mpath in raw_by_path:
                    continue
                raw_by_path[mpath] = self._run_model(
                    mpath, frame, _conf_floor(mpath), is_pose, core_id)
                new_infer = True

        # 本轮有新推理且提供了帧标识 → 更新帧缓存
        if new_infer and camera_id is not None and frame_seq is not None:
            self._raw_cache = {"key": (camera_id, frame_seq), "value": dict(raw_by_path)}

        # 第二遍：逐算法判定
        for dtype in detection_types:
            type_def = registry.get(dtype)
            if type_def is None:
                continue
            if type_def.get("post_process") == "yolo_pose":
                inst = next((raw_by_path[m["model_path"]] for m in type_def.get("models", [])
                             if m.get("model_path") in raw_by_path), None)
                results[dtype] = _process_yolo_pose(inst, type_def, frame)
                continue

            raw_by_model = {m["model_key"]: raw_by_path.get(m.get("model_path"), [])
                            for m in type_def.get("models", [])}
            # relation：ROI 预过滤所有目标框
            if roi_map and dtype in roi_map:
                roi, roi_invert = roi_map[dtype]
                if roi:
                    h, w = frame.shape[:2]
                    raw_by_model = {mk: _filter_boxes_by_roi(boxes, roi, roi_invert, w, h)
                                    for mk, boxes in raw_by_model.items()}
            r = evaluate_rule(raw_by_model, type_def.get("rule") or {})
            if roi_map and dtype in roi_map:
                r["roi_applied"] = True
            results[dtype] = r

        return results
```

测试断言要点：同一次 detect 内 `raw_by_path` 去重保证同模型只跑一遍；跨调用帧缓存命中时 `_run_model` 不被调用——`calls == ["chem.pt"]` 与 `calls == ["chem.pt", "chem.pt"]` 两条断言即验证这两点。

3e. ROI 框过滤辅助（放 inference_engine 底部，复用 detector_core 的多边形逻辑——直接 import）：

```python
from backend.safety_detection.detector_core import filter_by_roi as _filter_result_by_roi

def _filter_boxes_by_roi(boxes: list, roi: list, roi_invert: bool, w: int, h: int) -> list:
    """raw box 列表按 ROI 过滤（复用 detector_core.filter_by_roi 的判定）"""
    if not boxes:
        return boxes
    result = {"boxes": [b["xyxy"] for b in boxes],
              "scores": [b.get("confidence", 0) for b in boxes],
              "detected": True}
    kept = _filter_result_by_roi(result, roi, roi_invert, w, h)
    kept_set = set(map(tuple, kept["boxes"]))
    return [b for b in boxes if tuple(b["xyxy"]) in kept_set]
```

注意 detector_core 不 import inference_engine，无循环引用；若引入循环则把 filter_by_roi 的 polygon 判定抽到小工具模块。

3f. `ensure_models_loaded` / `_load_model`：`type_def["model_path"]` 读取处改为遍历 models：

```python
            for dtype in detection_types:
                type_def = registry.get(dtype)
                if type_def is None:
                    continue
                for m in type_def.get("models", []):
                    mpath = m.get("model_path")
                    if not mpath or mpath in loaded_paths:
                        continue
                    loaded_paths.add(mpath)
                    self._load_model(mpath, device)
```

`_load_model` 内 `_first_dtype_by_path` 改为匹配 models 内的 model_path。

- [ ] **Step 4: 运行新测试 + 相关旧测试**

Run: `python -m pytest tests/test_relation_engine.py tests/test_frame_seq.py -x -q` → PASS

- [ ] **Step 5: Commit**

```bash
git add backend/inference_engine.py tests/test_relation_engine.py
git commit -m "feat: 多模型去重推理 + 帧序号缓存 + yolo_relation 接线"
```

---

### Task 5: detector_core 接线 + 清理 box_count + GPU 调度排除

**Files:**
- Modify: `backend/safety_detection/detector_core.py`（TypeSchedule 72-99、check_box_count 148-175、_build_schedule 410-428、_process_camera 552-608、_handle_standard_detection 639-）
- Modify: `backend/config.py:95-130`
- Modify: `backend/main_multi.py:479-517`
- Test: `tests/test_camera_config_api.py`、现有 detector 相关测试

**Interfaces:**
- Consumes: Task 4 的 `detect(..., camera_id, frame_seq, roi_map)`。

- [ ] **Step 1: 删 box_count 机制**

- `TypeSchedule` 删除 `min_box_count/max_box_count/box_count_mode` 三字段。
- 删除 `check_box_count` 函数（无其他引用，先 grep 确认）。
- `_build_schedule` 删除三个字段的拷贝行。
- `_handle_standard_detection`：删除 box_count 调用段与 `has_max_box_trigger`/`has_outside_trigger` 分支，判定简化为：

```python
        detected = result.get("detected", False)
        max_conf = max(result.get("scores", [0]) or [0])
        if not detected or max_conf < schedule.threshold:
            ...（重置逻辑不变）
```

注：relation 纯全局命中时 max_confidence=1.0 且 scores 为空——`max(result.get("scores", [0]) or [0])` 对空 scores 得 0，会误挡。**改为** `max_conf = result.get("max_confidence", 0.0)`（yolo_relation 与 yolo_pose 均输出该字段）。

- ROI 段：relation 结果已在 detect 内过滤，跳过二次过滤：

```python
        if schedule.roi and not result.get("roi_applied"):
            h, w = frame.shape[:2]
            result = filter_by_roi(result, schedule.roi, schedule.roi_invert, w, h)
```

- 静态过滤触发段加无框守卫：`if schedule.static_filter and result.get("boxes"):`（缓存段已有守卫，确认触发段 `regions` 为空列表时也跳过——`check_static_filter([])` 的行为改为在调用前判断）。

- [ ] **Step 2: _process_camera 传 camera_id/frame_seq/roi_map**

```python
        frame = self.camera_manager.get_latest_frame(camera_id)
        if frame is None:
            return
        frame_seq = self.camera_manager.get_frame_seq(camera_id)
        ...
        due_types = self._get_due_types(camera_id, now)
        roi_map = {}
        with self._lock:
            for dt in due_types:
                s = self._schedules.get(camera_id, {}).get(dt)
                if s and s.roi:
                    roi_map[dt] = (s.roi, s.roi_invert)
        results = self.safety_detector.detect(frame, due_types, core_id=core_id,
                                              camera_id=camera_id, frame_seq=frame_seq,
                                              roi_map=roi_map)
```

- [ ] **Step 3: config.py 清理**

`get_default_type_config` 与 fallback 字典删除 `min_box_count/max_box_count/box_count_mode` 三键。grep 确认 `backend/config.py` 无其他引用。

- [ ] **Step 4: main_multi GPU 调度排除 relation 算法**

`main_multi.py:487-498` 的 model_configs 构建循环改为跳过 relation 类型（它们走 MultiDetector 自身 detect，不进 GPU 动态调度器；一期边界，见 spec）：

```python
            for dtype in registry.all_types():
                type_def = registry.get(dtype)
                if type_def is None or type_def.get("post_process") != "yolo_pose":
                    continue  # 一期：GPU 动态调度器只接管 pose 类型；relation 由 MultiDetector 检测
                ...
```

注意：现有 fire/smoke 等老 yolo_box 类型此前由 GPU 调度器接管，迁移后变为 yolo_relation → 回到 MultiDetector 串行/核心绑定路径。这是已接受的一期行为，在提交信息中注明。

- [ ] **Step 5: 更新受影响测试并全量回归**

先跑 `python -m pytest tests/ -x -q`，逐个修复因 box_count 删除、config 键删除失败的用例（主要是 test_camera_config_api.py 中 box_count 相关断言——删除或改为数量条件语义）。

- [ ] **Step 6: Commit**

```bash
git add backend/safety_detection/detector_core.py backend/config.py backend/main_multi.py tests/
git commit -m "refactor: 删除 box_count 机制，relation 算法接入调度链路"
```

---

### Task 6: API 层更新

**Files:**
- Modify: `backend/safety_detection/api.py`（list/get/create/update 端点、_algo_to_response 343-360、structural_fields 95/390、allowed_defaults）
- Test: `tests/test_models_api.py`、`tests/test_camera_config_api.py`

**Interfaces:**
- Consumes: Task 2 的 `registry.validate_rule`、Task 1 的 `OPERATORS`/`COUNT_CMPS`。

- [ ] **Step 1: 算子元数据端点**

```python
@router.get("/algorithms/operators")
async def list_operators():
    """规则算子元数据（前端条件编辑器下拉）"""
    from backend.safety_detection.relation_rules import OPERATORS, COUNT_CMPS
    return {"operators": OPERATORS, "count_cmps": COUNT_CMPS}
```

注意路由顺序：该端点必须注册在 `/algorithms/{key}` 之前，避免被路径参数捕获。

- [ ] **Step 2: 响应结构与字段集**

- `_algo_to_response` / `to_api_list` 已在 Task 2 改为输出 `models`/`rule`；api.py 中三处内联响应（get_detection_type 75-85、create_detection_type 133-144）同步删除 `classes`/`model_confidence` 行，改输出 `models`/`rule`。
- `update_detection_type` / `update_algorithm` 的 `structural_fields` 改为 `{"label", "color", "models", "rule", "vlm_prompt", "inspection_label", "alarm_description"}`；`allowed_defaults` 删除三个 box_count 键。
- create/update 端点对 `rule` 非空的请求调用 `registry.validate_rule(data["rule"])`，有错误返回 400 与错误列表。

- [ ] **Step 3: 更新 API 测试**

`tests/test_models_api.py` 中算法 CRUD 用例改为新结构请求体（models+rule），断言响应含 models/rule；box_count 相关用例删除。跑 `python -m pytest tests/test_models_api.py tests/test_camera_config_api.py -x -q` 至全绿。

- [ ] **Step 4: Commit**

```bash
git add backend/safety_detection/api.py tests/test_models_api.py tests/test_camera_config_api.py
git commit -m "feat: 算法 API 支持 models+rule 结构与算子元数据下发"
```

---

### Task 7: 前端算法弹窗重做（algorithms.html）

**Files:**
- Modify: `frontend/safety_detection/algorithms.html`（弹窗 175-260 区、脚本 290-430 区）

**Interfaces:**
- Consumes: `GET /algorithms`（models/rule 结构）、`GET /algorithms/operators`、`GET/PUT/POST /algorithms`。
- 数据模型（dialog 对象）：

```js
dialog = {
  label: "", color: "#888888",
  models: [{ model_key: "", model_confidence: 0.5 }],   // 至少一行
  groups: [{ conditions: [ newCond() ] }],              // 至少一组
  vlm_prompt: "", inspection_label: "", alarm_description: "",
  defaults: { enabled: false, interval: 1, threshold: 0.5,
              consecutive_required: 3, cooldown: 60,
              use_vlm: false, static_filter: false, static_diff_threshold: 0.02 },
}
// newCond() = { left: {model_key:"", classes:[], conf:null}, op: "exists",
//               right: {model_key:"", classes:[], conf:null}, iou: 0.3, ratio: 0.5,
//               cmp: "eq", value: 0, min: null, max: null }
```

- [ ] **Step 1: 模型列表编辑器**

替换"模型"单选下拉为可增删行：每行 = 模型下拉 + 模型置信度输入 + 删除按钮；底部"+ 添加模型"。类别多选的数据源：该模型在 `models`（页面已加载的模型列表，含 class_names）中的类别表。

- [ ] **Step 2: 条件组编辑器**

- 每个条件组一张卡片：条件行列表 + "+ 添加条件" + 删除组按钮；卡片之间显示"或"分隔提示；底部"+ 添加条件组"。
- 条件行渲染逻辑：
  - 左侧：`模型下拉 → 类别多选（checkbox 组）→ conf 输入（可空）`
  - 算子下拉：从 `/algorithms/operators` 渲染，按 group 分组（关系/全局）
  - `op` 属于 relation 组 → 渲染右侧类型选择器 + 参数输入（param 为 iou 显示 IoU 阈值，ratio 显示包含占比）
  - `op === "count"` → 渲染 cmp 下拉（count_cmps）+ value 输入；cmp=outside 时渲染 min/max 两个输入
  - `op` 为 exists/absent → 无右侧
- 组内左侧相同的条件旁显示提示文案："同组内左侧相同的条件将判定同一对象"。
- 保存时组装 `rule = {groups: [...]}`，清理空条件（未选模型/类别的行）后提交；`models` 列表去重校验（同一 model_key 出现两次提示错误）。

- [ ] **Step 3: 卡片列表与回填**

- 算法卡片 meta 行：`模型: {{ t.models.map(m => m.model_key).join(' + ') }}`；策略显示 `t.post_process`。
- 打开编辑：`models`/`groups` 从 `t.rule.groups` 深拷贝回填，conditions 补齐缺省字段（iou/ratio/cmp 等）。
- 删除弹窗中的"报警条件"（box_count_mode/a/b）整块 UI 与对应 JS 字段。

- [ ] **Step 4: 浏览器手动验证**

用 playwright 打开 algorithms 页：新增一个双模型组合算法（化工模型 person 重叠 welding）→ 保存 → 重新打开编辑确认回填正确 → 删除。截图确认布局。

- [ ] **Step 5: Commit**

```bash
git add frontend/safety_detection/algorithms.html
git commit -m "feat: 算法弹窗支持多模型与条件组规则编辑"
```

---

### Task 8: 全量回归 + 文档收尾

- [ ] **Step 1:** `python -m pytest tests/ -q` 全量回归，修复残余失败（重点是依赖老算法结构的测试）。
- [ ] **Step 2:** grep 残留清理确认：`grep -rn "min_box_count\|max_box_count\|box_count_mode\|yolo_box" backend/ frontend/ tests/ docs/` —— 除 migration 注释与 spec/plan 文档外应无引用；docs/api/ 下若有 box_count 字段说明，删除。
- [ ] **Step 3:** 启动后端冒烟：注册表迁移日志、validate 无警告、algorithms API 返回新结构。
- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: 规则引擎收尾（文档清理 + 全量回归）"
```

---

## Self-Review 结论

- Spec 覆盖：算子表(T1)、对象绑定(T1)、组或(T1)、置信度规则(T1/T4)、ROI 统一过滤(T4/T5)、无框静态过滤跳过(T5)、持续计时=interval×consecutive(无需任务)、多模型去重+帧缓存(T3/T4)、迁移(T2)、注册表校验(T2/T6)、前端(T7)、GPU 调度排除(T5)。无缺口。
- 类型一致性：`evaluate_rule`、`detect` 签名、`get_frame_seq`、`validate_rule`、`OPERATORS` 在各 Task 间一致。
- 已知边界（spec 已声明）：GPU 动态调度器一期不接管 relation 算法；摄像头级老 box_count 覆盖配置迁移后失效（未上线，接受）。
