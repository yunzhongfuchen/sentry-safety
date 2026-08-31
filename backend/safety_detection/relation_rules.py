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
    """命中返回 [(left_box, score), ...]，未命中返回 None；纯全局命中且无框返回 []"""
    obj_conds = []
    global_boxes = {}  # id(box) -> (box, score)
    for c in conditions:
        if c["op"] in GLOBAL_OPS:
            side_b = _side_boxes(c["left"], raw_by_model)
            n = len(side_b)
            op = c["op"]
            hit = (n >= 1) if op == "exists" else (n == 0) if op == "absent" else _count_hit(n, c)
            if not hit:
                return None
            if op in ("exists", "count"):
                for b in side_b:
                    bid = id(b)
                    score = b.get("confidence", 1.0)
                    if bid not in global_boxes or global_boxes[bid][1] < score:
                        global_boxes[bid] = (b, score)
        else:
            obj_conds.append(c)
    if not obj_conds:
        return list(global_boxes.values())

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
