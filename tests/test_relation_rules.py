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
    # 人中心 y=60 高于脚手架顶部 y=100，且与脚手架有重叠（inter/area=0.167）
    raw = {"m": [_box(100, 0, 160, 120, 1, 0.9), _box(90, 100, 300, 300, 4, 0.8)]}
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

def test_same_left_binds_same_object():
    # 张三(在脚手架上,含安全带) + 李四(不在脚手架上,不含安全带) → 不得拼合命中
    zhang = _box(100, 0, 160, 120, 1, 0.9)    # 在脚手架上
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
    li = _box(100, 0, 160, 120, 1, 0.9)
    zhang = _box(400, 300, 460, 500, 1, 0.9)
    scaffold = _box(90, 100, 300, 300, 4, 0.8)
    harness = _box(410, 320, 450, 360, 5, 0.8)  # 张三(地面)的安全带
    raw = {"m": [li, zhang, scaffold, harness]}
    rule = _rule(
        {"left": _side(1), "op": "above", "right": _side(4), "iou": 0.001},
        {"left": _side(1), "op": "not_contain", "right": _side(5), "ratio": 0.5},
    )
    r = evaluate_rule(raw, rule)
    assert r["detected"] is True and r["boxes"] == [[100, 0, 160, 120]]

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
