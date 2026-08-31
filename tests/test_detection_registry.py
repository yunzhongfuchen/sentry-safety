import json

import pytest


def _exists_rule(mkey, classes=None):
    """构造最小合法规则：单组单 exists 条件"""
    return {"groups": [{"conditions": [
        {"left": {"model_key": mkey, "classes": classes}, "op": "exists"}]}]}


def test_hex_to_bgr_standard():
    from backend.detection_registry import hex_to_bgr
    assert hex_to_bgr("#ef4444") == (68, 68, 239)


def test_hex_to_bgr_without_hash():
    from backend.detection_registry import hex_to_bgr
    assert hex_to_bgr("ef4444") == (68, 68, 239)


def test_hex_to_bgr_white():
    from backend.detection_registry import hex_to_bgr
    assert hex_to_bgr("#ffffff") == (255, 255, 255)


def test_hex_to_bgr_black():
    from backend.detection_registry import hex_to_bgr
    assert hex_to_bgr("#000000") == (0, 0, 0)


class TestDetectionTypeRegistry:
    """注册表核心功能测试（models+rule 新结构）"""

    def _make_registry(self, tmp_path, monkeypatch, data=None, models=None, seed_defaults=True):
        import backend.model_registry as mmod
        import backend.detection_registry as dmod
        monkeypatch.setattr(mmod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(mmod, "MODELS_FILE", tmp_path / "models.json")
        monkeypatch.setattr(mmod, "WEIGHTS_DIR", tmp_path / "weights")
        mr = mmod.ModelRegistry()
        mr.load()
        for m in (models or []):
            mr.add_model(**m)
        monkeypatch.setattr(dmod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(dmod, "ALGORITHMS_FILE", tmp_path / "algorithms.json")
        monkeypatch.setattr(dmod, "REGISTRY_FILE", tmp_path / "detection_types.json")
        monkeypatch.setattr(dmod, "model_registry", mr)
        if data is not None:
            (tmp_path / "algorithms.json").write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
        elif seed_defaults:
            # 模拟播种内置默认算法用于测试
            seeded = dmod._migrate_type_dicts(dmod.DEFAULT_DETECTION_TYPE_REGISTRY)
            (tmp_path / "algorithms.json").write_text(
                json.dumps(seeded, ensure_ascii=False), encoding="utf-8"
            )
        r = dmod.DetectionTypeRegistry()
        r.load()
        return r, mr

    def test_load_generates_empty_file_when_missing(self, tmp_path, monkeypatch):
        r, _ = self._make_registry(tmp_path, monkeypatch, seed_defaults=False)
        assert (tmp_path / "algorithms.json").exists()
        assert r.all_types() == []

    def test_builtin_types_use_models_and_rule(self, tmp_path, monkeypatch):
        """内置类型新结构：models[] + rule.groups，post_process 为 yolo_relation/yolo_pose"""
        r, _ = self._make_registry(tmp_path, monkeypatch)
        fire = r.get("fire")
        assert fire["post_process"] == "yolo_relation"
        assert fire["models"][0]["model_key"] == "fire_smoke"
        assert fire["models"][0]["model_path"] == "fire_smoke.pt"
        cond = fire["rule"]["groups"][0]["conditions"][0]
        assert cond["op"] == "exists"
        assert cond["left"]["model_key"] == "fire_smoke"
        assert cond["left"]["classes"] == [0]
        sleep = r.get("sleep")
        assert sleep["post_process"] == "yolo_pose"
        assert sleep["models"][0]["model_key"] == "yolov8n-pose"
        assert sleep["models"][0]["model_path"] == "yolov8n-pose.pt"
        assert sleep["rule"] is None
        # defaults 不再含 box_count 三键
        for td_key in r.all_types():
            d = r.get_defaults(td_key)
            assert "min_box_count" not in d
            assert "max_box_count" not in d
            assert "box_count_mode" not in d

    def test_load_preserves_existing_and_backfills(self, tmp_path, monkeypatch):
        partial = {"fire": {"label": "自定义火焰", "color": "#ff0000",
                            "post_process": "yolo_relation",
                            "models": [{"model_key": "fire_smoke", "model_confidence": 0.5}],
                            "rule": _exists_rule("fire_smoke", [0]),
                            "vlm_prompt": "fire_review", "inspection_label": "火",
                            "defaults": {"enabled": True, "threshold": 0.9}}}
        r, _ = self._make_registry(tmp_path, monkeypatch, data=partial)
        fire = r.get("fire")
        assert fire["label"] == "自定义火焰"
        assert fire["defaults"]["enabled"] is True
        assert fire["defaults"]["threshold"] == 0.9
        # backfilled fields
        assert "cooldown" in fire["defaults"]
        assert "min_box_count" not in fire["defaults"]

    def test_get_unknown_returns_none(self, tmp_path, monkeypatch):
        r, _ = self._make_registry(tmp_path, monkeypatch)
        assert r.get("unknown_type") is None

    def test_get_returns_deepcopy(self, tmp_path, monkeypatch):
        """get() 深拷贝：修改返回值不污染注册表缓存"""
        r, _ = self._make_registry(tmp_path, monkeypatch)
        fire = r.get("fire")
        fire["models"][0]["model_key"] = "polluted"
        fire["rule"]["groups"][0]["conditions"][0]["op"] = "absent"
        assert r.get("fire")["models"][0]["model_key"] == "fire_smoke"
        assert r.get("fire")["rule"]["groups"][0]["conditions"][0]["op"] == "exists"

    def test_all_types_returns_six(self, tmp_path, monkeypatch):
        r, _ = self._make_registry(tmp_path, monkeypatch)
        assert len(r.all_types()) == 6
        assert set(r.all_types()) == {"fire", "smoke", "uniform", "mask", "cigarette", "sleep"}

    def test_get_types_by_model_shared(self, tmp_path, monkeypatch):
        r, mr = self._make_registry(tmp_path, monkeypatch, models=[
            {"file": "fire_smoke.pt", "name": "烟火模型", "post_process": "yolo_box",
             "class_names": {"0": "fire", "1": "smoke"}}])
        mkey = mr.all_models()[0]
        k1 = r.add_type({"label": "火", "models": [{"model_key": mkey}],
                         "rule": _exists_rule(mkey, [0])})
        k2 = r.add_type({"label": "烟", "models": [{"model_key": mkey}],
                         "rule": _exists_rule(mkey, [1])})
        assert set(r.get_types_by_model(mkey)) == {k1, k2}

    def test_get_types_by_model_unique(self, tmp_path, monkeypatch):
        r, mr = self._make_registry(tmp_path, monkeypatch, models=[
            {"file": "mask.pt", "name": "口罩模型", "post_process": "yolo_box",
             "class_names": {"0": "mask"}}])
        mkey = mr.all_models()[0]
        key = r.add_type({"label": "口罩佩戴", "models": [{"model_key": mkey}],
                          "rule": _exists_rule(mkey)})
        assert r.get_types_by_model(mkey) == [key]

    def test_get_color_bgr(self, tmp_path, monkeypatch):
        r, _ = self._make_registry(tmp_path, monkeypatch)
        assert r.get_color_bgr("fire") == (68, 68, 239)

    def test_get_defaults(self, tmp_path, monkeypatch):
        r, _ = self._make_registry(tmp_path, monkeypatch)
        d = r.get_defaults("sleep")
        assert d["interval"] == 60
        assert d["threshold"] == 0.7
        assert "min_box_count" not in d

    def test_merge_camera_config(self, tmp_path, monkeypatch):
        r, _ = self._make_registry(tmp_path, monkeypatch)
        merged = r.merge_camera_config("fire", {"enabled": True, "threshold": 0.8,
                                                  "roi": [[0.1, 0.1], [0.9, 0.9]]})
        assert merged["enabled"] is True
        assert merged["threshold"] == 0.8
        assert merged["roi"] == [[0.1, 0.1], [0.9, 0.9]]
        # inherited defaults
        assert merged["cooldown"] == 60
        assert merged["consecutive_required"] == 3

    def test_to_api_list(self, tmp_path, monkeypatch):
        r, mr = self._make_registry(tmp_path, monkeypatch)
        api = r.to_api_list()
        assert len(api) == 6
        fire_entry = next(e for e in api if e["key"] == "fire")
        assert fire_entry["label"] == "明火"
        assert fire_entry["color"] == "#ef4444"
        assert fire_entry["post_process"] == "yolo_relation"
        assert "defaults" in fire_entry
        # models 数组带 model_path/model_name/class_names 供前端渲染
        mkey = next(k for k in mr.all_models() if mr.get(k)["file"] == "fire_smoke.pt")
        m0 = fire_entry["models"][0]
        assert m0["model_key"] == mkey
        assert m0["model_path"] == "fire_smoke.pt"
        assert "model_name" in m0
        assert "class_names" in m0
        assert fire_entry["rule"]["groups"][0]["conditions"][0]["op"] == "exists"
        # 顶层不再输出老结构字段
        assert "model_key" not in fire_entry
        assert "classes" not in fire_entry
        assert "model_confidence" not in fire_entry
        assert "npu_model_path" not in fire_entry
        assert "明火" in fire_entry["vlm_prompt"]
        assert fire_entry["inspection_label"] == "明火"

    def test_validate_warns_missing_model(self, tmp_path, monkeypatch):
        r, mr = self._make_registry(tmp_path, monkeypatch, models=[
            {"file": "missing.pt", "name": "缺失模型", "post_process": "yolo_box",
             "class_names": {}}])
        mkey = mr.all_models()[0]
        r.add_type({"label": "缺模型算法", "models": [{"model_key": mkey}],
                    "rule": _exists_rule(mkey)})
        warnings = r.validate()
        # 模型文件未落到 weights/，应产生警告
        assert len(warnings) > 0
        assert any(mkey in w for w in warnings)

    def test_update_defaults_persists(self, tmp_path, monkeypatch):
        r, _ = self._make_registry(tmp_path, monkeypatch)
        r.update_defaults("fire", {"threshold": 0.99, "cooldown": 120})
        assert r.get_defaults("fire")["threshold"] == 0.99
        assert r.get_defaults("fire")["cooldown"] == 120
        # verify persistence
        with open(tmp_path / "algorithms.json", "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["fire"]["defaults"]["threshold"] == 0.99

    def test_update_defaults_ignores_structural_fields(self, tmp_path, monkeypatch):
        r, _ = self._make_registry(tmp_path, monkeypatch)
        models_before = r.get("fire")["models"]
        r.update_defaults("fire", {"models": [], "threshold": 0.1})
        assert r.get("fire")["models"] == models_before
        assert r.get_defaults("fire")["threshold"] == 0.1

    def test_load_backfills_custom_type_defaults(self, tmp_path, monkeypatch):
        custom = {"custom_type": {"label": "自定义", "color": "#123456",
                                  "post_process": "yolo_relation",
                                  "models": [{"model_key": "custom", "model_confidence": 0.5}],
                                  "rule": _exists_rule("custom", [0]),
                                  "vlm_prompt": "custom_review", "inspection_label": "自定义"}}
        r, _ = self._make_registry(tmp_path, monkeypatch, data=custom)
        assert "custom_type" in r.all_types()
        d = r.get_defaults("custom_type")
        assert d["enabled"] is False
        assert d["interval"] == 1
        assert d["threshold"] == 0.5
        assert d["consecutive_required"] == 3
        assert d["cooldown"] == 60
        assert d["use_vlm"] is False
        assert "min_box_count" not in d
        assert "max_box_count" not in d

    def test_get_color_bgr_unknown_returns_green(self, tmp_path, monkeypatch):
        r, _ = self._make_registry(tmp_path, monkeypatch)
        assert r.get_color_bgr("unknown") == (0, 255, 0)

    def test_get_defaults_unknown_returns_empty(self, tmp_path, monkeypatch):
        r, _ = self._make_registry(tmp_path, monkeypatch)
        assert r.get_defaults("unknown") == {}

    def test_merge_camera_config_unknown_returns_overrides(self, tmp_path, monkeypatch):
        r, _ = self._make_registry(tmp_path, monkeypatch)
        overrides = {"enabled": True, "threshold": 0.8}
        assert r.merge_camera_config("unknown", overrides) == overrides

    def test_update_defaults_unknown_is_noop(self, tmp_path, monkeypatch):
        r, _ = self._make_registry(tmp_path, monkeypatch)
        r.update_defaults("unknown", {"threshold": 0.1})
        assert r.get("unknown") is None

    def test_get_types_by_model_missing_model(self, tmp_path, monkeypatch):
        custom = {"custom_no_model": {"label": "自定义"}}
        r, _ = self._make_registry(tmp_path, monkeypatch, data=custom)
        # should not raise
        assert r.get_types_by_model("anything.pt") == []
        assert r.get_types_by_model("") == []

    def test_to_api_list_missing_structural_fields(self, tmp_path, monkeypatch):
        custom = {"custom_sparse": {"defaults": {"enabled": True},
                                    "post_process": "yolo_relation", "models": [],
                                    "rule": None}}
        r, _ = self._make_registry(tmp_path, monkeypatch, data=custom)
        entry = next(e for e in r.to_api_list() if e["key"] == "custom_sparse")
        assert entry["label"] == "custom_sparse"
        assert entry["color"] == "#888888"
        assert entry["post_process"] == "yolo_relation"
        assert entry["models"] == []
        assert entry["rule"] is None
        assert entry["defaults"]["enabled"] is True
        assert entry["defaults"]["interval"] == 1
        assert entry["defaults"]["threshold"] == 0.5

    def test_load_malformed_json_initializes_empty(self, tmp_path, monkeypatch):
        (tmp_path / "algorithms.json").write_text("not json at all", encoding="utf-8")
        import backend.detection_registry as mod
        monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(mod, "ALGORITHMS_FILE", tmp_path / "algorithms.json")
        r = mod.DetectionTypeRegistry()
        r.load()
        assert r.all_types() == []
        # corrupted file overwritten with empty dict
        with open(tmp_path / "algorithms.json", "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved == {}


def test_hex_to_bgr_invalid_hex_raises():
    from backend.detection_registry import hex_to_bgr
    with pytest.raises(ValueError):
        hex_to_bgr("#zz4444")


def test_hex_to_bgr_invalid_type_raises():
    from backend.detection_registry import hex_to_bgr
    with pytest.raises(ValueError):
        hex_to_bgr(None)


def test_hex_to_bgr_wrong_length_raises():
    from backend.detection_registry import hex_to_bgr
    with pytest.raises(ValueError):
        hex_to_bgr("#ef444")


def _make_registry_with_model(tmp_path, monkeypatch):
    """构建算法注册表 + 模型注册表，返回 (registry, model_registry, model_key)"""
    import backend.model_registry as mmod
    import backend.detection_registry as dmod
    monkeypatch.setattr(mmod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(mmod, "MODELS_FILE", tmp_path / "models.json")
    monkeypatch.setattr(mmod, "WEIGHTS_DIR", tmp_path / "weights")
    mr = mmod.ModelRegistry()
    mr.load()
    mkey = mr.add_model(file="test.pt", name="测试模型", post_process="yolo_box",
                        class_names={"0": "x"})
    monkeypatch.setattr(dmod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(dmod, "ALGORITHMS_FILE", tmp_path / "algorithms.json")
    monkeypatch.setattr(dmod, "REGISTRY_FILE", tmp_path / "detection_types.json")
    monkeypatch.setattr(dmod, "model_registry", mr)
    r = dmod.DetectionTypeRegistry()
    r.load()
    return r, mr, mkey


def test_add_type_generates_key_and_saves(tmp_path, monkeypatch):
    r, mr, mkey = _make_registry_with_model(tmp_path, monkeypatch)
    key = r.add_type({"label": "测试类型", "color": "#ff0000",
                      "models": [{"model_key": mkey}], "rule": _exists_rule(mkey, [0])})
    assert key is not None
    assert r.get(key)["label"] == "测试类型"


def test_add_type_missing_models_raises(tmp_path, monkeypatch):
    r, mr, mkey = _make_registry_with_model(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="models"):
        r.add_type({"label": "无模型", "rule": _exists_rule(mkey)})


def test_add_type_unknown_op_in_rule_raises(tmp_path, monkeypatch):
    r, mr, mkey = _make_registry_with_model(tmp_path, monkeypatch)
    bad_rule = {"groups": [{"conditions": [
        {"left": {"model_key": mkey, "classes": [0]}, "op": "teleport"}]}]}
    with pytest.raises(ValueError, match="未知算子"):
        r.add_type({"label": "坏规则", "models": [{"model_key": mkey}], "rule": bad_rule})


def test_add_type_duplicate_label_raises(tmp_path, monkeypatch):
    r, mr, mkey = _make_registry_with_model(tmp_path, monkeypatch)
    r.add_type({"label": "明火", "color": "#ff0000",
                "models": [{"model_key": mkey}], "rule": _exists_rule(mkey)})
    with pytest.raises(ValueError, match="already exists"):
        r.add_type({"label": "明火", "color": "#ff0000",
                    "models": [{"model_key": mkey}], "rule": _exists_rule(mkey)})


def test_delete_type_removes_and_saves(tmp_path, monkeypatch):
    r, mr, mkey = _make_registry_with_model(tmp_path, monkeypatch)
    key = r.add_type({"label": "临时类型", "color": "#00ff00",
                      "models": [{"model_key": mkey}], "rule": _exists_rule(mkey)})
    assert r.get(key) is not None
    r.delete_type(key)
    assert r.get(key) is None


def test_update_type_structural_fields(tmp_path, monkeypatch):
    r, mr, mkey = _make_registry_with_model(tmp_path, monkeypatch)
    mkey2 = mr.add_model(file="new.pt", name="新模型", post_process="yolo_pose",
                         class_names={"0": "y"})
    key = r.add_type({"label": "旧名称", "color": "#111111",
                      "models": [{"model_key": mkey}], "rule": _exists_rule(mkey)})
    r.update_type(key, {"label": "新名称", "color": "#222222",
                        "models": [{"model_key": mkey2}], "rule": None})
    td = r.get(key)
    assert td["label"] == "新名称"
    assert td["color"] == "#222222"
    assert td["models"][0]["model_key"] == mkey2
    assert td["models"][0]["model_path"] == "new.pt"
    assert td["rule"] is None
    # post_process 跟随模型
    assert td["post_process"] == "yolo_pose"


class TestAlgorithmModelKey:
    @pytest.fixture
    def both(self, tmp_path, monkeypatch):
        import backend.model_registry as mmod
        import backend.detection_registry as dmod
        monkeypatch.setattr(mmod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(mmod, "MODELS_FILE", tmp_path / "models.json")
        monkeypatch.setattr(mmod, "WEIGHTS_DIR", tmp_path / "weights")
        mr = mmod.ModelRegistry()
        mr.load()
        mkey = mr.add_model(file="leak.pt", name="漏液模型", post_process="yolo_box",
                            class_names={"0": "leak"})
        monkeypatch.setattr(dmod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(dmod, "ALGORITHMS_FILE", tmp_path / "algorithms.json")
        monkeypatch.setattr(dmod, "REGISTRY_FILE", tmp_path / "detection_types.json")
        monkeypatch.setattr(dmod, "model_registry", mr)
        r = dmod.DetectionTypeRegistry()
        r.load()
        return r, mr, mkey

    def test_get_injects_model_path(self, both):
        r, mr, mkey = both
        key = r.add_type({"label": "漏液", "color": "#facc15",
                          "models": [{"model_key": mkey, "model_confidence": 0.6}],
                          "rule": _exists_rule(mkey, [0]),
                          "defaults": {"threshold": 0.5}})
        td = r.get(key)
        assert td["models"][0]["model_key"] == mkey
        assert td["models"][0]["model_path"] == "leak.pt"
        assert td["models"][0]["model_confidence"] == 0.6
        assert td["post_process"] == "yolo_relation"

    def test_add_type_unknown_model_key_raises(self, both):
        r, mr, mkey = both
        with pytest.raises(ValueError, match="Unknown model"):
            r.add_type({"label": "X", "color": "#fff",
                        "models": [{"model_key": "nonexistent"}],
                        "rule": _exists_rule("nonexistent")})

    def test_add_type_copies_post_process_from_model(self, both):
        r, mr, mkey = both
        mr.update_model(mkey, {"post_process": "yolo_pose"})
        key = r.add_type({"label": "姿态类", "color": "#fff",
                          "models": [{"model_key": mkey}], "rule": None})
        assert r.get(key)["post_process"] == "yolo_pose"

    def test_add_type_pose_multi_model_raises(self, both):
        r, mr, mkey = both
        mr.update_model(mkey, {"post_process": "yolo_pose"})
        mkey2 = mr.add_model(file="pose2.pt", name="姿态2", post_process="yolo_pose",
                             class_names={"0": "z"})
        with pytest.raises(ValueError, match="yolo_pose"):
            r.add_type({"label": "多姿态", "models": [{"model_key": mkey},
                                                      {"model_key": mkey2}]})

    def test_update_type_models_validated(self, both):
        r, mr, mkey = both
        key = r.add_type({"label": "漏液", "color": "#facc15",
                          "models": [{"model_key": mkey}], "rule": _exists_rule(mkey)})
        with pytest.raises(ValueError, match="Unknown model"):
            r.update_type(key, {"models": [{"model_key": "ghost"}]})

    def test_update_type_rule_validated(self, both):
        r, mr, mkey = both
        key = r.add_type({"label": "漏液", "color": "#facc15",
                          "models": [{"model_key": mkey}], "rule": _exists_rule(mkey)})
        bad_rule = {"groups": [{"conditions": [{"left": {"model_key": mkey}, "op": "fly"}]}]}
        with pytest.raises(ValueError, match="未知算子"):
            r.update_type(key, {"rule": bad_rule})

    def test_to_api_list_has_models_with_path(self, both):
        r, mr, mkey = both
        key = r.add_type({"label": "漏液", "color": "#facc15",
                          "models": [{"model_key": mkey}], "rule": _exists_rule(mkey)})
        item = [t for t in r.to_api_list() if t["key"] == key][0]
        assert item["models"][0]["model_key"] == mkey
        assert item["models"][0]["model_path"] == "leak.pt"
        assert item["models"][0]["class_names"] == {"0": "leak"}

    def test_get_model_keys_in_use(self, both):
        r, mr, mkey = both
        assert mkey not in r.get_model_keys_in_use()
        r.add_type({"label": "漏液", "color": "#facc15",
                    "models": [{"model_key": mkey}], "rule": _exists_rule(mkey)})
        assert mkey in r.get_model_keys_in_use()

    def test_get_model_usage_counts_multiple_algorithms(self, both):
        """同一模型被多个算法引用时返回真实引用数（used_by 展示用）"""
        r, mr, mkey = both
        assert mkey not in r.get_model_usage_counts()
        r.add_type({"label": "安全帽", "color": "#facc15",
                    "models": [{"model_key": mkey}], "rule": _exists_rule(mkey)})
        r.add_type({"label": "反光背心", "color": "#facc15",
                    "models": [{"model_key": mkey}], "rule": _exists_rule(mkey)})
        assert r.get_model_usage_counts()[mkey] == 2

    def test_model_path_none_when_model_deleted(self, both):
        r, mr, mkey = both
        key = r.add_type({"label": "漏液", "color": "#facc15",
                          "models": [{"model_key": mkey}], "rule": _exists_rule(mkey)})
        mr.delete_model(mkey)
        assert r.get(key)["models"][0]["model_path"] is None


class TestValidateRule:
    @pytest.fixture
    def reg(self, tmp_path, monkeypatch):
        r, mr, mkey = _make_registry_with_model(tmp_path, monkeypatch)
        return r, mkey

    def test_valid_rule_returns_empty(self, reg):
        r, mkey = reg
        assert r.validate_rule(_exists_rule(mkey, [0])) == []

    def test_empty_groups(self, reg):
        r, mkey = reg
        assert r.validate_rule(None) != []
        assert r.validate_rule({}) != []
        assert r.validate_rule({"groups": []}) != []

    def test_unknown_op(self, reg):
        r, mkey = reg
        errors = r.validate_rule({"groups": [{"conditions": [
            {"left": {"model_key": mkey}, "op": "teleport"}]}]})
        assert any("未知算子" in e and "组1条件1" in e for e in errors)

    def test_missing_model_key(self, reg):
        r, mkey = reg
        errors = r.validate_rule({"groups": [{"conditions": [
            {"left": {"classes": [0]}, "op": "exists"}]}]})
        assert any("缺少 model_key" in e for e in errors)

    def test_unknown_model(self, reg):
        r, mkey = reg
        errors = r.validate_rule(_exists_rule("ghost"))
        assert any("未知模型" in e for e in errors)

    def test_class_not_in_model(self, reg):
        r, mkey = reg
        errors = r.validate_rule(_exists_rule(mkey, [7]))
        assert any("类别 7" in e for e in errors)

    def test_relation_op_requires_right(self, reg):
        r, mkey = reg
        errors = r.validate_rule({"groups": [{"conditions": [
            {"left": {"model_key": mkey, "classes": [0]}, "op": "overlap"}]}]})
        assert any("right 缺少 model_key" in e for e in errors)

    def test_count_bad_cmp(self, reg):
        r, mkey = reg
        errors = r.validate_rule({"groups": [{"conditions": [
            {"left": {"model_key": mkey}, "op": "count", "cmp": "around", "value": 2}]}]})
        assert any("未知数量比较符" in e for e in errors)

    def test_count_valid_cmp(self, reg):
        r, mkey = reg
        assert r.validate_rule({"groups": [{"conditions": [
            {"left": {"model_key": mkey}, "op": "count", "cmp": "ge", "value": 2}]}]}) == []


class TestLegacyMigration:
    def test_migrates_types_to_models_and_algorithms(self, tmp_path, monkeypatch):
        import backend.model_registry as mmod
        import backend.detection_registry as dmod
        monkeypatch.setattr(mmod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(mmod, "MODELS_FILE", tmp_path / "models.json")
        monkeypatch.setattr(mmod, "WEIGHTS_DIR", tmp_path / "weights")
        monkeypatch.setattr(dmod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(dmod, "ALGORITHMS_FILE", tmp_path / "algorithms.json")
        monkeypatch.setattr(dmod, "REGISTRY_FILE", tmp_path / "detection_types.json")
        legacy = {
            "fire": {"label": "明火", "color": "#ef4444", "model_path": "fire_smoke.pt",
                     "post_process": "yolo_box", "classes": [0], "model_confidence": 0.5,
                     "vlm_prompt": "p1", "inspection_label": "明火",
                     "defaults": {"threshold": 0.6}},
            "smoke": {"label": "烟雾", "color": "#f97316", "model_path": "fire_smoke.pt",
                      "post_process": "yolo_box", "classes": [1], "model_confidence": 0.5,
                      "vlm_prompt": "p2", "inspection_label": "烟雾",
                      "defaults": {"threshold": 0.55}},
        }
        (tmp_path / "detection_types.json").write_text(
            json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
        mr = mmod.ModelRegistry()
        monkeypatch.setattr(dmod, "model_registry", mr)
        r = dmod.DetectionTypeRegistry()
        r.load()
        # 模型去重：fire/smoke 共享 fire_smoke.pt → 只有一个模型条目
        assert len(mr.all_models()) == 1
        mkey = mr.all_models()[0]
        assert mr.get(mkey)["file"] == "fire_smoke.pt"
        # 算法升级为 v2：models 指向同一模型，rule 带 exists 条件
        assert r.get("fire")["models"][0]["model_key"] == mkey
        assert r.get("smoke")["models"][0]["model_key"] == mkey
        assert r.get("fire")["post_process"] == "yolo_relation"
        cond = r.get("fire")["rule"]["groups"][0]["conditions"][0]
        assert cond["op"] == "exists"
        assert cond["left"]["classes"] == [0]
        assert r.get("fire")["defaults"]["threshold"] == 0.6
        # 旧文件改名 .bak
        assert not (tmp_path / "detection_types.json").exists()
        assert (tmp_path / "detection_types.json.bak").exists()

    def test_no_legacy_file_does_not_migrate(self, tmp_path, monkeypatch):
        """无 detection_types.json 时不执行迁移，返回 False"""
        import backend.model_registry as mmod
        import backend.detection_registry as dmod
        monkeypatch.setattr(mmod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(mmod, "MODELS_FILE", tmp_path / "models.json")
        monkeypatch.setattr(mmod, "WEIGHTS_DIR", tmp_path / "weights")
        monkeypatch.setattr(dmod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(dmod, "ALGORITHMS_FILE", tmp_path / "algorithms.json")
        monkeypatch.setattr(dmod, "REGISTRY_FILE", tmp_path / "detection_types.json")
        mr = mmod.ModelRegistry()
        monkeypatch.setattr(dmod, "model_registry", mr)
        assert dmod.migrate_legacy_registry() is False
        assert not (tmp_path / "algorithms.json").exists()


class TestMigrateAlgorithmsV2:
    """老结构 algorithms.json（顶层 model_key/classes/model_confidence）→ v2 迁移"""

    def _run_migration(self, tmp_path, monkeypatch, stored):
        import backend.model_registry as mmod
        import backend.detection_registry as dmod
        monkeypatch.setattr(mmod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(mmod, "MODELS_FILE", tmp_path / "models.json")
        monkeypatch.setattr(mmod, "WEIGHTS_DIR", tmp_path / "weights")
        monkeypatch.setattr(dmod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(dmod, "ALGORITHMS_FILE", tmp_path / "algorithms.json")
        (tmp_path / "algorithms.json").write_text(
            json.dumps(stored, ensure_ascii=False), encoding="utf-8")
        dmod._migrate_algorithms_v2()
        return json.loads((tmp_path / "algorithms.json").read_text(encoding="utf-8"))

    def test_basic_exists_and_structure(self, tmp_path, monkeypatch):
        out = self._run_migration(tmp_path, monkeypatch, {
            "fire": {"label": "明火", "model_key": "fire_smoke", "classes": [0],
                     "model_confidence": 0.5, "post_process": "yolo_box",
                     "defaults": {"threshold": 0.6, "min_box_count": 1,
                                  "max_box_count": None, "box_count_mode": None}},
        })
        td = out["fire"]
        assert td["post_process"] == "yolo_relation"
        assert td["models"] == [{"model_key": "fire_smoke", "model_confidence": 0.5}]
        assert "model_key" not in td and "classes" not in td
        conds = td["rule"]["groups"][0]["conditions"]
        # 默认 min=1 已被 exists 覆盖，只有一条
        assert conds == [{"left": {"model_key": "fire_smoke", "classes": [0]}, "op": "exists"}]
        # box_count 三键从 defaults 移除
        assert "min_box_count" not in td["defaults"]
        assert "max_box_count" not in td["defaults"]
        assert "box_count_mode" not in td["defaults"]
        assert td["defaults"]["threshold"] == 0.6

    def test_min_gt1_becomes_ge(self, tmp_path, monkeypatch):
        out = self._run_migration(tmp_path, monkeypatch, {
            "gather": {"label": "聚集", "model_key": "person", "classes": [0],
                       "post_process": "yolo_box",
                       "defaults": {"min_box_count": 3}},
        })
        conds = out["gather"]["rule"]["groups"][0]["conditions"]
        assert {"left": {"model_key": "person", "classes": [0]},
                "op": "count", "cmp": "ge", "value": 3} in conds

    def test_max_becomes_le(self, tmp_path, monkeypatch):
        out = self._run_migration(tmp_path, monkeypatch, {
            "few": {"label": "少人", "model_key": "person", "classes": None,
                    "post_process": "yolo_box",
                    "defaults": {"min_box_count": 1, "max_box_count": 2}},
        })
        conds = out["few"]["rule"]["groups"][0]["conditions"]
        assert {"left": {"model_key": "person", "classes": None},
                "op": "count", "cmp": "le", "value": 2} in conds
        # min=1 不生成 ge 条件
        assert not any(c.get("cmp") == "ge" for c in conds)

    def test_outside_mode(self, tmp_path, monkeypatch):
        out = self._run_migration(tmp_path, monkeypatch, {
            "abnormal": {"label": "异常数量", "model_key": "person", "classes": [0],
                         "post_process": "yolo_box",
                         "defaults": {"min_box_count": 2, "max_box_count": 5,
                                      "box_count_mode": "outside"}},
        })
        conds = out["abnormal"]["rule"]["groups"][0]["conditions"]
        assert {"left": {"model_key": "person", "classes": [0]},
                "op": "count", "cmp": "outside", "min": 2, "max": 5} in conds

    def test_between_becomes_ge_and_le(self, tmp_path, monkeypatch):
        out = self._run_migration(tmp_path, monkeypatch, {
            "range": {"label": "区间", "model_key": "person", "classes": [0],
                      "post_process": "yolo_box",
                      "defaults": {"min_box_count": 2, "max_box_count": 5,
                                   "box_count_mode": "between"}},
        })
        conds = out["range"]["rule"]["groups"][0]["conditions"]
        assert {"left": {"model_key": "person", "classes": [0]},
                "op": "count", "cmp": "ge", "value": 2} in conds
        assert {"left": {"model_key": "person", "classes": [0]},
                "op": "count", "cmp": "le", "value": 5} in conds

    def test_pose_type_no_rule(self, tmp_path, monkeypatch):
        out = self._run_migration(tmp_path, monkeypatch, {
            "sleep": {"label": "睡岗", "model_key": "yolov8n-pose", "classes": None,
                      "model_confidence": 0.1, "post_process": "yolo_pose",
                      "defaults": {"interval": 60}},
        })
        td = out["sleep"]
        assert td["post_process"] == "yolo_pose"
        assert td["models"] == [{"model_key": "yolov8n-pose", "model_confidence": 0.1}]
        assert td["rule"] is None

    def test_already_v2_is_noop(self, tmp_path, monkeypatch):
        v2 = {"fire": {"label": "明火", "post_process": "yolo_relation",
                       "models": [{"model_key": "fire_smoke", "model_confidence": 0.5}],
                       "rule": _exists_rule("fire_smoke", [0]), "defaults": {}}}
        out = self._run_migration(tmp_path, monkeypatch, v2)
        assert out == v2

    def test_missing_file_is_noop(self, tmp_path, monkeypatch):
        import backend.detection_registry as dmod
        monkeypatch.setattr(dmod, "ALGORITHMS_FILE", tmp_path / "algorithms.json")
        dmod._migrate_algorithms_v2()  # 不抛异常
        assert not (tmp_path / "algorithms.json").exists()
