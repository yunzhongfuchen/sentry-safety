import json

import pytest


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
    """注册表核心功能测试"""

    def _make_registry(self, tmp_path, monkeypatch, data=None, models=None):
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
        r = dmod.DetectionTypeRegistry()
        r.load()
        return r, mr

    def test_load_generates_file_when_missing(self, tmp_path, monkeypatch):
        r, _ = self._make_registry(tmp_path, monkeypatch)
        assert (tmp_path / "algorithms.json").exists()
        assert "fire" in r.all_types()
        assert "sleep" in r.all_types()

    def test_load_preserves_existing_and_backfills(self, tmp_path, monkeypatch):
        partial = {"fire": {"label": "自定义火焰", "color": "#ff0000", "model_path": "custom.pt",
                            "post_process": "yolo_box", "classes": [0], "model_confidence": 0.5,
                            "vlm_prompt": "fire_review", "inspection_label": "火",
                            "defaults": {"enabled": True, "threshold": 0.9}}}
        r, _ = self._make_registry(tmp_path, monkeypatch, data=partial)
        fire = r.get("fire")
        assert fire["label"] == "自定义火焰"
        assert fire["defaults"]["enabled"] is True
        assert fire["defaults"]["threshold"] == 0.9
        # backfilled fields
        assert "cooldown" in fire["defaults"]
        assert "min_box_count" in fire["defaults"]

    def test_get_unknown_returns_none(self, tmp_path, monkeypatch):
        r, _ = self._make_registry(tmp_path, monkeypatch)
        assert r.get("unknown_type") is None

    def test_all_types_returns_six(self, tmp_path, monkeypatch):
        r, _ = self._make_registry(tmp_path, monkeypatch)
        assert len(r.all_types()) == 6
        assert set(r.all_types()) == {"fire", "smoke", "uniform", "mask", "cigarette", "sleep"}

    def test_get_types_by_model_shared(self, tmp_path, monkeypatch):
        r, mr = self._make_registry(tmp_path, monkeypatch, models=[
            {"file": "fire_smoke.pt", "name": "烟火模型", "post_process": "yolo_box",
             "class_names": {"0": "fire", "1": "smoke"}}])
        mkey = mr.all_models()[0]
        k1 = r.add_type({"label": "火", "model_key": mkey})
        k2 = r.add_type({"label": "烟", "model_key": mkey})
        assert set(r.get_types_by_model(mkey)) == {k1, k2}

    def test_get_types_by_model_unique(self, tmp_path, monkeypatch):
        r, mr = self._make_registry(tmp_path, monkeypatch, models=[
            {"file": "mask.pt", "name": "口罩模型", "post_process": "yolo_box",
             "class_names": {"0": "mask"}}])
        mkey = mr.all_models()[0]
        key = r.add_type({"label": "口罩佩戴", "model_key": mkey})
        assert r.get_types_by_model(mkey) == [key]

    def test_get_color_bgr(self, tmp_path, monkeypatch):
        r, _ = self._make_registry(tmp_path, monkeypatch)
        assert r.get_color_bgr("fire") == (68, 68, 239)

    def test_get_defaults(self, tmp_path, monkeypatch):
        r, _ = self._make_registry(tmp_path, monkeypatch)
        d = r.get_defaults("sleep")
        assert d["interval"] == 60
        assert d["threshold"] == 0.7
        assert d["min_box_count"] == 1

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
        assert "defaults" in fire_entry
        # structural fields exposed so edit dialog can round-trip them
        # 默认类型经迁移已关联模型（fire/smoke 共享 fire_smoke.pt）
        mkey = next(k for k in mr.all_models() if mr.get(k)["file"] == "fire_smoke.pt")
        assert "model_key" in fire_entry
        assert fire_entry["model_key"] == mkey
        assert fire_entry["model_path"] == "fire_smoke.pt"
        assert "npu_model_path" not in fire_entry
        assert fire_entry["classes"] == [0]
        assert fire_entry["model_confidence"] == 0.5
        assert "明火" in fire_entry["vlm_prompt"]
        assert fire_entry["inspection_label"] == "明火"

    def test_validate_warns_missing_model(self, tmp_path, monkeypatch):
        r, mr = self._make_registry(tmp_path, monkeypatch, models=[
            {"file": "missing.pt", "name": "缺失模型", "post_process": "yolo_box",
             "class_names": {}}])
        mkey = mr.all_models()[0]
        r.add_type({"label": "缺模型算法", "model_key": mkey})
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
        mkey_before = r.get("fire").get("model_key")
        r.update_defaults("fire", {"model_key": "hacked", "threshold": 0.1})
        assert r.get("fire").get("model_key") == mkey_before
        assert r.get_defaults("fire")["threshold"] == 0.1

    def test_load_backfills_custom_type_defaults(self, tmp_path, monkeypatch):
        custom = {"custom_type": {"label": "自定义", "color": "#123456", "model_path": "custom.pt",
                                  "post_process": "yolo_box", "classes": [0], "model_confidence": 0.5,
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
        assert d["min_box_count"] == 1
        assert d["max_box_count"] is None

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

    def test_get_types_by_model_missing_model_path(self, tmp_path, monkeypatch):
        custom = {"custom_no_model": {"label": "自定义"}}
        r, _ = self._make_registry(tmp_path, monkeypatch, data=custom)
        # should not raise
        assert r.get_types_by_model("anything.pt") == []
        assert r.get_types_by_model("") == []

    def test_to_api_list_missing_structural_fields(self, tmp_path, monkeypatch):
        custom = {"custom_sparse": {"defaults": {"enabled": True}}}
        r, _ = self._make_registry(tmp_path, monkeypatch, data=custom)
        entry = next(e for e in r.to_api_list() if e["key"] == "custom_sparse")
        assert entry["label"] == "custom_sparse"
        assert entry["color"] == "#888888"
        assert entry["post_process"] == "yolo_box"
        assert entry["defaults"]["enabled"] is True
        assert entry["defaults"]["interval"] == 1
        assert entry["defaults"]["threshold"] == 0.5

    def test_load_malformed_json_regenerates_defaults(self, tmp_path, monkeypatch):
        (tmp_path / "algorithms.json").write_text("not json at all", encoding="utf-8")
        import backend.detection_registry as mod
        monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(mod, "ALGORITHMS_FILE", tmp_path / "algorithms.json")
        r = mod.DetectionTypeRegistry()
        r.load()
        assert set(r.all_types()) == {"fire", "smoke", "uniform", "mask", "cigarette", "sleep"}
        # corrupted file overwritten
        with open(tmp_path / "algorithms.json", "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert "fire" in saved


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
    key = r.add_type({"label": "测试类型", "color": "#ff0000", "model_key": mkey})
    assert key is not None
    assert r.get(key)["label"] == "测试类型"


def test_add_type_duplicate_label_raises(tmp_path, monkeypatch):
    r, mr, mkey = _make_registry_with_model(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="already exists"):
        r.add_type({"label": "明火", "color": "#ff0000", "model_key": mkey})


def test_delete_type_removes_and_saves(tmp_path, monkeypatch):
    r, mr, mkey = _make_registry_with_model(tmp_path, monkeypatch)
    key = r.add_type({"label": "临时类型", "color": "#00ff00", "model_key": mkey})
    assert r.get(key) is not None
    r.delete_type(key)
    assert r.get(key) is None


def test_update_type_structural_fields(tmp_path, monkeypatch):
    r, mr, mkey = _make_registry_with_model(tmp_path, monkeypatch)
    mkey2 = mr.add_model(file="new.pt", name="新模型", post_process="yolo_pose",
                         class_names={"0": "y"})
    key = r.add_type({"label": "旧名称", "color": "#111111", "model_key": mkey})
    r.update_type(key, {"label": "新名称", "color": "#222222", "model_key": mkey2})
    td = r.get(key)
    assert td["label"] == "新名称"
    assert td["color"] == "#222222"
    assert td["model_key"] == mkey2
    assert td["model_path"] == "new.pt"
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
        key = r.add_type({"label": "漏液", "color": "#facc15", "model_key": mkey,
                          "classes": [0], "defaults": {"threshold": 0.5}})
        td = r.get(key)
        assert td["model_key"] == mkey
        assert td["model_path"] == "leak.pt"
        assert td["post_process"] == "yolo_box"

    def test_add_type_unknown_model_key_raises(self, both):
        r, mr, mkey = both
        with pytest.raises(ValueError, match="Unknown model"):
            r.add_type({"label": "X", "color": "#fff", "model_key": "nonexistent"})

    def test_add_type_copies_post_process_from_model(self, both):
        r, mr, mkey = both
        mr.update_model(mkey, {"post_process": "yolo_pose"})
        key = r.add_type({"label": "姿态类", "color": "#fff", "model_key": mkey})
        assert r.get(key)["post_process"] == "yolo_pose"

    def test_update_type_model_key_validated(self, both):
        r, mr, mkey = both
        key = r.add_type({"label": "漏液", "color": "#facc15", "model_key": mkey})
        with pytest.raises(ValueError, match="Unknown model"):
            r.update_type(key, {"model_key": "ghost"})

    def test_to_api_list_has_model_key_and_path(self, both):
        r, mr, mkey = both
        key = r.add_type({"label": "漏液", "color": "#facc15", "model_key": mkey})
        item = [t for t in r.to_api_list() if t["key"] == key][0]
        assert item["model_key"] == mkey
        assert item["model_path"] == "leak.pt"

    def test_get_model_keys_in_use(self, both):
        r, mr, mkey = both
        assert mkey not in r.get_model_keys_in_use()
        r.add_type({"label": "漏液", "color": "#facc15", "model_key": mkey})
        assert mkey in r.get_model_keys_in_use()

    def test_model_path_none_when_model_deleted(self, both):
        r, mr, mkey = both
        key = r.add_type({"label": "漏液", "color": "#facc15", "model_key": mkey})
        mr.delete_model(mkey)
        assert r.get(key)["model_path"] is None


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
        # 算法 key 不变，model_key 指向同一模型
        assert r.get("fire")["model_key"] == mkey
        assert r.get("smoke")["model_key"] == mkey
        assert r.get("fire")["defaults"]["threshold"] == 0.6
        # 旧文件改名 .bak
        assert not (tmp_path / "detection_types.json").exists()
        assert (tmp_path / "detection_types.json.bak").exists()

    def test_fresh_install_migrates_builtin_defaults(self, tmp_path, monkeypatch):
        """全新安装：无 detection_types.json 时从内置默认值迁移（不生成 .bak）"""
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
        assert dmod.migrate_legacy_registry() is True
        # 内置 6 类型去重后 5 个模型（fire/smoke 共享 fire_smoke.pt）
        assert len(mr.all_models()) == 5
        saved = json.loads((tmp_path / "algorithms.json").read_text(encoding="utf-8"))
        assert "model_path" not in saved["fire"]
        assert saved["fire"]["model_key"] == saved["smoke"]["model_key"]
        # 无旧文件 → 不生成 .bak
        assert not (tmp_path / "detection_types.json.bak").exists()
        # load() 后算法经 model_key 解析出 model_path
        r = dmod.DetectionTypeRegistry()
        r.load()
        assert r.get("fire")["model_path"] == "fire_smoke.pt"
        assert r.get("fire")["model_key"] == saved["fire"]["model_key"]
        # 幂等：algorithms.json 已存在 → 不再迁移
        assert dmod.migrate_legacy_registry() is False
