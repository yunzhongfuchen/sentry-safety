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

    def _make_registry(self, tmp_path, monkeypatch, data=None):
        import backend.detection_registry as mod
        monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(mod, "REGISTRY_FILE", tmp_path / "detection_types.json")
        r = mod.DetectionTypeRegistry()
        if data is not None:
            (tmp_path / "detection_types.json").write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
        r.load()
        return r

    def test_load_generates_file_when_missing(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        assert (tmp_path / "detection_types.json").exists()
        assert "fire" in r.all_types()
        assert "sleep" in r.all_types()

    def test_load_preserves_existing_and_backfills(self, tmp_path, monkeypatch):
        partial = {"fire": {"label": "自定义火焰", "color": "#ff0000", "model_path": "custom.pt",
                            "post_process": "yolo_box", "classes": [0], "model_confidence": 0.5,
                            "vlm_prompt_key": "fire_review", "inspection_label": "火",
                            "defaults": {"enabled": True, "threshold": 0.9}}}
        r = self._make_registry(tmp_path, monkeypatch, data=partial)
        fire = r.get("fire")
        assert fire["label"] == "自定义火焰"
        assert fire["defaults"]["enabled"] is True
        assert fire["defaults"]["threshold"] == 0.9
        # backfilled fields
        assert "cooldown" in fire["defaults"]
        assert "min_box_count" in fire["defaults"]

    def test_get_unknown_returns_none(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        assert r.get("unknown_type") is None

    def test_all_types_returns_six(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        assert len(r.all_types()) == 6
        assert set(r.all_types()) == {"fire", "smoke", "uniform", "mask", "cigarette", "sleep"}

    def test_get_types_by_model_shared(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        shared = r.get_types_by_model("fire_smoke.pt")
        assert set(shared) == {"fire", "smoke"}

    def test_get_types_by_model_unique(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        assert r.get_types_by_model("mask.pt") == ["mask"]

    def test_get_color_bgr(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        assert r.get_color_bgr("fire") == (68, 68, 239)

    def test_get_defaults(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        d = r.get_defaults("sleep")
        assert d["interval"] == 60
        assert d["threshold"] == 0.7
        assert d["min_box_count"] == 1

    def test_merge_camera_config(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        merged = r.merge_camera_config("fire", {"enabled": True, "threshold": 0.8,
                                                  "roi": [[0.1, 0.1], [0.9, 0.9]]})
        assert merged["enabled"] is True
        assert merged["threshold"] == 0.8
        assert merged["roi"] == [[0.1, 0.1], [0.9, 0.9]]
        # inherited defaults
        assert merged["cooldown"] == 60
        assert merged["consecutive_required"] == 3

    def test_to_api_list(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        api = r.to_api_list()
        assert len(api) == 6
        fire_entry = next(e for e in api if e["key"] == "fire")
        assert fire_entry["label"] == "明火"
        assert fire_entry["color"] == "#ef4444"
        assert "defaults" in fire_entry
        # structural fields exposed so edit dialog can round-trip them
        assert fire_entry["model_path"] == "fire_smoke.pt"
        assert "npu_model_path" not in fire_entry
        assert fire_entry["classes"] == [0]
        assert fire_entry["model_confidence"] == 0.5
        assert fire_entry["vlm_prompt_key"] == "fire_review"
        assert fire_entry["inspection_label"] == "明火"

    def test_validate_warns_missing_model(self, tmp_path, monkeypatch):
        import backend.detection_registry as mod
        monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
        r = self._make_registry(tmp_path, monkeypatch)
        warnings = r.validate()
        # models don't exist in test env, so should have warnings
        assert len(warnings) > 0
        assert any("fire_smoke.pt" in w for w in warnings)

    def test_update_defaults_persists(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        r.update_defaults("fire", {"threshold": 0.99, "cooldown": 120})
        assert r.get_defaults("fire")["threshold"] == 0.99
        assert r.get_defaults("fire")["cooldown"] == 120
        # verify persistence
        with open(tmp_path / "detection_types.json", "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["fire"]["defaults"]["threshold"] == 0.99

    def test_update_defaults_ignores_structural_fields(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        r.update_defaults("fire", {"model_path": "hacked.pt", "threshold": 0.1})
        assert r.get("fire")["model_path"] != "hacked.pt"
        assert r.get_defaults("fire")["threshold"] == 0.1

    def test_load_backfills_custom_type_defaults(self, tmp_path, monkeypatch):
        custom = {"custom_type": {"label": "自定义", "color": "#123456", "model_path": "custom.pt",
                                  "post_process": "yolo_box", "classes": [0], "model_confidence": 0.5,
                                  "vlm_prompt_key": "custom_review", "inspection_label": "自定义"}}
        r = self._make_registry(tmp_path, monkeypatch, data=custom)
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
        r = self._make_registry(tmp_path, monkeypatch)
        assert r.get_color_bgr("unknown") == (0, 255, 0)

    def test_get_defaults_unknown_returns_empty(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        assert r.get_defaults("unknown") == {}

    def test_merge_camera_config_unknown_returns_overrides(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        overrides = {"enabled": True, "threshold": 0.8}
        assert r.merge_camera_config("unknown", overrides) == overrides

    def test_update_defaults_unknown_is_noop(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        r.update_defaults("unknown", {"threshold": 0.1})
        assert r.get("unknown") is None

    def test_get_types_by_model_missing_model_path(self, tmp_path, monkeypatch):
        custom = {"custom_no_model": {"label": "自定义"}}
        r = self._make_registry(tmp_path, monkeypatch, data=custom)
        # should not raise
        assert r.get_types_by_model("anything.pt") == []
        assert r.get_types_by_model("") == []

    def test_to_api_list_missing_structural_fields(self, tmp_path, monkeypatch):
        custom = {"custom_sparse": {"defaults": {"enabled": True}}}
        r = self._make_registry(tmp_path, monkeypatch, data=custom)
        entry = next(e for e in r.to_api_list() if e["key"] == "custom_sparse")
        assert entry["label"] == "custom_sparse"
        assert entry["color"] == "#888888"
        assert entry["icon"] == ""
        assert entry["post_process"] == "yolo_box"
        assert entry["defaults"]["enabled"] is True
        assert entry["defaults"]["interval"] == 1
        assert entry["defaults"]["threshold"] == 0.5

    def test_load_malformed_json_regenerates_defaults(self, tmp_path, monkeypatch):
        (tmp_path / "detection_types.json").write_text("not json at all", encoding="utf-8")
        import backend.detection_registry as mod
        monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(mod, "REGISTRY_FILE", tmp_path / "detection_types.json")
        r = mod.DetectionTypeRegistry()
        r.load()
        assert set(r.all_types()) == {"fire", "smoke", "uniform", "mask", "cigarette", "sleep"}
        # corrupted file overwritten
        with open(tmp_path / "detection_types.json", "r", encoding="utf-8") as f:
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


def test_add_type_generates_key_and_saves(tmp_path, monkeypatch):
    import backend.detection_registry as mod
    monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(mod, "REGISTRY_FILE", tmp_path / "detection_types.json")
    r = mod.DetectionTypeRegistry()
    r.load()
    key = r.add_type({"label": "测试类型", "color": "#ff0000", "model_path": "test.pt", "post_process": "yolo_box"})
    assert key is not None
    assert r.get(key)["label"] == "测试类型"


def test_add_type_duplicate_label_raises(tmp_path, monkeypatch):
    import backend.detection_registry as mod
    monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(mod, "REGISTRY_FILE", tmp_path / "detection_types.json")
    r = mod.DetectionTypeRegistry()
    r.load()
    with pytest.raises(ValueError, match="already exists"):
        r.add_type({"label": "明火", "color": "#ff0000", "model_path": "x.pt", "post_process": "yolo_box"})


def test_delete_type_removes_and_saves(tmp_path, monkeypatch):
    import backend.detection_registry as mod
    monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(mod, "REGISTRY_FILE", tmp_path / "detection_types.json")
    r = mod.DetectionTypeRegistry()
    r.load()
    key = r.add_type({"label": "临时类型", "color": "#00ff00", "model_path": "tmp.pt", "post_process": "yolo_box"})
    assert r.get(key) is not None
    r.delete_type(key)
    assert r.get(key) is None


def test_update_type_structural_fields(tmp_path, monkeypatch):
    import backend.detection_registry as mod
    monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(mod, "REGISTRY_FILE", tmp_path / "detection_types.json")
    r = mod.DetectionTypeRegistry()
    r.load()
    key = r.add_type({"label": "旧名称", "color": "#111111", "model_path": "old.pt", "post_process": "yolo_box"})
    r.update_type(key, {"label": "新名称", "color": "#222222", "model_path": "new.pt"})
    td = r.get(key)
    assert td["label"] == "新名称"
    assert td["color"] == "#222222"
    assert td["model_path"] == "new.pt"


def test_save_model_writes_file(tmp_path, monkeypatch):
    import backend.detection_registry as mod
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    r = mod.DetectionTypeRegistry()
    content = b"fake model content"
    path = r.save_model("test_model.pt", content)
    assert path.exists()
    assert path.read_bytes() == content
    assert path.parent.name == "weights"


def test_save_model_sanitizes_path_traversal(tmp_path, monkeypatch):
    """save_model('../evil.pt') 必须写入 weights/evil.pt，不能逃逸到 weights/ 之外"""
    import backend.detection_registry as mod
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    r = mod.DetectionTypeRegistry()
    content = b"evil payload"
    path = r.save_model("../evil.pt", content)
    assert path.name == "evil.pt"
    assert path.parent.name == "weights"
    assert path.parent.parent == tmp_path
    assert path.exists()
    assert path.read_bytes() == content
    # 确认没有文件被写到 weights/ 之外
    assert not (tmp_path / "evil.pt").exists()
