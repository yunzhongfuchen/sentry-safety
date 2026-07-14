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
        # structural fields not exposed
        assert "model_path" not in fire_entry

    def test_validate_warns_missing_model(self, tmp_path, monkeypatch):
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
