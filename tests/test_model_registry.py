"""模型注册表测试"""

import pytest
import backend.model_registry as mod


@pytest.fixture
def reg(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(mod, "MODELS_FILE", tmp_path / "models.json")
    monkeypatch.setattr(mod, "WEIGHTS_DIR", tmp_path / "weights")
    r = mod.ModelRegistry()
    r.load()
    return r


class TestAddModel:
    def test_add_model_generates_key_from_filename(self, reg):
        key = reg.add_model(file="fire_smoke.pt", name="火焰烟雾", post_process="yolo_box", class_names={"0": "fire", "1": "smoke"})
        assert key == "fire_smoke"
        m = reg.get(key)
        assert m["file"] == "fire_smoke.pt"
        assert m["class_names"] == {"0": "fire", "1": "smoke"}

    def test_add_model_duplicate_filename_gets_suffix(self, reg):
        reg.add_model(file="leak.pt", name="漏液", post_process="yolo_box", class_names={})
        key2 = reg.add_model(file="leak.pt", name="漏液v2", post_process="yolo_box", class_names={})
        assert key2 == "leak_1"

    def test_persisted_to_file(self, reg, tmp_path):
        reg.add_model(file="a.pt", name="A", post_process="yolo_box", class_names={})
        r2 = mod.ModelRegistry()
        r2.load()
        assert r2.get("a") is not None


class TestUpdateDelete:
    def test_update_model_metadata(self, reg):
        key = reg.add_model(file="m.rknn", name="旧名", post_process="yolo_box", class_names={})
        reg.update_model(key, {"name": "新名", "class_names": {"0": "leak"}})
        assert reg.get(key)["name"] == "新名"
        assert reg.get(key)["class_names"] == {"0": "leak"}

    def test_delete_model(self, reg):
        key = reg.add_model(file="d.pt", name="D", post_process="yolo_box", class_names={})
        reg.delete_model(key)
        assert reg.get(key) is None

    def test_delete_unknown_raises(self, reg):
        with pytest.raises(KeyError):
            reg.delete_model("nonexistent")


class TestFileOps:
    def test_save_model_file_strips_path(self, reg, tmp_path):
        p = reg.save_model_file("../evil.pt", b"data")
        assert p.name == "evil.pt"
        assert p.parent == tmp_path / "weights"

    def test_file_exists_and_resolve(self, reg, tmp_path):
        key = reg.add_model(file="x.pt", name="X", post_process="yolo_box", class_names={})
        assert reg.file_exists(key) is False
        (tmp_path / "weights").mkdir(parents=True, exist_ok=True)
        (tmp_path / "weights" / "x.pt").write_bytes(b"fake")
        assert reg.file_exists(key) is True
        assert reg.resolve_file(key).endswith("x.pt")


class TestParseMetadata:
    def test_parse_failure_returns_empty(self, reg, tmp_path):
        bad = tmp_path / "bad.pt"
        bad.write_bytes(b"not a model")
        assert mod.parse_model_metadata(bad) == {}
