"""
config.py 注册表驱动改造测试
验证 DEFAULT_TYPE_CONFIG 和 display_detection_types 从注册表动态生成
"""
import json
import pytest


@pytest.fixture
def setup_registry(tmp_path, monkeypatch):
    """初始化测试注册表"""
    import backend.detection_registry as reg_mod
    import backend.model_registry as mreg_mod
    # Redirect all config paths to tmp_path so no real config files are read
    monkeypatch.setattr(reg_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(reg_mod, "REGISTRY_FILE", tmp_path / "detection_types.json")
    monkeypatch.setattr(reg_mod, "ALGORITHMS_FILE", tmp_path / "algorithms.json")
    monkeypatch.setattr(mreg_mod, "MODELS_FILE", tmp_path / "models.json")
    # Reset model registry state so migration starts clean
    mreg_mod.model_registry._models = {}
    reg_mod.registry.load()
    return reg_mod.registry


class TestGetDefaultTypeConfig:
    """get_default_type_config() 从注册表动态生成"""

    def test_returns_all_types(self, setup_registry):
        from backend.config import get_default_type_config
        dtc = get_default_type_config()
        assert set(dtc.keys()) == {"fire", "smoke", "uniform", "mask", "cigarette", "sleep"}

    def test_values_match_registry_defaults(self, setup_registry):
        from backend.config import get_default_type_config
        dtc = get_default_type_config()
        assert dtc["fire"]["threshold"] == 0.6
        assert dtc["fire"]["interval"] == 1
        assert dtc["fire"]["cooldown"] == 60
        assert dtc["sleep"]["interval"] == 60
        assert dtc["sleep"]["threshold"] == 0.7

    def test_includes_all_default_fields(self, setup_registry):
        from backend.config import get_default_type_config
        dtc = get_default_type_config()
        required_fields = {"enabled", "interval", "threshold", "consecutive_required", "cooldown", "use_vlm"}
        for dtype, cfg in dtc.items():
            for field in required_fields:
                assert field in cfg, f"{dtype} missing field: {field}"

    def test_excludes_structural_fields(self, setup_registry):
        """不包含 model_path 等结构性字段"""
        from backend.config import get_default_type_config
        dtc = get_default_type_config()
        structural = {"model_path", "post_process", "classes", "model_confidence"}
        for dtype, cfg in dtc.items():
            for field in structural:
                assert field not in cfg, f"{dtype} should not contain {field}"


class TestDefaultGlobalSettings:
    """display_detection_types 从注册表动态生成"""

    def test_display_types_matches_registry(self, setup_registry):
        from backend.config import get_default_global_settings
        settings = get_default_global_settings()
        ddt = settings["display_detection_types"]
        assert set(ddt.keys()) == {"fire", "smoke", "uniform", "mask", "cigarette", "sleep"}
        # 所有类型默认显示
        for dtype, enabled in ddt.items():
            assert enabled is True


class TestApplyCameraGlobals:
    """apply_camera_globals 使用注册表类型"""

    def test_fills_all_registry_types(self, setup_registry, tmp_path, monkeypatch):
        """空摄像头配置应填充注册表中所有类型的默认值"""
        monkeypatch.setattr("backend.config.CONFIG_DIR", tmp_path)
        from backend.config import apply_camera_globals
        cam = {"camera_id": "test", "source": "rtsp://test"}
        result = apply_camera_globals(cam)
        dt = result["detection_types"]
        assert set(dt.keys()) == {"fire", "smoke", "uniform", "mask", "cigarette", "sleep"}

    def test_preserves_camera_overrides(self, setup_registry, tmp_path, monkeypatch):
        """摄像头级配置覆盖注册表默认值"""
        monkeypatch.setattr("backend.config.CONFIG_DIR", tmp_path)
        from backend.config import apply_camera_globals
        cam = {
            "camera_id": "test",
            "detection_types": {
                "fire": {"enabled": True, "threshold": 0.99},
            },
        }
        result = apply_camera_globals(cam)
        assert result["detection_types"]["fire"]["enabled"] is True
        assert result["detection_types"]["fire"]["threshold"] == 0.99
        # 缺失字段从注册表默认值补全
        assert result["detection_types"]["fire"]["cooldown"] == 60


class TestLoadCameraConfigs:
    """load_camera_configs 迁移逻辑使用注册表默认值"""

    def test_migration_uses_registry_defaults(self, setup_registry, tmp_path, monkeypatch):
        """旧配置迁移时使用注册表默认值"""
        cameras_file = tmp_path / "cameras.json"
        old_cam = {
            "camera_id": "old",
            "source": "rtsp://old",
            # 没有 detection_types/algorithms 字段 → 应从注册表填充
        }
        cameras_file.write_text(
            json.dumps({"cameras": [old_cam]}, ensure_ascii=False),
            encoding="utf-8",
        )
        monkeypatch.setattr("backend.config.CAMERAS_CONFIG_FILE", cameras_file)
        from backend.config import load_camera_configs
        cameras = load_camera_configs()
        cam = cameras[0]
        assert "algorithms" in cam
        # 所有注册表类型都应存在
        assert set(cam["algorithms"].keys()) >= {"fire", "smoke", "uniform", "mask", "cigarette", "sleep"}
        # 注入段只保留 enabled 覆盖
        assert set(cam["algorithms"]["fire"].keys()) == {"enabled"}
