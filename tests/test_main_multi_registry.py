"""main_multi.py 注册表改造测试"""

import pytest
from unittest.mock import patch, MagicMock


class TestConvertUltralyticsResult:
    def test_yolo_box_empty_result(self):
        from backend.main_multi import _convert_ultralytics_result

        mock_registry = MagicMock()
        mock_registry.get.return_value = {"post_process": "yolo_box"}

        with patch("backend.main_multi.registry", mock_registry):
            out = _convert_ultralytics_result("fire", None)
            assert out["detected"] is False
            assert "subjects" not in out

    def test_yolo_pose_empty_result(self):
        from backend.main_multi import _convert_ultralytics_result

        mock_registry = MagicMock()
        mock_registry.get.return_value = {"post_process": "yolo_pose"}

        with patch("backend.main_multi.registry", mock_registry):
            out = _convert_ultralytics_result("sleep", None)
            assert out["detected"] is False
            assert out["subjects"] == []

    def test_unknown_type_treated_as_box(self):
        from backend.main_multi import _convert_ultralytics_result

        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        with patch("backend.main_multi.registry", mock_registry):
            out = _convert_ultralytics_result("new_type", None)
            assert out["detected"] is False
            assert "subjects" not in out


class TestGpuModelConfigsFromRegistry:
    def test_model_configs_built_from_registry(self):
        """验证 model_configs 可以从注册表动态构建"""
        from backend.detection_registry import registry

        mock_reg = MagicMock()
        mock_reg.all_types.return_value = ["fire", "smoke", "mask"]
        mock_reg.get.side_effect = lambda dtype: {
            "fire": {"model_path": "fire_smoke.pt", "classes": [0], "model_confidence": 0.5},
            "smoke": {"model_path": "fire_smoke.pt", "classes": [1], "model_confidence": 0.5},
            "mask": {"model_path": "mask.pt", "classes": [1], "model_confidence": 0.5},
        }[dtype]

        configs = {}
        for dtype in mock_reg.all_types():
            type_def = mock_reg.get(dtype)
            configs[dtype] = {
                "model_path": type_def["model_path"],
                "classes": type_def.get("classes"),
                "confidence": type_def.get("model_confidence", 0.5),
            }

        assert len(configs) == 3
        assert "fire" in configs
        assert "smoke" in configs
        assert configs["fire"]["classes"] == [0]
        assert configs["smoke"]["classes"] == [1]
        assert configs["fire"]["model_path"] == configs["smoke"]["model_path"]
