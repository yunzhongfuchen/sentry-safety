"""understander.py 注册表改造测试"""

import pytest
from unittest.mock import patch, MagicMock


class TestBuildInspectionPromptRegistry:
    def test_uses_registry_inspection_label(self):
        from backend.understander import VideoUnderstander

        mock_registry = MagicMock()
        mock_registry.get.side_effect = lambda dtype: {
            "fire": {"inspection_label": "明火"},
            "smoke": {"inspection_label": "烟雾"},
        }.get(dtype)

        vu = VideoUnderstander.__new__(VideoUnderstander)

        with patch("backend.understander.registry", mock_registry):
            prompt = vu._build_inspection_prompt({"enabled_types": ["fire", "smoke"]})
            assert "明火" in prompt
            assert "烟雾" in prompt

    def test_unknown_type_uses_key_as_fallback(self):
        from backend.understander import VideoUnderstander

        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        vu = VideoUnderstander.__new__(VideoUnderstander)

        with patch("backend.understander.registry", mock_registry):
            prompt = vu._build_inspection_prompt({"enabled_types": ["new_type"]})
            assert "new_type" in prompt

    def test_new_type_with_inspection_label(self):
        from backend.understander import VideoUnderstander

        mock_registry = MagicMock()
        mock_registry.get.return_value = {"inspection_label": "未戴安全帽"}

        vu = VideoUnderstander.__new__(VideoUnderstander)

        with patch("backend.understander.registry", mock_registry):
            prompt = vu._build_inspection_prompt({"enabled_types": ["helmet"]})
            assert "未戴安全帽" in prompt
