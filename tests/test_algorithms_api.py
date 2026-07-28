"""算法管理 API 测试"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.safety_detection.api import router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    app.state.camera_manager = MagicMock()
    app.state.camera_manager.get_camera_ids_with_type.return_value = []
    return TestClient(app)


class TestCreateAlgorithm:
    def test_create_with_model_key(self, client):
        mock_reg = MagicMock()
        mock_reg.add_type.return_value = "leak_abc123"
        mock_reg.get.return_value = {
            "label": "漏液-高灵敏", "color": "#facc15", "model_key": "leak",
            "model_path": "leak.pt", "post_process": "yolo_box",
            "classes": [0], "model_confidence": 0.5, "vlm_prompt": "",
            "inspection_label": "漏液", "defaults": {"threshold": 0.4},
        }
        with patch("backend.safety_detection.api.registry", mock_reg):
            resp = client.post("/algorithms", json={
                "label": "漏液-高灵敏", "color": "#facc15", "model_key": "leak",
                "classes": [0], "defaults": {"threshold": 0.4},
            })
            assert resp.status_code == 200
            assert resp.json()["key"] == "leak_abc123"
            # add_type 收到的 payload 含 model_key，不含 model_path
            call_arg = mock_reg.add_type.call_args[0][0]
            assert call_arg["model_key"] == "leak"
            assert "model_path" not in call_arg

    def test_create_unknown_model_returns_400(self, client):
        mock_reg = MagicMock()
        mock_reg.add_type.side_effect = ValueError("Unknown model: ghost")
        with patch("backend.safety_detection.api.registry", mock_reg):
            resp = client.post("/algorithms", json={"label": "X", "model_key": "ghost"})
            assert resp.status_code == 400


class TestLegacyUploadCompat:
    def test_legacy_upload_sets_model_key(self, client):
        """旧 /detector/types/{dtype}/model 上传改为复用模型注册表"""
        import io
        mock_reg = MagicMock()
        mock_reg.get.return_value = {"label": "明火"}
        mock_mr = MagicMock()
        saved = MagicMock()
        saved.name = "fire_smoke.pt"
        saved.stem = "fire_smoke"
        saved.suffix = ".pt"
        mock_mr.save_model_file.return_value = saved
        mock_mr.add_model.return_value = "fire_smoke"
        with patch("backend.safety_detection.api.registry", mock_reg), \
             patch("backend.safety_detection.api.model_registry", mock_mr), \
             patch("backend.safety_detection.api.parse_model_metadata", return_value={}):
            resp = client.post("/detector/types/fire/model",
                               files={"file": ("fire_smoke.pt", io.BytesIO(b"f"), "application/octet-stream")})
            assert resp.status_code == 200
            mock_reg.update_type.assert_called_once()
            updates = mock_reg.update_type.call_args[0][1]
            assert updates == {"model_key": "fire_smoke"}


class TestListAlgorithms:
    def test_list_response_includes_alarm_description(self, client):
        mock_reg = MagicMock()
        mock_reg.to_api_list.return_value = [
            {
                "key": "fire", "label": "明火", "color": "#ef4444",
                "model_key": "fire_smoke", "model_path": "fire_smoke.pt",
                "post_process": "yolo_box", "classes": [0, 1],
                "model_confidence": 0.5, "vlm_prompt": "",
                "inspection_label": "明火", "alarm_description": "检测到明火",
                "defaults": {},
            }
        ]
        with patch("backend.safety_detection.api.registry", mock_reg):
            resp = client.get("/algorithms")
            assert resp.status_code == 200
            algorithms = {a["key"]: a for a in resp.json()["algorithms"]}
            assert "alarm_description" in algorithms["fire"]
            assert algorithms["fire"]["alarm_description"] == "检测到明火"
