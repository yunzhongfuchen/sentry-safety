"""detector types API 端点测试"""

import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.safety_detection.api import router
from fastapi import FastAPI


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    app.state.camera_manager = MagicMock()
    app.state.camera_manager.get_camera_ids_with_type.return_value = []
    return TestClient(app)


class TestListDetectionTypes:
    def test_returns_types_list(self, client):
        mock_registry = MagicMock()
        mock_registry.to_api_list.return_value = [
            {"key": "fire", "label": "明火", "color": "#ef4444", "icon": "flame",
             "post_process": "yolo_box", "defaults": {"enabled": False}},
        ]

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.get("/detector/types")
            assert resp.status_code == 200
            data = resp.json()
            assert "types" in data
            assert len(data["types"]) == 1
            assert data["types"][0]["key"] == "fire"


class TestGetDetectionType:
    def test_existing_type(self, client):
        mock_registry = MagicMock()
        mock_registry.get.return_value = {
            "label": "明火", "color": "#ef4444", "icon": "flame",
            "post_process": "yolo_box", "defaults": {"enabled": False},
        }

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.get("/detector/types/fire")
            assert resp.status_code == 200
            assert resp.json()["key"] == "fire"

    def test_unknown_type_404(self, client):
        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.get("/detector/types/nonexistent")
            assert resp.status_code == 404


class TestUpdateDetectionType:
    def test_update_defaults(self, client):
        mock_registry = MagicMock()
        mock_registry.get.return_value = {"defaults": {"enabled": False, "threshold": 0.5}, "label": "明火"}
        mock_registry.get_defaults.return_value = {"enabled": True, "threshold": 0.8}

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.put("/detector/types/fire", json={"enabled": True, "threshold": 0.8})
            assert resp.status_code == 200
            assert resp.json()["success"] is True
            mock_registry.update_defaults.assert_called_once()

    def test_update_structural_fields(self, client):
        mock_registry = MagicMock()
        mock_registry.get.return_value = {"defaults": {"enabled": False}, "label": "明火"}

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.put("/detector/types/fire", json={"model_path": "evil.pt"})
            assert resp.status_code == 200
            mock_registry.update_type.assert_called_once()

    def test_unknown_type_404(self, client):
        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.put("/detector/types/nope", json={"enabled": True})
            assert resp.status_code == 404

    def test_invalid_values_return_400(self, client):
        mock_registry = MagicMock()
        mock_registry.get.return_value = {"defaults": {"enabled": False, "threshold": 0.5}, "label": "明火"}

        with patch("backend.safety_detection.api.registry", mock_registry):
            invalid_cases = [
                {"enabled": "yes"},
                {"interval": 0},
                {"threshold": -0.1},
                {"threshold": 1.5},
                {"consecutive_required": 0},
                {"cooldown": -1},
                {"use_vlm": "true"},
                {"min_box_count": -1},
                {"max_box_count": -1},
                {"min_box_count": True},
            ]
            for payload in invalid_cases:
                resp = client.put("/detector/types/fire", json=payload)
                assert resp.status_code == 400, f"expected 400 for {payload}, got {resp.status_code}"
                mock_registry.update_defaults.reset_mock()


class TestCreateDetectionType:
    def test_create_type_returns_key(self, client):
        payload = {"label": "新类型", "color": "#123456", "model_path": "new.pt", "post_process": "yolo_box"}
        mock_registry = MagicMock()
        mock_registry.add_type.return_value = "xin_lei_xing_123abc"
        mock_registry.get.return_value = {
            "label": "新类型", "color": "#123456", "icon": "",
            "post_process": "yolo_box", "defaults": {},
        }

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.post("/detector/types", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert "key" in data
            assert data["label"] == "新类型"

    def test_create_type_duplicate_label_400(self, client):
        payload = {"label": "明火", "color": "#123456", "model_path": "x.pt", "post_process": "yolo_box"}
        mock_registry = MagicMock()
        mock_registry.add_type.side_effect = ValueError("label '明火' already exists")

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.post("/detector/types", json=payload)
            assert resp.status_code == 400

    def test_create_missing_label_400(self, client):
        mock_registry = MagicMock()
        mock_registry.add_type.side_effect = ValueError("label is required")

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.post("/detector/types", json={"color": "#123456"})
            assert resp.status_code == 400


class TestDeleteDetectionType:
    def test_delete_type_success(self, client):
        mock_registry = MagicMock()
        mock_registry.add_type.return_value = "dai_shan_chu_123abc"
        mock_registry.get.return_value = {"label": "待删除"}

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.post("/detector/types", json={"label": "待删除", "color": "#000000", "model_path": "d.pt", "post_process": "yolo_box"})
            key = resp.json()["key"]
            client.app.state.camera_manager.get_camera_ids_with_type.return_value = []
            resp = client.delete(f"/detector/types/{key}")
            assert resp.status_code == 200
            mock_registry.delete_type.assert_called_once()

    def test_delete_referenced_type_409(self, client):
        mock_registry = MagicMock()
        mock_registry.get.return_value = {"label": "明火"}
        client.app.state.camera_manager.get_camera_ids_with_type.return_value = ["cam1"]

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.delete("/detector/types/fire")
            assert resp.status_code == 409

    def test_delete_unknown_type_404(self, client):
        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.delete("/detector/types/nonexistent")
            assert resp.status_code == 404


class TestUploadModel:
    def test_upload_model_success(self, client, tmp_path):
        mock_registry = MagicMock()
        mock_registry.get.return_value = {"label": "明火", "model_path": "old.pt", "npu_model_path": "old.rknn"}

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.post(
                "/detector/types/fire/model",
                files={"file": ("test_model.pt", io.BytesIO(b"fake"), "application/octet-stream")}
            )
            assert resp.status_code == 200
            assert resp.json()["model_path"] == "test_model.pt"
            mock_registry.save_model.assert_called_once_with("test_model.pt", b"fake")
            mock_registry.update_type.assert_called_with("fire", {"model_path": "test_model.pt"})

    def test_upload_rknn_model_success(self, client, tmp_path):
        mock_registry = MagicMock()
        mock_registry.get.return_value = {"label": "明火", "model_path": "old.pt", "npu_model_path": "old.rknn"}

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.post(
                "/detector/types/fire/model",
                files={"file": ("test_model.rknn", io.BytesIO(b"rknn"), "application/octet-stream")}
            )
            assert resp.status_code == 200
            assert resp.json()["model_path"] == "test_model.rknn"
            mock_registry.save_model.assert_called_once_with("test_model.rknn", b"rknn")
            mock_registry.update_type.assert_called_with("fire", {"npu_model_path": "test_model.rknn"})

    def test_upload_invalid_extension_400(self, client):
        mock_registry = MagicMock()
        mock_registry.get.return_value = {"label": "明火"}

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.post(
                "/detector/types/fire/model",
                files={"file": ("model.onnx", io.BytesIO(b"fake"), "application/octet-stream")}
            )
            assert resp.status_code == 400
            mock_registry.save_model.assert_not_called()

    def test_upload_unknown_type_404(self, client):
        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.post(
                "/detector/types/nonexistent/model",
                files={"file": ("model.pt", io.BytesIO(b"fake"), "application/octet-stream")}
            )
            assert resp.status_code == 404
            mock_registry.save_model.assert_not_called()
