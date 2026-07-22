"""模型管理 API 测试"""

import io
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
    return TestClient(app)


class TestListModels:
    def test_returns_models_with_usage(self, client):
        mock_mr = MagicMock()
        mock_mr.to_api_list.return_value = [
            {"key": "leak", "name": "漏液模型", "file": "leak.pt",
             "post_process": "yolo_box", "class_names": {"0": "leak"},
             "file_size": 100, "uploaded_at": "2026-07-21"}
        ]
        mock_reg = MagicMock()
        mock_reg.get_model_keys_in_use.return_value = {"leak"}
        with patch("backend.safety_detection.api.model_registry", mock_mr), \
             patch("backend.safety_detection.api.registry", mock_reg):
            resp = client.get("/models")
            assert resp.status_code == 200
            m = resp.json()["models"][0]
            assert m["key"] == "leak"
            assert m["used_by"] == 1


class TestUploadModel:
    def test_upload_pt_parses_metadata(self, client):
        mock_mr = MagicMock()
        mock_mr.save_model_file.return_value = MagicMock(name="leak.pt", stem="leak", suffix=".pt")
        mock_mr.add_model.return_value = "leak"
        mock_mr.get.return_value = {"post_process": "yolo_box", "class_names": {"0": "leak"}}
        with patch("backend.safety_detection.api.model_registry", mock_mr), \
             patch("backend.safety_detection.api.parse_model_metadata",
                   return_value={"post_process": "yolo_box", "class_names": {"0": "leak"}}):
            resp = client.post("/models/upload",
                               files={"file": ("leak.pt", io.BytesIO(b"fake"), "application/octet-stream")})
            assert resp.status_code == 200
            data = resp.json()
            assert data["key"] == "leak"
            assert data["class_names"] == {"0": "leak"}

    def test_upload_rejects_bad_extension(self, client):
        resp = client.post("/models/upload",
                           files={"file": ("x.exe", io.BytesIO(b"f"), "application/octet-stream")})
        assert resp.status_code == 400


class TestDeleteModel:
    def test_delete_referenced_model_returns_409(self, client):
        mock_mr = MagicMock()
        mock_mr.get.return_value = {"name": "M", "file": "m.pt"}
        mock_reg = MagicMock()
        mock_reg.get_model_keys_in_use.return_value = {"m"}
        with patch("backend.safety_detection.api.model_registry", mock_mr), \
             patch("backend.safety_detection.api.registry", mock_reg):
            resp = client.delete("/models/m")
            assert resp.status_code == 409

    def test_delete_unreferenced_succeeds(self, client):
        mock_mr = MagicMock()
        mock_mr.get.return_value = {"name": "M", "file": "m.pt"}
        mock_reg = MagicMock()
        mock_reg.get_model_keys_in_use.return_value = set()
        with patch("backend.safety_detection.api.model_registry", mock_mr), \
             patch("backend.safety_detection.api.registry", mock_reg):
            resp = client.delete("/models/m")
            assert resp.status_code == 200
