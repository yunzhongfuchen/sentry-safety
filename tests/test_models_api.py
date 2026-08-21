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
        mock_reg.get_model_usage_counts.return_value = {"leak": 1}
        with patch("backend.safety_detection.api.model_registry", mock_mr), \
             patch("backend.safety_detection.api.registry", mock_reg):
            resp = client.get("/models")
            assert resp.status_code == 200
            m = resp.json()["models"][0]
            assert m["key"] == "leak"
            assert m["used_by"] == 1

    def test_used_by_counts_multiple_referencing_algorithms(self, client):
        """模型被多个算法引用时 used_by 返回真实数量而非布尔 1"""
        mock_mr = MagicMock()
        mock_mr.to_api_list.return_value = [
            {"key": "ppe", "name": "PPE", "file": "ppe.pt",
             "post_process": "yolo_box", "class_names": {}, "file_size": 1,
             "uploaded_at": "2026-08-06"}
        ]
        mock_reg = MagicMock()
        mock_reg.get_model_usage_counts.return_value = {"ppe": 3}
        with patch("backend.safety_detection.api.model_registry", mock_mr), \
             patch("backend.safety_detection.api.registry", mock_reg):
            resp = client.get("/models")
            assert resp.status_code == 200
            assert resp.json()["models"][0]["used_by"] == 3


class TestUploadModel:
    def test_upload_pt_parses_metadata(self, client):
        mock_mr = MagicMock()
        mock_mr.save_model_file.return_value = MagicMock(name="leak.pt", stem="leak", suffix=".pt")
        mock_mr.add_model.return_value = "leak"
        mock_mr.get.return_value = {"post_process": "yolo_relation", "class_names": {"0": "leak"}}
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


class TestReplaceModel:
    def test_replace_model_file_updates_entry(self, client):
        mock_mr = MagicMock()
        mock_mr.get.side_effect = lambda k: {"name": "M", "file": "m.pt",
                                              "post_process": "yolo_box",
                                              "class_names": {"0": "old"}} if k == "m" else None
        mock_mr.replace_model_file.return_value = MagicMock(name="m_new.pt")
        with patch("backend.safety_detection.api.model_registry", mock_mr), \
             patch("backend.safety_detection.api.parse_model_metadata",
                   return_value={"post_process": "yolo_pose", "class_names": {"0": "person"}}):
            resp = client.post("/models/m/replace",
                               files={"file": ("m_new.pt", io.BytesIO(b"fake"), "application/octet-stream")})
            assert resp.status_code == 200
            data = resp.json()
            assert data["key"] == "m"
            mock_mr.replace_model_file.assert_called_once()

    def test_replace_unknown_model_returns_404(self, client):
        mock_mr = MagicMock()
        mock_mr.get.return_value = None
        with patch("backend.safety_detection.api.model_registry", mock_mr):
            resp = client.post("/models/missing/replace",
                               files={"file": ("x.pt", io.BytesIO(b"fake"), "application/octet-stream")})
            assert resp.status_code == 404

    def test_replace_rejects_bad_extension(self, client):
        mock_mr = MagicMock()
        mock_mr.get.return_value = {"name": "M", "file": "m.pt"}
        with patch("backend.safety_detection.api.model_registry", mock_mr):
            resp = client.post("/models/m/replace",
                               files={"file": ("x.exe", io.BytesIO(b"fake"), "application/octet-stream")})
            assert resp.status_code == 400
