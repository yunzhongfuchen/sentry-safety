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
            {"key": "fire", "label": "明火", "color": "#ef4444",
             "post_process": "yolo_relation", "defaults": {"enabled": False}},
        ]

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.get("/detector/types")
            assert resp.status_code == 200
            data = resp.json()
            assert "types" in data
            assert len(data["types"]) == 1
            assert data["types"][0]["key"] == "fire"

    def test_list_includes_structural_fields(self, client):
        """Verify to_api_list returns all structural fields needed by the edit dialog."""
        from backend.detection_registry import DetectionTypeRegistry
        reg = DetectionTypeRegistry()
        reg._types = {
            "fire": {
                "label": "明火", "color": "#ef4444",
                "post_process": "yolo_relation",
                "models": [{"model_key": "fire_smoke", "model_confidence": 0.5}],
                "rule": {"groups": [{"conditions": [{"left": {"model_key": "fire_smoke", "classes": [0]}, "op": "exists"}]}]},
                "vlm_prompt": "fire_review", "inspection_label": "明火",
                "defaults": {"enabled": False},
            }
        }
        mock_mr = MagicMock()
        mock_mr.get.return_value = {"file": "fire_smoke.pt", "post_process": "yolo_box", "class_names": {"0": "fire"}}
        with patch("backend.detection_registry.model_registry", mock_mr):
            result = reg.to_api_list()
        assert len(result) == 1
        t = result[0]
        assert t["key"] == "fire"
        assert t["models"][0]["model_key"] == "fire_smoke"
        assert t["models"][0]["model_path"] == "fire_smoke.pt"
        assert t["models"][0]["class_names"] == {"0": "fire"}
        assert t["post_process"] == "yolo_relation"
        assert t["vlm_prompt"] == "fire_review"
        assert t["inspection_label"] == "明火"


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

    def test_get_includes_structural_fields(self, client):
        """GET /detector/types/{dtype} must return all structural fields so edit-save does not wipe them."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = {
            "label": "明火", "color": "#ef4444",
            "post_process": "yolo_relation",
            "models": [{"model_key": "fire_smoke", "model_confidence": 0.5}],
            "rule": {"groups": [{"conditions": [
                {"left": {"model_key": "fire_smoke", "classes": [0]}, "op": "exists"}]}]},
            "vlm_prompt": "fire_review", "inspection_label": "明火",
            "defaults": {"enabled": False},
        }

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.get("/detector/types/fire")
            assert resp.status_code == 200
            data = resp.json()
            assert data["key"] == "fire"
            assert data["models"][0]["model_key"] == "fire_smoke"
            assert data["post_process"] == "yolo_relation"
            assert data["rule"]["groups"][0]["conditions"][0]["op"] == "exists"
            assert data["vlm_prompt"] == "fire_review"
            assert data["inspection_label"] == "明火"

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
            resp = client.put("/detector/types/fire", json={"models": [{"model_key": "evil", "model_confidence": 0.5}]})
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
            ]
            for payload in invalid_cases:
                resp = client.put("/detector/types/fire", json=payload)
                assert resp.status_code == 400, f"expected 400 for {payload}, got {resp.status_code}"
                mock_registry.update_defaults.reset_mock()


class TestCreateDetectionType:
    def test_create_type_returns_key(self, client):
        payload = {"label": "新类型", "color": "#123456",
                   "models": [{"model_key": "new", "model_confidence": 0.5}],
                   "rule": {"groups": [{"conditions": [{"left": {"model_key": "new"}, "op": "exists"}]}]}}
        mock_registry = MagicMock()
        mock_registry.validate_rule.return_value = []
        mock_registry.add_type.return_value = "xin_lei_xing_123abc"
        mock_registry.get.return_value = {
            "label": "新类型", "color": "#123456",
            "post_process": "yolo_relation", "defaults": {},
        }

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.post("/detector/types", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert "key" in data
            assert data["label"] == "新类型"

    def test_create_type_duplicate_label_400(self, client):
        payload = {"label": "明火", "color": "#123456",
                   "models": [{"model_key": "x", "model_confidence": 0.5}],
                   "rule": {"groups": [{"conditions": [{"left": {"model_key": "x"}, "op": "exists"}]}]}}
        mock_registry = MagicMock()
        mock_registry.validate_rule.return_value = []
        mock_registry.add_type.side_effect = ValueError("label '明火' already exists")

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.post("/detector/types", json=payload)
            assert resp.status_code == 400

    def test_create_missing_label_400(self, client):
        mock_registry = MagicMock()
        mock_registry.validate_rule.return_value = []
        mock_registry.add_type.side_effect = ValueError("label is required")

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.post("/detector/types", json={"color": "#123456"})
            assert resp.status_code == 400


class TestDeleteDetectionType:
    def test_delete_type_success(self, client):
        mock_registry = MagicMock()
        mock_registry.validate_rule.return_value = []
        mock_registry.add_type.return_value = "dai_shan_chu_123abc"
        mock_registry.get.return_value = {"label": "待删除"}

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.post("/detector/types", json={"label": "待删除", "color": "#000000",
                                                          "models": [{"model_key": "d", "model_confidence": 0.5}],
                                                          "rule": {"groups": [{"conditions": [{"left": {"model_key": "d"}, "op": "exists"}]}]}})
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
    def _mock_model_registry(self, saved_name):
        from pathlib import Path
        mock_mr = MagicMock()
        mock_mr.save_model_file.side_effect = lambda filename, content: Path(filename)
        mock_mr.to_api_list.return_value = []
        mock_mr.add_model.return_value = Path(saved_name).stem
        mock_mr.get.return_value = {"post_process": "yolo_relation"}
        return mock_mr

    def test_upload_model_success(self, client, tmp_path):
        mock_registry = MagicMock()
        mock_registry.get.return_value = {"label": "明火", "models": [{"model_key": "test_model", "model_confidence": 0.3}]}
        mock_mr = self._mock_model_registry("test_model.pt")

        with patch("backend.safety_detection.api.registry", mock_registry), \
             patch("backend.safety_detection.api.model_registry", mock_mr), \
             patch("backend.safety_detection.api.parse_model_metadata", return_value={}):
            resp = client.post(
                "/detector/types/fire/model",
                files={"file": ("test_model.pt", io.BytesIO(b"fake"), "application/octet-stream")}
            )
            assert resp.status_code == 200
            assert resp.json()["model_key"] == "test_model"
            mock_mr.save_model_file.assert_called_once_with("test_model.pt", b"fake")
            mock_registry.update_type.assert_called_once()
            call = mock_registry.update_type.call_args
            assert call[0][0] == "fire"
            assert call[0][1]["models"] == [{"model_key": "test_model", "model_confidence": 0.5}]
            assert call[0][1]["rule"] == {"groups": [{"conditions": [{"left": {"model_key": "test_model"}, "op": "exists"}]}]}

    def test_upload_rknn_model_success(self, client, tmp_path):
        mock_registry = MagicMock()
        mock_registry.get.return_value = {"label": "明火", "models": [{"model_key": "test_model", "model_confidence": 0.3}]}
        mock_mr = self._mock_model_registry("test_model.rknn")

        with patch("backend.safety_detection.api.registry", mock_registry), \
             patch("backend.safety_detection.api.model_registry", mock_mr), \
             patch("backend.safety_detection.api.parse_model_metadata", return_value={}):
            resp = client.post(
                "/detector/types/fire/model",
                files={"file": ("test_model.rknn", io.BytesIO(b"rknn"), "application/octet-stream")}
            )
            assert resp.status_code == 200
            assert resp.json()["model_key"] == "test_model"
            mock_mr.save_model_file.assert_called_once_with("test_model.rknn", b"rknn")
            mock_registry.update_type.assert_called_once()
            call = mock_registry.update_type.call_args
            assert call[0][1]["models"] == [{"model_key": "test_model", "model_confidence": 0.5}]
            assert call[0][1]["rule"] == {"groups": [{"conditions": [{"left": {"model_key": "test_model"}, "op": "exists"}]}]}

    def test_upload_reuses_existing_model_entry(self, client):
        mock_registry = MagicMock()
        mock_registry.get.return_value = {"label": "明火", "models": []}
        mock_mr = self._mock_model_registry("test_model.pt")
        mock_mr.to_api_list.return_value = [{"key": "existing", "file": "test_model.pt"}]

        with patch("backend.safety_detection.api.registry", mock_registry), \
             patch("backend.safety_detection.api.model_registry", mock_mr), \
             patch("backend.safety_detection.api.parse_model_metadata", return_value={}):
            resp = client.post(
                "/detector/types/fire/model",
                files={"file": ("test_model.pt", io.BytesIO(b"fake"), "application/octet-stream")}
            )
            assert resp.status_code == 200
            assert resp.json()["model_key"] == "existing"
            mock_mr.add_model.assert_not_called()
            mock_registry.update_type.assert_called_once()
            call = mock_registry.update_type.call_args
            assert call[0][1]["models"] == [{"model_key": "existing", "model_confidence": 0.5}]
            assert call[0][1]["rule"] == {"groups": [{"conditions": [{"left": {"model_key": "existing"}, "op": "exists"}]}]}

    def test_upload_invalid_extension_400(self, client):
        mock_registry = MagicMock()
        mock_registry.get.return_value = {"label": "明火", "models": []}
        mock_mr = MagicMock()

        with patch("backend.safety_detection.api.registry", mock_registry), \
             patch("backend.safety_detection.api.model_registry", mock_mr):
            resp = client.post(
                "/detector/types/fire/model",
                files={"file": ("model.onnx", io.BytesIO(b"fake"), "application/octet-stream")}
            )
            assert resp.status_code == 400
            mock_mr.save_model_file.assert_not_called()

    def test_upload_unknown_type_404(self, client):
        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        mock_mr = MagicMock()

        with patch("backend.safety_detection.api.registry", mock_registry), \
             patch("backend.safety_detection.api.model_registry", mock_mr):
            resp = client.post(
                "/detector/types/nonexistent/model",
                files={"file": ("model.pt", io.BytesIO(b"fake"), "application/octet-stream")}
            )
            assert resp.status_code == 404
            mock_mr.save_model_file.assert_not_called()
