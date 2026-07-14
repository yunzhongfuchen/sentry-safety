"""detector types API 端点测试"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.safety_detection.api import router
from fastapi import FastAPI


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
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
        mock_registry.get.return_value = {"defaults": {"enabled": False, "threshold": 0.5}}
        mock_registry.get_defaults.return_value = {"enabled": True, "threshold": 0.8}

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.put("/detector/types/fire", json={"enabled": True, "threshold": 0.8})
            assert resp.status_code == 200
            assert resp.json()["success"] is True
            mock_registry.update_defaults.assert_called_once()

    def test_structural_fields_ignored(self, client):
        mock_registry = MagicMock()
        mock_registry.get.return_value = {"defaults": {"enabled": False}}

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.put("/detector/types/fire", json={"model_path": "evil.pt"})
            assert resp.status_code == 400

    def test_unknown_type_404(self, client):
        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.put("/detector/types/nope", json={"enabled": True})
            assert resp.status_code == 404

    def test_invalid_values_return_400(self, client):
        mock_registry = MagicMock()
        mock_registry.get.return_value = {"defaults": {"enabled": False, "threshold": 0.5}}

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
