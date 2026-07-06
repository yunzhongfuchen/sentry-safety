from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def test_select_main_camera():
    with patch("backend.main_multi.camera_manager") as cm, \
         patch("backend.main_multi.stream_server") as ss, \
         patch("backend.main_multi.set_main_camera") as set_main:
        cm._cameras = {"cam_01": MagicMock()}
        from backend.main_multi import app
        client = TestClient(app)
        response = client.post("/cameras/cam_01/select")
        assert response.status_code == 200
        assert response.json()["main_camera"] == "cam_01"
        set_main.assert_called_once_with("cam_01")


def test_select_main_camera_not_found():
    with patch("backend.main_multi.camera_manager") as cm, \
         patch("backend.main_multi.stream_server") as ss, \
         patch("backend.main_multi.set_main_camera") as set_main:
        cm._cameras = {"cam_01": MagicMock()}
        from backend.main_multi import app
        client = TestClient(app)
        response = client.post("/cameras/unknown/select")
        assert response.status_code == 404
        assert response.json()["error"] == "Camera not found"
        set_main.assert_not_called()


def test_main_camera_stream():
    with patch("backend.main_multi.camera_manager") as cm, \
         patch("backend.main_multi.stream_server") as ss, \
         patch("backend.main_multi.generate_camera_frames") as gen:
        cm._cameras = {"cam_01": MagicMock(), "cam_02": MagicMock()}
        cm.get_main_camera.return_value = "cam_01"
        gen.return_value = iter([])
        from backend.main_multi import app
        client = TestClient(app)
        response = client.get("/cameras/cam_01/stream")
        assert response.status_code == 200
        gen.assert_called_once_with("cam_01", raw=False)


def test_non_main_camera_stream():
    with patch("backend.main_multi.camera_manager") as cm, \
         patch("backend.main_multi.stream_server") as ss, \
         patch("backend.main_multi.generate_camera_frames") as gen:
        cm._cameras = {"cam_01": MagicMock(), "cam_02": MagicMock()}
        cm.get_main_camera.return_value = "cam_01"
        from backend.main_multi import app
        client = TestClient(app)
        response = client.get("/cameras/cam_02/stream")
        assert response.status_code == 404
        assert response.json()["error"] == "Camera is not the main stream"
        gen.assert_not_called()
