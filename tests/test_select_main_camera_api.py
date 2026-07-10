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


def test_set_main_camera_keeps_old_stream_buffer_registered():
    from backend import main_multi

    cm = MagicMock()
    cm.get_main_camera.return_value = "cam_01"
    ss = MagicMock()
    selected_display = MagicMock()

    with patch.object(main_multi, "camera_manager", cm), \
         patch.object(main_multi, "stream_server", ss), \
         patch.object(main_multi, "selected_camera_display", selected_display):
        main_multi.set_main_camera("cam_02")

    ss.unregister_camera.assert_not_called()
    ss.register_camera.assert_called_once_with("cam_02")
    cm.set_main_camera.assert_called_once_with("cam_02")
    selected_display.set_selected_camera.assert_called_once_with("cam_02")


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


def test_get_display_types_returns_types_and_interval():
    with patch("backend.main_multi.app_config.load_global_settings") as load_gs, \
         patch("backend.main_multi.app_config.DEFAULT_GLOBAL_SETTINGS", {
             "display_detection_types": {"fire": True},
             "display_detection_interval": 1.0,
         }):
        load_gs.return_value = {
            "display_detection_types": {"fire": True, "smoke": False},
            "display_detection_interval": 2.5,
        }
        from backend.main_multi import app
        client = TestClient(app)
        response = client.get("/display-types")
        assert response.status_code == 200
        data = response.json()
        assert data["display_detection_types"] == {"fire": True, "smoke": False}
        assert data["display_detection_interval"] == 2.5


def test_update_display_types_clamps_interval():
    with patch("backend.main_multi.app_config.load_global_settings") as load_gs, \
         patch("backend.main_multi.app_config.save_global_settings") as save_gs, \
         patch("backend.main_multi.selected_camera_display") as scd, \
         patch("backend.main_multi.app_config.DEFAULT_GLOBAL_SETTINGS", {
             "display_detection_types": {"fire": True},
             "display_detection_interval": 1.0,
         }):
        load_gs.return_value = {
            "display_detection_types": {"fire": True},
            "display_detection_interval": 1.0,
        }
        from backend.main_multi import app
        client = TestClient(app)
        response = client.post("/display-types", json={
            "display_detection_types": {"fire": False},
            "display_detection_interval": 0.05,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["display_detection_interval"] == 0.1
        assert data["display_detection_types"] == {"fire": False}
        saved = save_gs.call_args.args[0]
        assert saved["display_detection_interval"] == 0.1
        scd.set_display_config.assert_called_once_with({"fire": False}, 0.1)


def test_update_display_types_rejects_invalid_interval():
    with patch("backend.main_multi.app_config.load_global_settings") as load_gs, \
         patch("backend.main_multi.app_config.save_global_settings") as save_gs, \
         patch("backend.main_multi.selected_camera_display") as scd, \
         patch("backend.main_multi.app_config.DEFAULT_GLOBAL_SETTINGS", {
             "display_detection_types": {"fire": True},
             "display_detection_interval": 1.0,
         }):
        load_gs.return_value = {
            "display_detection_types": {"fire": True},
            "display_detection_interval": 1.0,
        }
        from backend.main_multi import app
        client = TestClient(app)
        response = client.post("/display-types", json={
            "display_detection_types": {"fire": True},
            "display_detection_interval": "fast",
        })
        assert response.status_code == 400
        save_gs.assert_not_called()
        scd.set_display_config.assert_not_called()
