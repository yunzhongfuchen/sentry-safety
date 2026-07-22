from backend.camera_manager import CameraConfig, CameraState


def test_camera_state_has_last_decode_time():
    cfg = CameraConfig(camera_id="cam_01", source="0")
    state = CameraState(config=cfg)
    assert hasattr(state, "last_decode_time")
    assert state.last_decode_time == 0.0


def test_camera_state_has_decode_queued():
    cfg = CameraConfig(camera_id="cam_01", source="0")
    state = CameraState(config=cfg)
    assert hasattr(state, "decode_queued")
    assert state.decode_queued is False


class TestCameraConfigMigration:
    def test_detection_types_section_migrated(self, tmp_path, monkeypatch):
        """cameras.json 的 detection_types 段改名 algorithms，参数被剔除"""
        import json
        import backend.config as cmod
        cfg_file = tmp_path / "cameras.json"
        cfg_file.write_text(json.dumps({
            "cameras": [{
                "camera_id": "cam1", "name": "测试", "source": "rtsp://x",
                "detection_types": {
                    "fire": {"enabled": True, "threshold": 0.9, "interval": 5,
                             "roi": [[0, 0], [1, 0], [1, 1]], "roi_invert": False}
                }
            }]
        }, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(cmod, "CAMERAS_CONFIG_FILE", cfg_file)
        cmod.load_camera_configs()
        data = json.loads(cfg_file.read_text(encoding="utf-8"))
        cam = data["cameras"][0]
        assert "detection_types" not in cam
        assert cam["algorithms"]["fire"] == {
            "enabled": True,
            "roi": [[0, 0], [1, 0], [1, 1]],
            "roi_invert": False,
        }
