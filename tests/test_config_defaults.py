def test_load_camera_configs_migrates_detection_types_section(tmp_path, monkeypatch):
    import json
    from backend.config import load_camera_configs, CAMERAS_CONFIG_FILE

    # 旧格式摄像头配置：detection_types 段带参数（level/use_vlm/threshold 等）
    old_camera = {
        "camera_id": "cam_old",
        "source": "rtsp://test",
        "detection_types": {
            "fire": {"enabled": True, "interval": 1, "threshold": 0.6, "consecutive_required": 2, "level": "P0", "use_vlm": True},
            "mask": {"enabled": True, "interval": 1, "threshold": 0.5, "consecutive_required": 1, "level": "P1", "use_vlm": False,
                     "roi": [[0, 0], [1, 0], [1, 1]], "roi_invert": True},
        }
    }

    # Write old-style config to temp file
    temp_cameras_file = tmp_path / "cameras.json"
    with open(temp_cameras_file, "w", encoding="utf-8") as f:
        json.dump({"cameras": [old_camera]}, f, ensure_ascii=False, indent=2)

    # Monkeypatch CAMERAS_CONFIG_FILE to point to temp file
    monkeypatch.setattr("backend.config.CAMERAS_CONFIG_FILE", temp_cameras_file)

    # Call load_camera_configs() — should migrate old config
    cameras = load_camera_configs()

    # Verify migration results
    assert len(cameras) == 1
    cam = cameras[0]
    assert cam["camera_id"] == "cam_old"
    assert cam["source"] == "rtsp://test"

    # detection_types 段改名 algorithms，只保留 enabled/roi/roi_invert
    assert "detection_types" not in cam
    assert cam["algorithms"]["fire"] == {"enabled": True}
    assert cam["algorithms"]["mask"] == {
        "enabled": True,
        "roi": [[0, 0], [1, 0], [1, 1]],
        "roi_invert": True,
    }

    # Verify the migrated config was saved back to file
    with open(temp_cameras_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    saved_cam = saved_data["cameras"][0]
    assert "detection_types" not in saved_cam
    assert saved_cam["algorithms"]["fire"] == {"enabled": True}


def test_default_global_settings_contains_display_interval():
    from backend.config import DEFAULT_GLOBAL_SETTINGS
    assert "display_detection_interval" in DEFAULT_GLOBAL_SETTINGS
    assert DEFAULT_GLOBAL_SETTINGS["display_detection_interval"] == 1.0


def test_global_settings_has_save_image_timestamp():
    from backend import config
    assert "save_image_timestamp" in config.DEFAULT_GLOBAL_SETTINGS
    assert config.DEFAULT_GLOBAL_SETTINGS["save_image_timestamp"] is True

