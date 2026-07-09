def test_load_camera_configs_migrates_old_level_config(tmp_path, monkeypatch):
    import json
    from backend.config import load_camera_configs, save_camera_configs, DEFAULT_TYPE_CONFIG, CAMERAS_CONFIG_FILE

    # Create old-style camera config with 'level' but no 'cooldown'
    old_camera = {
        "camera_id": "cam_old",
        "source": "rtsp://test",
        "detection_types": {
            "fire": {"enabled": True, "interval": 1, "threshold": 0.6, "consecutive_required": 2, "level": "P0", "use_vlm": True},
            "mask": {"enabled": True, "interval": 1, "threshold": 0.5, "consecutive_required": 1, "level": "P1", "use_vlm": False},
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

    # Verify 'level' removed and 'cooldown' added for all detection types
    for dtype, cfg in cam["detection_types"].items():
        assert "level" not in cfg, f"{dtype}: 'level' should be removed"
        assert "cooldown" in cfg, f"{dtype}: 'cooldown' should be added"
        assert cfg["cooldown"] == DEFAULT_TYPE_CONFIG[dtype]["cooldown"], f"{dtype}: cooldown default mismatch"

    # Verify other fields preserved
    fire_cfg = cam["detection_types"]["fire"]
    assert fire_cfg["enabled"] is True
    assert fire_cfg["interval"] == 1
    assert fire_cfg["threshold"] == 0.6
    assert fire_cfg["consecutive_required"] == 2
    assert fire_cfg["use_vlm"] is True

    mask_cfg = cam["detection_types"]["mask"]
    assert mask_cfg["enabled"] is True
    assert mask_cfg["interval"] == 1
    assert mask_cfg["threshold"] == 0.5
    assert mask_cfg["consecutive_required"] == 1
    assert mask_cfg["use_vlm"] is False

    # Verify the migrated config was saved back to file
    with open(temp_cameras_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    saved_cam = saved_data["cameras"][0]
    for dtype, cfg in saved_cam["detection_types"].items():
        assert "level" not in cfg
        assert "cooldown" in cfg


def test_default_global_settings_contains_display_interval():
    from backend.config import DEFAULT_GLOBAL_SETTINGS
    assert "display_detection_interval" in DEFAULT_GLOBAL_SETTINGS
    assert DEFAULT_GLOBAL_SETTINGS["display_detection_interval"] == 1.0

