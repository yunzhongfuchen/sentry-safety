def test_default_type_config_shape():
    from backend.config import DEFAULT_TYPE_CONFIG, DEFAULT_GLOBAL_SETTINGS
    for dtype, cfg in DEFAULT_TYPE_CONFIG.items():
        assert "level" not in cfg
        assert "cooldown" in cfg
        assert isinstance(cfg["cooldown"], (int, float))
        assert "use_vlm" in cfg
    assert "use_vlm" not in DEFAULT_GLOBAL_SETTINGS
    assert "p0_alert_cooldown" not in DEFAULT_GLOBAL_SETTINGS
    assert "p1_alert_cooldown" not in DEFAULT_GLOBAL_SETTINGS
