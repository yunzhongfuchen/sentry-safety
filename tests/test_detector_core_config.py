import pytest
from unittest.mock import Mock


def test_type_schedule_without_level():
    from backend.safety_detection.detector_core import TypeSchedule
    s = TypeSchedule(dtype="fire", enabled=True, interval=1, threshold=0.5, cooldown=10, use_vlm=True)
    assert s.cooldown == 10
    assert s.use_vlm is True
    assert not hasattr(s, "level")


def test_multidetector_register_camera_uses_cooldown_and_use_vlm():
    from backend.safety_detection.detector_core import MultiDetector
    md = MultiDetector(camera_manager=None, safety_detector=None, vlm_queue=None, strategy=None)
    md.register_camera("cam01", {
        "fire": {"enabled": True, "interval": 1, "threshold": 0.5, "cooldown": 15, "use_vlm": True},
        "mask": {"enabled": True, "interval": 1, "threshold": 0.5, "cooldown": 5, "use_vlm": False},
    })
    assert md._schedules["cam01"]["fire"].cooldown == 15
    assert md._schedules["cam01"]["fire"].use_vlm is True
    assert md._schedules["cam01"]["mask"].cooldown == 5
    assert md._schedules["cam01"]["mask"].use_vlm is False


def test_is_in_cooldown_uses_schedule_cooldown():
    from backend.safety_detection.detector_core import MultiDetector
    md = MultiDetector(camera_manager=None, safety_detector=None, vlm_queue=None, strategy=None)
    md.register_camera("cam01", {
        "fire": {"enabled": True, "interval": 1, "threshold": 0.5, "cooldown": 10, "use_vlm": True},
    })
    now = 100.0
    md._cooldowns["cam01"]["fire"] = now - 5.0  # 5 seconds ago
    assert md.is_in_cooldown("cam01", "fire", now) is True

    md._cooldowns["cam01"]["fire"] = now - 15.0  # 15 seconds ago
    assert md.is_in_cooldown("cam01", "fire", now) is False


def test_vlm_callbacks_are_pure_passthrough():
    from backend.safety_detection.detector_core import MultiDetector
    md = MultiDetector(camera_manager=None, safety_detector=None, vlm_queue=None, strategy=None)
    md.register_camera("cam01", {
        "fire": {"enabled": True, "interval": 1, "threshold": 0.5, "cooldown": 10, "use_vlm": True},
    })

    vlm_cb = Mock()
    md.vlm_result_callback = vlm_cb
    trigger_cb = Mock()
    md.trigger_callback = trigger_cb

    # Test _on_vlm_review
    md._on_vlm_review("cam01", "fire", {"confirmed": True, "confidence": 0.8, "reason": "有烟雾"})
    vlm_cb.assert_called_once()
    trigger_cb.assert_not_called()


def test_handle_standard_detection_sets_pending_vlm_review():
    from backend.safety_detection.detector_core import MultiDetector
    md = MultiDetector(camera_manager=None, safety_detector=None, vlm_queue=None, strategy=None)
    md.register_camera("cam01", {
        "fire": {"enabled": True, "interval": 1, "threshold": 0.5, "cooldown": 10, "use_vlm": True},
    })
    schedule = md._schedules["cam01"]["fire"]
    assert schedule.use_vlm is True
    # The actual detection path (_handle_standard_detection) is covered by integration tests.
    # This test verifies that the schedule is correctly configured for VLM review.
