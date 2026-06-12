import pytest


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
