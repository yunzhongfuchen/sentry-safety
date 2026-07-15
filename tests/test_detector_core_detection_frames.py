import time
import numpy as np
import pytest
from unittest.mock import Mock
from backend.safety_detection.detector_core import MultiDetector, TypeSchedule


def make_frame():
    return np.zeros((60, 80, 3), dtype=np.uint8)


def test_standard_detection_collects_frames():
    camera_manager = Mock()
    camera_manager.get_detection_frames.return_value = []
    md = MultiDetector(camera_manager=camera_manager, safety_detector=None, vlm_queue=None, strategy=None)
    md.register_camera("cam1", {
        "fire": {"enabled": True, "interval": 1, "threshold": 0.5, "cooldown": 60, "consecutive_required": 3, "use_vlm": False},
    })
    schedule = md._schedules["cam1"]["fire"]
    result = {"detected": True, "scores": [0.9]}

    md._handle_standard_detection("cam1", "fire", make_frame(), result, schedule)
    md._handle_standard_detection("cam1", "fire", make_frame(), result, schedule)
    md._handle_standard_detection("cam1", "fire", make_frame(), result, schedule)

    assert schedule.consecutive_count == 0  # 第 3 次触发后计数清零
    assert camera_manager.add_detection_frame.call_count == 3
    assert camera_manager.get_detection_frames.called


def test_no_detection_clears_frames():
    camera_manager = Mock()
    md = MultiDetector(camera_manager=camera_manager, safety_detector=None, vlm_queue=None, strategy=None)
    md.register_camera("cam1", {
        "fire": {"enabled": True, "interval": 1, "threshold": 0.5, "cooldown": 60, "consecutive_required": 3, "use_vlm": False},
    })
    schedule = md._schedules["cam1"]["fire"]
    md._handle_standard_detection("cam1", "fire", make_frame(), {"detected": True, "scores": [0.9]}, schedule)
    md._handle_standard_detection("cam1", "fire", make_frame(), {"detected": False}, schedule)
    assert camera_manager.clear_detection_frames.called
    assert schedule.consecutive_count == 0


def test_vlm_review_gets_recent_five_frames():
    vlm_queue = Mock()
    camera_manager = Mock()
    md = MultiDetector(camera_manager=camera_manager, safety_detector=None, vlm_queue=vlm_queue, strategy=None)
    md.register_camera("cam1", {
        "fire": {"enabled": True, "interval": 1, "threshold": 0.5, "cooldown": 60, "consecutive_required": 7, "use_vlm": True},
    })
    schedule = md._schedules["cam1"]["fire"]
    frames = [(time.time() + i, b"frame" + str(i).encode()) for i in range(7)]
    camera_manager.get_detection_frames.return_value = frames

    result = {"detected": True, "scores": [0.9]}
    for _ in range(7):
        md._handle_standard_detection("cam1", "fire", make_frame(), result, schedule)

    submitted = vlm_queue.submit.call_args[1]["task"]
    assert len(submitted["frames"]) == 5


def test_sleep_detection_uses_standard_logic_and_generic_reason():
    """睡岗检测走标准检测逻辑，reason 使用通用文案。"""
    camera_manager = Mock()
    camera_manager.get_detection_frames.return_value = []
    md = MultiDetector(camera_manager=camera_manager, safety_detector=None, vlm_queue=None, strategy=None)
    md.register_camera("cam1", {
        "sleep": {"enabled": True, "interval": 1, "threshold": 0.5, "cooldown": 60, "consecutive_required": 3, "use_vlm": False},
    })
    schedule = md._schedules["cam1"]["sleep"]
    result = {"detected": True, "scores": [0.9]}

    md._handle_standard_detection("cam1", "sleep", make_frame(), result, schedule)
    md._handle_standard_detection("cam1", "sleep", make_frame(), result, schedule)
    md._handle_standard_detection("cam1", "sleep", make_frame(), result, schedule)

    assert schedule.consecutive_count == 0  # 第 3 次触发后计数清零
    assert camera_manager.add_detection_frame.call_count == 3
    assert camera_manager.get_detection_frames.called
    assert "检测到 sleep" in result["reason"]

    md._handle_standard_detection("cam1", "sleep", make_frame(), {"detected": False}, schedule)
    assert camera_manager.clear_detection_frames.called
    assert schedule.consecutive_count == 0
