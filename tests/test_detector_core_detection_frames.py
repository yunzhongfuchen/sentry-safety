import time
import numpy as np
import pytest
from unittest.mock import Mock
from backend.safety_detection.detector_core import MultiDetector, TypeSchedule


class FakeRegistry:
    """中性 defaults 的假注册表：register_camera 现在以注册表 defaults 为基底，
    测试需隔离用户本地 algorithms.json 的影响（与真实 merge_camera_config 行为一致）"""

    NEUTRAL_DEFAULTS = {
        "enabled": False,
        "interval": 1.0, "threshold": 0.5, "cooldown": 60.0,
        "verification_frame_count": 1, "verification_frame_interval": 1.0,
        "consecutive_required": 3, "use_vlm": False,
        "static_filter": False, "static_diff_threshold": 0.02,
    }
    LABELS = {"fire": "明火", "sleep": "睡岗"}

    def get(self, dtype):
        return {"label": self.LABELS.get(dtype, dtype), "alarm_description": ""}

    def merge_camera_config(self, dtype, overrides):
        result = dict(self.NEUTRAL_DEFAULTS)
        for k, v in (overrides or {}).items():
            if k in result or k in ("roi", "roi_invert"):
                result[k] = v
        return result


@pytest.fixture(autouse=True)
def fake_registry(monkeypatch):
    monkeypatch.setattr("backend.safety_detection.detector_core.registry", FakeRegistry())


def make_frame():
    return np.zeros((60, 80, 3), dtype=np.uint8)


def test_multi_frame_round_collects_all_frames_before_counting_hit():
    camera_manager = Mock()
    camera_manager.get_detection_frames.return_value = []
    md = MultiDetector(camera_manager=camera_manager, safety_detector=None, vlm_queue=None, strategy=None)
    md.register_camera("cam1", {
        "fire": {
            "enabled": True, "verification_frame_count": 3,
            "verification_frame_interval": 1, "consecutive_required": 2,
        },
    })
    schedule = md._schedules["cam1"]["fire"]
    result = {"detected": True, "boxes": [[1, 1, 10, 10]], "scores": [0.9], "max_confidence": 0.9}

    md._handle_standard_detection("cam1", "fire", make_frame(), result.copy(), schedule, now=0)
    md._handle_standard_detection("cam1", "fire", make_frame(), result.copy(), schedule, now=1)
    assert schedule.consecutive_count == 0
    assert schedule.sampled_frame_count == 2

    md._handle_standard_detection("cam1", "fire", make_frame(), result.copy(), schedule, now=2)
    assert schedule.consecutive_count == 1
    assert schedule.sampled_frame_count == 0
    assert camera_manager.add_detection_frame.call_count == 3
    assert camera_manager.add_detection_frame.call_args.kwargs["maxlen"] == 6


def test_failed_sample_clears_current_and_previous_round_frames():
    camera_manager = Mock()
    md = MultiDetector(camera_manager=camera_manager, safety_detector=None, vlm_queue=None, strategy=None)
    md.register_camera("cam1", {
        "fire": {"enabled": True, "verification_frame_count": 2, "consecutive_required": 2},
    })
    schedule = md._schedules["cam1"]["fire"]
    hit = {"detected": True, "boxes": [[1, 1, 10, 10]], "scores": [0.9], "max_confidence": 0.9}

    md._handle_standard_detection("cam1", "fire", make_frame(), hit.copy(), schedule, now=0)
    md._handle_standard_detection("cam1", "fire", make_frame(), hit.copy(), schedule, now=1)
    assert schedule.consecutive_count == 1

    md._handle_standard_detection("cam1", "fire", make_frame(), hit.copy(), schedule, now=10)
    md._handle_standard_detection("cam1", "fire", make_frame(), {"detected": False}, schedule, now=11)

    assert schedule.consecutive_count == 0
    assert schedule.sampled_frame_count == 0
    assert schedule.sampling_active is False
    assert camera_manager.clear_detection_frames.called


def test_sampling_interval_applies_inside_round_only():
    schedule = TypeSchedule(
        dtype="fire", enabled=True, interval=10, threshold=0.5, cooldown=60,
        verification_frame_count=3, verification_frame_interval=2,
    )
    schedule.last_run = 100
    assert schedule.is_due(109.9) is False
    assert schedule.is_due(110) is True

    schedule.sampling_active = True
    schedule.last_sample_time = 110
    assert schedule.is_due(111.9) is False
    assert schedule.is_due(112) is True


def test_standard_detection_collects_frames():
    camera_manager = Mock()
    camera_manager.get_detection_frames.return_value = []
    md = MultiDetector(camera_manager=camera_manager, safety_detector=None, vlm_queue=None, strategy=None)
    md.register_camera("cam1", {
        "fire": {"enabled": True, "interval": 1, "threshold": 0.5, "cooldown": 60, "consecutive_required": 3, "use_vlm": False},
    })
    schedule = md._schedules["cam1"]["fire"]
    result = {"detected": True, "scores": [0.9], "max_confidence": 0.9}

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

    result = {"detected": True, "scores": [0.9], "max_confidence": 0.9}
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
    result = {"detected": True, "scores": [0.9], "max_confidence": 0.9}

    md._handle_standard_detection("cam1", "sleep", make_frame(), result, schedule)
    md._handle_standard_detection("cam1", "sleep", make_frame(), result, schedule)
    md._handle_standard_detection("cam1", "sleep", make_frame(), result, schedule)

    assert schedule.consecutive_count == 0  # 第 3 次触发后计数清零
    assert camera_manager.add_detection_frame.call_count == 3
    assert camera_manager.get_detection_frames.called
    assert result["reason"] == "检测到睡岗异常"

    md._handle_standard_detection("cam1", "sleep", make_frame(), {"detected": False}, schedule)
    assert camera_manager.clear_detection_frames.called
    assert schedule.consecutive_count == 0
