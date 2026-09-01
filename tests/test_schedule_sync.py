"""算法 defaults 热同步回归测试

- register_camera 以注册表 defaults 为基底构建 TypeSchedule（不再用硬编码兜底）
- refresh_type_schedule 在算法 defaults 变更后热同步所有启用该算法的摄像头，保留摄像头级 roi/roi_invert
"""
import threading
from collections import defaultdict

import pytest

from unittest.mock import Mock

from backend.safety_detection.detector_core import MultiDetector


class FakeRegistry:
    """镜像 DetectionTypeRegistry 的 defaults/merge_camera_config 行为"""

    def __init__(self, defaults):
        self._defaults = dict(defaults)

    def get(self, dtype):
        return {"label": dtype, "defaults": self._defaults}

    def merge_camera_config(self, dtype, overrides):
        result = dict(self._defaults)
        for k, v in (overrides or {}).items():
            if k in result or k in ("roi", "roi_invert"):
                result[k] = v
        return result

    def set_defaults(self, **kw):
        self._defaults.update(kw)


DEFAULTS = {
    "enabled": True,
    "interval": 5.0,
    "threshold": 0.9,
    "cooldown": 120.0,
    "consecutive_required": 4,
    "use_vlm": True,
    "static_filter": True,
    "static_diff_threshold": 0.05,
}

ROI = [[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]]


def make_detector():
    md = MultiDetector.__new__(MultiDetector)
    md._lock = threading.RLock()
    md._schedules = {}
    md._alert_states = defaultdict(dict)
    md._cooldowns = defaultdict(dict)
    md.camera_manager = Mock()
    return md


@pytest.fixture
def fake_registry(monkeypatch):
    reg = FakeRegistry(DEFAULTS)
    monkeypatch.setattr("backend.safety_detection.detector_core.registry", reg)
    return reg


def test_register_camera_uses_registry_defaults(fake_registry):
    """注册表 defaults 应作为基底，摄像头级只覆盖 roi/roi_invert"""
    md = make_detector()
    md.register_camera("cam1", {"fire": {"enabled": True, "roi": ROI, "roi_invert": True}})

    s = md._schedules["cam1"]["fire"]
    assert s.interval == 5.0
    assert s.threshold == 0.9
    assert s.cooldown == 120.0
    assert s.consecutive_required == 4
    assert s.use_vlm is True
    assert s.static_filter is True
    assert s.static_diff_threshold == 0.05
    assert s.roi == ROI
    assert s.roi_invert is True


def test_refresh_type_schedule_hot_updates_defaults(fake_registry):
    """defaults 变更后热同步启用该算法的摄像头，保留摄像头级 roi"""
    md = make_detector()
    md.register_camera("cam1", {"fire": {"enabled": True, "roi": ROI}})
    md.register_camera("cam2", {"fire": {"enabled": True}, "smoke": {"enabled": True}})

    fake_registry.set_defaults(
        threshold=0.7, interval=2.0, use_vlm=False,
        verification_frame_count=3, verification_frame_interval=1.5,
    )
    camera_manager = md.camera_manager
    synced = md.refresh_type_schedule("fire")

    assert synced == 2
    assert camera_manager.clear_detection_frames.call_count == 2
    for cam in ("cam1", "cam2"):
        s = md._schedules[cam]["fire"]
        assert s.threshold == 0.7
        assert s.interval == 2.0
        assert s.use_vlm is False
        assert s.verification_frame_count == 3
        assert s.verification_frame_interval == 1.5
        assert s.consecutive_count == 0
        assert s.sampling_active is False
        assert s.sampled_frame_count == 0
    assert md._schedules["cam1"]["fire"].roi == ROI
    # 未刷新的类型保持原快照
    assert md._schedules["cam2"]["smoke"].threshold == 0.9


def test_refresh_type_schedule_no_camera_returns_zero(fake_registry):
    """没有摄像头启用该算法时返回 0"""
    md = make_detector()
    md.register_camera("cam1", {"fire": {"enabled": True}})

    assert md.refresh_type_schedule("smoke") == 0
    assert md._schedules["cam1"]["fire"].threshold == 0.9
