import time
import pytest
from backend.camera_manager import CameraManager, CameraConfig


def _register_and_start(cm, cid="cam1"):
    cfg = CameraConfig(camera_id=cid, source="dummy")
    cm.register_camera(cfg)
    # 不需要真启动视频流，直接操作内部 state
    return cm._cameras[cid]


def test_add_detection_frame_and_get():
    cm = CameraManager()
    _register_and_start(cm)
    ts = time.time()
    cm.add_detection_frame("cam1", "fire", ts, b"frame1", maxlen=3)
    frames = cm.get_detection_frames("cam1", "fire")
    assert len(frames) == 1
    assert frames[0] == (ts, b"frame1")


def test_detection_frames_maxlen():
    cm = CameraManager()
    _register_and_start(cm)
    for i in range(5):
        cm.add_detection_frame("cam1", "fire", time.time() + i, f"frame{i}".encode(), maxlen=3)
    frames = cm.get_detection_frames("cam1", "fire")
    assert len(frames) == 3
    assert frames[0][1] == b"frame2"


def test_clear_detection_frames():
    cm = CameraManager()
    _register_and_start(cm)
    cm.add_detection_frame("cam1", "fire", time.time(), b"f", maxlen=3)
    cm.clear_detection_frames("cam1", "fire")
    assert cm.get_detection_frames("cam1", "fire") == []


def test_clear_all_detection_frames():
    cm = CameraManager()
    _register_and_start(cm)
    ts = time.time()
    cm.add_detection_frame("cam1", "fire", ts, b"f1", maxlen=3)
    cm.add_detection_frame("cam1", "smoke", ts, b"s1", maxlen=3)
    assert len(cm.get_detection_frames("cam1", "fire")) == 1
    assert len(cm.get_detection_frames("cam1", "smoke")) == 1
    cm.clear_all_detection_frames("cam1")
    assert cm.get_detection_frames("cam1", "fire") == []
    assert cm.get_detection_frames("cam1", "smoke") == []


def test_unregister_clears_detection_frames():
    cm = CameraManager()
    _register_and_start(cm)
    cm.add_detection_frame("cam1", "fire", time.time(), b"f", maxlen=3)
    cm.unregister_camera("cam1")
    assert "cam1" not in cm._cameras


def test_frame_history_removed():
    cm = CameraManager()
    state = _register_and_start(cm)
    assert not hasattr(state, "frame_history")
