import time
import queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.camera_manager import CameraConfig, CameraManager


def test_camera_manager_has_decode_scheduler():
    cm = CameraManager()
    assert hasattr(cm, "decode_scheduler")
    assert cm.decode_scheduler is not None


def test_camera_manager_get_latest_frame_returns_none_when_empty():
    cm = CameraManager()
    assert cm.get_latest_frame("nonexistent") is None


def test_camera_manager_set_main_camera():
    cm = CameraManager()
    cm.register_camera(CameraConfig(camera_id="cam_01", source="0"))
    cm.set_main_camera("cam_01")
    assert cm.get_main_camera() == "cam_01"


def test_camera_manager_set_main_camera_unknown_clears_main():
    cm = CameraManager()
    cm.register_camera(CameraConfig(camera_id="cam_01", source="0"))
    cm.set_main_camera("cam_01")
    assert cm.set_main_camera("missing") is False
    assert cm.get_main_camera() is None


def test_get_latest_frame_returns_copy():
    cm = CameraManager()
    cm.register_camera(CameraConfig(camera_id="cam_01", source="0"))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cm._cameras["cam_01"].current_frame = frame

    latest = cm.get_latest_frame("cam_01")

    assert latest is not frame
    assert np.array_equal(latest, frame)


def test_get_latest_frame_returns_none_when_no_frame():
    cm = CameraManager()
    cm.register_camera(CameraConfig(camera_id="cam_01", source="0"))
    assert cm.get_latest_frame("cam_01") is None


def test_get_latest_frame_copy_is_independent():
    cm = CameraManager()
    cm.register_camera(CameraConfig(camera_id="cam_01", source="0"))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cm._cameras["cam_01"].current_frame = frame

    latest = cm.get_latest_frame("cam_01")
    latest[0, 0, 0] = 255

    assert cm._cameras["cam_01"].current_frame[0, 0, 0] == 0


def test_decode_scheduler_empty_queue_returns_gracefully():
    cm = CameraManager()
    cm.register_camera(CameraConfig(camera_id="cam_01", source="0"))
    state = cm._cameras["cam_01"]
    state.running = True

    scheduler = cm.decode_scheduler
    for _ in range(31):
        scheduler._decode_one_frame("cam_01", time.time())

    # reader 线程未产出帧时，decode worker 应直接返回，不记错误、不触发重连
    assert state.error_count == 0
    assert state.reader_error_count == 0
    assert state.reconnect_attempts == 0


def test_main_camera_writes_frame_history():
    cm = CameraManager()
    cm.register_camera(CameraConfig(camera_id="cam_01", source="0"))
    cm.set_main_camera("cam_01")
    state = cm._cameras["cam_01"]
    state.running = True

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    state.reader_queue.put_nowait(frame.copy())

    scheduler = cm.decode_scheduler
    scheduler._decode_one_frame("cam_01", time.time())

    assert len(state.frame_history) == 1
    ts, hist_frame = state.frame_history[0]
    assert np.array_equal(hist_frame, frame)
