from unittest.mock import MagicMock

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
