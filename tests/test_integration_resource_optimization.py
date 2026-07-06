from unittest.mock import MagicMock, patch
import numpy as np
import time

from backend.camera_manager import CameraManager, CameraConfig


def test_main_camera_only_stream_registered():
    """验证只有主画面注册流缓冲"""
    cm = CameraManager()
    cm.register_camera(CameraConfig(camera_id="cam_01", source="0"))
    cm.register_camera(CameraConfig(camera_id="cam_02", source="0"))

    with patch.object(cm.decode_scheduler, "start"):
        cm.start_camera("cam_01")
        cm.start_camera("cam_02")

    cm.set_main_camera("cam_01")
    assert cm.get_main_camera() == "cam_01"
    assert cm.decode_scheduler._main_camera == "cam_01"

    cm.stop_all()


def test_get_latest_frame_returns_copy():
    """验证 get_latest_frame 返回帧副本"""
    cm = CameraManager()
    cm.register_camera(CameraConfig(camera_id="cam_01", source="0"))

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    cm._cameras["cam_01"].current_frame = dummy

    frame = cm.get_latest_frame("cam_01")
    assert frame is not None
    assert frame.shape == dummy.shape
    assert frame is not dummy
