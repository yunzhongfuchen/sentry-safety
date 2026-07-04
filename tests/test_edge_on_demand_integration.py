import time
import numpy as np
from unittest.mock import MagicMock, patch


def test_scheduled_camera_only_decodes_on_request():
    """验证 SCHEDULED 摄像头不在请求时持续解码"""
    from backend.camera_manager import CameraManager, CameraConfig, DecodeMode

    cm = CameraManager()
    cfg = CameraConfig(camera_id="cam_01", source="0")
    cm.register_camera(cfg)

    # 模拟 cap 未打开，直接验证模式
    assert cm._cameras["cam_01"].decode_mode == DecodeMode.SCHEDULED

    # request_frame 超时（没有真实解码线程）
    frame = cm.request_frame("cam_01", timeout=0.05)
    assert frame is None


def test_main_camera_gets_continuous_mode():
    """验证设置主画面后模式变为 CONTINUOUS"""
    from backend.camera_manager import CameraManager, CameraConfig, DecodeMode

    cm = CameraManager()
    cm.register_camera(CameraConfig(camera_id="cam_01", source="0"))
    cm.set_main_camera("cam_01")

    assert cm.get_main_camera() == "cam_01"
    assert cm._cameras["cam_01"].decode_mode == DecodeMode.CONTINUOUS
