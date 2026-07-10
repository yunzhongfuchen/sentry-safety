from unittest.mock import MagicMock, patch

import numpy as np

from backend.camera_manager import CameraManager, CameraConfig
from backend import main_multi


def test_main_camera_stream_buffers_are_kept_across_switches():
    """验证切换主画面时保留旧流缓冲，避免前端 stream 重建竞态。"""
    cm = CameraManager()
    cm.register_camera(CameraConfig(camera_id="cam_01", source="0"))
    cm.register_camera(CameraConfig(camera_id="cam_02", source="0"))

    with patch.object(CameraManager, "_open_capture"):
        with patch.object(cm.decode_scheduler, "start"):
            cm.start_camera("cam_01")
            cm.start_camera("cam_02")

    # 给 cap 赋值 mock，避免 stop_camera / 调度器访问真实硬件
    for camera_id in ("cam_01", "cam_02"):
        cm._cameras[camera_id].cap = MagicMock()

    original_camera_manager = main_multi.camera_manager
    main_multi.camera_manager = cm
    try:
        with patch.object(main_multi, "stream_server") as mock_stream_server:
            main_multi.set_main_camera("cam_01")
            assert cm.get_main_camera() == "cam_01"
            assert cm.decode_scheduler._main_camera == "cam_01"
            mock_stream_server.register_camera.assert_called_once_with("cam_01")
            mock_stream_server.unregister_camera.assert_not_called()

            mock_stream_server.reset_mock()
            main_multi.set_main_camera("cam_02")
            assert cm.get_main_camera() == "cam_02"
            assert cm.decode_scheduler._main_camera == "cam_02"
            mock_stream_server.unregister_camera.assert_not_called()
            mock_stream_server.register_camera.assert_called_once_with("cam_02")
    finally:
        main_multi.camera_manager = original_camera_manager
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
