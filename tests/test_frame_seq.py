import time

import numpy as np

from backend.camera_manager import CameraConfig, CameraManager


def test_frame_seq_starts_zero_and_getter():
    cm = CameraManager()
    assert cm.get_frame_seq("nonexistent") == -1

    cm.register_camera(CameraConfig(camera_id="cam_01", source="0"))
    assert cm.get_frame_seq("cam_01") == 0


def test_frame_seq_increments_after_decode_write():
    cm = CameraManager()
    cm.register_camera(CameraConfig(camera_id="cam_01", source="0"))
    state = cm._cameras["cam_01"]
    state.running = True

    scheduler = cm.decode_scheduler
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    assert cm.get_frame_seq("cam_01") == 0

    state.reader_queue.put_nowait(frame)
    scheduler._decode_one_frame("cam_01", time.time())
    assert cm.get_frame_seq("cam_01") == 1
    assert state.current_frame is not None

    state.reader_queue.put_nowait(frame)
    scheduler._decode_one_frame("cam_01", time.time())
    assert cm.get_frame_seq("cam_01") == 2
