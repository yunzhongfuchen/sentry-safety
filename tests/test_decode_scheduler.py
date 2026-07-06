import time
from unittest.mock import MagicMock
import numpy as np
import pytest

from backend.decode_scheduler import DecodeScheduler


def test_scheduler_has_main_camera_attribute():
    cm = MagicMock()
    cm._cameras = {}
    scheduler = DecodeScheduler(cm, num_workers=1)
    assert scheduler._main_camera is None


def test_scheduler_set_main_camera():
    cm = MagicMock()
    cm._cameras = {}
    scheduler = DecodeScheduler(cm, num_workers=1)
    scheduler.set_main_camera("cam_01")
    assert scheduler._main_camera == "cam_01"


def test_scheduler_start_stop_cleans_threads():
    cm = MagicMock()
    cm._cameras = {}
    scheduler = DecodeScheduler(cm, num_workers=2)
    scheduler.start()
    assert scheduler._running is True
    assert len(scheduler._worker_threads) == 2
    assert scheduler._scheduler_thread is not None

    scheduler.stop()
    assert scheduler._running is False
    assert scheduler._worker_threads == []


def test_scheduler_prioritizes_main_camera():
    cm = MagicMock()
    cfg_main = MagicMock()
    cfg_main.width = 640
    cfg_main.height = 480
    cfg_non = MagicMock()
    cfg_non.width = 640
    cfg_non.height = 480

    main_state = MagicMock()
    main_state.running = True
    main_state.cap = MagicMock()
    main_state.cap.isOpened.return_value = True
    main_state.cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    main_state.last_decode_time = 0.0
    main_state.decode_queued = False
    main_state.lock = MagicMock()
    main_state.__enter__ = MagicMock(return_value=(MagicMock(), None))
    main_state.__exit__ = MagicMock(return_value=False)
    main_state.config = cfg_main
    main_state.current_frame = None
    main_state.frame_count = 0
    main_state.frame_history = []
    main_state.error_count = 0

    non_state = MagicMock()
    non_state.running = True
    non_state.cap = MagicMock()
    non_state.cap.isOpened.return_value = True
    non_state.cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    non_state.last_decode_time = 0.0
    non_state.decode_queued = False
    non_state.lock = MagicMock()
    non_state.__enter__ = MagicMock(return_value=(MagicMock(), None))
    non_state.__exit__ = MagicMock(return_value=False)
    non_state.config = cfg_non
    non_state.current_frame = None
    non_state.frame_count = 0
    non_state.frame_history = []
    non_state.error_count = 0

    cm._cameras = {"main_cam": main_state, "non_main_cam": non_state}
    scheduler = DecodeScheduler(cm, num_workers=1)
    scheduler.set_main_camera("main_cam")

    # 手动调用一次调度循环以验证优先级队列内容
    scheduler._running = True
    scheduler._schedule_loop_single()

    # 检查队列中主画面优先级为 0，非主画面优先级为 1
    items = []
    while not scheduler._queue.empty():
        items.append(scheduler._queue.get())
    priorities = {cam_id: priority for priority, due_time, cam_id in items}
    assert priorities["main_cam"] == 0
    assert priorities["non_main_cam"] == 1
