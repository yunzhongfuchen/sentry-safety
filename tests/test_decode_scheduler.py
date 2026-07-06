import time
from unittest.mock import MagicMock
import numpy as np
import pytest

from backend.decode_scheduler import DecodeScheduler, _MAX_FRAME_HISTORY


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
    assert scheduler._running.is_set() is True
    assert len(scheduler._worker_threads) == 2
    assert scheduler._scheduler_thread is not None

    scheduler.stop()
    assert scheduler._running.is_set() is False
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
    non_state.config = cfg_non
    non_state.current_frame = None
    non_state.frame_count = 0
    non_state.frame_history = []
    non_state.error_count = 0

    cm._cameras = {"main_cam": main_state, "non_main_cam": non_state}
    scheduler = DecodeScheduler(cm, num_workers=1)
    scheduler.set_main_camera("main_cam")

    # 手动调用一次调度循环以验证优先级队列内容
    scheduler._running.set()
    scheduler._schedule_loop_single()

    # 检查队列中主画面优先级为 0，非主画面优先级为 1
    items = []
    while not scheduler._queue.empty():
        items.append(scheduler._queue.get())
    priorities = {cam_id: priority for priority, _counter, due_time, cam_id in items}
    assert priorities["main_cam"] == 0
    assert priorities["non_main_cam"] == 1


def test_decode_one_frame_updates_state_under_lock():
    cm = MagicMock()
    cfg = MagicMock()
    cfg.width = 640
    cfg.height = 480

    state = MagicMock()
    state.running = True
    state.cap = MagicMock()
    state.cap.isOpened.return_value = True
    state.cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    state.last_decode_time = 0.0
    state.decode_queued = True
    state.lock = MagicMock()
    state.config = cfg
    state.current_frame = None
    state.frame_count = 0
    state.frame_history = []
    state.error_count = 0

    cm._cameras = {"cam_01": state}
    scheduler = DecodeScheduler(cm, num_workers=1)

    scheduler._decode_one_frame("cam_01", time.time())

    assert state.decode_queued is False
    assert state.frame_count == 1
    assert state.error_count == 0
    assert len(state.frame_history) == 1


def test_decode_one_frame_bounds_frame_history():
    cm = MagicMock()
    cfg = MagicMock()
    cfg.width = 640
    cfg.height = 480

    state = MagicMock()
    state.running = True
    state.cap = MagicMock()
    state.cap.isOpened.return_value = True
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    state.cap.read.return_value = (True, frame)
    state.last_decode_time = 0.0
    state.decode_queued = True
    state.lock = MagicMock()
    state.config = cfg
    state.current_frame = None
    state.frame_count = 0
    state.frame_history = []
    state.error_count = 0

    cm._cameras = {"cam_01": state}
    scheduler = DecodeScheduler(cm, num_workers=1)

    for _ in range(_MAX_FRAME_HISTORY + 5):
        scheduler._decode_one_frame("cam_01", time.time())

    assert len(state.frame_history) == _MAX_FRAME_HISTORY


def test_decode_one_frame_increments_error_count_on_failure():
    cm = MagicMock()
    cfg = MagicMock()
    cfg.width = 640
    cfg.height = 480

    state = MagicMock()
    state.running = True
    state.cap = MagicMock()
    state.cap.isOpened.return_value = True
    state.cap.read.return_value = (False, None)
    state.last_decode_time = 0.0
    state.decode_queued = True
    state.lock = MagicMock()
    state.config = cfg
    state.current_frame = None
    state.frame_count = 0
    state.frame_history = []
    state.error_count = 3

    cm._cameras = {"cam_01": state}
    scheduler = DecodeScheduler(cm, num_workers=1)

    scheduler._decode_one_frame("cam_01", time.time())

    assert state.error_count == 4
    assert state.decode_queued is False
    assert state.frame_count == 0


def test_worker_breaks_on_sentinel():
    cm = MagicMock()
    cm._cameras = {}
    scheduler = DecodeScheduler(cm, num_workers=1)
    scheduler._running.set()

    # 放入一个 sentinel 任务
    scheduler._queue.put((-1, 0, -1, None))
    scheduler._running.clear()

    # 手动运行 worker_loop：它应处理 sentinel 并退出循环
    scheduler._worker_loop()

    assert scheduler._queue.empty() is True
