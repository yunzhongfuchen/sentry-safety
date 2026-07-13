import time
import queue
from collections import deque
from unittest.mock import MagicMock
import numpy as np
import pytest

from backend.decode_scheduler import DecodeScheduler, _MAX_FRAME_HISTORY, _MAIN_CAMERA_INTERVAL


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


def test_scheduler_prioritizes_main_camera_with_shorter_interval():
    cm = MagicMock()
    cfg = MagicMock()
    cfg.width = 640
    cfg.height = 480

    now = time.time()
    # 主画面已到期，非主画面还未到期
    main_state = MagicMock()
    main_state.running = True
    main_state.cap = MagicMock()
    main_state.cap.isOpened.return_value = True
    main_state.last_decode_time = now - _MAIN_CAMERA_INTERVAL - 0.01
    main_state.decode_queued = False
    main_state.lock = MagicMock()
    main_state.config = cfg

    non_state = MagicMock()
    non_state.running = True
    non_state.cap = MagicMock()
    non_state.cap.isOpened.return_value = True
    non_state.last_decode_time = now - 0.5
    non_state.decode_queued = False
    non_state.lock = MagicMock()
    non_state.config = cfg

    cm._cameras = {"main_cam": main_state, "non_main_cam": non_state}
    scheduler = DecodeScheduler(cm, num_workers=1)
    scheduler.set_main_camera("main_cam")

    scheduler._running.set()
    scheduler._schedule_loop_single()

    items = []
    while not scheduler._queue.empty():
        items.append(scheduler._queue.get())
    scheduled_cams = {cam_id for *_rest, cam_id in items}
    assert scheduled_cams == {"main_cam"}


def test_decode_one_frame_updates_state_under_lock():
    cm = MagicMock()
    cfg = MagicMock()
    cfg.width = 640
    cfg.height = 480

    state = MagicMock()
    state.running = True
    state.reader_queue = queue.Queue(maxsize=1)
    state.reader_queue.put_nowait(np.zeros((480, 640, 3), dtype=np.uint8))
    state.last_decode_time = 0.0
    state.decode_queued = True
    state.lock = MagicMock()
    state.config = cfg
    state.current_frame = None
    state.frame_count = 0
    state.frame_history = deque(maxlen=_MAX_FRAME_HISTORY)
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
    state.reader_queue = queue.Queue(maxsize=1)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    state.last_decode_time = 0.0
    state.decode_queued = True
    state.lock = MagicMock()
    state.config = cfg
    state.current_frame = None
    state.frame_count = 0
    state.frame_history = deque(maxlen=_MAX_FRAME_HISTORY)
    state.error_count = 0

    cm._cameras = {"cam_01": state}
    scheduler = DecodeScheduler(cm, num_workers=1)

    for _ in range(_MAX_FRAME_HISTORY + 5):
        state.reader_queue.put_nowait(frame.copy())
        scheduler._decode_one_frame("cam_01", time.time())

    assert len(state.frame_history) == _MAX_FRAME_HISTORY


def test_decode_one_frame_returns_gracefully_when_queue_empty():
    cm = MagicMock()
    cfg = MagicMock()
    cfg.width = 640
    cfg.height = 480

    state = MagicMock()
    state.running = True
    state.reader_queue = queue.Queue(maxsize=1)
    state.last_decode_time = 0.0
    state.decode_queued = True
    state.lock = MagicMock()
    state.config = cfg
    state.current_frame = None
    state.frame_count = 0
    state.frame_history = deque(maxlen=_MAX_FRAME_HISTORY)
    state.error_count = 3

    cm._cameras = {"cam_01": state}
    scheduler = DecodeScheduler(cm, num_workers=1)

    scheduler._decode_one_frame("cam_01", time.time())

    # 空队列不应导致错误计数增加（错误统计已下沉到 reader 线程）
    assert state.error_count == 3
    assert state.decode_queued is False
    assert state.frame_count == 0


def test_worker_breaks_on_sentinel():
    cm = MagicMock()
    cm._cameras = {}
    scheduler = DecodeScheduler(cm, num_workers=1)
    scheduler._running.set()

    # 放入一个 sentinel 任务（优先级应低于真实解码任务）
    scheduler._queue.put((2, 0, -1, None))
    scheduler._running.clear()

    # 手动运行 worker_loop：它应处理 sentinel 并退出循环
    scheduler._worker_loop()

    assert scheduler._queue.empty() is True


def test_worker_does_not_skip_real_work_for_sentinel():
    cm = MagicMock()
    cfg = MagicMock()
    cfg.width = 640
    cfg.height = 480

    state = MagicMock()
    state.running = True
    state.reader_queue = queue.Queue(maxsize=1)
    state.reader_queue.put_nowait(np.zeros((480, 640, 3), dtype=np.uint8))
    state.last_decode_time = 0.0
    state.decode_queued = False
    state.lock = MagicMock()
    state.config = cfg
    state.current_frame = None
    state.frame_count = 0
    state.frame_history = deque(maxlen=_MAX_FRAME_HISTORY)
    state.error_count = 0

    cm._cameras = {"cam_01": state}
    scheduler = DecodeScheduler(cm, num_workers=1)
    scheduler._running.set()

    # 先放入真实解码任务，再放入 sentinel；sentinel 优先级更低
    now = time.time()
    scheduler._queue.put((0, 0, now, "cam_01"))
    scheduler._queue.put((2, 1, -1, None))

    scheduler._worker_loop()

    assert state.frame_count == 1
    assert scheduler._queue.empty() is True
