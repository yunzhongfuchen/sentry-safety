"""
统一解码线程池调度器
- 所有摄像头共享固定大小的解码线程池
- 主画面摄像头按 25 FPS 调度
- 非主画面摄像头按 1 FPS 调度
"""

import logging
import queue
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DecodeScheduler:
    """统一解码线程池"""

    def __init__(self, camera_manager, num_workers: int = 4):
        self.camera_manager = camera_manager
        self.num_workers = num_workers
        self._main_camera: Optional[str] = None
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._worker_threads: list[threading.Thread] = []

    def set_main_camera(self, camera_id: Optional[str]):
        """设置主画面摄像头"""
        self._main_camera = camera_id

    def start(self):
        """启动调度器和 worker 线程"""
        if self._running:
            return
        self._running = True

        self._scheduler_thread = threading.Thread(
            target=self._schedule_loop,
            daemon=True,
            name="decode-scheduler"
        )
        self._scheduler_thread.start()

        for i in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name=f"decode-worker-{i}"
            )
            t.start()
            self._worker_threads.append(t)

        logger.info(f"DecodeScheduler started with {self.num_workers} workers")

    def stop(self):
        """停止调度器和 worker 线程"""
        if not self._running:
            return
        self._running = False

        # 唤醒所有可能在 get 上阻塞的 worker
        for _ in range(self.num_workers * 2):
            try:
                self._queue.put_nowait((-1, -1, None))
            except queue.Full:
                pass

        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=2)
        for t in self._worker_threads:
            if t.is_alive():
                t.join(timeout=2)
        self._worker_threads.clear()
        logger.info("DecodeScheduler stopped")

    def _schedule_loop(self):
        """调度循环：把到期摄像头放入优先队列"""
        while self._running:
            self._schedule_loop_single()
            time.sleep(0.01)

    def _schedule_loop_single(self):
        """执行一轮调度，便于单元测试"""
        now = time.time()
        try:
            cameras = getattr(self.camera_manager, "_cameras", {})
            for cam_id, state in cameras.items():
                if not getattr(state, "running", False) or getattr(state, "cap", None) is None:
                    continue

                is_main = cam_id == self._main_camera
                interval = 1.0 / 25 if is_main else 1.0
                due_time = state.last_decode_time + interval

                if now >= due_time and not state.decode_queued:
                    priority = 0 if is_main else 1
                    self._queue.put((priority, due_time, cam_id))
                    state.decode_queued = True
        except Exception as e:
            logger.error(f"Decode scheduler error: {e}")

    def _worker_loop(self):
        """Worker 循环：从队列取任务并解码一帧"""
        while self._running:
            try:
                priority, due_time, cam_id = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if cam_id is None:
                continue

            try:
                self._decode_one_frame(cam_id, due_time)
            except Exception as e:
                logger.error(f"Decode worker error [{cam_id}]: {e}")

    def _decode_one_frame(self, cam_id: str, due_time: float):
        """解码一帧并更新状态"""
        cameras = getattr(self.camera_manager, "_cameras", {})
        state = cameras.get(cam_id)
        if state is None:
            return

        cap = getattr(state, "cap", None)
        if cap is None or not getattr(cap, "isOpened", lambda: False)():
            with state.lock:
                state.decode_queued = False
                state.last_decode_time = due_time
            return

        ret, frame = cap.read()

        with state.lock:
            state.decode_queued = False
            state.last_decode_time = due_time

        if not ret or frame is None:
            # 读帧失败，由 CameraManager 的重连逻辑处理
            state.error_count += 1
            return

        state.error_count = 0

        # 等比例缩放
        src_h, src_w = frame.shape[:2]
        max_w = getattr(state.config, "width", 640)
        max_h = getattr(state.config, "height", 480)
        if src_w > max_w or src_h > max_h:
            scale = min(max_w / src_w, max_h / src_h)
            new_w = int(src_w * scale)
            new_h = int(src_h * scale)
            frame = __import__("cv2").resize(frame, (new_w, new_h))

        current_time = time.time()
        with state.lock:
            state.current_frame = frame
            state.last_frame_time = current_time
            state.frame_count += 1
            if cam_id != self._main_camera:
                state.frame_history.append((current_time, frame.copy()))

        # 全局回调
        global_callback = getattr(self.camera_manager, "_global_frame_callback", None)
        if global_callback:
            try:
                global_callback(cam_id, frame)
            except Exception as e:
                logger.error(f"Frame callback error: {e}")
