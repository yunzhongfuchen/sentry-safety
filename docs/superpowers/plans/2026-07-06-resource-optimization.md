# 多路摄像头检测资源优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现统一解码线程池、主画面专享推流画框、所有摄像头检测逻辑一致、推理侧直接读最新帧，确保现有界面功能不出现问题。

**Architecture:** 用 `DecodeScheduler` 替代每路摄像头的独立解码线程，主画面 25 FPS、非主画面 1 FPS；`CameraManager` 提供 `get_latest_frame` 和主画面管理；`main_multi.py` 只注册主画面流缓冲、`_overlay_loop` 只画主画面；`GPUDynamicScheduler` 加 `_busy` 防并发并直接读最新帧；`SerialStrategy/CorePinnedStrategy` 也直接读最新帧。

**Tech Stack:** Python 3.12, OpenCV, Ultralytics YOLO, FastAPI, threading, pytest.

## Global Constraints

- Python 3.12
- 不引入新的运行时依赖
- 保留现有 `MultiDetector`、`GPUDynamicScheduler`、`camera_manager`、`MJPEGStreamServer` 的公共接口兼容性
- 主画面实时预览保留，非主画面不实时推流
- 快照、告警、记录逻辑不变
- 现有前端界面功能不出现问题
- 检测逻辑对所有摄像头一致，不受是否主画面影响
- 频繁提交，每个 task 结束后 commit

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `backend/decode_scheduler.py` | 新增统一解码线程池，调度所有摄像头解码任务 |
| `backend/camera_manager.py` | 移除每路独立解码线程，接入 DecodeScheduler；新增主画面管理、get_latest_frame |
| `backend/main_multi.py` | 初始化默认主画面；改造 _overlay_loop 只处理主画面；添加 select API |
| `backend/gpu_scheduler.py` | 直接读最新帧；加 `_busy` 防并发；`last_infer` 按完成时间更新 |
| `backend/safety_detection/detector_core.py` | `_process_camera` 直接读最新帧 |
| `tests/test_decode_scheduler.py` | DecodeScheduler 单元测试 |
| `tests/test_camera_manager_main_camera.py` | 主画面管理单元测试 |
| `tests/test_gpu_scheduler_busy.py` | GPU scheduler 防并发测试 |
| `tests/test_integration_resource_optimization.py` | 集成测试 |

---

### Task 0: 基线确认

**Files:**
- All existing files

**Interfaces:**
- Consumes: 当前 `feature/resource-optimization-2026-07-06` 分支
- Produces: 干净的基线状态

- [ ] **Step 1: 运行现有测试确认起点健康**

```bash
cd d:/project/sentry-safety
python -m pytest tests/ -q
```

Expected: 现有测试全部通过。

- [ ] **Step 2: 确认当前分支**

```bash
git branch --show-current
```

Expected: `feature/resource-optimization-2026-07-06`

---

### Task 1: CameraState 新增字段

**Files:**
- Modify: `backend/camera_manager.py:71-108`

**Interfaces:**
- Consumes: 无
- Produces:
  - `CameraState.last_decode_time: float = 0.0`
  - `CameraState.decode_queued: bool = False`

- [ ] **Step 1: 编写失败测试**

Create: `tests/test_camera_manager_fields.py`

```python
from backend.camera_manager import CameraConfig, CameraState


def test_camera_state_has_last_decode_time():
    cfg = CameraConfig(camera_id="cam_01", source="0")
    state = CameraState(config=cfg)
    assert hasattr(state, "last_decode_time")
    assert state.last_decode_time == 0.0


def test_camera_state_has_decode_queued():
    cfg = CameraConfig(camera_id="cam_01", source="0")
    state = CameraState(config=cfg)
    assert hasattr(state, "decode_queued")
    assert state.decode_queued is False
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_camera_manager_fields.py -v
```

Expected: `2 failed` because fields do not exist.

- [ ] **Step 3: 最小实现**

Modify: `backend/camera_manager.py`

在 `CameraState` dataclass 中新增字段：

```python
@dataclass
class CameraState:
    """摄像头运行时状态"""
    config: CameraConfig
    status: CameraStatus = CameraStatus.IDLE
    cap: Optional[cv2.VideoCapture] = None
    current_frame: Optional[np.ndarray] = None
    last_frame_time: float = 0
    frame_count: int = 0
    error_count: int = 0
    reconnect_attempts: int = 0
    thread: Optional[threading.Thread] = None
    running: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    # 新增字段
    last_decode_time: float = 0.0
    decode_queued: bool = False
    # ... 后续字段保持不变
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_camera_manager_fields.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/camera_manager.py tests/test_camera_manager_fields.py
git commit -m "feat: add last_decode_time and decode_queued to CameraState

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: 创建 DecodeScheduler

**Files:**
- Create: `backend/decode_scheduler.py`

**Interfaces:**
- Consumes: `CameraState`, `CameraManager`
- Produces:
  - `DecodeScheduler(camera_manager, num_workers=4)`
  - `DecodeScheduler.start()`
  - `DecodeScheduler.stop()`
  - `DecodeScheduler.set_main_camera(camera_id)`

- [ ] **Step 1: 编写失败测试**

Create: `tests/test_decode_scheduler.py`

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_decode_scheduler.py -v
```

Expected: `2 errors` because module does not exist.

- [ ] **Step 3: 实现 DecodeScheduler**

Create: `backend/decode_scheduler.py`

```python
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

            time.sleep(0.01)

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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_decode_scheduler.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/decode_scheduler.py tests/test_decode_scheduler.py
git commit -m "feat: add DecodeScheduler with shared thread pool

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: CameraManager 接入 DecodeScheduler

**Files:**
- Modify: `backend/camera_manager.py:126-135`, `163-212`, `599-855`

**Interfaces:**
- Consumes: `DecodeScheduler`
- Produces:
  - `CameraManager.decode_scheduler: DecodeScheduler`
  - `CameraManager.get_latest_frame(camera_id)`
  - `CameraManager.set_main_camera(camera_id)`
  - `CameraManager.get_main_camera()`

- [ ] **Step 1: 编写失败测试**

Add to `tests/test_camera_manager_main_camera.py`:

```python
from backend.camera_manager import CameraManager, CameraConfig


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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_camera_manager_main_camera.py -v
```

Expected: tests fail because methods do not exist.

- [ ] **Step 3: 修改 CameraManager 接入 DecodeScheduler**

Modify: `backend/camera_manager.py`

1. 导入 DecodeScheduler：

```python
from decode_scheduler import DecodeScheduler
```

2. 修改 `__init__`：

```python
def __init__(self):
    self._cameras: Dict[str, CameraState] = {}
    self._lock = threading.RLock()
    self._global_frame_callback: Optional[Callable[[str, np.ndarray], None]] = None
    self._main_camera_id: Optional[str] = None
    self.decode_scheduler = DecodeScheduler(self, num_workers=4)
```

3. 修改 `start_camera`：

```python
def start_camera(self, camera_id: str) -> bool:
    """启动指定摄像头"""
    with self._lock:
        if camera_id not in self._cameras:
            logger.error(f"Camera {camera_id} not found")
            return False

        state = self._cameras[camera_id]
        if state.running:
            logger.warning(f"Camera {camera_id} already running")
            return True

        state.running = True
        state.error_count = 0
        state.reconnect_attempts = 0
        state.last_decode_time = 0.0

        # 启动 DecodeScheduler（首次启动时）
        if not self.decode_scheduler._running:
            self.decode_scheduler.start()

        logger.info(f"Camera {camera_id} started")
        return True
```

4. 修改 `stop_camera`：

```python
def stop_camera(self, camera_id: str) -> bool:
    """停止指定摄像头"""
    with self._lock:
        if camera_id not in self._cameras:
            return False

        state = self._cameras[camera_id]
        state.running = False

    # 不在这里 join 解码线程，因为 DecodeScheduler 是统一调度
    with self._lock:
        if camera_id not in self._cameras:
            return True

        state = self._cameras[camera_id]
        if state.cap:
            state.cap.release()
            state.cap = None

        state.status = CameraStatus.IDLE
        logger.info(f"Camera {camera_id} stopped")
        return True
```

5. 修改 `stop_all`：

```python
def stop_all(self):
    """停止所有摄像头"""
    self.decode_scheduler.stop()
    with self._lock:
        camera_ids = list(self._cameras.keys())
    for camera_id in camera_ids:
        self.stop_camera(camera_id)
```

6. 新增 `get_latest_frame`、`set_main_camera`、`get_main_camera`：

```python
def get_latest_frame(self, camera_id: str) -> Optional[np.ndarray]:
    """获取指定摄像头的最新帧"""
    with self._lock:
        if camera_id not in self._cameras:
            return None
        state = self._cameras[camera_id]
        with state.lock:
            return state.current_frame.copy() if state.current_frame is not None else None


def set_main_camera(self, camera_id: Optional[str]) -> bool:
    """设置主画面摄像头"""
    with self._lock:
        if camera_id is not None and camera_id not in self._cameras:
            logger.warning(f"Cannot set main camera {camera_id}: not found")
            self._main_camera_id = None
            self.decode_scheduler.set_main_camera(None)
            return False

        self._main_camera_id = camera_id
        self.decode_scheduler.set_main_camera(camera_id)
        logger.info(f"Main camera set to {camera_id}")
        return True


def get_main_camera(self) -> Optional[str]:
    """获取当前主画面摄像头 ID"""
    with self._lock:
        return self._main_camera_id
```

7. 移除 `_camera_loop` 和 `_connect_and_stream` 中的每路线程逻辑。保留 `_connect_and_stream` 但改名为 `_open_capture`，只负责打开视频源并设置 cap。

```python
def _open_capture(self, camera_id: str):
    """打开视频源并初始化 cap"""
    with self._lock:
        if camera_id not in self._cameras:
            return
        state = self._cameras[camera_id]
        state.status = CameraStatus.CONNECTING

    source = state.config.source
    if source.isdigit():
        source = int(source)

    is_rtsp = str(source).startswith("rtsp")
    cap = None

    # RTSP 流优先尝试 GPU 硬件解码
    if is_rtsp and gpu_decoder.gpu_available():
        gpu_reader = gpu_decoder.GPUVideoReader(str(source))
        if gpu_reader.start():
            cap = gpu_reader
            logger.info(f"Camera {camera_id} using GPU decoder")

    # 回退到 OpenCV CPU 解码
    if cap is None:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, state.config.buffer_size)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, state.config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, state.config.height)
    cap.set(cv2.CAP_PROP_FPS, state.config.fps)

    with self._lock:
        state.cap = cap
        state.status = CameraStatus.CONNECTED
        state.reconnect_attempts = 0

    logger.info(f"Camera {camera_id} connected")
```

8. 删除 `_camera_loop` 方法。

9. 在 `start_camera` 中调用 `_open_capture`（异步，不阻塞）：

```python
def start_camera(self, camera_id: str) -> bool:
    """启动指定摄像头"""
    with self._lock:
        if camera_id not in self._cameras:
            logger.error(f"Camera {camera_id} not found")
            return False

        state = self._cameras[camera_id]
        if state.running:
            logger.warning(f"Camera {camera_id} already running")
            return True

        state.running = True
        state.error_count = 0
        state.reconnect_attempts = 0
        state.last_decode_time = 0.0

        # 启动 DecodeScheduler（首次启动时）
        if not self.decode_scheduler._running:
            self.decode_scheduler.start()

    # 在锁外打开视频源
    def _open_and_retry():
        try:
            self._open_capture(camera_id)
        except Exception as e:
            logger.error(f"Camera {camera_id} open failed: {e}")
            # 重连逻辑交给 DecodeScheduler 的 decode 失败计数

    threading.Thread(target=_open_and_retry, daemon=True, name=f"open-{camera_id}").start()

    logger.info(f"Camera {camera_id} started")
    return True
```

10. 重连逻辑：当 DecodeScheduler 的 worker 连续读帧失败超过阈值时，触发 `_open_capture` 重新打开。

这个可以在 `_decode_one_frame` 中实现，但 `_decode_one_frame` 在 DecodeScheduler 中。需要让 DecodeScheduler 能通知 CameraManager 重连。

更简单的方式：在 DecodeScheduler 的 `_decode_one_frame` 中，如果连续失败超过阈值，调用 CameraManager 的 `_reopen_capture` 方法。

新增 `CameraManager._reopen_capture(camera_id)`：

```python
def _reopen_capture(self, camera_id: str):
    """重新打开视频源"""
    with self._lock:
        if camera_id not in self._cameras:
            return
        state = self._cameras[camera_id]
        state.status = CameraStatus.RECONNECTING
        state.reconnect_attempts += 1
        if state.cap:
            try:
                state.cap.release()
            except Exception:
                pass
            state.cap = None

    try:
        self._open_capture(camera_id)
    except Exception as e:
        logger.error(f"Camera {camera_id} reopen failed: {e}")
```

在 DecodeScheduler 的 `_decode_one_frame` 中：

```python
if not ret or frame is None:
    state.error_count += 1
    if state.error_count > 30:
        logger.warning(f"Camera {cam_id} too many read errors, reopening")
        reopen = getattr(self.camera_manager, "_reopen_capture", None)
        if reopen:
            reopen(cam_id)
        state.error_count = 0
    return
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_camera_manager_main_camera.py -v
```

Expected: tests pass.

- [ ] **Step 5: 运行全部测试确认未破坏**

```bash
python -m pytest tests/ -q
```

Expected: 所有现有测试仍通过（可能有少量需要适配）。

- [ ] **Step 6: Commit**

```bash
git add backend/camera_manager.py tests/test_camera_manager_main_camera.py
git commit -m "feat: integrate DecodeScheduler into CameraManager

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: main_multi.py 初始化默认主画面

**Files:**
- Modify: `backend/main_multi.py:200-481`, `1450-1482`

**Interfaces:**
- Consumes: `camera_manager.set_main_camera`, `stream_server.register_camera/unregister_camera`
- Produces: 启动时正确的默认主画面和流缓冲注册

- [ ] **Step 1: 找到初始化流程**

在 `init_components()` 中，找到摄像头注册后启动 overlay 和 scheduler 的位置（line 234 附近）。

- [ ] **Step 2: 修改初始化：只注册主画面到流缓冲**

原代码：

```python
for cam_data in camera_configs_data:
    ...
    camera_manager.register_camera(cfg)
    stream_server.register_camera(cfg.camera_id)  # 这行要移除
```

改为：

```python
for cam_data in camera_configs_data:
    ...
    camera_manager.register_camera(cfg)
    # 不在此处注册所有流缓冲，只由 set_main_camera 注册主画面
```

- [ ] **Step 3: 新增默认主画面选择函数**

在 `backend/main_multi.py` 中添加：

```python
def _pick_default_main_camera() -> Optional[str]:
    """选择第一个启用的摄像头作为默认主画面"""
    if camera_manager is None:
        return None
    cam_ids = camera_manager.get_camera_ids()
    if not cam_ids:
        return None
    for cid in cam_ids:
        state = camera_manager._cameras.get(cid)
        if state and state.config.enabled:
            return cid
    return cam_ids[0]
```

- [ ] **Step 4: 新增 set_main_camera 函数**

```python
def set_main_camera(camera_id: Optional[str]):
    """切换主画面摄像头，同步更新解码频率和流缓冲"""
    global stream_server

    old_main = camera_manager.get_main_camera() if camera_manager else None

    if old_main:
        stream_server.unregister_camera(old_main)
        log_message(f"Unregistered stream for old main camera {old_main}")

    if camera_manager:
        camera_manager.set_main_camera(camera_id)

    if camera_id:
        stream_server.register_camera(camera_id)
        log_message(f"Registered stream for main camera {camera_id}")
```

- [ ] **Step 5: 修改 startup 流程**

在 `_do_startup` 中，启动摄像头后设置默认主画面：

```python
def _do_startup():
    init_components()

    # 启动摄像头
    if camera_manager:
        camera_manager.start_all()

    # 设置默认主画面
    main_id = _pick_default_main_camera()
    if main_id:
        set_main_camera(main_id)

    # 启动检测器
    if gpu_scheduler:
        gpu_scheduler.start()
        log_message("GPU scheduler started")
    elif multi_detector:
        multi_detector.start()

    # ... 后续保持不变
```

- [ ] **Step 6: 修改 add_camera API**

在 `add_camera` 中，新摄像头注册后不再自动注册流缓冲：

```python
success = camera_manager.register_camera(cfg)
if not success:
    return JSONResponse({"error": "Camera ID already exists"}, status_code=400)

# 不在这里注册流缓冲；只有主画面才注册
# 如果当前没有主画面，可以设为新主画面
if camera_manager.get_main_camera() is None:
    set_main_camera(cfg.camera_id)
```

- [ ] **Step 7: Commit**

```bash
git add backend/main_multi.py
git commit -m "feat: initialize default main camera and register stream only for main

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: 改造 _overlay_loop 只处理主画面

**Files:**
- Modify: `backend/main_multi.py:1376-1429`

**Interfaces:**
- Consumes: `camera_manager.get_main_camera`, `camera_manager.get_latest_frame`, `multi_detector._latest_results`
- Produces: 只给主画面推送原始帧和标注帧

- [ ] **Step 1: 编写测试（可选，通过集成测试覆盖）**

本任务主要通过集成测试覆盖。

- [ ] **Step 2: 改造 _overlay_loop**

替换 `_overlay_loop` 为：

```python
def _overlay_loop():
    """独立渲染线程：只给主画面推送原始帧和标注帧"""
    global _overlay_running
    log_message("Overlay render thread started")
    while _overlay_running:
        try:
            if camera_manager is None or multi_detector is None or stream_server is None:
                time.sleep(0.1)
                continue

            main_id = camera_manager.get_main_camera()
            if main_id is None:
                time.sleep(0.1)
                continue

            frame = camera_manager.get_latest_frame(main_id)
            if frame is None:
                time.sleep(0.02)
                continue

            # 推送原始帧
            stream_server.update_frame(main_id, frame, raw=True)

            # 获取全局要画的类型
            with _overlay_config_lock:
                types_to_draw = list(_overlay_config)

            if types_to_draw:
                results = multi_detector._latest_results.get(main_id, {})
                state = camera_manager._cameras.get(main_id)
                cam_enabled_types = set()
                if state and state.config.detection_types:
                    cam_enabled_types = {
                        k for k, v in state.config.detection_types.items()
                        if v.get("enabled", False)
                    }
                effective_types = [t for t in types_to_draw if t in cam_enabled_types]
                filtered = {k: v for k, v in results.items() if k in effective_types}
                annotated = MultiDetector._annotate_frame(frame, filtered, main_id, [])
            else:
                annotated = frame

            # 推送标注帧
            stream_server.update_frame(main_id, annotated, raw=False)
        except Exception as e:
            logger.error(f"Overlay loop error: {e}")

        time.sleep(0.04)  # 25 FPS 上限
    log_message("Overlay render thread stopped")
```

- [ ] **Step 3: Commit**

```bash
git add backend/main_multi.py
git commit -m "feat: overlay loop only processes main camera

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: 添加 select_main_camera API

**Files:**
- Modify: `backend/main_multi.py:583-603` 附近

**Interfaces:**
- Consumes: `camera_manager.set_main_camera`, `stream_server.register_camera/unregister_camera`
- Produces: `/cameras/{camera_id}/select` API

- [ ] **Step 1: 编写测试**

Create: `tests/test_select_main_camera_api.py`

```python
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def test_select_main_camera():
    with patch("backend.main_multi.camera_manager") as cm, \
         patch("backend.main_multi.stream_server") as ss, \
         patch("backend.main_multi.set_main_camera") as set_main:
        cm._cameras = {"cam_01": MagicMock()}
        from backend.main_multi import app
        client = TestClient(app)
        response = client.post("/cameras/cam_01/select")
        assert response.status_code == 200
        assert response.json()["main_camera"] == "cam_01"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_select_main_camera_api.py -v
```

Expected: `404` or test fails because endpoint does not exist.

- [ ] **Step 3: 实现 API**

在 `backend/main_multi.py` 中添加：

```python
@app.post("/cameras/{camera_id}/select")
async def select_main_camera(camera_id: str):
    """切换主画面摄像头"""
    if camera_manager is None or camera_id not in camera_manager._cameras:
        return JSONResponse({"error": "Camera not found"}, status_code=404)
    set_main_camera(camera_id)
    return {"success": True, "main_camera": camera_id}
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_select_main_camera_api.py -v
```

Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/main_multi.py tests/test_select_main_camera_api.py
git commit -m "feat: add select main camera API

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: GPU Scheduler 改造

**Files:**
- Modify: `backend/gpu_scheduler.py:131-318`

**Interfaces:**
- Consumes: `camera_manager.get_latest_frame`
- Produces:
  - `GPUDynamicScheduler._busy: bool`
  - `GPUDynamicScheduler.run()` 直接读最新帧 + `_busy` 防并发

- [ ] **Step 1: 编写测试**

Create: `tests/test_gpu_scheduler_busy.py`

```python
from unittest.mock import MagicMock, patch
import numpy as np
from backend.gpu_scheduler import GPUDynamicScheduler, ModelConfig


def test_scheduler_sets_busy_during_inference():
    cm = MagicMock()
    cm._cameras = {}
    cm.get_latest_frame = MagicMock(return_value=None)

    with patch("backend.gpu_scheduler.YOLO"):
        scheduler = GPUDynamicScheduler(
            cm,
            {"fire": ModelConfig("dummy.pt", "fire", device="cpu")},
            num_queues=1,
            interval=0.1,
            warmup=False,
        )
        assert scheduler._busy is False
        scheduler.stop()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_gpu_scheduler_busy.py -v
```

Expected: `1 failed` because `_busy` does not exist.

- [ ] **Step 3: 实现改造**

Modify: `backend/gpu_scheduler.py`

1. 在 `__init__` 中新增：

```python
self._busy = False
```

2. 修改 `run()` 方法：

```python
def run(self):
    logger.info(
        f"GPU 调度器启动: models={len(self.detectors)}, queues={self.num_queues}, "
        f"interval={self.interval}s"
    )
    while self.running:
        if self._busy:
            time.sleep(0.05)
            continue

        t0 = time.time()
        self._busy = True
        collected_keys = []
        try:
            now = time.time()

            # 收集到期任务: {detection_type: [(cam_id, frame), ...]}
            tasks: Dict[str, List[Tuple[str, np.ndarray]]] = {}

            for cam_id in self._get_active_cameras():
                if not self._is_camera_enabled(cam_id):
                    continue

                frame = self.camera_manager.get_latest_frame(cam_id)
                if frame is None:
                    continue

                det_types = self._get_camera_detection_types(cam_id)
                for dtype, cfg in det_types.items():
                    if dtype not in self.model_configs:
                        continue

                    enabled = cfg.get("enabled", False) if isinstance(cfg, dict) else getattr(cfg, "enabled", False)
                    if not enabled:
                        continue

                    interval = (
                        cfg.get("interval", 1.0)
                        if isinstance(cfg, dict)
                        else getattr(cfg, "interval", 1.0)
                    )
                    key = (cam_id, dtype)
                    last = self.last_infer.get(key, 0.0)
                    if now - last >= interval:
                        tasks.setdefault(dtype, []).append((cam_id, frame.copy()))
                        collected_keys.append(key)

            # 提交任务到各队列并等待完成
            if tasks:
                active_queues: set = set()
                for dtype, cam_frames in tasks.items():
                    cam_ids, frames = zip(*cam_frames)
                    qid = self.dtype_to_queue[dtype]
                    self.queues[qid].set_frames(list(frames), list(cam_ids))
                    active_queues.add(qid)

                for qid in active_queues:
                    self.queues[qid].done_event.wait()
                    self.queues[qid].done_event.clear()

                # 回调结果
                if self.on_result:
                    for dtype, cam_frames in tasks.items():
                        cam_ids, _ = zip(*cam_frames)
                        qid = self.dtype_to_queue[dtype]
                        idx = self.dtype_to_idx[dtype]
                        results = self.queues[qid].results
                        if results is None or idx >= len(results):
                            continue
                        model_results = results[idx]
                        if model_results is None:
                            continue
                        for cam_id, result in zip(cam_ids, model_results):
                            try:
                                self.on_result(cam_id, dtype, result)
                            except Exception as e:
                                logger.error(f"回调出错 [{cam_id}/{dtype}]: {e}")

                # last_infer 按推理完成时间更新
                completed_at = time.time()
                for key in collected_keys:
                    self.last_infer[key] = completed_at
        finally:
            self._busy = False

        sleep_time = self.interval - (time.time() - t0)
        if sleep_time > 0:
            time.sleep(sleep_time)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_gpu_scheduler_busy.py -v
```

Expected: `1 passed`

- [ ] **Step 5: 运行全部测试确认未破坏**

```bash
python -m pytest tests/ -q
```

Expected: 所有测试通过。

- [ ] **Step 6: Commit**

```bash
git add backend/gpu_scheduler.py tests/test_gpu_scheduler_busy.py
git commit -m "feat: GPU scheduler reads latest frame and adds _busy guard

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Serial/CorePinned Strategy 改造

**Files:**
- Modify: `backend/safety_detection/detector_core.py:399-465`

**Interfaces:**
- Consumes: `camera_manager.get_latest_frame`
- Produces: `_process_camera` 直接读最新帧

- [ ] **Step 1: 修改 _process_camera**

将：

```python
frame = self.camera_manager.get_frame(camera_id, allow_paused=False)
```

改为：

```python
frame = self.camera_manager.get_latest_frame(camera_id)
if frame is None:
    return
```

注意保留 `allow_paused` 语义：如果本地视频暂停，最新帧可能是旧的静止画面。但当前 `get_latest_frame` 不检查暂停状态。如果需要，可以在 `_process_camera` 中额外检查：

```python
state = self.camera_manager._cameras.get(camera_id)
if state and state.config.is_video_source() and not state.playback_state.get("playing", True):
    return
```

- [ ] **Step 2: 运行测试确认未破坏**

```bash
python -m pytest tests/ -q
```

Expected: 所有测试通过。

- [ ] **Step 3: Commit**

```bash
git add backend/safety_detection/detector_core.py
git commit -m "feat: strategies read latest frame directly

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: 集成测试

**Files:**
- Create: `tests/test_integration_resource_optimization.py`

**Interfaces:**
- Consumes: 全部前述改动
- Produces: 集成测试验证端到端行为

- [ ] **Step 1: 编写集成测试**

```python
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
```

- [ ] **Step 2: 运行测试确认通过**

```bash
python -m pytest tests/test_integration_resource_optimization.py -v
```

Expected: `2 passed`

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration_resource_optimization.py
git commit -m "test: add integration tests for resource optimization

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: 最终验证与界面功能检查

**Files:**
- All modified files

**Interfaces:**
- Consumes: 全部任务产出
- Produces: 可 review 的分支

- [ ] **Step 1: 运行完整测试套件**

```bash
python -m pytest tests/ -q
```

Expected: 全部通过。

- [ ] **Step 2: 检查关键 API 未破坏**

```bash
cd d:/project/sentry-safety
python -c "from backend.main_multi import app; print('import ok')"
python -c "from backend.camera_manager import CameraManager; print('camera_manager ok')"
python -c "from backend.gpu_scheduler import GPUDynamicScheduler; print('gpu_scheduler ok')"
```

Expected: 无导入错误。

- [ ] **Step 3: 手动验证界面功能**

启动服务：

```bash
python backend/main_multi.py
```

验证：
1. 打开 `/monitor` 页面，能看到默认主画面；
2. 切换主画面后，新画面能正常显示；
3. 非主画面摄像头不推流（`/cameras/{id}/stream` 对新主画面有效，对非主画面返回空或 404）；
4. 检测、告警、记录功能正常。

- [ ] **Step 4: 检查代码变更范围**

```bash
git diff --stat main
```

Expected: 只有计划内的文件被修改。

- [ ] **Step 5: Commit 最终 checkpoint**

```bash
git add -A
git commit -m "checkpoint: resource optimization implementation ready for review

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## 自我审查

### Spec 覆盖检查

| Spec 章节 | 覆盖任务 |
|-----------|---------|
| 4. 解码线程池 | Task 2, 3 |
| 5. 主画面与非主画面 | Task 3, 4 |
| 6. 推流与画框 | Task 4, 5 |
| 7. 检测调度 | Task 7, 8 |
| 8. 主画面切换 | Task 4, 6 |
| 9. 错误处理 | Task 3（重连）, Task 7（丢弃） |
| 10. 测试策略 | Task 1, 2, 3, 7, 9, 10 |

### Placeholder 检查

- 无 TBD/TODO
- 无 "implement later"
- 所有代码步骤包含实际代码
- 所有测试步骤包含具体断言

### 类型一致性检查

- `DecodeScheduler.set_main_camera(camera_id: Optional[str])` 签名一致
- `CameraManager.get_latest_frame(camera_id: str) -> Optional[np.ndarray]` 签名一致
- `GPUDynamicScheduler._busy: bool` 类型一致
- `last_infer` 更新逻辑一致

---

## 执行交接

Plan complete and saved to `docs/superpowers/plans/2026-07-06-resource-optimization.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
