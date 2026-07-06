# 多路摄像头检测资源优化设计

**日期**: 2026-07-06  
**状态**: 设计待审  
**目标**: 前端只显示一路主画面时，仅对主画面推流和画框；非主画面降低解码频率；检测逻辑对所有摄像头保持一致。

---

## 1. 背景与问题

当前 `camera_manager.py` 为每个已注册摄像头启动一个独立解码线程，持续以配置帧率（默认 15 FPS）调用 `cap.read()`。多路同时运行时存在以下资源浪费：

- **解码资源浪费**：非主画面摄像头也在持续解码，但前端只显示一路主画面。
- **画框资源浪费**：`_overlay_loop` 遍历多路摄像头生成带框标注帧，但前端只消费其中一路。
- **流缓冲内存浪费**：`MJPEGStreamServer` 为多路摄像头维护 `raw + annotated` 双缓冲区。
- **线程开销**：每路一个解码线程，96 路场景下线程数和上下文切换开销较大。

本设计将解码、推流、画框与检测逻辑分离：

- **主画面**只影响「推流 + 画框」；
- **检测逻辑**对所有摄像头一致，按各自配置 `interval` 执行；
- **非主画面**不解码时不推流、不画框。

---

## 2. 设计目标

1. 主画面实时推流保留，最高 25 FPS。
2. 画框只在主画面上进行，画框内容可由用户设置。
3. 非主画面固定 1 秒解码一次，不解码时不空转。
4. 所有摄像头的检测逻辑一致，不受是否主画面影响。
5. 降低解码线程数：从「每路一个线程」改为「统一解码线程池」。
6. 推理侧直接读最新帧，不追旧帧，避免积压。
7. 保留告警、快照、记录逻辑不变。

---

## 3. 总体架构

```
┌─────────────────┐
│  前端 monitor   │── 只订阅 /cameras/{mainId}/stream
└─────────────────┘
         │
         ▼
┌─────────────────┐      持续解码（≤25 FPS）    ┌─────────────────┐
│ DecodeScheduler │◄──── 主画面摄像头 ──────────►│  stream_server  │
│  (thread pool)  │                            │ (raw+annotated) │
└─────────────────┘                            └────────┬────────┘
         │                                              │
         │ 获取最新帧                                    │ MJPEG
         ▼                                              ▼
┌─────────────────┐                            ┌─────────────────┐
│   _overlay_loop │── 画框 ───────────────────►│     前端预览    │
│  (仅主画面)      │                            └─────────────────┘
└─────────────────┘

┌─────────────────┐
│  非主画面摄像头  │── 1 秒解码一次 ─────────►  current_frame
└─────────────────┘                              │
                                                 ▼
                                       ┌─────────────────┐
                                       │  检测调度器      │
                                       │  GPU: batch     │
                                       │  非 GPU: 串行   │
                                       └────────┬────────┘
                                                │
                                                ▼
                                       ┌─────────────────┐
                                       │  _latest_results │
                                       │  告警/录像/记录   │
                                       └─────────────────┘
```

---

## 4. 解码线程池（DecodeScheduler）

所有摄像头共享一个解码线程池，不再每路一个线程。

### 4.1 新增组件

```python
class DecodeScheduler:
    def __init__(self, camera_manager, num_workers: int = 4):
        self.camera_manager = camera_manager
        self.num_workers = num_workers
        self._main_camera: Optional[str] = None
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._running = False
```

`CameraState` 新增字段：

```python
last_decode_time: float = 0.0
decode_queued: bool = False  # 避免同一摄像头重复入队
```

### 4.2 调度逻辑

一个调度线程持续把所有待解码的摄像头放入优先队列：

```python
def _schedule_loop(self):
    while self._running:
        now = time.time()
        for cam_id, state in self.camera_manager.iter_cameras():
            if not state.running or state.cap is None:
                continue

            is_main = cam_id == self._main_camera
            interval = 1.0 / 25 if is_main else 1.0
            due_time = state.last_decode_time + interval

            if now >= due_time and not state.decode_queued:
                priority = 0 if is_main else 1
                self._queue.put((priority, due_time, cam_id))
                state.decode_queued = True

        time.sleep(0.01)
```

### 4.3 Worker 逻辑

线程池中的 worker 从队列取任务，读取一帧并更新状态：

```python
def _worker_loop(self):
    while self._running:
        try:
            priority, due_time, cam_id = self._queue.get(timeout=0.1)
        except queue.Empty:
            continue

        state = self.camera_manager._cameras.get(cam_id)
        if state is None or not state.running or state.cap is None:
            continue

        ret, frame = state.cap.read()

        with state.lock:
            state.decode_queued = False
            state.last_decode_time = due_time

        if not ret or frame is None:
            # 原有错误/重连逻辑
            continue

        with state.lock:
            state.current_frame = frame
            state.frame_count += 1
            if cam_id != self._main_camera:
                state.frame_history.append((time.time(), frame.copy()))
```

### 4.4 线程池大小

固定大小，通过配置可调整，默认 4。96 路场景下可配到 8。

---

## 5. 主画面与非主画面

### 5.1 主画面

- 解码频率：最高 25 FPS；
- 注册 `MJPEGStreamServer` 缓冲；
- `_overlay_loop` 只给主画面画框并推流；
- 检测仍按自身配置的 `interval` 执行；
- 画框内容由用户设置决定。

### 5.2 非主画面

- 解码频率：固定 1 秒一次；
- 不注册流缓冲、不画框、不推流；
- 检测按自身配置的 `interval` 执行；
- 解码帧顺手写入 `frame_history`，用于告警前 5 秒窗口回溯。

### 5.3 关键原则

**是否主画面只影响「推流 + 画框」，不影响「检测调度」。**

---

## 6. 推流与画框

### 6.1 只给主画面注册流缓冲

```python
def set_main_camera(camera_id: Optional[str]):
    old = camera_manager.get_main_camera()
    if old:
        stream_server.unregister_camera(old)

    camera_manager.set_main_camera(camera_id)

    if camera_id:
        stream_server.register_camera(camera_id)
```

### 6.2 `_overlay_loop` 只处理主画面

```python
def _overlay_loop():
    while _overlay_running:
        main_id = camera_manager.get_main_camera()
        if main_id is None:
            time.sleep(0.1)
            continue

        frame = camera_manager.get_latest_frame(main_id)
        if frame is None:
            time.sleep(0.02)
            continue

        stream_server.update_frame(main_id, frame, raw=True)

        results = multi_detector.get_latest_results(main_id)
        annotated = multi_detector.annotate(frame, results, main_id)
        stream_server.update_frame(main_id, annotated, raw=False)

        time.sleep(0.04)  # 25 FPS 上限
```

---

## 7. 检测调度

所有摄像头检测逻辑一致，按配置的 `interval` 到期取最新帧推理。

### 7.1 GPU 模式：GPUDynamicScheduler

条件：`detect_best_device()` 返回 cuda，或配置显式指定 `gpu`。

改造点：

1. **直接读最新帧**：每次收集到期任务时，直接取 `current_frame`。
2. **`_busy` 防并发**：上一轮推理未完成时，跳过本轮，避免多个 batch 同时占 GPU。
3. **`last_infer` 按完成时间更新**：推理完成后统一更新。

```python
def run(self):
    while self.running:
        if self._busy:
            time.sleep(0.05)
            continue

        t0 = time.time()
        self._busy = True
        try:
            now = time.time()
            tasks = self._collect_due_frames(now)
            if tasks:
                self._infer_batch(tasks)
                completed_at = time.time()
                for key in collected_keys:
                    self.last_infer[key] = completed_at
        finally:
            self._busy = False

        sleep_time = self.interval - (time.time() - t0)
        if sleep_time > 0:
            time.sleep(sleep_time)
```

### 7.2 非 GPU 模式：串行调度器

条件：未检测到 cuda，或配置显式指定 `serial`。

单线程按顺序遍历摄像头，对每路只处理已到期的检测类型：

```python
def _worker_loop(self):
    while self.running:
        for cam_id in self.camera_manager.get_camera_ids():
            if not self._is_camera_enabled(cam_id):
                continue

            frame = self.camera_manager.get_latest_frame(cam_id)
            if frame is None:
                continue

            for dtype in self._get_due_types(cam_id):
                result = self._infer(cam_id, frame, dtype)
                self._update_last_infer(cam_id, dtype)
                self._on_result(cam_id, dtype, result)

        time.sleep(0.1)
```

### 7.3 统一取帧接口

`CameraManager` 提供：

```python
def get_latest_frame(self, camera_id: str) -> Optional[np.ndarray]:
    state = self._cameras.get(camera_id)
    if state is None:
        return None
    with state.lock:
        return state.current_frame.copy() if state.current_frame is not None else None
```

---

## 8. 主画面切换

用户在前端手动选择主画面后，后端执行：

```python
@app.post("/cameras/{camera_id}/select")
async def select_main_camera(camera_id: str):
    if camera_id not in camera_manager._cameras:
        return JSONResponse({"error": "Camera not found"}, status_code=404)
    set_main_camera(camera_id)
    return {"success": True, "main_camera": camera_id}
```

切换流程：

1. 旧主画面：注销流缓冲，降为 1 秒解码；
2. 新主画面：升为 25 FPS 解码，注册流缓冲；
3. 前端切到新的 `/cameras/{newId}/stream` URL；
4. 若新主画面离线，前端显示离线状态，不保留旧画面。

---

## 9. 错误处理

| 场景 | 处理 |
|---|---|
| 新主画面离线 | 返回离线状态，前端显示离线，不保留旧画面 |
| 推理慢/堆积 | 直接读最新帧，不追旧帧；GPU 模式加 `_busy` 防并发 |
| 某路 `cap.read()` 阻塞 | 线程池还有其他 worker，不影响全局；超时时跳过 |
| 非主画面检测轮次被跳过 | 可接受，下一周期用最新帧继续 |
| 系统启动/热更新 | 停止线程池和推理调度，按新配置重新初始化 |

---

## 10. 测试策略

### 10.1 单元测试

- `DecodeScheduler` 按主画面 25 FPS、非主画面 1 FPS 调度；
- `set_main_camera` 切换后旧画面降级、新画面升级；
- `get_latest_frame` 返回最新帧；
- `GPUDynamicScheduler` 推理期间 `_busy` 为 True。

### 10.2 集成测试

- 启动 4 路模拟摄像头；
- 验证只有主画面注册流缓冲；
- 切换主画面，验证流 URL 变化、旧摄像头降级；
- 触发告警，验证快照和 `frame_history` 正常。

### 10.3 压力测试

- 96 路 mock 视频源，运行 10 分钟；
- 验证解码线程数稳定（等于线程池大小）；
- 验证内存无持续增长；
- 验证非主画面约 1 FPS 解码。

---

## 11. 设计决策记录

| 决策 | 选项 | 选择 | 理由 |
|---|---|---|---|
| 非主画面解码频率 | A. 按最短检测 interval；B. 固定 1 秒；C. 持续解码 | B | 实现简单，足够覆盖大多数检测需求 |
| 解码线程模型 | A. 每路一个线程；B. 统一单线程；C. 线程池 | C | 96 路场景下大幅降低线程数，同时避免单线程阻塞 |
| 主画面是否影响检测 | A. 主画面更频繁检测；B. 所有摄像头检测一致 | B | 主画面只影响推流/画框，简化逻辑 |
| 主画面线程安排 | A. 独立线程；B. 在线程池内统一调度 | B | 架构统一，调度通过频率区分 |
| 推理取帧策略 | A. 收集并等待特定帧；B. 直接读最新帧 | B | 简单，天然不追旧帧 |
| GPU 防并发 | A. 不限制；B. `_busy` 标志跳过 | B | 避免多个 batch 同时占 GPU |
| 切换失败 fallback | A. 保留旧画面；B. 显示离线 | B | 用户要求直接显示离线 |

---

*设计文档待审，审批通过后进入 implementation plan 阶段。*
