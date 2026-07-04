# 96 路 GPU 边缘部署按需解码设计

**日期**: 2026-07-04  
**状态**: 设计待审  
**目标**: 在保留现有检测、告警、快照、录像功能的前提下，为 96 路摄像头场景优化边缘端 GPU 资源消耗，并为后续算能（Sophon/BM1684X）NPU 迁移预留扩展点。

---

## 1. 背景与问题

当前 `camera_manager.py` 为每个已注册摄像头启动一个独立解码线程，持续以配置帧率（默认 15 FPS）调用 `cap.read()`。在 96 路边缘部署场景下，这会导致：

- **解码资源浪费**：非主画面摄像头也在全速解码，但前端只显示一路主画面。
- **画框资源浪费**：`_overlay_loop` 每 40ms 遍历全部摄像头，为每路生成带框标注帧，但前端只消费其中一路。
- **流缓冲内存浪费**：`MJPEGStreamServer` 为每路摄像头维护 `raw + annotated` 双缓冲区。
- **推理侧等待**：`MultiDetector` / `GPUDynamicScheduler` 从 `camera_manager.current_frame` 取帧时，大部分帧并不对应任何到期的检测任务。

本设计将解码、画框、推理、流服务按“主画面”和“非主画面”分离，实现**按需解码 + batch 推理 + 主动丢弃**。

---

## 2. 设计目标

1. **不堆积**：推理压力大时，主动丢弃到期帧，不形成队列或旧帧积压。
2. **推理效率高**：96 路场景下优先使用 `GPUDynamicScheduler` 按检测类型做 batch 推理。
3. **保留检测功能**：所有摄像头继续按配置的 `interval` 进行检测，告警、快照、记录逻辑不变。
4. **保留直播功能**：前端主画面仍能实时预览，且最高 25 FPS。
5. **兼容快照与稀疏窗口**：触发快照使用检测帧；触发前 5 秒窗口对非主画面为稀疏窗口（FPS = 1 / 最短检测间隔）。
6. **可迁移到算能 NPU**：推理后端抽象，调度逻辑与设备无关。

---

## 3. 总体架构

```
┌─────────────────┐
│  前端 monitor   │── 只订阅 /cameras/{mainId}/stream
└─────────────────┘
         │
         ▼
┌─────────────────┐     持续解码（≤25 FPS）     ┌─────────────────┐
│ camera_manager  │◄──── 主画面摄像头 ────────►│  stream_server  │
│  (main camera)  │                            │ (raw+annotated) │
└─────────────────┘                            └────────┬────────┘
         │                                              │
         │ 获取最新帧                                    │ MJPEG
         ▼                                              ▼
┌─────────────────┐                            ┌─────────────────┐
│   _overlay_loop │── 画框 ───────────────────►│     前端预览    │
│  (仅主画面)      │                            └─────────────────┘
└─────────────────┘

┌─────────────────┐
│  95 路非主画面  │── 按需解码（SCHEDULED）──►  request_frame()
└─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ GPUDynamicScheduler │
                    │  - 收集到期帧      │
                    │  - batch 推理      │
                    │  - 丢弃旧帧/未完成 │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  _latest_results │
                    │  告警/录像/记录   │
                    └─────────────────┘
```

---

## 4. 解码模式（CameraState）

新增 `DecodeMode`：

```python
class DecodeMode(Enum):
    CONTINUOUS = "continuous"   # 主画面：持续解码，最高 25 FPS
    SCHEDULED = "scheduled"     # 非主画面：按需解码，解完就睡
```

`CameraState` 新增字段：

```python
decode_mode: DecodeMode = DecodeMode.SCHEDULED
frame_request_event: threading.Event = field(default_factory=threading.Event)
frame_ready_event: threading.Event = field(default_factory=threading.Event)
current_scheduled_frame: Optional[np.ndarray] = None
```

### 4.1 CONTINUOUS 模式

- 保持现有循环，但限制最大帧率为 25 FPS：

```python
loop_start = time.perf_counter()
ret, frame = cap.read()
if ret and frame is not None:
    with state.lock:
        state.current_frame = frame
        state.frame_history.append((time.time(), frame.copy()))

# 限制最大 25 FPS
elapsed = time.perf_counter() - loop_start
sleep_time = max(0, 1.0 / 25 - elapsed)
if sleep_time > 0:
    time.sleep(sleep_time)
```

### 4.2 SCHEDULED 模式

- 解码线程等待 `frame_request_event`，被唤醒后读取一帧，写入 `current_scheduled_frame`，然后继续等待。
- 不主动维持 `frame_history`，由请求方在拿到帧后决定是否追加。

```python
while state.running:
    state.frame_request_event.wait()
    if not state.running:
        break
    state.frame_request_event.clear()

    ret, frame = cap.read()
    if ret and frame is not None:
        with state.lock:
            state.current_scheduled_frame = frame
        state.frame_ready_event.set()
```

### 4.3 request_frame 统一入口

```python
def request_frame(camera_id: str, timeout: float = 1.0,
                  store_history: bool = True) -> Optional[np.ndarray]:
    state = self._cameras.get(camera_id)
    if state is None:
        return None

    if state.decode_mode == DecodeMode.CONTINUOUS:
        with state.lock:
            return state.current_frame.copy() if state.current_frame is not None else None

    # SCHEDULED：触发一帧解码
    state.frame_request_event.set()
    if state.frame_ready_event.wait(timeout=timeout):
        state.frame_ready_event.clear()
        with state.lock:
            frame = state.current_scheduled_frame
            state.current_scheduled_frame = None

        if frame is not None and store_history:
            # 顺手写入历史窗口，触发告警时可回溯稀疏窗口
            state.frame_history.append((time.time(), frame.copy()))
        return frame
    return None
```

**注**：SCHEDULED 模式下，一帧出来后会同时用于：
1. 当前检测输入；
2. `frame_history` 稀疏窗口（可选，默认开启）；
3. 临时切为主画面时的最新预览帧。

若某摄像头启用了多种检测类型且 `interval` 不同，解码频率由**最短 interval** 决定，因为任一类型到期都会触发 `request_frame()`。

---

## 5. 主画面切换（promote / demote）

当用户在前端切换主画面时，后端执行：

```python
def set_main_camera(camera_id: Optional[str]):
    old_main = _main_camera_id

    if old_main and old_main in camera_manager._cameras:
        camera_manager.set_decode_mode(old_main, DecodeMode.SCHEDULED)
        stream_server.unregister_camera(old_main)
        log_message(f"Demoted {old_main} to scheduled decode")

    if camera_id and camera_id in camera_manager._cameras:
        camera_manager.set_decode_mode(camera_id, DecodeMode.CONTINUOUS)
        stream_server.register_camera(camera_id)
        _main_camera_id = camera_id
        log_message(f"Promoted {camera_id} to main decode")
```

- 旧主画面降级：释放流缓冲，停止持续解码。
- 新主画面升级：注册流缓冲，开始持续解码（≤25 FPS）。
- 切换期间前端显示占位黑屏，约 0.3-1s 后恢复。

---

## 6. 推理侧：GPUDynamicScheduler + 丢弃策略

### 6.1 为什么用 GPUDynamicScheduler

`SerialStrategy` 虽然简单，但单帧 GPU 推理开销大，96 路 1 秒间隔难以满足。`GPUDynamicScheduler` 能把同一检测类型的多路帧 batch 进一次模型调用，显著提高效率。

### 6.2 防堆积设计

在现有 `GPUDynamicScheduler` 基础上增加三条规则：

#### 规则 1：推理未完成时跳过新一轮

```python
if self._busy:
    time.sleep(0.05)
    continue
self._busy = True
try:
    due_frames = self._collect_due_frames()
    if due_frames:
        self._infer_batch(due_frames)
finally:
    self._busy = False
```

#### 规则 2：丢弃过旧帧

```python
frame_age = now - frame_capture_time
if frame_age > MAX_FRAME_AGE:  # 建议 0.5s
    continue
```

#### 规则 3：last_infer 按推理完成时间更新

```python
# 收集时只记录 key
# 推理完成后统一更新
for key in collected_keys:
    self.last_infer[key] = time.time()
```

这样推理慢时，中间到期的帧会被直接丢弃，系统用最新帧继续，不会处理 3 秒前的旧帧。

### 6.3 与 MultiDetector 的协作

- `GPUDynamicScheduler` 负责收集帧、batch 推理、回调结果。
- `MultiDetector` 负责维护 `_latest_results`、告警状态机、连续计数、冷却等逻辑。
- 被 `GPUDynamicScheduler` 接管的检测类型，在 `MultiDetector` 中标记为 `externally_managed=True`，避免重复推理：

```python
# GPUDynamicScheduler 只接管有对应模型配置的类型
managed_dtypes = set(model_configs.keys())
for cam_id in camera_manager.get_camera_ids():
    dtypes = camera_manager.get_detection_types(cam_id)
    to_manage = [d for d in dtypes if d in managed_dtypes]
    multi_detector.mark_externally_managed(cam_id, to_manage)
```

- 未被接管的类型（如依赖 VLM 或特殊后处理的类型）仍由 `MultiDetector` 内部调度处理。

---

## 7. 流服务与画框优化

### 7.1 只给主画面注册流缓冲

```python
# 初始化时注册主画面
stream_server.register_camera(main_camera_id)

# 非主画面不注册
# 只有主画面切换时才注册/注销
```

### 7.2 _overlay_loop 只处理主画面

```python
def _overlay_loop():
    while _overlay_running:
        if _main_camera_id is None:
            time.sleep(0.1)
            continue

        frame = camera_manager.request_frame(_main_camera_id)
        if frame is None:
            continue

        stream_server.update_frame(_main_camera_id, frame, raw=True)

        results = multi_detector._latest_results.get(_main_camera_id, {})
        annotated = MultiDetector._annotate_frame(frame, results, _main_camera_id, [])
        stream_server.update_frame(_main_camera_id, annotated, raw=False)

        time.sleep(0.04)  # 25 FPS 上限
```

### 7.3 内存收益

| 项目 | 改造前 | 改造后 |
|------|--------|--------|
| 流缓冲 | 96 路 × 2 缓冲 | 1 路 × 2 缓冲 |
| overlay CPU | 每 40ms 处理 96 路 | 每 40ms 处理 1 路 |
| 解码线程 | 96 路持续跑 | 1 路持续跑 + 95 路按需唤醒 |

---

## 8. 功能兼容性

| 功能 | 影响 | 处理 |
|------|------|------|
| 快照 | 无影响 | 触发时检测帧仍存在，正常保存 |
| 触发前 5 秒窗口 | 非主画面变稀疏 | 按需解码出的帧顺手追加到 `frame_history`，稀疏 FPS = 1 / 最短检测间隔；窗口为空时 fallback 到触发帧 |
| 推理间隔设置 | 无影响，更直接 | interval 直接控制解码频率 |
| 多路摄像头 | 无影响 | 注册/注销/状态查询/热更新全部保留 |
| 睡岗连续计数 | 无影响 | sleep 本身 60s 一次，按需解码频率足够 |
| VLM 巡检 | 可选关闭 | 边缘端建议 `vlm_inspection_interval=0` |
| 事件录像 | 保留 | 触发帧和窗口帧正常保存 |
| 24h 全量录像 | 需额外处理 | 若需要，对录像摄像头单独设为 CONTINUOUS 低帧率模式 |

---

## 9. GPU → 算能（Sophon）迁移扩展

### 9.1 后端抽象接口

新增推理后端抽象，第一步先提供接口而不替换现有 `ModelDetector`，避免一次改动过大：

```python
class InferenceBackend(ABC):
    @abstractmethod
    def load_model(self, model_path: str, device: str): ...

    @abstractmethod
    def predict_batch(self, frames: List[np.ndarray], dtype: str) -> List[Any]: ...

class YoloCudaBackend(InferenceBackend):
    """包装当前 ModelDetector，先保证行为一致"""
    def __init__(self, cfg: ModelConfig):
        self._detector = ModelDetector(cfg)

    def predict_batch(self, frames, dtype):
        return self._detector.predict(frames)

class SophonBMBackend(InferenceBackend):
    """后续实现，替换 YOLO/Ultralytics 为 sophon.Inference"""
    ...
```

### 9.2 迁移步骤

1. **第一阶段**：引入 `InferenceBackend` 接口，`YoloCudaBackend` 内部仍调用现有 `ModelDetector`，验证行为一致。
2. **第二阶段**：`GPUDynamicScheduler` 改为接收 `InferenceBackend` 实例列表，不再直接依赖 `ModelDetector`。
3. **第三阶段**：实现 `SophonBMBackend`，替换模型加载和 `predict_batch`，调度、解码、告警逻辑不变。

解码、流服务、告警、记录逻辑无需改动。

---

## 10. 错误处理

| 场景 | 处理 |
|------|------|
| 主画面切换时新摄像头离线 | 显示占位黑屏；升级失败则保留旧主画面 |
| 按需解码请求超时 | `request_frame` 返回 None，本轮跳过该摄像头 |
| 推理耗时超过 interval | 本轮丢弃，等下一轮；`last_infer` 按完成时间更新 |
| 摄像头模式切换 | 复用已打开的 `cap`，不重新连接 |
| 系统重启/热更新 | 停止 overlay 和 scheduler，按新配置重新注册模式 |
| GPU 内存不足 | 降低 batch size 或分辨率，或 fallback 到串行 |

---

## 11. 测试策略

### 11.1 单元测试

- `request_frame()` 在 CONTINUOUS 和 SCHEDULED 模式下返回正确帧。
- `set_main_camera()` 切换后模式、流缓冲注册/注销正确。
- `GPUDynamicScheduler` 在推理慢时不会重复收集同一摄像头。
- 非主画面按需解码时，`frame_history` 按间隔写入。

### 11.2 集成测试

- 启动 4 路模拟摄像头，验证非主画面按 interval 解码。
- 切换主画面，验证 stream URL 正常、旧摄像头降级。
- 触发告警，验证快照和稀疏窗口帧正常保存。

### 11.3 压力测试

- 96 路 mock 视频源，运行 10 分钟，验证：
  - 内存无持续增长
  - 非主画面解码线程无空转
  - GPU 利用率在合理范围
  - 无旧帧积压（检查 `frame_age`）

---

## 12. 后续实施计划

设计审批通过后，将按以下顺序实现：

1. 改造 `CameraState` 和 `camera_manager.py`，支持 `CONTINUOUS` / `SCHEDULED` 模式。
2. 实现 `request_frame()` 和主画面切换 `promote/demote`。
3. 改造 `_overlay_loop` 和 `MJPEGStreamServer`，只处理主画面。
4. 改造 `GPUDynamicScheduler`，加入防堆积策略。
5. 在按需解码路径中追加 `frame_history` 兼容。
6. 增加 `InferenceBackend` 抽象接口。
7. 补充单元测试和集成测试。
8. 在 GPU 环境做 96 路压力测试并调参。

---

## 13. 设计决策记录

| 决策 | 选项 | 选择 | 理由 |
|------|------|------|------|
| 非主画面解码策略 | A. 持续降频到 1 FPS；B. 按需解码；C. 运动触发 | B | 精确匹配检测 interval，最省资源 |
| 推理模式 | A. 串行；B. GPUDynamicScheduler batch | B | 96 路 GPU 串行难以满足 1s 间隔 |
| 防堆积 | A. 无限排队；B. 主动丢弃 | B | 边缘端不能积压旧帧 |
| 画框范围 | A. 所有摄像头；B. 仅主画面 | B | 前端只显示一路，画其他路是浪费 |
| 稀疏窗口 | A. 持续解码补窗口；B. 用检测帧凑稀疏窗口 | B | 不增加解码成本 |

---

*设计文档待审，审批通过后进入 implementation plan 阶段。*
