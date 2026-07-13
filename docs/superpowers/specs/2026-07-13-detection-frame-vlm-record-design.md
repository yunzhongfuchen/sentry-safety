# 检测帧统一驱动 VLM 复核与记录帧序列设计

## 日期

2026-07-13

## 背景

当前 VLM 复核只提交触发瞬间的一张原始解码帧，告警记录里的视频帧序列则来自解码历史窗口（`CameraState.frame_history`）。这两者都不是“检测真正用到的那帧”，导致：

- VLM 复核缺少连续命中过程的上下文；
- 记录帧序列可能包含与告警无关的帧；
- 原始解码历史占用大量内存。

本设计目标：让 VLM 复核和记录帧序列都使用每次检测命中时的帧，按 `consecutive_required` 收集，并统一压缩存储。

## 目标

1. VLM 复核收到 `consecutive_required` 张检测帧（最多 5 张），例如间隔 10s、连续帧 3 时收到第 0s、10s、20s 的检测帧。
2. 告警记录的视频帧序列改用同样的检测帧序列。
3. 快照（snapshot）保持不变，仍为最后一张带检测框的帧。
4. 新增系统设置开关 `save_image_timestamp`，控制保存的快照和记录帧是否叠加时间戳，默认开启。
5. 移除原始解码帧历史 `CameraState.frame_history`，降低内存占用。

## 非目标

- 不改前端 `/monitor` 视频流展示逻辑。
- 不改检测模型推理流程本身。
- 不改动 VLM 提示词模板机制。

## 关键概念

- **检测帧**：每次检测调度到期时，真正送进 `safety_detector.detect()` 的那一张解码帧。
- **检测帧缓存**：每个摄像头、每个启用类型各自维护的近期命中帧队列，存 JPEG 字节，避免内存膨胀。
- **`consecutive_required`**：检测类型配置里的连续命中次数，决定缓存长度。
- **`MAX_VLM_REVIEW_FRAMES`**：VLM 复核最多提交的帧数，固定为 5，超过时取最近 5 张。

## 架构设计

### 数据流

```
DecodeScheduler 解码帧
       │
       ▼
MultiDetector._process_camera()
       │
       ├── 检测命中 ──► CameraManager.add_detection_frame(cid, dtype, ts, jpeg)
       │                   （未命中或阈值不足时清空）
       │
       └── 达到 consecutive_required 且过冷却 ──► 触发告警
                               │
                               ▼
              result["detection_frames"] = CameraManager.get_detection_frames(cid, dtype)
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
           snapshot       record frames       VLM review
         （带框+可选      （JPEG 字节直接      （最近最多 5 张
          时间戳）          落盘）              解码后提交）
```

### 帧存储位置

| 缓存       | 位置                                    | 内容                        | 生命周期                    |
| ---------- | --------------------------------------- | --------------------------- | --------------------------- |
| 当前解码帧 | `CameraState.current_frame`           | 1 张 numpy 原图             | 每次解码更新                |
| 检测帧缓存 | `CameraState.detection_frames[dtype]` | `deque[(ts, jpeg_bytes)]` | 命中写入，未命中/触发后清空 |

原始 `CameraState.frame_history` 将被移除。

## 详细设计

### 1. CameraState 与 CameraManager 改动

**新增字段**

```python
@dataclass
class CameraState:
    config: CameraConfig
    ...
    detection_frames: Dict[str, "deque[Tuple[float, bytes]]"] = field(default_factory=dict)
```

**新增方法**

```python
def add_detection_frame(
    self,
    camera_id: str,
    dtype: str,
    timestamp: float,
    jpeg_bytes: bytes,
    maxlen: int,
) -> None:
    """把一次命中检测帧写入该摄像头该类型的缓存队列。"""


def clear_detection_frames(self, camera_id: str, dtype: str) -> None:
    """清空指定类型的检测帧缓存。"""


def get_detection_frames(
    self,
    camera_id: str,
    dtype: str,
) -> List[Tuple[float, bytes]]:
    """按时间升序返回检测帧序列。"""
```

`add_detection_frame` 内部应检查该类型现有 deque 的 `maxlen` 是否与传入的 `maxlen` 一致；不一致时创建新的 deque 并保留最近 `maxlen` 个元素，避免配置变更后缓存长度错误。

**缓存清空时机**

除检测未命中、阈值不足、告警触发外，以下生命周期变化时也应清空 `detection_frames`：

- `CameraManager.unregister_camera(camera_id)` —— 摄像头注销，释放所有类型缓存；
- `CameraManager.stop_camera(camera_id)` —— 摄像头停止，流已关闭，缓存失效；
- `MultiDetector.register_camera` / `update_camera_config` 前 —— 重新注册该摄像头检测类型前，先清空旧缓存。

**移除**

- `CameraState.frame_history`
- `CameraManager._frame_history_maxlen()`
- `CameraManager._recreate_frame_history()`
- `CameraManager.get_window_frames()`

### 2. DecodeScheduler 改动

移除 `_decode_one_frame` 中把帧写入 `state.frame_history` 的代码。

### 3. MultiDetector 改动

#### 3.1 类型配置

`TypeSchedule` 移除 `history_frames`，睡岗检测也统一使用 `CameraState.detection_frames`。所有检测帧统一下沉到 `CameraState.detection_frames`。`register_camera` 时清空该摄像头的 `detection_frames`。

#### 3.2 编码辅助与时间戳工具

新增独立工具函数（建议放在 `backend/frame_utils.py`），避免 `detector_core` 反向依赖 `main_multi`：

```python
def draw_timestamp_on_frame(frame: np.ndarray, timestamp: float) -> np.ndarray:
    """在帧右上角绘制时间戳（YYYY-MM-DD HH:MM:SS，白字黑边）。"""


def encode_frame_to_jpg(frame: np.ndarray, quality: int, draw_ts: bool, timestamp: float) -> bytes:
    """把检测帧编码为 JPEG 字节，按需叠加时间戳。"""
```

`MultiDetector` 在初始化时持有全局设置引用，或每次调用 `config.load_global_settings()` 获取 `frame_quality` 与 `save_image_timestamp`。

#### 3.3 标准检测处理

```python
def _handle_standard_detection(...):
    detected = result.get("detected", False)
    max_conf = max(result.get("scores", [0]) or [0])

    if not detected or max_conf < schedule.threshold:
        self.camera_manager.clear_detection_frames(camera_id, dtype)
        schedule.consecutive_count = 0
        return

    schedule.consecutive_count += 1

    # 编码并写入检测帧缓存
    ts = time.time()
    settings = config.load_global_settings()
    jpeg_bytes = encode_frame_to_jpg(
        frame,
        quality=settings.get("frame_quality", 60),
        draw_ts=settings.get("save_image_timestamp", True),
        timestamp=ts,
    )
    self.camera_manager.add_detection_frame(
        camera_id, dtype, ts, jpeg_bytes, maxlen=schedule.consecutive_required
    )

    if schedule.consecutive_count < schedule.consecutive_required:
        return

    if self.is_in_cooldown(...):
        return

    # 触发告警
    self._cooldowns[camera_id][dtype] = time.time()
    result["level"] = "small_model_alarm"
    result["detection_frames"] = self.camera_manager.get_detection_frames(camera_id, dtype)

    if schedule.use_vlm:
        result["pending_vlm_review"] = True
        vlm_frames = result["detection_frames"][-MAX_VLM_REVIEW_FRAMES:]
        self._submit_vlm_review(camera_id, dtype, vlm_frames, schedule, result)

    if self.trigger_callback:
        self.trigger_callback(camera_id, dtype, frame, result)

    # 触发后清空缓存
    self.camera_manager.clear_detection_frames(camera_id, dtype)
```

#### 3.4 睡岗检测处理

睡岗检测不再使用独立的“宽容递减”状态机，统一为标准检测逻辑：

- 命中时 `consecutive_count += 1`，并把检测帧写入 `detection_frames["sleep"]`；
- 未命中或阈值不足时，清空 `consecutive_count` 和 `detection_frames["sleep"]`；
- 达到 `consecutive_required` 且过冷却时触发告警，触发后清空缓存。

原 `TypeSchedule.history_frames` 及睡岗专用历史帧逻辑删除。

### 4. VLM 复核改动

`_submit_vlm_review` 签名改为接收帧列表：

```python
def _submit_vlm_review(
    self,
    camera_id: str,
    dtype: str,
    frames: List[Tuple[float, bytes]],
    schedule: TypeSchedule,
    result: dict,
) -> None:
    numpy_frames = [cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR) for _, jpg in frames]
    task = {
        ...
        "frames": numpy_frames,
        "prompt_type": f"{dtype}_review",
        ...
    }
```

### 5. 记录保存改动

`on_trigger` 不再调用 `get_window_frames`，改为：

```python
detection_frames = result.get("detection_frames", [])
record["frame_count"] = len(detection_frames)

if _save_executor is not None:
    _save_executor.submit(
        _save_detection_frames_async,
        record_id,
        detection_frames,
        _global_settings.get("frame_quality", 60),
    )
else:
    _save_detection_frames_async(record_id, detection_frames, ...)
```

`_save_detection_frames_async` 直接写 JPEG 字节到磁盘，每张帧文件名保留时间戳或序号。

快照逻辑保持不变，仅在 `_annotate_frame` 后根据 `save_image_timestamp` 开关决定是否叠加时间戳。

### 6. 系统设置改动

#### 6.1 后端

`config.py` 的 `DEFAULT_GLOBAL_SETTINGS` 增加：

```python
"save_image_timestamp": True,
```

#### 6.2 前端

`frontend/safety_detection/settings.html` 的“图像质量”卡片增加：

```html
<label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
    <input type="checkbox" v-model="settings.save_image_timestamp" />
    保存图像时叠加时间戳
</label>
```

`saveSystemSettings()` 的 payload 增加 `save_image_timestamp`。

## 内存估算

假设 96 路全部启用、每路 6 个类型、默认 `consecutive_required=3`：

| 项目       | 原设计                                 | 新设计                         |
| ---------- | -------------------------------------- | ------------------------------ |
| 解码历史帧 | 非主画面 96 × 10 × 0.88 MB ≈ 845 MB | 移除                           |
| 检测帧缓存 | 无                                     | 96 × 6 × 3 × 50 KB ≈ 84 MB |
| 当前解码帧 | 96 × 0.88 MB ≈ 84 MB                 | 96 × 0.88 MB ≈ 84 MB         |

整体内存显著下降。

## 风险与回退

| 风险                                       | 缓解                                                                                |
| ------------------------------------------ | ----------------------------------------------------------------------------------- |
| JPEG 压缩后再解码送给 VLM 有轻微画质损失   | 复核场景对画质不敏感，可接受                                                        |
| 每次命中都进行 JPEG 编码增加 CPU 开销      | 仅在命中时编码，且可通过降低`frame_quality` 控制；96 路全开高频命中场景需实际压测 |
| 连续命中过程中摄像头切换主画面导致帧率变化 | 检测调度已按`last_run` 对齐，不影响                                               |
| 缓存清空时机错误导致 VLM 收到旧帧          | 单元测试覆盖命中/中断/触发三种状态                                                  |

## 测试计划

1. `tests/test_detection_frame_buffer.py`

   - 命中 3 次后 `detection_frames` 长度为 3；
   - 未命中一次后缓存清空；
   - 阈值未通过时缓存清空。
2. `tests/test_vlm_review_frames.py`

   - `consecutive_required=7` 时 VLM 任务只收到最近 5 张；
   - `consecutive_required=3` 时收到 3 张。
3. `tests/test_record_detection_frames.py`

   - 告警记录 `frame_count` 等于 `consecutive_required`；
   - 保存的文件数与 `frame_count` 一致。
4. `tests/test_save_image_timestamp.py`

   - `save_image_timestamp=True` 时保存帧包含时间戳；
   - `save_image_timestamp=False` 时不包含。

## 待实现文件清单

- `backend/camera_manager.py`
- `backend/camera_state.py`（如 `CameraState` 已独立）
- `backend/decode_scheduler.py`
- `backend/safety_detection/detector_core.py`
- `backend/main_multi.py`
- `backend/config.py`
- `frontend/safety_detection/settings.html`
- `tests/test_detection_frame_buffer.py`
- `tests/test_vlm_review_frames.py`
- `tests/test_record_detection_frames.py`
- `tests/test_save_image_timestamp.py`
