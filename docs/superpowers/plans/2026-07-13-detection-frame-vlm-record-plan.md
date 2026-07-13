# 检测帧统一驱动 VLM 复核与记录帧序列实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 VLM 复核和告警记录帧序列统一使用每次检测命中的帧，移除原始解码帧历史，并新增保存图像时间戳开关。

**Architecture:** 在 `CameraState` 新增按检测类型隔离的 `detection_frames` 缓存，命中时由 `MultiDetector` 写入 JPEG 字节；触发告警时把缓存交给 VLM 和记录保存；同时移除 `frame_history` 及其消费路径。

**Tech Stack:** Python 3.12, OpenCV, FastAPI, Vue 3

## Global Constraints

- Python 3.12，Conda 环境 `py312`。
- 不改动前端 `/monitor` 视频流展示逻辑。
- 不改动检测模型推理流程本身。
- 不改动 VLM 提示词模板机制。
- 每次变更后运行相关测试，全部通过后再进入下一步。
- 优先修改现有文件，新增文件需配合同步新增测试。

---

## File Map

| 文件 | 职责 |
|------|------|
| `backend/frame_utils.py`（新建） | 时间戳绘制、JPEG 编码辅助函数 |
| `backend/camera_manager.py` | `CameraState.detection_frames` 缓存；移除 `frame_history` |
| `backend/decode_scheduler.py` | 停止向 `frame_history` 写入 |
| `backend/safety_detection/detector_core.py` | 命中时写检测帧缓存；VLM 复核接收多帧；移除 `history_frames` |
| `backend/main_multi.py` | 告警记录帧序列改从 `detection_frames` 取；快照时间戳开关；移除窗口帧保存 |
| `backend/config.py` | 默认全局设置增加 `save_image_timestamp` |
| `frontend/safety_detection/settings.html` | 系统设置增加时间戳开关 |

---

### Task 1: 创建帧工具模块

**Files:**
- Create: `backend/frame_utils.py`
- Modify: `backend/main_multi.py`（后续 Task 5 替换 `_draw_timestamp_on_frame` 引用）
- Test: `tests/test_frame_utils.py`

**Interfaces:**
- Produces: `draw_timestamp_on_frame(frame: np.ndarray, timestamp: float) -> np.ndarray`
- Produces: `encode_frame_to_jpg(frame: np.ndarray, quality: int, draw_ts: bool, timestamp: float) -> bytes`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_frame_utils.py
import time
import cv2
import numpy as np
from backend.frame_utils import draw_timestamp_on_frame, encode_frame_to_jpg


def test_draw_timestamp_on_frame_adds_text():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    ts = time.mktime(time.strptime("2026-07-13 12:00:00", "%Y-%m-%d %H:%M:%S"))
    out = draw_timestamp_on_frame(frame, ts)
    assert out.shape == frame.shape


def test_encode_frame_to_jpg_without_timestamp():
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    data = encode_frame_to_jpg(frame, quality=80, draw_ts=False, timestamp=time.time())
    assert isinstance(data, bytes)
    assert len(data) > 0
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None


def test_encode_frame_to_jpg_with_timestamp():
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    ts = time.mktime(time.strptime("2026-07-13 12:00:00", "%Y-%m-%d %H:%M:%S"))
    data = encode_frame_to_jpg(frame, quality=80, draw_ts=True, timestamp=ts)
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_frame_utils.py -v
```

Expected: `ModuleNotFoundError: No module named 'backend.frame_utils'`

- [ ] **Step 3: 实现最小代码**

```python
# backend/frame_utils.py
import time
import cv2
import numpy as np


def draw_timestamp_on_frame(frame: np.ndarray, timestamp: float) -> np.ndarray:
    """在帧右上角绘制时间戳（白字黑边）。"""
    h, w = frame.shape[:2]
    text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(w, h) / 800.0)
    thickness = max(1, int(min(w, h) / 400))
    margin = max(8, int(min(w, h) * 0.015))

    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = w - text_w - margin
    y = text_h + margin

    cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def encode_frame_to_jpg(frame: np.ndarray, quality: int, draw_ts: bool, timestamp: float) -> bytes:
    """把帧编码为 JPEG 字节，按需叠加时间戳。"""
    img = frame.copy()
    if draw_ts:
        img = draw_timestamp_on_frame(img, timestamp)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_frame_utils.py -v
```

Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add backend/frame_utils.py tests/test_frame_utils.py
git commit -m "feat: add frame timestamp and jpeg encoding utilities"
```

---

### Task 2: CameraManager 新增 detection_frames 缓存并移除 frame_history

**Files:**
- Modify: `backend/camera_manager.py`
- Test: `tests/test_camera_manager_detection_frames.py`

**Interfaces:**
- Produces: `CameraManager.add_detection_frame(camera_id, dtype, timestamp, jpeg_bytes, maxlen) -> None`
- Produces: `CameraManager.clear_detection_frames(camera_id, dtype) -> None`
- Produces: `CameraManager.get_detection_frames(camera_id, dtype) -> List[Tuple[float, bytes]]`
- Produces: `CameraManager.clear_all_detection_frames(camera_id) -> None`
- Consumes: `deque` from `collections`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_camera_manager_detection_frames.py
import time
import pytest
from backend.camera_manager import CameraManager, CameraConfig


def _register_and_start(cm, cid="cam1"):
    cfg = CameraConfig(camera_id=cid, source="dummy")
    cm.register_camera(cfg)
    # 不需要真启动视频流，直接操作内部 state
    return cm._cameras[cid]


def test_add_detection_frame_and_get():
    cm = CameraManager()
    _register_and_start(cm)
    ts = time.time()
    cm.add_detection_frame("cam1", "fire", ts, b"frame1", maxlen=3)
    frames = cm.get_detection_frames("cam1", "fire")
    assert len(frames) == 1
    assert frames[0] == (ts, b"frame1")


def test_detection_frames_maxlen():
    cm = CameraManager()
    _register_and_start(cm)
    for i in range(5):
        cm.add_detection_frame("cam1", "fire", time.time() + i, f"frame{i}".encode(), maxlen=3)
    frames = cm.get_detection_frames("cam1", "fire")
    assert len(frames) == 3
    assert frames[0][1] == b"frame2"


def test_clear_detection_frames():
    cm = CameraManager()
    _register_and_start(cm)
    cm.add_detection_frame("cam1", "fire", time.time(), b"f", maxlen=3)
    cm.clear_detection_frames("cam1", "fire")
    assert cm.get_detection_frames("cam1", "fire") == []


def test_unregister_clears_detection_frames():
    cm = CameraManager()
    _register_and_start(cm)
    cm.add_detection_frame("cam1", "fire", time.time(), b"f", maxlen=3)
    cm.unregister_camera("cam1")
    assert "cam1" not in cm._cameras


def test_frame_history_removed():
    cm = CameraManager()
    state = _register_and_start(cm)
    assert not hasattr(state, "frame_history")
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_camera_manager_detection_frames.py -v
```

Expected: 属性/方法不存在错误。

- [ ] **Step 3: 修改 CameraState 与 CameraManager**

在 `backend/camera_manager.py` 中：

1. `CameraState` 移除 `frame_history` 字段，新增：

```python
# 检测命中帧缓存：dtype -> deque[(timestamp, jpeg_bytes)]
detection_frames: Dict[str, "deque[Tuple[float, bytes]]"] = field(default_factory=dict)
```

2. 删除 `_frame_history_maxlen` 和 `_recreate_frame_history` 方法。

3. 删除 `get_window_frames` 方法。

4. `start_camera` 中删除创建 `state.frame_history` 的代码。

5. `stop_camera` 中在停止线程后清空 `state.detection_frames`：

```python
with self._lock:
    if camera_id in self._cameras:
        self._cameras[camera_id].detection_frames.clear()
```

6. `unregister_camera` 中已有 `pop`，无需额外处理，但确保 `stop_camera` 先被调用。

7. 新增方法：

```python
def add_detection_frame(
    self,
    camera_id: str,
    dtype: str,
    timestamp: float,
    jpeg_bytes: bytes,
    maxlen: int,
) -> None:
    with self._lock:
        if camera_id not in self._cameras:
            return
        state = self._cameras[camera_id]
        with state.lock:
            existing = state.detection_frames.get(dtype)
            if existing is None or existing.maxlen != maxlen:
                existing = deque(maxlen=maxlen)
                state.detection_frames[dtype] = existing
            existing.append((timestamp, jpeg_bytes))


def clear_detection_frames(self, camera_id: str, dtype: str) -> None:
    with self._lock:
        if camera_id not in self._cameras:
            return
        state = self._cameras[camera_id]
        with state.lock:
            if dtype in state.detection_frames:
                state.detection_frames[dtype].clear()


def get_detection_frames(self, camera_id: str, dtype: str) -> List[Tuple[float, bytes]]:
    with self._lock:
        if camera_id not in self._cameras:
            return []
        state = self._cameras[camera_id]
        with state.lock:
            return list(state.detection_frames.get(dtype, []))


def clear_all_detection_frames(self, camera_id: str) -> None:
    with self._lock:
        if camera_id not in self._cameras:
            return
        state = self._cameras[camera_id]
        with state.lock:
            state.detection_frames.clear()
```

- [ ] **Step 4: 运行新增测试与现有 CameraManager 测试**

```bash
pytest tests/test_camera_manager_detection_frames.py tests/test_camera_manager_fields.py tests/test_camera_manager_main_camera.py -v
```

Expected: all passed

- [ ] **Step 5: 提交**

```bash
git add backend/camera_manager.py tests/test_camera_manager_detection_frames.py
git commit -m "feat: per-camera per-type detection frame buffer, remove decoded frame history"
```

---

### Task 3: DecodeScheduler 停止写入 frame_history

**Files:**
- Modify: `backend/decode_scheduler.py`
- Test: `tests/test_decode_scheduler.py`（更新现有断言）

**Interfaces:**
- Consumes: `CameraState` 不再期望有 `frame_history`。

- [ ] **Step 1: 修改 DecodeScheduler**

在 `backend/decode_scheduler.py` 的 `_decode_one_frame` 中，删除类似代码：

```python
state.frame_history.append((current_time, frame.copy()))
```

只保留更新 `state.current_frame`、`state.frame_count`、`fps_stats` 的逻辑。

- [ ] **Step 2: 更新现有测试**

打开 `tests/test_decode_scheduler.py`，删除或修改所有检查 `frame_history` 长度/内容的断言。确保测试只验证：

- 解码后 `current_frame` 被更新；
- `frame_count` 增加；
- `decode_queued` 状态正确。

- [ ] **Step 3: 运行测试**

```bash
pytest tests/test_decode_scheduler.py -v
```

Expected: all passed

- [ ] **Step 4: 提交**

```bash
git add backend/decode_scheduler.py tests/test_decode_scheduler.py
git commit -m "refactor: stop writing decoded frames to frame_history"
```

---

### Task 4: MultiDetector 使用 detection_frames 并更新 VLM 复核

**Files:**
- Modify: `backend/safety_detection/detector_core.py`
- Test: `tests/test_detector_core_detection_frames.py`

**Interfaces:**
- Consumes: `CameraManager.add_detection_frame`, `clear_detection_frames`, `get_detection_frames`
- Consumes: `backend.frame_utils.encode_frame_to_jpg`
- Modifies: `_submit_vlm_review(camera_id, dtype, frames, schedule, result)` 其中 `frames: List[Tuple[float, bytes]]`
- Removes: `TypeSchedule.history_frames`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_detector_core_detection_frames.py
import time
import numpy as np
import pytest
from unittest.mock import Mock
from backend.safety_detection.detector_core import MultiDetector, TypeSchedule


def make_frame():
    return np.zeros((60, 80, 3), dtype=np.uint8)


def test_standard_detection_collects_frames():
    camera_manager = Mock()
    camera_manager.get_detection_frames.return_value = []
    md = MultiDetector(camera_manager=camera_manager, safety_detector=None, vlm_queue=None, strategy=None)
    md.register_camera("cam1", {
        "fire": {"enabled": True, "interval": 1, "threshold": 0.5, "cooldown": 60, "consecutive_required": 3, "use_vlm": False},
    })
    schedule = md._schedules["cam1"]["fire"]
    result = {"detected": True, "scores": [0.9]}

    md._handle_standard_detection("cam1", "fire", make_frame(), result, schedule)
    md._handle_standard_detection("cam1", "fire", make_frame(), result, schedule)
    md._handle_standard_detection("cam1", "fire", make_frame(), result, schedule)

    assert schedule.consecutive_count == 3
    assert camera_manager.add_detection_frame.call_count == 3
    assert camera_manager.get_detection_frames.called


def test_no_detection_clears_frames():
    camera_manager = Mock()
    md = MultiDetector(camera_manager=camera_manager, safety_detector=None, vlm_queue=None, strategy=None)
    md.register_camera("cam1", {
        "fire": {"enabled": True, "interval": 1, "threshold": 0.5, "cooldown": 60, "consecutive_required": 3, "use_vlm": False},
    })
    schedule = md._schedules["cam1"]["fire"]
    md._handle_standard_detection("cam1", "fire", make_frame(), {"detected": True, "scores": [0.9]}, schedule)
    md._handle_standard_detection("cam1", "fire", make_frame(), {"detected": False}, schedule)
    assert camera_manager.clear_detection_frames.called
    assert schedule.consecutive_count == 0


def test_vlm_review_gets_recent_five_frames():
    vlm_queue = Mock()
    camera_manager = Mock()
    md = MultiDetector(camera_manager=camera_manager, safety_detector=None, vlm_queue=vlm_queue, strategy=None)
    md.register_camera("cam1", {
        "fire": {"enabled": True, "interval": 1, "threshold": 0.5, "cooldown": 60, "consecutive_required": 7, "use_vlm": True},
    })
    schedule = md._schedules["cam1"]["fire"]
    frames = [(time.time() + i, b"frame" + str(i).encode()) for i in range(7)]
    camera_manager.get_detection_frames.return_value = frames

    result = {"detected": True, "scores": [0.9]}
    for _ in range(7):
        md._handle_standard_detection("cam1", "fire", make_frame(), result, schedule)

    submitted = vlm_queue.submit.call_args[1]["task"]
    assert len(submitted["frames"]) == 5
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_detector_core_detection_frames.py -v
```

Expected: 方法不存在或断言失败。

- [ ] **Step 3: 修改 detector_core.py**

1. 导入：

```python
from backend.frame_utils import encode_frame_to_jpg
from backend import config
```

2. `TypeSchedule` 移除 `history_frames` 字段。

3. `register_camera` 注册完成后调用 `self.camera_manager.clear_all_detection_frames(camera_id)`（先实现该辅助方法，或循环调用 `clear_detection_frames`）。如果 `camera_manager` 为 `None`（测试场景），跳过。

4. `_handle_standard_detection` 按设计文档改造：命中且过阈值时编码并写入缓存；未命中/阈值不足时清空；触发时把缓存放入 `result["detection_frames"]`；`use_vlm` 时取最近最多 5 张调用 `_submit_vlm_review`。

5. `_handle_sleep_detection` 统一为标准逻辑：移除宽容递减，命中写缓存，未命中清空，触发后清空。

6. `_submit_vlm_review` 签名改为：

```python
def _submit_vlm_review(
    self, camera_id: str, dtype: str,
    frames: List[Tuple[float, bytes]],
    schedule: TypeSchedule, result: dict
) -> None:
```

内部解码：

```python
numpy_frames = [
    cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    for _, jpg in frames
]
```

- [ ] **Step 4: 运行测试**

```bash
pytest tests/test_detector_core_detection_frames.py tests/test_detector_core_config.py -v
```

Expected: all passed

- [ ] **Step 5: 提交**

```bash
git add backend/safety_detection/detector_core.py tests/test_detector_core_detection_frames.py
git commit -m "feat: collect detection frames and submit up to 5 to VLM review"
```

---

### Task 5: main_multi 集成 detection_frames 与时间戳开关

**Files:**
- Modify: `backend/main_multi.py`
- Test: `tests/test_main_multi_detection_frames.py`

**Interfaces:**
- Consumes: `backend.frame_utils.draw_timestamp_on_frame`
- Consumes: `result["detection_frames"]` from `on_trigger`
- Replaces: `_save_window_frames_async` -> `_save_detection_frames_async`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_main_multi_detection_frames.py
import time
import numpy as np
import pytest
from unittest.mock import Mock, patch
from backend import main_multi


def test_on_trigger_uses_detection_frames():
    main_multi._global_settings = {"frame_quality": 60, "save_image_timestamp": True, "max_records": 100, "emergency_cleanup_ratio": 0.2}
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    detection_frames = [(time.time(), b"f1"), (time.time(), b"f2")]
    result = {
        "detected": True,
        "scores": [0.9],
        "level": "small_model_alarm",
        "detection_frames": detection_frames,
    }
    with patch.object(main_multi.storage, "save_image") as mock_save:
        main_multi.on_trigger("cam1", "fire", frame, result)
        # snapshot + 2 detection frames
        assert mock_save.call_count == 3


def test_save_detection_frames_async_writes_bytes():
    main_multi._global_settings = {"frame_quality": 60, "save_image_timestamp": True}
    frames = [(time.time(), b"f1"), (time.time(), b"f2")]
    with patch.object(main_multi.storage, "save_image") as mock_save:
        main_multi._save_detection_frames_async("rec1", frames, 60)
        assert mock_save.call_count == 2
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_main_multi_detection_frames.py -v
```

Expected: 函数不存在或断言失败。

- [ ] **Step 3: 修改 main_multi.py**

1. 导入：

```python
from backend.frame_utils import draw_timestamp_on_frame
```

2. 删除旧的 `_draw_timestamp_on_frame` 函数定义。

3. `on_trigger` 改造：

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
    _save_detection_frames_async(
        record_id,
        detection_frames,
        _global_settings.get("frame_quality", 60),
    )
```

4. 替换 `_save_window_frames_async` 为 `_save_detection_frames_async`：

```python
def _save_detection_frames_async(
    record_id: str,
    detection_frames: List[Tuple[float, bytes]],
    quality: int,
):
    """后台保存检测帧序列。"""
    try:
        for i, (ts, jpg_bytes) in enumerate(detection_frames):
            storage.save_image(record_id, "frame", jpg_bytes, i)
        log_message(f"Saved {len(detection_frames)} detection frames for {record_id}")
    except Exception as e:
        logger.error(f"Failed to save detection frames for {record_id}: {e}")
```

5. 快照时间戳开关：

```python
if frame is not None:
    trigger_results = {dtype: result}
    annotated = MultiDetector._annotate_frame(frame, trigger_results, camera_id, [])
    if _global_settings.get("save_image_timestamp", True):
        annotated = draw_timestamp_on_frame(annotated.copy(), trigger_ts)
    snapshot_bytes = encode_frame_to_bytes(annotated, quality=_global_settings.get("snapshot_quality", 70))
    storage.save_image(record_id, "snapshot", snapshot_bytes)
```

6. 删除 `get_window_frames` 调用及旧窗口帧逻辑。

- [ ] **Step 4: 运行测试**

```bash
pytest tests/test_main_multi_detection_frames.py tests/test_main_multi_records.py -v
```

Expected: all passed

- [ ] **Step 5: 提交**

```bash
git add backend/main_multi.py tests/test_main_multi_detection_frames.py
git commit -m "feat: use detection frames for record sequence and timestamp switch"
```

---

### Task 6: config.py 增加默认设置

**Files:**
- Modify: `backend/config.py`
- Test: `tests/test_config_defaults.py`

**Interfaces:**
- Produces: `DEFAULT_GLOBAL_SETTINGS["save_image_timestamp"]` = `True`

- [ ] **Step 1: 修改 config.py**

在 `DEFAULT_GLOBAL_SETTINGS` 中增加：

```python
"save_image_timestamp": True,
```

- [ ] **Step 2: 更新测试**

在 `tests/test_config_defaults.py` 中增加断言：

```python
def test_global_settings_has_save_image_timestamp():
    from backend import config
    assert "save_image_timestamp" in config.DEFAULT_GLOBAL_SETTINGS
    assert config.DEFAULT_GLOBAL_SETTINGS["save_image_timestamp"] is True
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/test_config_defaults.py -v
```

Expected: all passed

- [ ] **Step 4: 提交**

```bash
git add backend/config.py tests/test_config_defaults.py
git commit -m "feat: add save_image_timestamp default global setting"
```

---

### Task 7: 前端系统设置增加时间戳开关

**Files:**
- Modify: `frontend/safety_detection/settings.html`

**Interfaces:**
- `settings.save_image_timestamp` boolean
- `saveSystemSettings()` payload includes `save_image_timestamp`

- [ ] **Step 1: 在图像质量卡片增加开关**

在 `frontend/safety_detection/settings.html` 的“图像质量” `div.glass-card` 中，在“帧质量”输入框下方增加：

```html
<div class="gc-form-field" style="display: flex; align-items: center; justify-content: space-between;">
    <label>保存图像时叠加时间戳</label>
    <input type="checkbox" v-model="settings.save_image_timestamp" />
</div>
```

- [ ] **Step 2: 在 saveSystemSettings payload 中增加字段**

```javascript
const payload = {
    ...
    snapshot_quality: settings.value.snapshot_quality,
    frame_quality: settings.value.frame_quality,
    save_image_timestamp: settings.value.save_image_timestamp,
    ...
};
```

- [ ] **Step 3: 手动验证**

启动服务，打开 `/settings.html?tab=system`，确认：

- “保存图像时叠加时间戳”复选框存在；
- 切换后点击保存，请求 payload 包含 `save_image_timestamp`；
- 后端 `/settings` 返回的 `settings` 包含该字段。

- [ ] **Step 4: 提交**

```bash
git add frontend/safety_detection/settings.html
git commit -m "feat: frontend toggle for save_image_timestamp"
```

---

### Task 8: 全量回归与内存验证

**Files:**
- All modified files

- [ ] **Step 1: 运行全量测试**

```bash
pytest tests/ -q
```

Expected: `79 passed`（或更多，含新增测试）

- [ ] **Step 2: 运行内存/集成冒烟测试**

```bash
pytest tests/test_integration_resource_optimization.py -v
```

Expected: all passed

- [ ] **Step 3: 启动服务做端到端验证**

```bash
bash start.sh
```

在浏览器中：

1. 触发一个 fire 检测（可用测试视频或模拟接口）；
2. 查看 `/records.html`，确认记录详情里有 `consecutive_required` 张帧；
3. 查看 `data/frames/` 目录，确认帧文件名为 `*_frame_000.jpg`、`*_frame_001.jpg` 等；
4. 在系统设置关闭时间戳开关，再次触发告警，确认帧上无时间戳。

- [ ] **Step 4: 提交任何微调**

```bash
git add ...
git commit -m "test: verify full detection-frame flow end-to-end"
```

---

## Self-Review Checklist

- [ ] Spec coverage：每个设计点（detection_frames 缓存、VLM 多帧、记录帧替换、时间戳开关、frame_history 移除）都有对应任务。
- [ ] Placeholder scan：计划中没有 TBD/TODO，每个步骤都有具体代码或命令。
- [ ] Type consistency：`add_detection_frame` / `clear_detection_frames` / `get_detection_frames` 签名在 Task 2 和 Task 4 中一致；`_submit_vlm_review` 的 `frames` 类型为 `List[Tuple[float, bytes]]`。
- [ ] Test coverage：每个任务都有新增或更新的测试。
- [ ] Memory：通过 JPEG 字节缓存和移除 `frame_history` 降低内存。

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-13-detection-frame-vlm-record-plan.md`.

Two execution options:

1. **Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
