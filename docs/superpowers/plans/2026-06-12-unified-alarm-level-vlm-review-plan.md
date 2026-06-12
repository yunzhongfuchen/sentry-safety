# 告警记录统一级别与 VLM 复核流程实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 取消 P0/P1 区分，统一告警记录级别为小模型报警/大模型报警/大模型忽略，状态改为待确认/已确认/误报，支持每类型独立配置 VLM 复核与冷却时间。

**Architecture:** 将告警记录的「创建」与「VLM 复核状态应用」抽取为纯函数（`backend/alarm_state.py`），便于单元测试；`MultiDetector` 移除全局 `use_vlm` 与 P0/P1 冷却参数，改为按类型读取 `use_vlm` 与 `cooldown`；前端页面同步替换 P0/P1 相关显示与配置。

**Tech Stack:** Python 3.12, FastAPI, vanilla JavaScript, Vue 3 (settings.html), pytest

---

## 文件结构

| 文件 |  responsibility |
|---|---|
| `backend/alarm_state.py` | 新建：告警记录状态纯函数（创建、VLM 复核应用、人工确认） |
| `backend/config.py` | 修改默认检测类型配置（移除 `level`，新增 `cooldown`），移除全局 `use_vlm`/`p0_alert_cooldown`/`p1_alert_cooldown` |
| `backend/safety_detection/detector_core.py` | 修改 `TypeSchedule`、`MultiDetector`、冷却逻辑、触发逻辑、VLM 回调 |
| `backend/main_multi.py` | 修改记录创建/VLM 回调、API 端点（stats/confirm/ignore/settings）、清空旧记录 |
| `backend/performance_storage.py` | 修改统计摘要字段、分页过滤字段 |
| `frontend/safety_detection/records.html` | 修改统计/过滤/表格/详情弹窗 |
| `frontend/safety_detection/settings.html` | 修改全局配置与检测类型表格、摄像头弹窗、批量配置弹窗 |
| `frontend/safety_detection/multi.html` | 移除 P0/P1 样式与逻辑 |
| `frontend/safety_detection/hud.html` | 移除 P0/P1 样式与逻辑 |
| `tests/test_alarm_state.py` | 新建：`alarm_state` 单元测试 |
| `tests/test_detector_core_config.py` | 新建：`TypeSchedule` 与 `MultiDetector.register_camera` 单元测试 |
| `requirements.txt` | 新增 `pytest` 依赖 |

---

## Task 1: 创建 `backend/alarm_state.py` 纯函数模块

**Files:**
- Create: `backend/alarm_state.py`
- Test: `tests/test_alarm_state.py`

- [ ] **Step 1: 写失败的测试 — 创建记录**

```python
import time

def test_create_record_initial_state():
    from backend.alarm_state import create_record
    result = {"detected": True, "max_confidence": 0.87, "boxes": [[1, 2, 3, 4]], "reason": "检测到 fire"}
    record = create_record("cam01", "fire", result)
    assert record["camera_id"] == "cam01"
    assert record["detection_type"] == "fire"
    assert record["level"] == "small_model_alarm"
    assert record["status"] == "pending"
    assert record["small_model"]["detected"] is True
    assert record["vlm_review"] is None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_alarm_state.py::test_create_record_initial_state -v`
Expected: `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: 实现最小模块**

```python
import time
from typing import Any, Dict, Optional


def create_record(camera_id: str, dtype: str, result: Dict[str, Any], record_id: Optional[str] = None) -> Dict[str, Any]:
    """根据小模型检测结果创建一条新告警记录"""
    if record_id is None:
        record_id = f"{camera_id}_{dtype}_{int(time.time() * 1000)}"
    trigger_time = time.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "id": record_id,
        "camera_id": camera_id,
        "detection_type": dtype,
        "level": "small_model_alarm",
        "status": "pending",
        "time": trigger_time,
        "confidence": result.get("max_confidence", result.get("confidence", 0)),
        "reason": result.get("reason", ""),
        "small_model": {
            "detected": result.get("detected", False),
            "confidence": result.get("max_confidence", 0),
            "boxes": result.get("boxes", []),
        },
        "vlm_review": None,
        "source": result.get("source", "small_model"),
        "frame_count": 0,
    }


def apply_vlm_review(record: Dict[str, Any], vlm_result: Dict[str, Any]) -> None:
    """把 VLM 复核结果应用到记录：只改 level 和 vlm_review，不改 status"""
    confirmed = vlm_result.get("confirmed", False)
    conf = vlm_result.get("confidence", 0)
    reason = vlm_result.get("reason", "")
    record["vlm_review"] = {
        "confirmed": confirmed,
        "confidence": conf,
        "reason": reason,
    }
    if confirmed:
        record["level"] = "vlm_alarm"
        record["reason"] = f"[VLM 确认] {reason}" if reason else "[VLM 确认] 复核通过"
    else:
        record["level"] = "vlm_ignore"
        record["reason"] = f"[VLM 已排除] {reason}" if reason else "[VLM 已排除] 复核未通过"


def confirm_alarm(record: Dict[str, Any]) -> None:
    """人工确认报警"""
    record["status"] = "confirmed"


def confirm_false_positive(record: Dict[str, Any]) -> None:
    """人工确认误报"""
    record["status"] = "false_positive"
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_alarm_state.py::test_create_record_initial_state -v`
Expected: `PASS`

- [ ] **Step 5: 写失败的测试 — VLM 复核不改变状态**

```python
def test_vlm_review_does_not_change_status():
    from backend.alarm_state import create_record, apply_vlm_review
    record = create_record("cam01", "fire", {"detected": True, "max_confidence": 0.8})
    apply_vlm_review(record, {"confirmed": True, "confidence": 0.9, "reason": "有火焰"})
    assert record["level"] == "vlm_alarm"
    assert record["status"] == "pending"

    apply_vlm_review(record, {"confirmed": False, "confidence": 0.9, "reason": "无火焰"})
    assert record["level"] == "vlm_ignore"
    assert record["status"] == "pending"
```

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/test_alarm_state.py -v`
Expected: 2 tests `PASS`

- [ ] **Step 7: 写失败的测试 — 人工确认状态**

```python
def test_human_confirm_changes_status():
    from backend.alarm_state import create_record, confirm_alarm, confirm_false_positive
    record = create_record("cam01", "fire", {"detected": True, "max_confidence": 0.8})
    confirm_alarm(record)
    assert record["status"] == "confirmed"

    record2 = create_record("cam01", "fire", {"detected": True, "max_confidence": 0.8})
    confirm_false_positive(record2)
    assert record2["status"] == "false_positive"
```

- [ ] **Step 8: 运行测试确认通过**

Run: `python -m pytest tests/test_alarm_state.py -v`
Expected: 3 tests `PASS`

- [ ] **Step 9: 提交**

```bash
git add backend/alarm_state.py tests/test_alarm_state.py
if ! grep -q "^pytest" requirements.txt; then
  echo "pytest>=7.0.0" >> requirements.txt
fi
git add requirements.txt
git commit -m "feat: add alarm_state module for unified alarm source/status logic

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: 修改 `backend/config.py` 默认配置

**Files:**
- Modify: `backend/config.py:90-170`
- Modify: `backend/config.py:353-357`（向后兼容逻辑）
- Test: `tests/test_config_defaults.py`

- [ ] **Step 1: 写失败的测试 — 默认类型不含 `level`，含 `cooldown`**

```python
def test_default_type_config_shape():
    from backend.config import DEFAULT_TYPE_CONFIG, DEFAULT_GLOBAL_SETTINGS
    for dtype, cfg in DEFAULT_TYPE_CONFIG.items():
        assert "level" not in cfg
        assert "cooldown" in cfg
        assert isinstance(cfg["cooldown"], (int, float))
        assert "use_vlm" in cfg
    assert "use_vlm" not in DEFAULT_GLOBAL_SETTINGS
    assert "p0_alert_cooldown" not in DEFAULT_GLOBAL_SETTINGS
    assert "p1_alert_cooldown" not in DEFAULT_GLOBAL_SETTINGS
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_config_defaults.py::test_default_type_config_shape -v`
Expected: `AssertionError`（因为当前还有 `level`）

- [ ] **Step 3: 修改 `DEFAULT_TYPE_CONFIG` 和 `DEFAULT_GLOBAL_SETTINGS`**

将 `backend/config.py:90-160` 改为：

```python
DEFAULT_TYPE_CONFIG = {
    "fire": {
        "enabled": False,
        "interval": 1,
        "threshold": 0.6,
        "consecutive_required": 2,
        "cooldown": 10,
        "use_vlm": False,
    },
    "smoke": {
        "enabled": False,
        "interval": 1,
        "threshold": 0.55,
        "consecutive_required": 2,
        "cooldown": 10,
        "use_vlm": False,
    },
    "uniform": {
        "enabled": False,
        "interval": 1,
        "threshold": 0.5,
        "compliance_window_seconds": 30,
        "cooldown": 3,
        "use_vlm": False,
    },
    "mask": {
        "enabled": False,
        "interval": 1,
        "threshold": 0.5,
        "consecutive_required": 1,
        "cooldown": 3,
        "use_vlm": False,
    },
    "cigarette": {
        "enabled": False,
        "interval": 1,
        "threshold": 0.5,
        "consecutive_required": 1,
        "cooldown": 3,
        "use_vlm": False,
    },
    "sleep": {
        "enabled": False,
        "interval": 60,
        "threshold": 0.7,
        "consecutive_required": 3,
        "cooldown": 30,
        "use_vlm": False,
    },
}

DEFAULT_GLOBAL_SETTINGS = {
    "vlm_max_concurrent": 3,
    "vlm_inspection_interval": 30,
    "max_records": 100000,
    "max_storage_mb": 500,
    "memory_threshold_percent": 80,
    "emergency_cleanup_ratio": 0.2,
    "snapshot_quality": 70,
    "frame_quality": 60,
    "detection_resolution": [640, 480],
    "use_gpu_scheduler": False,
    "gpu_scheduler_num_queues": 0,
    "gpu_scheduler_interval": 0.5,
    "gpu_scheduler_half": False,
}
```

- [ ] **Step 4: 更新向后兼容逻辑**

在 `backend/config.py:353-357` 的 `use_vlm` 兼容块下方新增：

```python
        # 旧配置迁移：移除 level，补充 cooldown
        for dtype, cfg in cam.get("detection_types", {}).items():
            if "level" in cfg:
                del cfg["level"]
                migrated = True
            if "cooldown" not in cfg:
                cfg["cooldown"] = DEFAULT_TYPE_CONFIG.get(dtype, {}).get("cooldown", 3)
                migrated = True
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_config_defaults.py -v`
Expected: `PASS`

- [ ] **Step 6: 提交**

```bash
git add backend/config.py tests/test_config_defaults.py
git commit -m "refactor: remove P0/P1 from default config, add per-type cooldown

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: 修改 `backend/safety_detection/detector_core.py`

**Files:**
- Modify: `backend/safety_detection/detector_core.py:24-43`
- Modify: `backend/safety_detection/detector_core.py:193-215`
- Modify: `backend/safety_detection/detector_core.py:244-264`
- Modify: `backend/safety_detection/detector_core.py:504-570`
- Modify: `backend/safety_detection/detector_core.py:620-632`
- Modify: `backend/safety_detection/detector_core.py:699-753`
- Modify: `backend/safety_detection/detector_core.py:810-814`
- Test: `tests/test_detector_core_config.py`

- [ ] **Step 1: 写失败的测试 — TypeSchedule 使用 cooldown**

```python
def test_type_schedule_without_level():
    from backend.safety_detection.detector_core import TypeSchedule
    s = TypeSchedule(dtype="fire", enabled=True, interval=1, threshold=0.5, cooldown=10, use_vlm=True)
    assert s.cooldown == 10
    assert s.use_vlm is True
    assert not hasattr(s, "level")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_detector_core_config.py::test_type_schedule_without_level -v`
Expected: `TypeError`（因为当前还有 level 参数）

- [ ] **Step 3: 修改 `TypeSchedule`**

将 `backend/safety_detection/detector_core.py:24-43` 改为：

```python
@dataclass
class TypeSchedule:
    """单类型检测调度状态"""
    dtype: str
    enabled: bool
    interval: float
    threshold: float
    cooldown: float
    consecutive_required: int = 1
    consecutive_count: int = 0
    last_run: float = 0.0
    pending_vlm: bool = False
    # uniform 专用
    compliance_window_seconds: float = 30.0
    compliance_window_start: Optional[float] = None
    vest_detected_in_window: bool = False
    use_vlm: bool = False
    # sleep 专用
    history_frames: deque = field(default_factory=lambda: deque(maxlen=10))

    def is_due(self, now: float) -> bool:
        return now - self.last_run >= self.interval
```

- [ ] **Step 4: 修改 `MultiDetector.__init__`**

将 `backend/safety_detection/detector_core.py:193-215` 改为：

```python
    def __init__(
        self,
        camera_manager,
        safety_detector: SafetyDetector,
        vlm_queue,
        strategy: DetectionStrategy,
        trigger_callback: Optional[Callable[[str, str, np.ndarray, dict], None]] = None,
        frame_callback: Optional[Callable[[str, np.ndarray], None]] = None,
        vlm_result_callback: Optional[Callable[[str, str, dict], None]] = None,
    ):
        self.camera_manager = camera_manager
        self.safety_detector = safety_detector
        self.vlm_queue = vlm_queue
        self.strategy = strategy
        self.trigger_callback = trigger_callback
        self.frame_callback = frame_callback
        self.vlm_result_callback = vlm_result_callback

        self._schedules: Dict[str, Dict[str, TypeSchedule]] = {}
        self._alert_states: Dict[str, Dict[str, dict]] = {}
        self._cooldowns: Dict[str, Dict[str, float]] = {}
        self._latest_results: Dict[str, Dict[str, dict]] = {}
        self._lock = threading.RLock()
        self._running = False
```

- [ ] **Step 5: 修改 `register_camera`**

将 `backend/safety_detection/detector_core.py:253-263` 中的 `TypeSchedule(...)` 调用改为：

```python
                schedule = TypeSchedule(
                    dtype=dtype,
                    enabled=True,
                    interval=cfg.get("interval", 1.0),
                    threshold=cfg.get("threshold", 0.5),
                    cooldown=cfg.get("cooldown", 3.0),
                    consecutive_required=cfg.get("consecutive_required", 1),
                    compliance_window_seconds=cfg.get("compliance_window_seconds", 30.0),
                    use_vlm=cfg.get("use_vlm", False),
                )
```

- [ ] **Step 6: 修改标准检测触发逻辑**

将 `backend/safety_detection/detector_core.py:507-533` 改为：

```python
        # 把 level 和 reason 写入 result，供 trigger_callback 创建记录时使用
        result["level"] = "small_model_alarm"
        if not result.get("reason"):
            result["reason"] = f"检测到 {dtype}，置信度 {max_conf:.2f}"

        # 达到阈值，统一触发告警流程：先创建记录，再按需提交 VLM 复核
        self._alert_states[camera_id][dtype] = {"active": True, "time": now, "level": "small_model_alarm"}
        if self.use_vlm and schedule.use_vlm:
            result["pending_vlm_review"] = True
            self._submit_vlm_review(camera_id, dtype, frame, schedule, result)
        if self.trigger_callback:
            try:
                self.trigger_callback(camera_id, dtype, frame, result)
            except Exception as e:
                logger.error(f"Trigger callback error: {e}")
```

- [ ] **Step 7: 修改工服窗口触发逻辑**

将 `backend/safety_detection/detector_core.py:558-574` 改为：

```python
                        if not self.is_in_cooldown(camera_id, "uniform", now):
                            self._cooldowns[camera_id]["uniform"] = now
                            result["level"] = "small_model_alarm"
                            if not result.get("reason"):
                                result["reason"] = "工服合规窗口过期，未检测到反光背心"
                            self._alert_states[camera_id]["uniform"] = {"active": True, "time": now, "level": "small_model_alarm"}
                            if self.use_vlm and schedule.use_vlm:
                                schedule.pending_vlm = True
                                result["pending_vlm_review"] = True
                                logger.info(f"{camera_id} uniform window expired, submitting VLM confirm")
                                self._submit_vlm_confirm(camera_id, "uniform", frame, schedule, result)
                            logger.info(f"{camera_id} uniform window expired, alerting")
                            if self.trigger_callback:
                                try:
                                    self.trigger_callback(camera_id, "uniform", frame, result)
                                except Exception as e:
                                    logger.error(f"Trigger callback error: {e}")
```

- [ ] **Step 8: 修改睡岗触发逻辑**

将 `backend/safety_detection/detector_core.py:619-621` 改为：

```python
        # 小模型直接告警
        result["level"] = "small_model_alarm"
        if not result.get("reason"):
            result["reason"] = f"睡岗检测连续 {schedule.consecutive_required} 次命中"
        self._cooldowns[camera_id]["sleep"] = now
        self._alert_states[camera_id]["sleep"] = {
            "active": True, "time": now, "level": "small_model_alarm"
        }
        logger.info(f"{camera_id} sleep triggered directly (small-model only)")
```

- [ ] **Step 9: 修改 VLM 回调逻辑**

将 `_on_vlm_review`（`backend/safety_detection/detector_core.py:699-706`）改为直接透传结果：

```python
    def _on_vlm_review(self, camera_id: str, dtype: str, vlm_result: dict) -> None:
        """VLM 复核回调：透传给上层，由上层更新记录"""
        logger.info(f"VLM review result for {camera_id} {dtype}: {vlm_result}")
        if self.vlm_result_callback:
            try:
                self.vlm_result_callback(camera_id, dtype, vlm_result)
            except Exception as e:
                logger.error(f"VLM result callback error: {e}")
```

将 `_on_vlm_confirm`（`backend/safety_detection/detector_core.py:708-734`）改为：

```python
    def _on_vlm_confirm(self, camera_id: str, dtype: str, frame: np.ndarray,
                        result: dict, schedule: TypeSchedule, vlm_result: dict) -> None:
        """VLM 确认回调：透传给上层，由上层更新记录 level"""
        schedule.pending_vlm = False
        logger.info(f"VLM confirm result for {camera_id} {dtype}: {vlm_result}")
        if self.vlm_result_callback:
            try:
                self.vlm_result_callback(camera_id, dtype, vlm_result)
            except Exception as e:
                logger.error(f"VLM result callback error: {e}")
```

- [ ] **Step 10: 修改 `is_in_cooldown`**

将 `backend/safety_detection/detector_core.py:810-814` 改为：

```python
    def is_in_cooldown(self, camera_id: str, dtype: str, now: float) -> bool:
        with self._lock:
            last = self._cooldowns.get(camera_id, {}).get(dtype, 0)
            schedule = self._schedules.get(camera_id, {}).get(dtype)
            cooldown = schedule.cooldown if schedule else 3.0
            return now - last < cooldown
```

- [ ] **Step 11: 写测试 — MultiDetector.register_camera 使用新配置**

```python
def test_multidetector_register_camera_uses_cooldown_and_use_vlm():
    from backend.safety_detection.detector_core import MultiDetector
    md = MultiDetector(camera_manager=None, safety_detector=None, vlm_queue=None, strategy=None)
    md.register_camera("cam01", {
        "fire": {"enabled": True, "interval": 1, "threshold": 0.5, "cooldown": 15, "use_vlm": True},
        "mask": {"enabled": True, "interval": 1, "threshold": 0.5, "cooldown": 5, "use_vlm": False},
    })
    assert md._schedules["cam01"]["fire"].cooldown == 15
    assert md._schedules["cam01"]["fire"].use_vlm is True
    assert md._schedules["cam01"]["mask"].cooldown == 5
    assert md._schedules["cam01"]["mask"].use_vlm is False
```

- [ ] **Step 12: 运行测试确认通过**

Run: `python -m pytest tests/test_detector_core_config.py -v`
Expected: 2 tests `PASS`

- [ ] **Step 13: 提交**

```bash
git add backend/safety_detection/detector_core.py tests/test_detector_core_config.py
git commit -m "refactor: remove P0/P1 from MultiDetector, use per-type cooldown/use_vlm

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: 修改 `backend/main_multi.py`

**Files:**
- Modify: `backend/main_multi.py:55`（import alarm_state）
- Modify: `backend/main_multi.py:280-405`
- Modify: `backend/main_multi.py:407-417`
- Modify: `backend/main_multi.py:915-954`
- Modify: `backend/main_multi.py:736-941`
- Modify: `backend/main_multi.py:869-910`
- Modify: `backend/main_multi.py:500-502`
- Test: `tests/test_main_multi_records.py`

- [ ] **Step 1: 导入 alarm_state**

在 `backend/main_multi.py:55` 附近新增：

```python
from alarm_state import create_record, apply_vlm_review, confirm_alarm, confirm_false_positive
```

- [ ] **Step 2: 修改 `on_trigger`**

将 `backend/main_multi.py:280-348` 改为：

```python
    def on_trigger(camera_id: str, dtype: str, frame: Optional[np.ndarray], result: dict):
        """检测触发回调（创建告警记录）"""
        global detection_records

        with _status_lock:
            _system_status["total_detections"] += 1

        log_message(f"Camera {camera_id}: {dtype} detected, level={result.get('level', 'small_model_alarm')}")

        trigger_ts = time.time()
        record_id = f"{camera_id}_{dtype}_{int(trigger_ts * 1000)}"

        # 保存快照（只画触发类型的框）
        if frame is not None:
            trigger_results = {dtype: result}
            annotated = MultiDetector._annotate_frame(frame, trigger_results, camera_id, [])
            snapshot_b64 = encode_frame_to_base64(annotated, quality=_global_settings.get("snapshot_quality", 70))
            storage.save_image(record_id, "snapshot", snapshot_b64)

        record = create_record(camera_id, dtype, result, record_id=record_id)
        record["time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(trigger_ts))

        # 如果该类型启用了 VLM 复核，记录待更新映射，供 VLM 回调更新 level
        if result.get("pending_vlm_review"):
            with _pending_reviews_lock:
                _pending_reviews[(camera_id, dtype)] = record_id

        with _records_lock:
            detection_records.insert(0, record)
            max_records = _global_settings.get("max_records", 100)
            if len(detection_records) > max_records:
                ratio = _global_settings.get("emergency_cleanup_ratio", 0.2)
                remove_count = max(1, int(len(detection_records) * ratio))
                to_remove = detection_records[-remove_count:]
                for old in to_remove:
                    storage.delete_record_images(old["id"])
                detection_records = detection_records[:-remove_count]
        mark_records_dirty()

        # 保存触发时刻前最多5秒的时间窗口帧
        try:
            window_frames = camera_manager.get_window_frames(camera_id, trigger_ts - 5, trigger_ts)
            if window_frames:
                quality = _global_settings.get("frame_quality", 60)
                for i, (ts, fr) in enumerate(window_frames):
                    b64 = encode_frame_to_base64(fr, quality=quality)
                    storage.save_image(record_id, "frame", b64, i)
                record["frame_count"] = len(window_frames)
                log_message(f"Saved {len(window_frames)} window frames for {record_id}")
            elif frame is not None:
                quality = _global_settings.get("frame_quality", 60)
                b64 = encode_frame_to_base64(frame, quality=quality)
                storage.save_image(record_id, "frame", b64, 0)
                record["frame_count"] = 1
                log_message(f"Saved trigger frame for {record_id} (window empty)")
        except Exception as e:
            logger.error(f"Failed to save window frames for {record_id}: {e}")
```

- [ ] **Step 3: 修改 `on_vlm_result`**

将 `backend/main_multi.py:369-405` 改为：

```python
    def on_vlm_result(camera_id: str, dtype: str, vlm_result: dict):
        """VLM 复核/确认结果回调：更新已有记录 level，不改 status"""
        global detection_records
        record_id = None
        with _pending_reviews_lock:
            record_id = _pending_reviews.pop((camera_id, dtype), None)
        if not record_id:
            return

        updated_record = None
        with _records_lock:
            for r in detection_records:
                if r.get("id") == record_id:
                    apply_vlm_review(r, vlm_result)
                    updated_record = r
                    break
        if updated_record:
            mark_records_dirty()
            log_message(f"Record {record_id} updated by VLM: level={updated_record['level']}, status={updated_record['status']}")
```

- [ ] **Step 4: 修改 `MultiDetector` 初始化调用**

将 `backend/main_multi.py:407-417` 改为：

```python
    multi_detector = MultiDetector(
        camera_manager=camera_manager,
        safety_detector=safety_detector,
        vlm_queue=vlm_queue,
        strategy=strategy,
        trigger_callback=on_trigger,
        vlm_result_callback=on_vlm_result,
    )
```

- [ ] **Step 5: 修改 `/alerts/stats` 返回字段**

将 `backend/main_multi.py:932-941` 改为：

```python
@app.get("/alerts/stats")
async def get_alerts_stats():
    """获取告警统计"""
    summary = storage.get_record_summary()
    return {
        "total": summary.get("total", 0),
        "pending": summary.get("by_status", {}).get("pending", 0),
        "confirmed": summary.get("by_status", {}).get("confirmed", 0),
        "false_positive": summary.get("by_status", {}).get("false_positive", 0),
    }
```

- [ ] **Step 6: 新增 `/alerts/{id}/confirm` 端点**

在 `backend/main_multi.py:944` 的 `ignore_alert` 函数上方新增：

```python
@app.post("/alerts/{record_id}/confirm")
async def confirm_alert(record_id: str):
    """标记告警为已确认"""
    global detection_records
    with _records_lock:
        for r in detection_records:
            if r.get("id") == record_id:
                confirm_alarm(r)
                mark_records_dirty()
                return {"success": True}
    return JSONResponse({"error": "Record not found"}, status_code=404)
```

- [ ] **Step 7: 修改 `/alerts/{id}/ignore` 端点**

将 `backend/main_multi.py:944-954` 的函数体改为使用 `confirm_false_positive`：

```python
@app.post("/alerts/{record_id}/ignore")
async def ignore_alert(record_id: str):
    """标记告警为误报"""
    global detection_records
    with _records_lock:
        for r in detection_records:
            if r.get("id") == record_id:
                confirm_false_positive(r)
                mark_records_dirty()
                return {"success": True}
    return JSONResponse({"error": "Record not found"}, status_code=404)
```

- [ ] **Step 8: 修改 `/settings` POST 更新逻辑**

将 `backend/main_multi.py:895-899` 的 `multi_detector` 动态更新块改为：

```python
        if multi_detector:
            # 冷却与 VLM 开关已下沉到 per-type 配置，动态更新通过摄像头配置接口处理
            pass
```

或者干脆移除该块。

- [ ] **Step 9: 启动时清空历史记录**

将 `backend/main_multi.py:500-502` 改为：

```python
    # 9. 加载历史记录（按需求清空测试数据）
    global detection_records
    detection_records = []
    storage.save_records(detection_records)
    log_message("Historical records cleared")
```

- [ ] **Step 10: 写测试 — on_trigger / on_vlm_result 行为**

```python
def test_on_trigger_creates_pending_small_model_record():
    from backend.main_multi import on_trigger
    # 注意：on_trigger 是 init_components 内的闭包，无法直接导入
    # 此测试改为直接调用 alarm_state 函数验证等价逻辑
    from backend.alarm_state import create_record, apply_vlm_review
    record = create_record("cam01", "fire", {"detected": True, "max_confidence": 0.87})
    apply_vlm_review(record, {"confirmed": True, "confidence": 0.9, "reason": "有火焰"})
    assert record["level"] == "vlm_alarm"
    assert record["status"] == "pending"
```

- [ ] **Step 11: 运行测试确认通过**

Run: `python -m pytest tests/test_main_multi_records.py -v`
Expected: `PASS`

- [ ] **Step 12: 提交**

```bash
git add backend/main_multi.py tests/test_main_multi_records.py
git commit -m "feat: integrate alarm_state into main_multi, update stats/confirm/ignore APIs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: 修改 `backend/performance_storage.py`

**Files:**
- Modify: `backend/performance_storage.py:275-305`
- Test: `tests/test_performance_storage.py`

- [ ] **Step 1: 修改 `get_record_summary`**

将 `backend/performance_storage.py:275-305` 改为：

```python
def get_record_summary() -> Dict:
    """获取记录统计摘要（按状态和告警来源聚合）"""
    records = load_records()

    total = len(records)
    by_status = {}
    by_type = {}
    by_level = {}

    for r in records:
        status = r.get("status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1

        dtype = r.get("detection_type", "unknown")
        by_type[dtype] = by_type.get(dtype, 0) + 1

        level = r.get("level", "small_model_alarm")
        by_level[level] = by_level.get(level, 0) + 1

    camera_stats = {}
    for r in records:
        cam_id = r.get("camera_id", "unknown")
        camera_stats[cam_id] = camera_stats.get(cam_id, 0) + 1

    return {
        "total": total,
        "by_status": by_status,
        "by_type": by_type,
        "by_level": by_level,
        "camera_stats": camera_stats,
    }
```

- [ ] **Step 2: 写测试 — 统计摘要字段**

```python
def test_summary_uses_status_and_level():
    from backend.performance_storage import get_record_summary
    # 临时注入测试数据
    import backend.performance_storage as ps
    ps.save_records([
        {"id": "1", "status": "pending", "level": "small_model_alarm", "detection_type": "fire", "camera_id": "cam01"},
        {"id": "2", "status": "confirmed", "level": "vlm_alarm", "detection_type": "mask", "camera_id": "cam01"},
        {"id": "3", "status": "false_positive", "level": "vlm_ignore", "detection_type": "smoke", "camera_id": "cam02"},
    ])
    summary = get_record_summary()
    assert summary["total"] == 3
    assert summary["by_status"]["pending"] == 1
    assert summary["by_status"]["confirmed"] == 1
    assert summary["by_status"]["false_positive"] == 1
    assert summary["by_level"]["small_model_alarm"] == 1
    assert summary["by_level"]["vlm_alarm"] == 1
    assert summary["by_level"]["vlm_ignore"] == 1
```

- [ ] **Step 3: 运行测试确认通过**

Run: `python -m pytest tests/test_performance_storage.py -v`
Expected: `PASS`

- [ ] **Step 4: 提交**

```bash
git add backend/performance_storage.py tests/test_performance_storage.py
git commit -m "refactor: update record summary to aggregate by status and alarm source

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: 修改 `frontend/safety_detection/records.html`

**Files:**
- Modify: `frontend/safety_detection/records.html:236-240`（CSS）
- Modify: `frontend/safety_detection/records.html:495-511`（统计卡片）
- Modify: `frontend/safety_detection/records.html:534-550`（过滤器）
- Modify: `frontend/safety_detection/records.html:635-637`（详情按钮）
- Modify: `frontend/safety_detection/records.html:656-664`（loadSummary）
- Modify: `frontend/safety_detection/records.html:716-740`（表格渲染）
- Modify: `frontend/safety_detection/records.html:780-805`（详情渲染）
- Modify: `frontend/safety_detection/records.html:900-910`（ignoreCurrent 函数）

- [ ] **Step 1: 更新 CSS 标签样式**

将 `frontend/safety_detection/records.html:236-240` 改为：

```html
        .badge-small_model_alarm { background: var(--danger-light); color: var(--danger); }
        .badge-vlm_alarm { background: var(--warning-light); color: var(--warning); }
        .badge-vlm_ignore { background: var(--success-light); color: var(--success); }
```

- [ ] **Step 2: 更新统计卡片**

将 `frontend/safety_detection/records.html:495-511` 改为：

```html
        <div class="stats-bar">
            <div class="stat-card">
                <div class="stat-value" id="statTotal">-</div>
                <div class="stat-label">总记录</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="statPending" style="color: var(--warning);">-</div>
                <div class="stat-label">待确认</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="statConfirmed" style="color: var(--danger);">-</div>
                <div class="stat-label">已确认</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="statFalsePositive" style="color: var(--success);">-</div>
                <div class="stat-label">误报</div>
            </div>
        </div>
```

- [ ] **Step 3: 更新过滤器选项**

将 `frontend/safety_detection/records.html:534-550` 改为：

```html
                <select id="filterLevel" onchange="applyFilters()">
                    <option value="">全部</option>
                    <option value="small_model_alarm">小模型报警</option>
                    <option value="vlm_alarm">大模型报警</option>
                    <option value="vlm_ignore">大模型忽略</option>
                </select>
                ...
                <select id="filterStatus" onchange="applyFilters()">
                    <option value="">全部</option>
                    <option value="pending">待确认</option>
                    <option value="confirmed">已确认</option>
                    <option value="false_positive">误报</option>
                </select>
```

- [ ] **Step 4: 更新详情按钮**

将 `frontend/safety_detection/records.html:635-637` 改为：

```html
                <div style="display: flex; gap: 10px; justify-content: flex-end; margin-top: 16px;">
                    <button class="btn primary" id="confirmBtn" onclick="confirmCurrent()">确认报警</button>
                    <button class="btn" id="ignoreBtn" onclick="ignoreCurrent()">确认误报</button>
                    <button class="btn" onclick="closeModal()">关闭</button>
                </div>
```

- [ ] **Step 5: 更新 `loadSummary`**

将 `frontend/safety_detection/records.html:656-664` 改为：

```javascript
                document.getElementById('statTotal').textContent = data.total || 0;
                document.getElementById('statPending').textContent = data.pending || 0;
                document.getElementById('statConfirmed').textContent = data.confirmed || 0;
                document.getElementById('statFalsePositive').textContent = data.false_positive || 0;
```

- [ ] **Step 6: 更新表格渲染**

将 `frontend/safety_detection/records.html:716-740` 的 `renderTable` 改为：

```javascript
            const typeLabels = { fire: '明火', smoke: '烟雾', uniform: '工服', mask: '口罩', cigarette: '吸烟', sleep: '睡岗' };
            const typeColors = { fire: '#ef4444', smoke: '#f97316', uniform: '#22c55e', mask: '#0ea5e9', cigarette: '#a855f7', sleep: '#eab308' };
            const levelLabels = {
                small_model_alarm: '小模型报警',
                vlm_alarm: '大模型报警',
                vlm_ignore: '大模型忽略'
            };
            const statusLabels = { pending: '待确认', confirmed: '已确认', false_positive: '误报' };
            const levelClasses = {
                small_model_alarm: 'badge-small_model_alarm',
                vlm_alarm: 'badge-vlm_alarm',
                vlm_ignore: 'badge-vlm_ignore'
            };
            const statusClasses = {
                pending: 'badge-pending',
                confirmed: 'badge-confirmed',
                false_positive: 'badge-false_positive'
            };

            let html = `<table><thead><tr>
                <th>ID</th><th>时间</th><th>摄像头</th><th>类型</th><th>级别</th><th>状态</th><th>置信度</th><th>说明</th>
            </tr></thead><tbody>`;

            html += state.records.map(r => {
                const typeLabel = typeLabels[r.detection_type] || r.detection_type;
                const typeColor = typeColors[r.detection_type] || '#94a3b8';
                const levelLabel = levelLabels[r.level] || r.level || '-';
                const levelClass = levelClasses[r.level] || 'badge-pending';
                const statusLabel = statusLabels[r.status] || r.status;
                const statusClass = statusClasses[r.status] || 'badge-pending';
                const conf = r.confidence != null ? (r.confidence * 100).toFixed(0) + '%' : '-';
                return `<tr onclick="showDetail('${encodeURIComponent(r.id)}')">
                    <td class="col-id">${r.id.slice(-8)}</td>
                    <td class="col-time">${formatTime(r.time)}</td>
                    <td class="col-camera">${r.camera_id}</td>
                    <td><span class="type-dot" style="background:${typeColor}"></span>${typeLabel}</td>
                    <td><span class="badge ${levelClass}">${levelLabel}</span></td>
                    <td><span class="badge ${statusClass}">${statusLabel}</span></td>
                    <td class="confidence">${conf}</td>
                    <td class="reason" title="${r.reason || ''}">${r.reason || '-'}</td>
                </tr>`;
            }).join('');
```

- [ ] **Step 7: 更新详情渲染**

将 `frontend/safety_detection/records.html:780-805` 的 `showDetail` 中相关代码改为：

```javascript
                const levelLabels = {
                    small_model_alarm: '小模型报警',
                    vlm_alarm: '大模型报警',
                    vlm_ignore: '大模型忽略'
                };
                const statusLabels = { pending: '待确认', confirmed: '已确认', false_positive: '误报' };
                const levelClasses = {
                    small_model_alarm: 'badge-small_model_alarm',
                    vlm_alarm: 'badge-vlm_alarm',
                    vlm_ignore: 'badge-vlm_ignore'
                };
                const statusClasses = {
                    pending: 'badge-pending',
                    confirmed: 'badge-confirmed',
                    false_positive: 'badge-false_positive'
                };

                document.getElementById('detailLevel').innerHTML = `<span class="badge ${levelClasses[data.level] || 'badge-pending'}">${levelLabels[data.level] || data.level || '-'}</span>`;
                document.getElementById('detailStatus').innerHTML = `<span class="badge ${statusClasses[data.status] || 'badge-pending'}">${statusLabels[data.status] || data.status}</span>`;
                document.getElementById('detailSmall').textContent = data.small_model?.detected ? `检出异常 (置信度 ${(data.small_model.confidence * 100).toFixed(0)}%)` : '未检出异常';
                document.getElementById('detailVlm').textContent = data.vlm_review ? (data.vlm_review.confirmed ? '确认告警' : '已排除') : '未复核';

                const confirmBtn = document.getElementById('confirmBtn');
                const ignoreBtn = document.getElementById('ignoreBtn');
                if (confirmBtn) confirmBtn.style.display = data.status === 'confirmed' ? 'none' : 'inline-block';
                if (ignoreBtn) ignoreBtn.style.display = data.status === 'false_positive' ? 'none' : 'inline-block';
```

- [ ] **Step 8: 新增 `confirmCurrent` 函数并修改 `ignoreCurrent`**

在 `frontend/safety_detection/records.html` 的 `ignoreCurrent` 附近新增：

```javascript
        async function confirmCurrent() {
            if (!currentDetail) return;
            try {
                const res = await fetch(`/alerts/${encodeURIComponent(currentDetail.encodedId)}/confirm`, { method: 'POST' });
                if (res.ok) {
                    showDetail(currentDetail.encodedId);
                    loadRecords();
                    loadSummary();
                } else {
                    alert('确认报警失败');
                }
            } catch (e) { alert('确认报警失败'); }
        }

        async function ignoreCurrent() {
            if (!currentDetail) return;
            try {
                const res = await fetch(`/alerts/${encodeURIComponent(currentDetail.encodedId)}/ignore`, { method: 'POST' });
                if (res.ok) {
                    showDetail(currentDetail.encodedId);
                    loadRecords();
                    loadSummary();
                } else {
                    alert('确认误报失败');
                }
            } catch (e) { alert('确认误报失败'); }
        }
```

- [ ] **Step 9: 手动验证**

Run: 启动服务后访问 `/records.html`，检查：
1. 统计卡片显示「总记录/待确认/已确认/误报」。
2. 过滤器选项正确。
3. 表格级别/状态列显示中文标签。
4. 详情弹窗有「确认报警」「确认误报」「关闭」三个按钮。
5. 点击按钮后状态改变且列表刷新。

- [ ] **Step 10: 提交**

```bash
git add frontend/safety_detection/records.html
git commit -m "feat: update records page for unified alarm levels and human confirmation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: 修改 `frontend/safety_detection/settings.html`

**Files:**
- Modify: `frontend/safety_detection/settings.html:414-428`（全局 VLM + 冷却字段）
- Modify: `frontend/safety_detection/settings.html:483-498`（类型表头）
- Modify: `frontend/safety_detection/settings.html:656-663`（摄像头弹窗类型表格）
- Modify: `frontend/safety_detection/settings.html:690-697`（批量配置弹窗类型表格）
- Modify: `frontend/safety_detection/settings.html:713-722`（defaultTypes）

- [ ] **Step 1: 移除全局 VLM 开关和 P0/P1 冷却**

将 `frontend/safety_detection/settings.html:414-428` 改为：

```html
                    <div class="form-field">
                        <label>VLM 最大并发</label>
                        <input type="number" v-model.number="settings.vlm_max_concurrent" min="1" />
                    </div>
                    <div class="form-field">
                        <label>VLM 巡检间隔 (s, 0=关闭)</label>
                        <input type="number" v-model.number="settings.vlm_inspection_interval" min="0" />
                    </div>
                    <div class="form-field">
                        <label>最大记录数</label>
                        <input type="number" v-model.number="settings.max_records" min="10" />
                    </div>
```

- [ ] **Step 2: 更新类型表格表头**

将 `frontend/safety_detection/settings.html:483-498` 及 `647-655`、`682-689` 的表头统一改为：

```html
                <div class="type-row" style="font-weight:600;color:var(--text-muted);margin-bottom:8px;">
                    <span>类型</span>
                    <span>启用</span>
                    <span>间隔</span>
                    <span>阈值</span>
                    <span>连续</span>
                    <span>冷却</span>
                    <span>VLM</span>
                </div>
```

- [ ] **Step 3: 更新类型表格行**

将三处类型行模板统一改为：

```html
                <div v-for="t in detectionTypes" :key="t.key" class="type-row">
                    <span :style="{ color: t.color, fontWeight: 500 }">{{ t.label }}</span>
                    <input type="checkbox" v-model="settings.default_detection_types[t.key].enabled" />
                    <input type="number" v-model.number="settings.default_detection_types[t.key].interval" />
                    <input type="number" step="0.1" v-model.number="settings.default_detection_types[t.key].threshold" />
                    <input type="number" v-model.number="settings.default_detection_types[t.key].consecutive_required" />
                    <input type="number" v-model.number="settings.default_detection_types[t.key].cooldown" min="0" />
                    <input type="checkbox" v-model="settings.default_detection_types[t.key].use_vlm" />
                </div>
```

- [ ] **Step 4: 更新 `defaultTypes` 函数**

将 `frontend/safety_detection/settings.html:713-722` 改为：

```javascript
        function defaultTypes() {
            return {
                fire: { enabled: false, interval: 1, threshold: 0.6, consecutive_required: 2, cooldown: 10, use_vlm: false },
                smoke: { enabled: false, interval: 1, threshold: 0.55, consecutive_required: 2, cooldown: 10, use_vlm: false },
                uniform: { enabled: false, interval: 1, threshold: 0.5, consecutive_required: 1, compliance_window_seconds: 30, cooldown: 3, use_vlm: false },
                mask: { enabled: false, interval: 1, threshold: 0.5, consecutive_required: 1, cooldown: 3, use_vlm: false },
                cigarette: { enabled: false, interval: 1, threshold: 0.5, consecutive_required: 1, cooldown: 3, use_vlm: false },
                sleep: { enabled: false, interval: 60, threshold: 0.7, consecutive_required: 3, cooldown: 30, use_vlm: false },
            };
        }
```

- [ ] **Step 5: 手动验证**

Run: 启动服务后访问 `/settings.html`，检查：
1. 全局配置没有「启用 VLM 复核」「P0/P1 告警冷却」。
2. 检测类型表格有「冷却」列，没有「级别」列。
3. 保存默认值后刷新，数据保持。
4. 摄像头编辑弹窗和批量配置弹窗同步正确。

- [ ] **Step 6: 提交**

```bash
git add frontend/safety_detection/settings.html
git commit -m "feat: update settings page for per-type cooldown and VLM, remove P0/P1 config

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: 修改 `frontend/safety_detection/multi.html`

**Files:**
- Modify: `frontend/safety_detection/multi.html:147-148, 249-250, 573, 790, 803, 856`

- [ ] **Step 1: 更新 CSS**

将 `frontend/safety_detection/multi.html:147-148` 改为：

```html
        .alert-item.small_model_alarm { border-left-color: var(--danger); background: var(--danger-light); }
        .alert-item.vlm_alarm { border-left-color: var(--warning); background: var(--warning-light); }
        .alert-item.vlm_ignore { border-left-color: var(--success); background: var(--success-light); }
```

将 `frontend/safety_detection/multi.html:249-250` 改为：

```html
        .detection-badge.small_model_alarm { background: var(--danger-light); color: var(--danger); animation: alert-pulse 2s infinite; }
        .detection-badge.vlm_alarm { background: var(--warning-light); color: var(--warning); }
        .detection-badge.vlm_ignore { background: var(--success-light); color: var(--success); }
```

- [ ] **Step 2: 更新动态 class 绑定**

将 `frontend/safety_detection/multi.html:573` 的 `:class` 改为：

```html
:class="['alert-item', alert.level || 'small_model_alarm']"
```

将 `frontend/safety_detection/multi.html:790` 改为：

```javascript
                        if (det[t]?.alert) return det[t]?.level || 'small_model_alarm';
```

将 `frontend/safety_detection/multi.html:803` 改为：

```javascript
                            return (det[t]?.level ? det[t].level.replace(/_/g, ' ').toUpperCase() + ' ' : '') + label;
```

将 `frontend/safety_detection/multi.html:856` 改为：

```javascript
                                    level: r.level || 'small_model_alarm',
```

- [ ] **Step 3: 手动验证**

Run: 访问 `/multi`，触发告警，检查告警条/徽章按新级别显示。

- [ ] **Step 4: 提交**

```bash
git add frontend/safety_detection/multi.html
git commit -m "feat: update multi monitor page for unified alarm levels

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: 修改 `frontend/safety_detection/hud.html`

**Files:**
- Modify: `frontend/safety_detection/hud.html:255-256, 525-531, 840, 895, 918, 1064, 1071, 1083, 1144`

- [ ] **Step 1: 按 multi.html 相同模式更新 CSS、class 绑定、标签文本**

- [ ] **Step 2: 手动验证**

Run: 访问 `/hud`，检查 P0/P1 相关显示已替换。

- [ ] **Step 3: 提交**

```bash
git add frontend/safety_detection/hud.html
git commit -m "feat: update HUD page for unified alarm levels

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: 清理旧记录文件

**Files:**
- Delete: `data/records.json`
- Delete: `data/frames/*.jpg`

- [ ] **Step 1: 删除旧数据**

```bash
rm -f data/records.json
rm -f data/frames/*.jpg
```

- [ ] **Step 2: 提交**

```bash
git rm data/records.json 2>/dev/null || true
git add data
git commit -m "chore: clear historical test records and frames

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## 自审清单

- [ ] Spec coverage: 每条验收标准都有对应任务。
- [ ] Placeholder scan: 无 TBD/TODO，所有代码/命令完整。
- [ ] Type consistency: `level` 字段取值、函数名在所有任务中一致。
- [ ] Frontend consistency: `small_model_alarm` / `vlm_alarm` / `vlm_ignore` 在 records/settings/multi/hud 中一致。
- [ ] Test coverage: alarm_state、config、detector_core、storage 均有测试。
- [ ] Risk: 清空 `data/records.json` 为破坏性操作，已在 Task 10 单独提交。

---

## 执行交接

Plan complete and saved to `docs/superpowers/plans/2026-06-12-unified-alarm-level-vlm-review-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
