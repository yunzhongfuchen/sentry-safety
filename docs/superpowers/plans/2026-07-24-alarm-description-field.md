# 算法“报警说明”字段 + 类别选择调整 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在算法管理中新增可配置的“报警说明”字段，并在小模型触发告警时优先使用该字段作为记录说明；同时调整编辑弹窗中“类别选择”的文案和勾选框样式。

**Architecture:** 在算法注册表（`backend/detection_registry.py`）新增 `alarm_description` 结构性字段；API 层（`backend/safety_detection/api.py`）透传该字段；前端 `algorithms.html` 在编辑弹窗提供输入框并随保存提交；后端 `detector_core.py` 在触发告警时从 `registry` 读取该字段并生成 `result["reason"]`。类别选择 UI 仅做文案和样式调整，不改变数据逻辑。

**Tech Stack:** Python 3.12, FastAPI, Vue 3 (global build), vanilla JS/CSS, pytest

## Global Constraints
- Python 3.12
- 最小代码原则：不改动无关字段和页面
- 向后兼容：旧算法无 `alarm_description` 时回退到默认模板
- VLM 复核后的 reason 覆盖逻辑保持不变
- 类别选择逻辑不变：全不勾选 = 全部选择（`classes` 保存为 `null`）

---

### Task 1: Backend — Add `alarm_description` to algorithm registry

**Files:**
- Modify: `backend/detection_registry.py`
- Test: `tests/test_detector_core_registry.py`（新增/扩展测试）

**Interfaces:**
- Consumes: `model_registry.get(mkey)` for model resolution
- Produces: `registry.get(dtype)["alarm_description"]`, `to_api_list()[i]["alarm_description"]`

- [ ] **Step 1: Write failing test for registry field persistence**

Add to `tests/test_detector_core_registry.py`:

```python
def test_alarm_description_persisted(tmp_path, monkeypatch):
    from backend.detection_registry import DetectionRegistry
    monkeypatch.setattr("backend.detection_registry.ALGORITHMS_FILE", tmp_path / "algorithms.json")
    reg = DetectionRegistry.__new__(DetectionRegistry)
    reg._types = {}
    # Use add_type to create
    from backend.detection_registry import model_registry
    # If model_registry needs a real model, mock it
    monkeypatch.setattr(model_registry, "get", lambda k: {"file": "x.pt", "post_process": "yolo_box"})
    monkeypatch.setattr(model_registry, "file_exists", lambda k: True)
    key = reg.add_type({
        "label": "测试报警",
        "color": "#888888",
        "model_key": "test",
        "alarm_description": "自定义报警说明",
    })
    assert reg.get(key)["alarm_description"] == "自定义报警说明"
    # Update
    reg.update_type(key, {"alarm_description": "更新后说明"})
    assert reg.get(key)["alarm_description"] == "更新后说明"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_detector_core_registry.py::test_alarm_description_persisted -v`
Expected: FAIL (field not persisted)

- [ ] **Step 3: Implement registry field support**

Modify `backend/detection_registry.py`:

In `to_api_list` (around line 415-427), add to the dict:
```python
"alarm_description": td.get("alarm_description", ""),
```

In `add_type` (around line 461-471), add to the dict:
```python
"alarm_description": type_def.get("alarm_description", ""),
```

In `update_type` (around line 493-495), extend the field tuple:
```python
for field in ("color", "classes", "model_confidence", "vlm_prompt", "inspection_label", "alarm_description"):
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_detector_core_registry.py::test_alarm_description_persisted -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/detection_registry.py tests/test_detector_core_registry.py
git commit -m "feat: add alarm_description field to algorithm registry"
```

---

### Task 2: Backend API — Expose `alarm_description`

**Files:**
- Modify: `backend/safety_detection/api.py`

**Interfaces:**
- Consumes: `registry.get(key)` containing `"alarm_description"`
- Produces: `_algo_to_response()` returns `"alarm_description"`; `update_algorithm` accepts `"alarm_description"`

- [ ] **Step 1: Write failing test for API response field**

Add to `tests/test_models_api.py` (or create `tests/test_algorithms_api.py` if appropriate):

```python
def test_algorithm_response_includes_alarm_description(client):
    # Assuming client fixture and existing algorithm 'fire'
    resp = client.get("/algorithms")
    assert resp.status_code == 200
    algorithms = {a["key"]: a for a in resp.json()["algorithms"]}
    assert "alarm_description" in algorithms["fire"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models_api.py::test_algorithm_response_includes_alarm_description -v`
Expected: FAIL (key missing)

- [ ] **Step 3: Implement API field exposure**

Modify `backend/safety_detection/api.py`:

In `_algo_to_response` (around line 341-355), add:
```python
"alarm_description": td.get("alarm_description", ""),
```

In `update_algorithm` (around line 385), add `"alarm_description"` to `structural_fields`:
```python
structural_fields = {"label", "color", "model_key", "classes", "model_confidence",
                     "vlm_prompt", "inspection_label", "alarm_description"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models_api.py::test_algorithm_response_includes_alarm_description -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/safety_detection/api.py tests/test_models_api.py
git commit -m "feat: expose alarm_description in algorithms API"
```

---

### Task 3: Backend — Use `alarm_description` when triggering small-model alarm

**Files:**
- Modify: `backend/safety_detection/detector_core.py`
- Test: `tests/test_detector_core_registry.py`

**Interfaces:**
- Consumes: `registry.get(dtype)["label"]` and `registry.get(dtype)["alarm_description"]`
- Produces: `result["reason"]` set to custom text or fallback template

- [ ] **Step 1: Write failing test for custom alarm description**

Add to `tests/test_detector_core_registry.py`:

```python
def test_alarm_description_used_in_reason(monkeypatch):
    from backend.safety_detection.detector_core import MultiDetector, TypeSchedule
    # Minimal stub
    md = MultiDetector.__new__(MultiDetector)
    md._cooldowns = {"cam1": {}}
    md._alert_states = {"cam1": {}}
    md._latest_results = {}
    md._static_regions = {}
    md.camera_manager = None
    md.trigger_callback = None
    md.vlm_result_callback = None
    md.vlm_queue = None
    md.safety_detector = None
    md.strategy = None
    md._running = False
    md._lock = threading.RLock()
    md._schedules = {"cam1": {"fire": TypeSchedule(dtype="fire", enabled=True, interval=1, threshold=0.5, cooldown=0)}}
    md._schedules["cam1"]["fire"].consecutive_count = 3

    # Mock registry
    class FakeRegistry:
        def get(self, dtype):
            return {"label": "明火", "alarm_description": "发现明火请处理"}
    monkeypatch.setattr("backend.safety_detection.detector_core.registry", FakeRegistry())

    result = {"detected": True, "boxes": [[1,2,3,4]], "scores": [0.9], "max_confidence": 0.9}
    frame = object()
    md._handle_standard_detection("cam1", "fire", frame, result, md._schedules["cam1"]["fire"])
    assert result["reason"] == "发现明火请处理"
    assert result["level"] == "small_model_alarm"
```

Add a second test for fallback:

```python
def test_alarm_description_fallback_when_empty(monkeypatch):
    from backend.safety_detection.detector_core import MultiDetector, TypeSchedule
    md = MultiDetector.__new__(MultiDetector)
    md._cooldowns = {"cam1": {}}
    md._alert_states = {"cam1": {}}
    md._latest_results = {}
    md._static_regions = {}
    md.camera_manager = None
    md.trigger_callback = None
    md.vlm_result_callback = None
    md.vlm_queue = None
    md.safety_detector = None
    md.strategy = None
    md._running = False
    md._lock = threading.RLock()
    md._schedules = {"cam1": {"smoke": TypeSchedule(dtype="smoke", enabled=True, interval=1, threshold=0.5, cooldown=0)}}
    md._schedules["cam1"]["smoke"].consecutive_count = 3

    class FakeRegistry:
        def get(self, dtype):
            return {"label": "烟雾", "alarm_description": ""}
    monkeypatch.setattr("backend.safety_detection.detector_core.registry", FakeRegistry())

    result = {"detected": True, "boxes": [[1,2,3,4]], "scores": [0.9], "max_confidence": 0.9}
    md._handle_standard_detection("cam1", "smoke", object(), result, md._schedules["cam1"]["smoke"])
    assert result["reason"] == "检测到烟雾异常"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_detector_core_registry.py::test_alarm_description_used_in_reason tests/test_detector_core_registry.py::test_alarm_description_fallback_when_empty -v`
Expected: FAIL

- [ ] **Step 3: Implement reason generation using registry**

Modify `backend/safety_detection/detector_core.py`:

At the top of the file, add import:
```python
from backend.detection_registry import registry
```

In `_handle_standard_detection`, replace the existing reason assignment (around line 733-734):

```python
# 把 level 和 reason 写入 result，供 trigger_callback 创建记录时使用
result["level"] = "small_model_alarm"
type_def = registry.get(dtype)
label = type_def.get("label", dtype) if type_def else dtype
alarm_description = type_def.get("alarm_description", "").strip() if type_def else ""
if alarm_description:
    result["reason"] = alarm_description
else:
    result["reason"] = f"检测到 {label} 异常"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_detector_core_registry.py::test_alarm_description_used_in_reason tests/test_detector_core_registry.py::test_alarm_description_fallback_when_empty -v`
Expected: PASS

- [ ] **Step 5: Run full backend test suite**

Run: `export PYTHONIOENCODING=utf-8 && conda run -n py312 pytest tests/ -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add backend/safety_detection/detector_core.py tests/test_detector_core_registry.py
git commit -m "feat: use algorithm alarm_description in small-model alarm reason"
```

---

### Task 4: Frontend — Add alarm description input in algorithm dialog

**Files:**
- Modify: `frontend/safety_detection/algorithms.html`

**Interfaces:**
- Consumes: API response field `alarm_description`
- Produces: payload field `alarm_description` sent to POST/PUT `/algorithms`

- [ ] **Step 1: Add input field to dialog template**

In `frontend/safety_detection/algorithms.html`, after the “巡检显示名” field (around line 234), add:

```html
<div class="type-card-field" style="grid-column: 1 / -1;">
    <label>报警说明（留空则使用默认模板：检测到xx异常）</label>
    <input v-model="dialog.alarm_description" placeholder="如：发现明火，请立即处理" />
</div>
```

- [ ] **Step 2: Initialize and save the field**

In `openDialog` for new algorithm (around line 344-348), add to the dialog object:
```javascript
alarm_description: '',
```

In `openDialog` for existing algorithm (around line 329-337), the `JSON.parse(JSON.stringify(t))` will already copy `alarm_description` if present. Verify by checking the field is available after loading.

In `saveType` payload (around line 355-360), add:
```javascript
alarm_description: d.alarm_description || '',
```

- [ ] **Step 3: Restart backend and verify in browser**

1. Restart backend to load new code.
2. Open `http://127.0.0.1:8000/algorithms.html`.
3. Edit an algorithm, fill in “报警说明”, save.
4. Reopen edit dialog and confirm the text persists.

- [ ] **Step 4: Commit**

```bash
git add frontend/safety_detection/algorithms.html
git commit -m "feat: add alarm_description input to algorithm dialog"
```

---

### Task 5: Frontend — Rename and restyle class filter section

**Files:**
- Modify: `frontend/safety_detection/algorithms.html`

**Interfaces:**
- No data change; only label text and checkbox styling

- [ ] **Step 1: Update label and use consistent checkbox styling**

In `frontend/safety_detection/algorithms.html`, replace the class filter block (around line 237-245) with:

```html
<div class="type-card-field" style="margin-top: 12px;">
    <label>类别选择（全不勾选 = 全部选择）</label>
    <div class="type-card-checkboxes" style="flex-wrap: wrap;">
        <label v-for="c in availableClasses" :key="c.id" class="type-card-checkbox">
            <input type="checkbox" :value="c.id" v-model="dialog.classesChecked" /> {{ c.id }}:{{ c.name }}
        </label>
        <span v-if="!availableClasses.length" style="font-size: 12px; color: var(--text-muted);">该模型无类别清单，保存后不过滤</span>
    </div>
</div>
```

Note: the only changes are the label text and the container class changed from inline `style` to `class="type-card-checkboxes"`. The existing `.type-card-checkbox` styles are already defined in the page's `<style>`.

- [ ] **Step 2: Verify in browser**

1. Open `http://127.0.0.1:8000/algorithms.html`.
2. Edit an algorithm that has a model with classes.
3. Confirm the section title is “类别选择（全不勾选 = 全部选择）”.
4. Confirm checkboxes look identical to those under “运行参数默认值”.

- [ ] **Step 3: Commit**

```bash
git add frontend/safety_detection/algorithms.html
git commit -m "feat: rename class filter to class selection and unify checkbox style"
```

---

### Task 6: End-to-end verification

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

Run: `export PYTHONIOENCODING=utf-8 && conda run -n py312 pytest tests/ -q`
Expected: all pass

- [ ] **Step 2: Browser verification checklist**

1. Algorithms page loads without console errors.
2. Create a new algorithm with `alarm_description`, save, reload — field persists.
3. Edit an existing algorithm, clear `alarm_description`, save — fallback template works on next alarm.
4. Trigger a small-model alarm (or use test endpoint) and verify record reason uses custom text or fallback.
5. Confirm VLM review still overrides reason after复核.
6. Confirm class selection UI label and checkbox style match spec.

- [ ] **Step 3: Final commit**

If any fixes were needed during verification, commit them.

```bash
git commit -m "fix: e2e verification adjustments for alarm_description and class selection"
```

---

## Self-Review

- Spec coverage:
  - `alarm_description` registry field → Task 1
  - API exposure → Task 2
  - Small-model reason generation → Task 3
  - Frontend input → Task 4
  - Category selection rename/style → Task 5
  - VLM unchanged → not a task, preserved by not touching VLM code
- No placeholders used.
- Type consistency: `alarm_description` is string, default `""`; `classesChecked` unchanged.
