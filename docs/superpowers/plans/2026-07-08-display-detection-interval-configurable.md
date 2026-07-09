# 主画面显示检测刷新频率可配置 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the main-display detection refresh interval configurable through the monitor page, persisting alongside display detection type toggles.

**Architecture:** Extend the existing `/display-types` API to carry a `display_detection_interval` field. Store the interval in `config/global.json` under the same global settings key. Thread the interval through `SelectedCameraDisplay` so its detection loop sleeps the configured duration instead of a hard-coded 1.0 second. Add a number input in `monitor.html` within the existing "显示检测类型" panel.

**Tech Stack:** Python 3.12, FastAPI, OpenCV, Vue 3 (CDN), pytest

## Global Constraints

- Interval must be configurable in the range **0.1 ~ 10.0 seconds**, inclusive.
- Interval is **global and uniform** across all display detection types.
- Backend must clamp out-of-range values to `[0.1, 10.0]` silently rather than erroring.
- Default interval is **1.0** second.
- Reuse the existing `/display-types` GET/POST endpoints; do not add a separate endpoint for interval.
- Persist interval alongside `display_detection_types` in `config/global.json`.
- Frontend and backend must stay consistent after a page reload.
- Existing global detection behavior must not change.
- All changes must have regression tests.

---

## File Structure

- `backend/config.py`
  - Add `display_detection_interval` to `DEFAULT_GLOBAL_SETTINGS`.
- `backend/main_multi.py`
  - Update `init_components()` to read the interval default.
  - Update `SelectedCameraDisplay` to accept and apply the interval.
  - Update `GET /display-types` and `POST /display-types` to include the interval.
- `frontend/safety_detection/monitor.html`
  - Add interval input in the display types panel.
  - Wire the input to the existing save flow.
- `tests/test_selected_camera_display.py`
  - Add tests for interval application and clamping.
- `tests/test_select_main_camera_api.py`
  - Update or extend API tests to verify the interval field in `/display-types`.

## Task 1: Add interval default to global settings

**Files:**
- Modify: `backend/config.py`
- Test: `tests/test_select_main_camera_api.py` (or a new lightweight config test if one exists)

**Interfaces:**
- Consumes: `app_config.DEFAULT_GLOBAL_SETTINGS`
- Produces: `app_config.DEFAULT_GLOBAL_SETTINGS["display_detection_interval"]` = `1.0`

- [ ] **Step 1: Write the failing test**

```python
from backend import config as app_config


def test_default_global_settings_contains_display_interval():
    assert "display_detection_interval" in app_config.DEFAULT_GLOBAL_SETTINGS
    assert app_config.DEFAULT_GLOBAL_SETTINGS["display_detection_interval"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_default_global_settings_contains_display_interval -v`
Expected: FAIL (test file may not exist yet; if it does, FAIL because key is missing)

- [ ] **Step 3: Write minimal implementation**

In `backend/config.py`, add inside `DEFAULT_GLOBAL_SETTINGS`:

```python
DEFAULT_GLOBAL_SETTINGS = {
    ...,
    "display_detection_types": {
        "fire": True,
        "smoke": True,
        "uniform": True,
        "mask": True,
        "cigarette": True,
        "sleep": True,
    },
    "display_detection_interval": 1.0,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::test_default_global_settings_contains_display_interval -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/config.py tests/test_config.py
git commit -m "feat: add display_detection_interval default"
```

## Task 2: Make SelectedCameraDisplay use a configurable interval

**Files:**
- Modify: `backend/main_multi.py`
- Test: `tests/test_selected_camera_display.py`

**Interfaces:**
- Consumes: `SelectedCameraDisplay.__init__(..., display_interval: float = 1.0)` and `set_display_config(display_types: Dict[str, bool], display_interval: Optional[float] = None)`
- Produces: `SelectedCameraDisplay._display_interval: float` used by `_detect_loop()`

- [ ] **Step 1: Write the failing test**

```python
from unittest.mock import MagicMock, patch

import numpy as np

from backend.main_multi import SelectedCameraDisplay


@patch("backend.main_multi.DisplayDetectionWorker")
def test_selected_camera_display_stores_display_interval(_mock_worker_cls):
    camera_manager = MagicMock()
    stream_server = MagicMock()
    display = SelectedCameraDisplay(
        camera_manager, stream_server, npu_cores=0, device="cpu", display_interval=2.5
    )
    assert display._display_interval == 2.5


@patch("backend.main_multi.DisplayDetectionWorker")
def test_selected_camera_display_clamps_interval_to_range(_mock_worker_cls):
    camera_manager = MagicMock()
    stream_server = MagicMock()

    display_low = SelectedCameraDisplay(
        camera_manager, stream_server, npu_cores=0, device="cpu", display_interval=0.05
    )
    assert display_low._display_interval == 0.1

    display_high = SelectedCameraDisplay(
        camera_manager, stream_server, npu_cores=0, device="cpu", display_interval=20.0
    )
    assert display_high._display_interval == 10.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_selected_camera_display.py::test_selected_camera_display_stores_display_interval tests/test_selected_camera_display.py::test_selected_camera_display_clamps_interval_to_range -v`
Expected: FAIL because `display_interval` parameter and `_display_interval` attribute do not exist

- [ ] **Step 3: Write minimal implementation**

Add to `SelectedCameraDisplay.__init__`:

```python
def __init__(
    self,
    camera_manager,
    stream_server,
    npu_cores: int,
    device: str,
    display_types: Optional[Dict[str, bool]] = None,
    display_interval: float = 1.0,
):
    ...
    self._display_types: Dict[str, bool] = dict(display_types) if display_types else {}
    self._display_interval: float = self._clamp_display_interval(display_interval)
    ...
```

Add a static helper:

```python
@staticmethod
def _clamp_display_interval(value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0
    if value < 0.1:
        return 0.1
    if value > 10.0:
        return 10.0
    return value
```

Update `_detect_loop()`:

```python
while self._running:
    try:
        interval = self._display_interval
        time.sleep(interval)
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_selected_camera_display.py::test_selected_camera_display_stores_display_interval tests/test_selected_camera_display.py::test_selected_camera_display_clamps_interval_to_range -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/main_multi.py tests/test_selected_camera_display.py
git commit -m "feat: configurable display detection interval in SelectedCameraDisplay"
```

## Task 3: Add runtime interval update method

**Files:**
- Modify: `backend/main_multi.py`
- Test: `tests/test_selected_camera_display.py`

**Interfaces:**
- Consumes: `set_display_config(display_types: Dict[str, bool], display_interval: Optional[float] = None)`
- Produces: `self._display_types` and `self._display_interval` updated atomically

- [ ] **Step 1: Write the failing test**

```python
@patch("backend.main_multi.DisplayDetectionWorker")
def test_set_display_config_updates_interval(_mock_worker_cls):
    camera_manager = MagicMock()
    stream_server = MagicMock()
    display = SelectedCameraDisplay(camera_manager, stream_server, npu_cores=0, device="cpu")

    display.set_display_config({"fire": True}, display_interval=3.0)

    assert display._display_types == {"fire": True}
    assert display._display_interval == 3.0


@patch("backend.main_multi.DisplayDetectionWorker")
def test_set_display_config_clamps_interval(_mock_worker_cls):
    camera_manager = MagicMock()
    stream_server = MagicMock()
    display = SelectedCameraDisplay(camera_manager, stream_server, npu_cores=0, device="cpu")

    display.set_display_config({"fire": True}, display_interval=0.0)
    assert display._display_interval == 0.1

    display.set_display_config({"fire": True}, display_interval=15.0)
    assert display._display_interval == 10.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_selected_camera_display.py::test_set_display_config_updates_interval tests/test_selected_camera_display.py::test_set_display_config_clamps_interval -v`
Expected: FAIL because `set_display_config` does not exist

- [ ] **Step 3: Write minimal implementation**

Replace `set_display_types` with `set_display_config`:

```python
def set_display_config(
    self, display_types: Dict[str, bool], display_interval: Optional[float] = None
):
    """更新显示类型开关和刷新频率"""
    with self._lock:
        self._display_types = dict(display_types)
        if display_interval is not None:
            self._display_interval = self._clamp_display_interval(display_interval)
        if not any(self._display_types.values()):
            self._last_detection_results = {}
            self._overlay_expires_at = 0.0
```

Keep a thin backward-compatible alias if other callers still use `set_display_types`:

```python
def set_display_types(self, display_types: Dict[str, bool]):
    """兼容旧调用：仅更新显示类型开关，不修改刷新频率"""
    self.set_display_config(display_types)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_selected_camera_display.py::test_set_display_config_updates_interval tests/test_selected_camera_display.py::test_set_display_config_clamps_interval -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/main_multi.py tests/test_selected_camera_display.py
git commit -m "feat: add set_display_config for interval updates"
```

## Task 4: Wire interval through init_components and API

**Files:**
- Modify: `backend/main_multi.py`
- Test: `tests/test_select_main_camera_api.py`

**Interfaces:**
- Consumes: `_global_settings["display_detection_interval"]`
- Produces: `SelectedCameraDisplay` constructed with interval; `/display-types` GET/POST include interval

- [ ] **Step 1: Write the failing test**

```python
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def test_get_display_types_includes_interval():
    with patch("backend.main_multi.app_config.load_global_settings") as mock_load, \
         patch("backend.main_multi.selected_camera_display") as mock_display:
        mock_load.return_value = {
            "display_detection_types": {"fire": True},
            "display_detection_interval": 2.5,
        }
        from backend.main_multi import app
        client = TestClient(app)
        response = client.get("/display-types")
        assert response.status_code == 200
        data = response.json()
        assert data["display_detection_interval"] == 2.5


def test_post_display_types_updates_interval():
    with patch("backend.main_multi.app_config.load_global_settings") as mock_load, \
         patch("backend.main_multi.app_config.save_global_settings") as mock_save, \
         patch("backend.main_multi.selected_camera_display") as mock_display:
        mock_load.return_value = {
            "display_detection_types": {"fire": True},
            "display_detection_interval": 1.0,
        }
        from backend.main_multi import app
        client = TestClient(app)
        response = client.post(
            "/display-types",
            json={"display_detection_types": {"fire": True}, "display_detection_interval": 0.5},
        )
        assert response.status_code == 200
        saved = mock_save.call_args[0][0]
        assert saved["display_detection_interval"] == 0.5
        mock_display.set_display_config.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_select_main_camera_api.py::test_get_display_types_includes_interval tests/test_select_main_camera_api.py::test_post_display_types_updates_interval -v`
Expected: FAIL because `/display-types` does not yet return or accept interval

- [ ] **Step 3: Write minimal implementation**

In `init_components()`:

```python
display_types = _global_settings.get("display_detection_types", app_config.DEFAULT_GLOBAL_SETTINGS["display_detection_types"])
display_interval = _global_settings.get("display_detection_interval", app_config.DEFAULT_GLOBAL_SETTINGS["display_detection_interval"])
selected_camera_display = SelectedCameraDisplay(
    camera_manager=camera_manager,
    stream_server=stream_server,
    npu_cores=npu_cores,
    device=device,
    display_types=display_types,
    display_interval=display_interval,
)
```

Update `GET /display-types`:

```python
@app.get("/display-types")
async def get_display_types():
    settings = app_config.load_global_settings()
    types = settings.get("display_detection_types", app_config.DEFAULT_GLOBAL_SETTINGS["display_detection_types"])
    interval = settings.get("display_detection_interval", app_config.DEFAULT_GLOBAL_SETTINGS["display_detection_interval"])
    return {**types, "display_detection_interval": interval}
```

Update `POST /display-types`:

```python
@app.post("/display-types")
async def update_display_types(data: dict):
    display_types = data.get("display_detection_types", {})
    if not isinstance(display_types, dict):
        return JSONResponse({"error": "Invalid display_detection_types"}, status_code=400)

    display_interval = data.get("display_detection_interval")
    if display_interval is not None:
        try:
            display_interval = float(display_interval)
        except (TypeError, ValueError):
            return JSONResponse({"error": "Invalid display_detection_interval"}, status_code=400)
        if display_interval < 0.1:
            display_interval = 0.1
        elif display_interval > 10.0:
            display_interval = 10.0

    settings = app_config.load_global_settings()
    settings["display_detection_types"] = display_types
    if display_interval is not None:
        settings["display_detection_interval"] = display_interval
    app_config.save_global_settings(settings)

    if selected_camera_display is not None:
        selected_camera_display.set_display_config(display_types, display_interval)

    log_message(f"Display types updated: {display_types}, interval={display_interval}")
    return {
        "success": True,
        "display_detection_types": display_types,
        "display_detection_interval": settings.get("display_detection_interval", 1.0),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_select_main_camera_api.py::test_get_display_types_includes_interval tests/test_select_main_camera_api.py::test_post_display_types_updates_interval -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/main_multi.py tests/test_select_main_camera_api.py
git commit -m "feat: expose display_detection_interval via /display-types API"
```

## Task 5: Add interval input to monitor.html

**Files:**
- Modify: `frontend/safety_detection/monitor.html`

**Interfaces:**
- Consumes: `GET /display-types` response with `display_detection_interval`
- Produces: `POST /display-types` body with `display_detection_types` and `display_detection_interval`

- [ ] **Step 1: Update template markup**

In the "显示检测类型" panel, after the toggle list, add:

```html
<div style="margin-top: 16px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 12px;">
    <div style="font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
        刷新频率
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <input type="number" class="clay-input"
               style="width: 80px; padding: 8px 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); background: rgba(0,0,0,0.15); color: var(--text-primary);"
               v-model.number="displayInterval"
               min="0.1" max="10" step="0.1" />
        <span style="color: var(--text-secondary); font-size: 13px;">秒</span>
        <button class="clay-button" style="margin-left: auto; padding: 6px 12px; font-size: 12px;"
                @click="saveDisplayConfig">保存</button>
    </div>
</div>
```

- [ ] **Step 2: Update Vue setup**

Add state:

```javascript
const displayInterval = ref(1.0);
```

Update `fetchDisplayTypes`:

```javascript
async function fetchDisplayTypes() {
    try {
        const data = await safeFetch('/display-types');
        if (data) {
            displayTypes.value = Object.fromEntries(
                Object.entries(data).filter(([k]) => k !== 'display_detection_interval')
            );
            displayInterval.value = data.display_detection_interval ?? 1.0;
        }
    } catch (e) { console.error('获取显示类型失败:', e); }
}
```

Update `toggleDisplayType` to use a unified save:

```javascript
async function toggleDisplayType(type) {
    const newTypes = { ...displayTypes.value, [type]: !displayTypes.value[type] };
    displayTypes.value = newTypes;
    await saveDisplayConfig();
}

async function saveDisplayConfig() {
    try {
        const data = await safeFetch('/display-types', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                display_detection_types: displayTypes.value,
                display_detection_interval: displayInterval.value,
            }),
        });
        if (data.display_detection_interval !== undefined) {
            displayInterval.value = data.display_detection_interval;
        }
    } catch (e) {
        console.error('更新显示配置失败:', e);
    }
}
```

Update the `return` object to expose `displayInterval` and `saveDisplayConfig`:

```javascript
return {
    sidebar, cameras, recentAlerts, displayTypes, displayInterval, detectionTypes, streamVisible, utcTime,
    ...
    toggleSettingsNav, selectCamera, onVideoError, toggleDisplayType, saveDisplayConfig
}
```

- [ ] **Step 3: Manual verification**

Start the app, open `/monitor`, and verify:
1. The refresh interval input shows the current value (default 1.0).
2. Changing the value and clicking save updates the interval.
3. Toggling a detection type still works and preserves the interval.
4. Refreshing the page restores the saved interval.
5. Values below 0.1 or above 10 are clamped after save.

- [ ] **Step 4: Commit**

```bash
git add frontend/safety_detection/monitor.html
git commit -m "feat: add display detection interval input on monitor page"
```

## Task 6: Full regression verification

**Files:**
- All modified files

- [ ] **Step 1: Run targeted tests**

```bash
pytest tests/test_selected_camera_display.py tests/test_select_main_camera_api.py -v
```

Expected: PASS

- [ ] **Step 2: Run full test suite**

```bash
pytest -q
```

Expected: PASS with only pre-existing warnings

- [ ] **Step 3: Manual end-to-end check**

1. Start the backend.
2. Open `/monitor`.
3. Enable a display detection type and set interval to 0.5s.
4. Observe that boxes refresh roughly every 0.5 seconds.
5. Set interval to 3.0s and verify the refresh slows down.
6. Reload the page and confirm the saved interval is displayed.
7. Disable all types and confirm no detection runs but the interval setting is retained.

- [ ] **Step 4: Commit any final fixes**

If manual testing reveals one small issue, fix it and re-run the relevant verification before committing.

```bash
git add ...
git commit -m "fix: address interval config edge cases"
```
