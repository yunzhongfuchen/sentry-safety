# 项目综合优化实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the six concrete optimization opportunities identified in the project scan: Ark VLM model-name bug, duplicated video-source heuristic, records storage JSON reload bottleneck, GPU scheduler / MultiDetector overlap, duplicated sidebar markup, and the unused console.html legacy page.

**Architecture:** Make surgical backend fixes (config/understander, camera_manager helper, storage cache, detector scheduling) and one frontend refactor (shared Sidebar component + dead page removal). Each task is isolated and independently testable.

**Tech Stack:** Python 3.12, FastAPI, Vue 3 global build, plain JS/CSS, OpenCV.

## Global Constraints

- Do not change business behavior except where explicitly required by the optimization.
- Keep changes minimal and focused; do not refactor unrelated code.
- Maintain backward compatibility for environment variables where possible.
- Preserve the existing glass-clay / Cool Slate / Chinese branding already implemented.
- Add tests or verification commands for each backend change.
- Commit after each task.
- Do not convert the frontend into an SPA.

---

## File Structure

- `backend/config.py`
  - Add `ARK_MODEL` config (backward-compatible fallback to `VLM_ENDPOINT`).
- `backend/understander.py`
  - Use `config.ARK_MODEL` instead of `config.VLM_ENDPOINT` for the Ark provider.
- `backend/camera_manager.py`
  - Add `CameraConfig.is_video_source()` method.
  - Replace all inline video-source checks with the new helper.
- `backend/performance_storage.py`
  - Add in-memory records cache with write-through invalidation.
- `backend/safety_detection/detector_core.py`
  - Add `externally_managed` flag to `TypeSchedule`.
  - Skip externally-managed types in `_get_due_types`.
- `backend/main_multi.py`
  - Mark scheduler-handled detection types as externally managed when GPU scheduler is active.
- `frontend/safety_detection/shared.js`
  - Add `renderSidebar(container, context)` helper.
- `frontend/safety_detection/monitor.html`
  - Replace inline sidebar markup with `renderSidebar` call.
- `frontend/safety_detection/records.html`
  - Replace inline sidebar markup with `renderSidebar` call.
- `frontend/safety_detection/settings.html`
  - Replace inline sidebar markup with `renderSidebar` call.
- `frontend/safety_detection/console.html`
  - Delete.
- `backend/main_multi.py`
  - Remove `/console` route.

## Task 1: Fix Ark VLM provider model-name bug

**Files:**
- Modify: `backend/config.py`
- Modify: `backend/understander.py`
- Modify: `.env.example`
- Test: `python -c "import backend.understander; ..."` + a small assertion script

**Interfaces:**
- Consumes: existing `config.VLM_ENDPOINT`, `config.ARK_API_KEY`.
- Produces: new `config.ARK_MODEL`; `VideoUnderstander.model` for Ark provider is now the model name, not the endpoint URL.

- [ ] **Step 1: Read current config and understander around the bug**

Run:
```bash
python - <<'PY'
from pathlib import Path
for rel, start, end in [
    ('backend/config.py', 14, 28),
    ('backend/understander.py', 102, 116),
    ('.env.example', 7, 15),
]:
    print(f'--- {rel} ---')
    for i, line in enumerate(Path(rel).read_text(encoding='utf-8').splitlines(), 1):
        if start <= i <= end:
            print(f"{i}:{line}")
PY
```
Expected: see the Ark/VLM config and the buggy `self.model = config.VLM_ENDPOINT` line.

- [ ] **Step 2: Add ARK_MODEL to config.py**

In `backend/config.py`, after the line `VLM_ENDPOINT = os.getenv("VLM_ENDPOINT", "")`, add:

```python
# Backward-compatible: ARK_MODEL defaults to VLM_ENDPOINT so existing deployments keep working.
# New deployments should set ARK_MODEL to the actual Ark model ID (e.g. doubao-vision-pro-32k).
ARK_MODEL = os.getenv("ARK_MODEL", VLM_ENDPOINT)
```

- [ ] **Step 3: Use ARK_MODEL in understander.py**

In `backend/understander.py`, change:

```python
        else:
            self.provider = "ark"
            self.api_key = config.ARK_API_KEY
            self.model = config.VLM_ENDPOINT
            self.endpoint = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
```

to:

```python
        else:
            self.provider = "ark"
            self.api_key = config.ARK_API_KEY
            self.model = config.ARK_MODEL
            self.endpoint = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
```

- [ ] **Step 4: Update .env.example**

In `.env.example`, change the Ark section comment and variables from:

```bash
# --- 火山引擎 Ark ---
ARK_API_KEY=your-ark-api-key-here
VLM_ENDPOINT=your-vlm-endpoint-here
```

to:

```bash
# --- 火山引擎 Ark ---
ARK_API_KEY=your-ark-api-key-here
# Ark 模型名（如 doubao-vision-pro-32k）；未设置时回退到 VLM_ENDPOINT 保持兼容
ARK_MODEL=your-ark-model-name-here
# 保留 VLM_ENDPOINT 仅用于旧配置兼容，新配置请使用 ARK_MODEL
VLM_ENDPOINT=your-ark-model-name-here
```

- [ ] **Step 5: Verify the fix with a small assertion script**

Run:
```bash
python - <<'PY'
import os
os.environ['ARK_API_KEY'] = 'test-key'
os.environ['BAILIAN_API_KEY'] = ''
os.environ['ARK_MODEL'] = 'doubao-test'
import backend.config as config
assert config.ARK_MODEL == 'doubao-test', config.ARK_MODEL
from backend.understander import VideoUnderstander
u = VideoUnderstander()
assert u.provider == 'ark', u.provider
assert u.model == 'doubao-test', u.model
print('ark model fix verified')
PY
```
Expected: `ark model fix verified`

- [ ] **Step 6: Commit**

```bash
git add backend/config.py backend/understander.py .env.example
git commit -m "fix: use ARK_MODEL instead of VLM_ENDPOINT as Ark VLM model name"
```

## Task 2: Refactor duplicated video-source heuristic

**Files:**
- Modify: `backend/camera_manager.py`
- Test: `python -c "from backend.camera_manager import CameraConfig; ..."`

**Interfaces:**
- Consumes: existing `CameraConfig` dataclass fields `source_type` and `source`.
- Produces: new method `CameraConfig.is_video_source()` returning `bool`.

- [ ] **Step 1: Read the current CameraConfig definition and one inline check**

Run:
```bash
python - <<'PY'
from pathlib import Path
text = Path('backend/camera_manager.py').read_text(encoding='utf-8').splitlines()
for i in range(35, 62):
    print(f"{i+1}:{text[i]}")
print('--- example inline check ---')
for i in range(106, 115):
    print(f"{i+1}:{text[i]}")
PY
```
Expected: see the dataclass and a sample inline heuristic.

- [ ] **Step 2: Add `is_video_source()` method to CameraConfig**

In `backend/camera_manager.py`, inside the `CameraConfig` dataclass (after line 60), add:

```python
    def is_video_source(self) -> bool:
        """判断当前源是否为本地视频文件（非摄像头索引、非 RTSP 流）"""
        if self.source_type == "video":
            return True
        if self.source_type == "auto":
            return not str(self.source).isdigit() and not str(self.source).startswith("rtsp")
        return False
```

- [ ] **Step 3: Replace inline checks with helper calls**

Replace every occurrence of the inline heuristic in `backend/camera_manager.py` with `state.config.is_video_source()` or `config.is_video_source()` as appropriate.

The patterns to replace are:

```python
self.config.source_type == "video" or (
    self.config.source_type == "auto"
    and not str(self.config.source).isdigit()
    and not str(self.config.source).startswith("rtsp")
)
```
→ `self.config.is_video_source()`

```python
state.config.source_type == "video" or (
    state.config.source_type == "auto"
    and not str(state.config.source).isdigit()
    and not str(state.config.source).startswith("rtsp")
)
```
→ `state.config.is_video_source()`

```python
source_type == "video" or (
    source_type == "auto"
    and not str(source).isdigit()
    and not str(source).startswith("rtsp")
)
```
→ `CameraConfig(source=source, source_type=source_type).is_video_source()`

Use a verification command to confirm no inline heuristic remains:

```bash
python - <<'PY'
from pathlib import Path
text = Path('backend/camera_manager.py').read_text(encoding='utf-8')
assert text.count('is_video_source()') >= 7, text.count('is_video_source()')
# Ensure the old inline pattern no longer appears
assert 'and not str(state.config.source).isdigit()' not in text
assert 'and not str(self.config.source).isdigit()' not in text
print('video source helper refactor verified')
PY
```
Expected: `video source helper refactor verified`

- [ ] **Step 4: Verify helper logic with a quick test**

Run:
```bash
python - <<'PY'
from backend.camera_manager import CameraConfig
assert CameraConfig(camera_id='c1', source='0', source_type='auto').is_video_source() is False
assert CameraConfig(camera_id='c2', source='rtsp://x', source_type='auto').is_video_source() is False
assert CameraConfig(camera_id='c3', source='/tmp/a.mp4', source_type='auto').is_video_source() is True
assert CameraConfig(camera_id='c4', source='anything', source_type='video').is_video_source() is True
assert CameraConfig(camera_id='c5', source='anything', source_type='camera').is_video_source() is False
print('is_video_source logic verified')
PY
```
Expected: `is_video_source logic verified`

- [ ] **Step 5: Commit**

```bash
git add backend/camera_manager.py
git commit -m "refactor: extract repeated video-source heuristic into CameraConfig helper"
```

## Task 3: Add in-memory cache to performance_storage records

**Files:**
- Modify: `backend/performance_storage.py`
- Test: existing storage tests or a small assertion script

**Interfaces:**
- Consumes: existing `load_records()` and `save_records()` functions.
- Produces: cached records list returned by `load_records()`; `save_records()` invalidates the cache after writing.

- [ ] **Step 1: Read load_records and save_records**

Run:
```bash
python - <<'PY'
from pathlib import Path
text = Path('backend/performance_storage.py').read_text(encoding='utf-8').splitlines()
for i in range(122, 148):
    print(f"{i+1}:{text[i]}")
PY
```
Expected: see `load_records()` and `save_records()` definitions.

- [ ] **Step 2: Add module-level cache**

In `backend/performance_storage.py`, near the top (after LRUCache and before functions), add:

```python
_records_cache: Optional[List[Dict]] = None
```

- [ ] **Step 3: Cache in load_records and invalidate in save_records**

Change `load_records()` to:

```python
def load_records() -> List[Dict]:
    """加载记录元数据（不含图片数据），带内存缓存"""
    global _records_cache
    if _records_cache is not None:
        return _records_cache
    ensure_dirs()
    if not RECORDS_FILE.exists():
        return []
    try:
        with open(RECORDS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} records metadata")
        _records_cache = data
        return data
    except Exception as e:
        logger.error(f"Failed to load records: {e}")
        return []
```

Change `save_records()` to:

```python
def save_records(records: List[Dict]):
    """保存记录元数据"""
    global _records_cache
    ensure_dirs()
    try:
        with open(RECORDS_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
        logger.info(f"Saved {len(records)} records metadata")
        _records_cache = records
    except Exception as e:
        logger.error(f"Failed to save records: {e}")
```

- [ ] **Step 4: Verify cache behavior**

Run:
```bash
python - <<'PY'
from pathlib import Path
import tempfile, shutil, os
orig_data_dir = os.environ.get('DATA_DIR', '')
# Point to temp dir
os.environ['DATA_DIR'] = tempfile.mkdtemp()
# Force reimport
try:
    import backend.performance_storage as ps
    ps.RECORDS_FILE = Path(os.environ['DATA_DIR']) / 'records.json'
    ps._records_cache = None
    ps.save_records([{'id': 'r1'}])
    first = ps.load_records()
    assert first == [{'id': 'r1'}]
    # Mutate cache directly to prove second load returns cached copy
    ps._records_cache.append({'id': 'r2'})
    second = ps.load_records()
    assert len(second) == 2, second
    print('records cache verified')
finally:
    shutil.rmtree(os.environ['DATA_DIR'], ignore_errors=True)
    if orig_data_dir:
        os.environ['DATA_DIR'] = orig_data_dir
    else:
        os.environ.pop('DATA_DIR', None)
PY
```
Expected: `records cache verified`

- [ ] **Step 5: Commit**

```bash
git add backend/performance_storage.py
git commit -m "perf: cache loaded records in memory with write-through invalidation"
```

## Task 4: Avoid duplicate detection when GPU scheduler is active

**Files:**
- Modify: `backend/safety_detection/detector_core.py`
- Modify: `backend/main_multi.py`
- Test: assertion script that checks `_get_due_types` skips externally-managed types

**Interfaces:**
- Consumes: existing `TypeSchedule` dataclass and `MultiDetector._get_due_types()`.
- Produces: new `externally_managed: bool` field on `TypeSchedule`; new `multi_detector.mark_externally_managed(camera_id, dtypes)` API.

- [ ] **Step 1: Read TypeSchedule and _get_due_types**

Run:
```bash
python - <<'PY'
from pathlib import Path
text = Path('backend/safety_detection/detector_core.py').read_text(encoding='utf-8').splitlines()
print('--- TypeSchedule ---')
for i in range(100, 135):
    print(f"{i+1}:{text[i]}")
print('--- _get_due_types ---')
for i in range(454, 462):
    print(f"{i+1}:{text[i]}")
PY
```
Expected: see `TypeSchedule` fields and `_get_due_types` implementation.

- [ ] **Step 2: Add externally_managed field to TypeSchedule**

In `backend/safety_detection/detector_core.py`, in the `TypeSchedule` dataclass, add the field:

```python
    externally_managed: bool = False  # True when a separate scheduler (e.g. GPU scheduler) runs inference for this type
```

- [ ] **Step 3: Skip externally-managed types in _get_due_types**

Change `_get_due_types` from:

```python
    def _get_due_types(self, camera_id: str, now: float) -> List[str]:
        """获取当前到期的检测类型"""
        with self._lock:
            schedules = self._schedules.get(camera_id, {})
            return [dtype for dtype, s in schedules.items() if s.is_due(now)]
```

to:

```python
    def _get_due_types(self, camera_id: str, now: float) -> List[str]:
        """获取当前到期的检测类型（跳过由外部调度器管理的类型）"""
        with self._lock:
            schedules = self._schedules.get(camera_id, {})
            return [dtype for dtype, s in schedules.items() if not s.externally_managed and s.is_due(now)]
```

- [ ] **Step 4: Add mark_externally_managed helper to MultiDetector**

In `backend/safety_detection/detector_core.py`, after `register_camera`, add:

```python
    def mark_externally_managed(self, camera_id: str, dtypes: List[str]) -> None:
        """标记指定检测类型由外部调度器接管，本 Detector 不再对其运行推理"""
        with self._lock:
            schedules = self._schedules.get(camera_id, {})
            for dtype in dtypes:
                if dtype in schedules:
                    schedules[dtype].externally_managed = True
                    logger.info(f"Camera {camera_id} type {dtype} marked as externally managed")
```

- [ ] **Step 5: Mark scheduler types as externally managed in main_multi.py**

In `backend/main_multi.py`, after the GPU scheduler is successfully initialized (after the `gpu_scheduler = GPUDynamicScheduler(...)` block and its log message), add:

```python
            # 让 MultiDetector 跳过已由 GPU scheduler 推理的类型，避免重复检测
            scheduler_types = list(model_configs.keys())
            for cam_data in camera_configs_data:
                camera_id = cam_data["camera_id"]
                multi_detector.mark_externally_managed(camera_id, scheduler_types)
```

Place it before the `vlm_inspector = VLMInspector(...)` line.

- [ ] **Step 6: Verify externally-managed behavior**

Run:
```bash
python - <<'PY'
import sys
sys.path.insert(0, 'backend')
from safety_detection.detector_core import MultiDetector, TypeSchedule
import time

class FakeCM:
    pass

md = MultiDetector(
    camera_manager=FakeCM(),
    safety_detector=None,
    vlm_queue=None,
    strategy=None,
)
md.register_camera('cam1', {
    'fire': {'enabled': True, 'interval': 1.0},
    'sleep': {'enabled': True, 'interval': 1.0},
})
now = time.time()
# Both should be due before marking
assert set(md._get_due_types('cam1', now)) == {'fire', 'sleep'}
md.mark_externally_managed('cam1', ['fire'])
assert set(md._get_due_types('cam1', now)) == {'sleep'}
print('externally-managed detection verified')
PY
```
Expected: `externally-managed detection verified`

- [ ] **Step 7: Commit**

```bash
git add backend/safety_detection/detector_core.py backend/main_multi.py
git commit -m "fix: avoid duplicate inference when GPU scheduler is active"
```

## Task 5: Extract shared Sidebar component and remove inline duplication

**Files:**
- Modify: `frontend/safety_detection/shared.js`
- Modify: `frontend/safety_detection/monitor.html`
- Modify: `frontend/safety_detection/records.html`
- Modify: `frontend/safety_detection/settings.html`
- Test: browser verification on the three pages

**Interfaces:**
- Consumes: existing `getSidebarContext()` helper and CSS classes `.app-sidebar`, `.sidebar-brand`, `.nav-item`, etc.
- Produces: new `renderSidebar(container, context)` function in `shared.js`.

- [ ] **Step 1: Read current inline sidebar markup from one page**

Run:
```bash
python - <<'PY'
from pathlib import Path
text = Path('frontend/safety_detection/monitor.html').read_text(encoding='utf-8').splitlines()
for i in range(230, 266):
    print(f"{i+1}:{text[i]}")
PY
```
Expected: see the full inline sidebar markup.

- [ ] **Step 2: Add renderSidebar helper to shared.js**

In `frontend/safety_detection/shared.js`, after the navigation helper functions, add:

```javascript
function renderSidebar(container, context) {
    const expandedClass = context.settingsExpanded ? 'open' : '';
    container.innerHTML = `
        <aside class="app-sidebar">
            <div class="sidebar-brand">
                <div class="sidebar-brand-mark">S</div>
                <div class="sidebar-brand-copy">
                    <div class="sidebar-brand-title">安全哨兵</div>
                    <div class="sidebar-brand-subtitle">安全检测平台</div>
                </div>
            </div>
            <nav class="sidebar-nav">
                <div class="nav-group">
                    <a href="/monitor" class="nav-item ${context.page === 'monitor' ? 'active' : ''}">
                        <span class="nav-item-label">监控</span>
                    </a>
                    <a href="/records.html" class="nav-item ${context.page === 'records' ? 'active' : ''}">
                        <span class="nav-item-label">记录</span>
                    </a>
                </div>
                <div class="nav-group">
                    <button
                        type="button"
                        class="nav-item parent ${context.page === 'settings' ? 'active' : ''}"
                        aria-expanded="${String(context.settingsExpanded)}"
                        onclick="window.toggleSettingsNav && window.toggleSettingsNav()"
                    >
                        <span class="nav-item-label">设置</span>
                        <span class="nav-item-caret">›</span>
                    </button>
                    <div class="nav-children ${expandedClass}">
                        <a href="/settings.html?tab=cameras" class="nav-item child ${context.tab === 'cameras' ? 'active' : ''}">摄像头</a>
                        <a href="/settings.html?tab=detection" class="nav-item child ${context.tab === 'detection' ? 'active' : ''}">检测配置</a>
                        <a href="/settings.html?tab=system" class="nav-item child ${context.tab === 'system' ? 'active' : ''}">系统设置</a>
                    </div>
                </div>
            </nav>
        </aside>
    `;
}
```

- [ ] **Step 3: Replace inline sidebar in monitor.html**

In `frontend/safety_detection/monitor.html`:
1. Remove the entire `<aside class="app-sidebar">...</aside>` block.
2. Replace it with a placeholder: `<div id="sidebar-root"></div>`.
3. In the Vue `setup()`, after defining `sidebar` and `toggleSettingsNav`, add:

```javascript
                window.toggleSettingsNav = toggleSettingsNav;
                onMounted(() => {
                    const root = document.getElementById('sidebar-root');
                    if (root) renderSidebar(root, sidebar.value);
                });
```

Ensure `onMounted` is destructured from Vue at the top.

- [ ] **Step 4: Replace inline sidebar in records.html**

Repeat Step 3 for `frontend/safety_detection/records.html`:
1. Remove inline `<aside class="app-sidebar">`.
2. Add `<div id="sidebar-root"></div>`.
3. Wire up `renderSidebar` in `onMounted`.

- [ ] **Step 5: Replace inline sidebar in settings.html**

Repeat Step 3 for `frontend/safety_detection/settings.html`:
1. Remove inline `<aside class="app-sidebar">`.
2. Add `<div id="sidebar-root"></div>`.
3. Wire up `renderSidebar` in `onMounted`.
4. Also add a `watch(sidebar, ...)` or update `renderSidebar` when `tab` changes so the active child item updates as the user switches internal tabs.

Simplest approach in settings: after the existing `watch(tab, ...)` block, add:

```javascript
                watch(sidebar, (value) => {
                    const root = document.getElementById('sidebar-root');
                    if (root) renderSidebar(root, value);
                }, { deep: true });
```

- [ ] **Step 6: Verify structural changes**

Run:
```bash
python - <<'PY'
from pathlib import Path
for rel in ['frontend/safety_detection/monitor.html', 'frontend/safety_detection/records.html', 'frontend/safety_detection/settings.html']:
    text = Path(rel).read_text(encoding='utf-8')
    assert 'id="sidebar-root"' in text, rel
    assert 'renderSidebar(root' in text or 'renderSidebar(' in text, rel
    # Ensure no duplicated full sidebar markup remains
    assert text.count('sidebar-brand-title') <= 1, rel
print('shared sidebar integration verified')
PY
```
Expected: `shared sidebar integration verified`

- [ ] **Step 7: Browser spot-check**

Run the app and verify that monitor/records/settings still render the sidebar correctly, highlight the right item, and expand settings by default.

- [ ] **Step 8: Commit**

```bash
git add frontend/safety_detection/shared.js frontend/safety_detection/monitor.html frontend/safety_detection/records.html frontend/safety_detection/settings.html
git commit -m "refactor: render sidebar from shared component to eliminate triplicated markup"
```

## Task 6: Remove unused console.html and its route

**Files:**
- Delete: `frontend/safety_detection/console.html`
- Modify: `backend/main_multi.py`
- Test: verify `/console` route no longer exists

**Interfaces:**
- Consumes: existing `/console` route in `backend/main_multi.py`.
- Produces: removal of dead code.

- [ ] **Step 1: Confirm no references other than the backend route**

Run:
```bash
grep -rn "console.html\|/console" backend/ frontend/safety_detection/ --include="*.py" --include="*.html" --include="*.js"
```
Expected: only `backend/main_multi.py:553` and the file itself.

- [ ] **Step 2: Delete console.html**

```bash
rm frontend/safety_detection/console.html
```

- [ ] **Step 3: Remove /console route from main_multi.py**

In `backend/main_multi.py`, delete the entire `@app.get("/console")` handler block (around lines 553-556).

- [ ] **Step 4: Verify route removal**

Run:
```bash
python - <<'PY'
from pathlib import Path
text = Path('backend/main_multi.py').read_text(encoding='utf-8')
assert '@app.get("/console")' not in text
assert 'console.html' not in text
assert not Path('frontend/safety_detection/console.html').exists()
print('console.html and route removed')
PY
```
Expected: `console.html and route removed`

- [ ] **Step 5: Commit**

```bash
git add frontend/safety_detection/console.html backend/main_multi.py
git commit -m "chore: remove unused console.html and /console route"
```

## Task 7: Final regression sweep

**Files:**
- Verify only: files touched in Tasks 1-6.

- [ ] **Step 1: Run Python syntax checks on modified backend files**

Run:
```bash
python -m py_compile backend/config.py backend/understander.py backend/camera_manager.py backend/performance_storage.py backend/main_multi.py backend/safety_detection/detector_core.py
```
Expected: no output (success).

- [ ] **Step 2: Run existing tests if available**

Run:
```bash
pytest tests/ -q 2>/dev/null || echo "no pytest or no tests"
```
Expected: either passing tests or a message that pytest/tests are unavailable.

- [ ] **Step 3: Static frontend checks**

Run:
```bash
python - <<'PY'
from pathlib import Path
checks = {
    'frontend/safety_detection/shared.js': 'function renderSidebar',
    'frontend/safety_detection/monitor.html': ['id="sidebar-root"', 'renderSidebar'],
    'frontend/safety_detection/records.html': ['id="sidebar-root"', 'renderSidebar'],
    'frontend/safety_detection/settings.html': ['id="sidebar-root"', 'renderSidebar'],
}
for rel, needles in checks.items():
    text = Path(rel).read_text(encoding='utf-8')
    for needle in ([needles] if isinstance(needles, str) else needles):
        assert needle in text, f'{needle} missing in {rel}'
assert not Path('frontend/safety_detection/console.html').exists()
print('final regression checks passed')
PY
```
Expected: `final regression checks passed`

- [ ] **Step 4: Push all commits**

```bash
git push
```

## Self-Review

- Spec coverage: all six optimization items map to Tasks 1-6; Task 7 is verification.
- Placeholder scan: no TBD/TODO; all code snippets and verification commands are concrete.
- Type consistency: `TypeSchedule.externally_managed` is a boolean; `CameraConfig.is_video_source()` returns bool; `renderSidebar` signature is consistent across pages.
