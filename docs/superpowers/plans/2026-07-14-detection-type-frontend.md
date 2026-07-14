# 检测类型注册表前端管理实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为检测类型注册表提供完整的前端管理能力，包括类型管理页面、摄像头弹窗优化、ROI 绘制和人数条件配置。

**Architecture:** 后端扩展注册表 CRUD 和模型上传 API，前端新增独立类型管理页面并优化摄像头配置弹窗。人数条件通过 `box_count_mode` 字段支持四种模式（≥、≤、区间内、区间外）。

**Tech Stack:** Python 3.12, FastAPI, Vue 3, pytest

## Global Constraints

- Python 3.12，conda 环境 `py312`
- 后端测试命令：`C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest <file> -v`
- 前端无构建步骤，直接编辑 HTML/JS
- 检测类型 key 由后端自动生成，不暴露给用户
- ROI 坐标使用归一化格式（0~1）
- 删除类型前必须检查摄像头引用，有引用则禁止删除

---

### Task 1: 后端注册表扩展 — 类型 CRUD 与模型保存

**Files:**
- Modify: `backend/detection_registry.py`
- Test: `tests/test_detection_registry.py`

**Interfaces:**
- Consumes: 现有 `DetectionTypeRegistry` 类
- Produces: `add_type()`, `delete_type()`, `save_model()`, `is_type_referenced()`

- [ ] **Step 1: Write the failing test**

```python
def test_add_type_generates_key_and_saves(self, tmp_path, monkeypatch):
    import backend.detection_registry as mod
    monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(mod, "REGISTRY_FILE", tmp_path / "detection_types.json")
    r = mod.DetectionTypeRegistry()
    r.load()
    key = r.add_type({"label": "测试类型", "color": "#ff0000", "model_path": "test.pt", "post_process": "yolo_box"})
    assert key is not None
    assert r.get(key)["label"] == "测试类型"

def test_add_type_duplicate_label_raises(self, tmp_path, monkeypatch):
    import backend.detection_registry as mod
    monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(mod, "REGISTRY_FILE", tmp_path / "detection_types.json")
    r = mod.DetectionTypeRegistry()
    r.load()
    with pytest.raises(ValueError, match="already exists"):
        r.add_type({"label": "明火", "color": "#ff0000", "model_path": "x.pt", "post_process": "yolo_box"})

def test_delete_type_removes_and_saves(self, tmp_path, monkeypatch):
    import backend.detection_registry as mod
    monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(mod, "REGISTRY_FILE", tmp_path / "detection_types.json")
    r = mod.DetectionTypeRegistry()
    r.load()
    key = r.add_type({"label": "临时类型", "color": "#00ff00", "model_path": "tmp.pt", "post_process": "yolo_box"})
    assert r.get(key) is not None
    r.delete_type(key)
    assert r.get(key) is None

def test_update_type_structural_fields(self, tmp_path, monkeypatch):
    import backend.detection_registry as mod
    monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(mod, "REGISTRY_FILE", tmp_path / "detection_types.json")
    r = mod.DetectionTypeRegistry()
    r.load()
    key = r.add_type({"label": "旧名称", "color": "#111111", "model_path": "old.pt", "post_process": "yolo_box"})
    r.update_type(key, {"label": "新名称", "color": "#222222", "model_path": "new.pt"})
    td = r.get(key)
    assert td["label"] == "新名称"
    assert td["color"] == "#222222"
    assert td["model_path"] == "new.pt"

def test_save_model_writes_file(self, tmp_path, monkeypatch):
    import backend.detection_registry as mod
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    r = mod.DetectionTypeRegistry()
    content = b"fake model content"
    path = r.save_model("test_model.pt", content)
    assert path.exists()
    assert path.read_bytes() == content
    assert path.parent.name == "weights"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest tests/test_detection_registry.py::TestDetectionTypeRegistry::test_add_type_generates_key_and_saves -v`
Expected: FAIL with "AttributeError: 'DetectionTypeRegistry' object has no attribute 'add_type'"

- [ ] **Step 3: Write minimal implementation**

在 `DetectionTypeRegistry` 中新增：

```python
def add_type(self, type_def: dict) -> str:
    """新增检测类型，自动生成唯一 key，返回 key"""
    label = type_def.get("label", "").strip()
    if not label:
        raise ValueError("label is required")
    for existing in self._types.values():
        if existing.get("label") == label:
            raise ValueError(f"label '{label}' already exists")
    import uuid
    base = label.lower().replace(" ", "_")
    key = f"{base}_{uuid.uuid4().hex[:6]}"
    while key in self._types:
        key = f"{base}_{uuid.uuid4().hex[:6]}"
    merged = copy.deepcopy(UNIVERSAL_DEFAULTS)
    merged.update(type_def.get("defaults", {}))
    self._types[key] = {
        "label": label,
        "color": type_def.get("color", "#888888"),
        "icon": type_def.get("icon", ""),
        "model_path": type_def.get("model_path"),
        "npu_model_path": type_def.get("npu_model_path"),
        "post_process": type_def.get("post_process", "yolo_box"),
        "classes": type_def.get("classes"),
        "model_confidence": type_def.get("model_confidence", 0.5),
        "vlm_prompt_key": type_def.get("vlm_prompt_key", ""),
        "inspection_label": type_def.get("inspection_label", label),
        "defaults": merged,
    }
    self._save(self._types)
    return key

def update_type(self, dtype: str, updates: dict) -> None:
    """更新检测类型的结构性字段（不更新 defaults）"""
    td = self.get(dtype)
    if td is None:
        raise KeyError(f"Unknown detection type: {dtype}")
    if "label" in updates:
        new_label = updates["label"].strip()
        if not new_label:
            raise ValueError("label is required")
        for k, existing in self._types.items():
            if k != dtype and existing.get("label") == new_label:
                raise ValueError(f"label '{new_label}' already exists")
        td["label"] = new_label
    for field in ("color", "icon", "model_path", "npu_model_path", "post_process",
                  "classes", "model_confidence", "vlm_prompt_key", "inspection_label"):
        if field in updates:
            td[field] = updates[field]
    self._save(self._types)

def delete_type(self, dtype: str) -> None:
    """删除检测类型"""
    if dtype not in self._types:
        raise KeyError(f"Unknown detection type: {dtype}")
    del self._types[dtype]
    self._save(self._types)

def save_model(self, filename: str, content: bytes) -> Path:
    """保存模型文件到 weights/ 目录"""
    weights_dir = PROJECT_ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    path = weights_dir / filename
    path.write_bytes(content)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest tests/test_detection_registry.py -v -k "add_type or delete_type or save_model"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/detection_registry.py tests/test_detection_registry.py
git commit -m "feat: add type CRUD and model save to detection registry"
```

---

### Task 2: 后端 API 扩展 — 类型管理与模型上传

**Files:**
- Modify: `backend/safety_detection/api.py`
- Test: `tests/test_detector_types_api.py`

**Interfaces:**
- Consumes: `DetectionTypeRegistry.add_type()`, `delete_type()`, `save_model()`, `update_type()`
- Produces: `POST /detector/types`, `PUT /detector/types/{dtype}`, `DELETE /detector/types/{dtype}`, `POST /detector/types/{dtype}/model`

- [ ] **Step 1: Write the failing test**

```python
def test_create_type_returns_key(self, client):
    payload = {"label": "新类型", "color": "#123456", "model_path": "new.pt", "post_process": "yolo_box"}
    resp = client.post("/detector/types", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "key" in data
    assert data["label"] == "新类型"

def test_create_type_duplicate_label_400(self, client):
    payload = {"label": "明火", "color": "#123456", "model_path": "x.pt", "post_process": "yolo_box"}
    resp = client.post("/detector/types", json=payload)
    assert resp.status_code == 400

def test_delete_type_success(self, client):
    # 先新增一个类型
    resp = client.post("/detector/types", json={"label": "待删除", "color": "#000000", "model_path": "d.pt", "post_process": "yolo_box"})
    key = resp.json()["key"]
    resp = client.delete(f"/detector/types/{key}")
    assert resp.status_code == 200

def test_delete_referenced_type_409(self, client):
    # fire 被摄像头配置引用
    resp = client.delete("/detector/types/fire")
    assert resp.status_code == 409

def test_upload_model_success(self, client, tmp_path):
    import io
    resp = client.post(
        "/detector/types/fire/model",
        files={"file": ("test_model.pt", io.BytesIO(b"fake"), "application/octet-stream")}
    )
    assert resp.status_code == 200
    assert resp.json()["model_path"] == "test_model.pt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest tests/test_detector_types_api.py -v -k "create_type or delete_type or upload_model"`
Expected: FAIL with 404/405

- [ ] **Step 3: Write minimal implementation**

在 `api.py` 中新增：

```python
from fastapi import UploadFile, File

@router.put("/detector/types/{dtype}")
async def update_detection_type(dtype: str, data: dict):
    """更新检测类型（结构性字段 + defaults）"""
    type_def = registry.get(dtype)
    if type_def is None:
        return JSONResponse({"error": f"Unknown detection type: {dtype}"}, status_code=404)

    structural_fields = {"label", "color", "icon", "model_path", "npu_model_path",
                        "post_process", "classes", "model_confidence", "vlm_prompt_key", "inspection_label"}
    structural_update = {k: v for k, v in data.items() if k in structural_fields}
    defaults_update = {k: v for k, v in data.items() if k not in structural_fields}

    try:
        if structural_update:
            registry.update_type(dtype, structural_update)
        if defaults_update:
            allowed_keys = {"enabled", "interval", "threshold", "consecutive_required",
                            "cooldown", "use_vlm", "min_box_count", "max_box_count"}
            for k, v in defaults_update.items():
                if k not in allowed_keys:
                    continue
                error = _validate_default_value(k, v)
                if error:
                    return JSONResponse({"error": error}, status_code=400)
            registry.update_defaults(dtype, defaults_update)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    return {"success": True, "dtype": dtype, "type": registry.get(dtype)}

@router.post("/detector/types")
async def create_detection_type(data: dict):
    """新增检测类型"""
    try:
        key = registry.add_type(data)
        type_def = registry.get(key)
        return {
            "key": key,
            "label": type_def.get("label", key),
            "color": type_def.get("color", "#888888"),
            "icon": type_def.get("icon", ""),
            "post_process": type_def.get("post_process", "yolo_box"),
            "defaults": type_def.get("defaults", {}),
        }
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@router.delete("/detector/types/{dtype}")
async def delete_detection_type(dtype: str, request: Request):
    """删除检测类型（检查摄像头引用）"""
    if registry.get(dtype) is None:
        return JSONResponse({"error": f"Unknown detection type: {dtype}"}, status_code=404)
    # 检查摄像头引用
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if camera_manager is not None:
        for cam_id, cam in camera_manager._cameras.items():
            if dtype in cam.config.detection_types:
                return JSONResponse({"error": f"Type '{dtype}' is referenced by camera '{cam_id}'"}, status_code=409)
    registry.delete_type(dtype)
    return {"success": True, "dtype": dtype}

@router.post("/detector/types/{dtype}/model")
async def upload_model(dtype: str, file: UploadFile = File(...)):
    """上传模型文件"""
    if registry.get(dtype) is None:
        return JSONResponse({"error": f"Unknown detection type: {dtype}"}, status_code=404)
    filename = file.filename
    if not filename.endswith((".pt", ".rknn")):
        return JSONResponse({"error": "Only .pt and .rknn files are allowed"}, status_code=400)
    content = await file.read()
    registry.save_model(filename, content)
    # 自动填入对应路径字段
    type_def = registry.get(dtype)
    if filename.endswith(".pt"):
        type_def["model_path"] = filename
    else:
        type_def["npu_model_path"] = filename
    registry._save(registry._types)
    return {"success": True, "model_path": filename, "dtype": dtype}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest tests/test_detector_types_api.py -v -k "create_type or delete_type or upload_model"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/safety_detection/api.py tests/test_detector_types_api.py
git commit -m "feat: add type management and model upload APIs"
```

---

### Task 3: 后端人数条件扩展 — box_count_mode

**Files:**
- Modify: `backend/safety_detection/detector_core.py`
- Test: `tests/test_detector_core_registry.py`

**Interfaces:**
- Consumes: 现有 `TypeSchedule`, `check_box_count()`
- Produces: `TypeSchedule.box_count_mode`, `check_box_count()` 支持 `outside` 模式

- [ ] **Step 1: Write the failing test**

```python
def test_check_box_count_outside_mode(self):
    from backend.safety_detection.detector_core import check_box_count
    result = {"boxes": [[0,0,10,10]] * 5, "scores": [0.9]*5, "detected": True}
    # outside: < 3 或 > 8 时报警，5 在区间内，不报警
    out = check_box_count(result, min_box_count=3, max_box_count=8, box_count_mode="outside")
    assert out["detected"] is False
    # outside: 2 < 3，报警
    result2 = {"boxes": [[0,0,10,10]] * 2, "scores": [0.9]*2, "detected": False}
    out2 = check_box_count(result2, min_box_count=3, max_box_count=8, box_count_mode="outside")
    assert out2["detected"] is True
    # outside: 10 > 8，报警
    result3 = {"boxes": [[0,0,10,10]] * 10, "scores": [0.9]*10, "detected": True}
    out3 = check_box_count(result3, min_box_count=3, max_box_count=8, box_count_mode="outside")
    assert out3["detected"] is True

def test_type_schedule_has_box_count_mode(self):
    from backend.safety_detection.detector_core import TypeSchedule
    s = TypeSchedule(dtype="fire", enabled=True, interval=1, threshold=0.5, cooldown=60)
    assert hasattr(s, "box_count_mode")
    assert s.box_count_mode is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest tests/test_detector_core_registry.py -v -k "box_count_mode or outside"`
Expected: FAIL with TypeError: check_box_count() got an unexpected keyword argument 'box_count_mode'

- [ ] **Step 3: Write minimal implementation**

在 `TypeSchedule` 中新增字段：

```python
@dataclass
class TypeSchedule:
    ...
    min_box_count: int = None
    max_box_count: int = None
    box_count_mode: str = None  # gte | lte | between | outside
```

修改 `check_box_count`：

```python
def check_box_count(result: dict, min_box_count: int = None,
                    max_box_count: int = None, box_count_mode: str = None) -> dict:
    """按框数量阈值判断检测结果"""
    box_count = len(result.get("boxes", []))

    if box_count_mode == "outside":
        # 目标数 < a 或 > b 时报警
        if min_box_count is not None and box_count < min_box_count:
            return {**result, "detected": True}
        if max_box_count is not None and box_count > max_box_count:
            return {**result, "detected": True}
        return {**result, "detected": False}

    # 原有逻辑（gte / lte / between）
    if min_box_count is not None and box_count < min_box_count:
        return {**result, "detected": False}

    if max_box_count is not None:
        if box_count > max_box_count:
            return {**result, "detected": False}
        return {**result, "detected": True}

    return result
```

修改 `_handle_standard_detection` 调用处：

```python
if schedule.min_box_count is not None or schedule.max_box_count is not None:
    result = check_box_count(result, schedule.min_box_count, schedule.max_box_count, schedule.box_count_mode)
```

修改 `register_camera` 读取 `box_count_mode`：

```python
schedule = TypeSchedule(
    ...
    min_box_count=cfg.get("min_box_count"),
    max_box_count=cfg.get("max_box_count"),
    box_count_mode=cfg.get("box_count_mode"),
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest tests/test_detector_core_registry.py -v -k "box_count_mode or outside"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/safety_detection/detector_core.py tests/test_detector_core_registry.py
git commit -m "feat: add box_count_mode support for outside detection"
```

---

### Task 4: 前端共享工具扩展 — 人数条件与 ROI 工具

**Files:**
- Modify: `frontend/safety_detection/shared.js`

**Interfaces:**
- Consumes: 现有 `DETECTION_TYPES`, `defaultDetectionTypes()`
- Produces: `boxCountModeToFields()`, `fieldsToBoxCountMode()`, `drawRoiOnCanvas()`

- [ ] **Step 1: Write the failing test**

前端 JavaScript 测试使用浏览器环境，这里用 Node 的 `node --test` 或直接在浏览器 console 验证。为保持简单，写一个独立的 HTML 测试文件 `tests/test_shared_js.html` 并在浏览器中验证。

但按照项目惯例，前端测试较少，这里改为：在 `shared.js` 中添加函数后，通过类型管理页面和摄像头弹窗的集成测试间接验证。

先写函数签名和 JSDoc：

```javascript
/**
 * 将人数条件模式转换为后端字段
 * @param {string} mode - 'gte' | 'lte' | 'between' | 'outside'
 * @param {number} a - 下界
 * @param {number} b - 上界（可选）
 * @returns {{min_box_count: number|null, max_box_count: number|null, box_count_mode: string}}
 */
function boxCountModeToFields(mode, a, b) {
    switch (mode) {
        case 'gte': return { min_box_count: a, max_box_count: null, box_count_mode: 'gte' };
        case 'lte': return { min_box_count: null, max_box_count: a, box_count_mode: 'lte' };
        case 'between': return { min_box_count: a, max_box_count: b, box_count_mode: 'between' };
        case 'outside': return { min_box_count: a, max_box_count: b, box_count_mode: 'outside' };
        default: return { min_box_count: null, max_box_count: null, box_count_mode: null };
    }
}

/**
 * 将后端字段转换为人数条件模式
 * @param {number|null} min - min_box_count
 * @param {number|null} max - max_box_count
 * @param {string|null} mode - box_count_mode
 * @returns {{mode: string, a: number|null, b: number|null}}
 */
function fieldsToBoxCountMode(min, max, mode) {
    if (mode === 'outside') return { mode: 'outside', a: min, b: max };
    if (min !== null && max !== null) return { mode: 'between', a: min, b: max };
    if (min !== null) return { mode: 'gte', a: min, b: null };
    if (max !== null) return { mode: 'lte', a: max, b: null };
    return { mode: 'gte', a: null, b: null };
}

/**
 * 在 canvas 上绘制 ROI 多边形
 * @param {HTMLCanvasElement} canvas
 * @param {Array<[number, number]>} points - 归一化坐标点 [[x1,y1], [x2,y2], ...]
 * @param {boolean} closed - 是否闭合
 */
function drawRoiOnCanvas(canvas, points, closed = false) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (points.length === 0) return;
    ctx.strokeStyle = '#22c55e';
    ctx.fillStyle = 'rgba(34, 197, 94, 0.2)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(points[0][0] * w, points[0][1] * h);
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i][0] * w, points[i][1] * h);
    }
    if (closed && points.length >= 3) {
        ctx.closePath();
        ctx.fill();
    }
    ctx.stroke();
    // 绘制顶点
    ctx.fillStyle = '#22c55e';
    for (const [x, y] of points) {
        ctx.beginPath();
        ctx.arc(x * w, y * h, 4, 0, Math.PI * 2);
        ctx.fill();
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

打开浏览器访问任意页面，在 console 中调用 `boxCountModeToFields('gte', 3)`，应报错函数不存在。

- [ ] **Step 3: Write minimal implementation**

将上述函数添加到 `shared.js` 末尾。

- [ ] **Step 4: Run test to verify it passes**

浏览器 console 中验证：
```javascript
boxCountModeToFields('gte', 3)
// {min_box_count: 3, max_box_count: null, box_count_mode: 'gte'}
fieldsToBoxCountMode(3, 8, 'outside')
// {mode: 'outside', a: 3, b: 8}
```

- [ ] **Step 5: Commit**

```bash
git add frontend/safety_detection/shared.js
git commit -m "feat: add box count mode and ROI drawing utilities to shared.js"
```

---

### Task 5: 前端类型管理页面 — types.html

**Files:**
- Create: `frontend/safety_detection/types.html`
- Modify: `frontend/safety_detection/shared.js`（导航栏增加类型管理入口）

**Interfaces:**
- Consumes: `GET /detector/types`, `POST /detector/types`, `DELETE /detector/types/{dtype}`, `POST /detector/types/{dtype}/model`
- Produces: 类型管理页面

- [ ] **Step 1: Write the failing test**

前端页面测试通过浏览器手动验证。先创建空页面，验证导航入口出现。

- [ ] **Step 2: Run test to verify it fails**

访问 `/types.html`，应 404。

- [ ] **Step 3: Write minimal implementation**

创建 `types.html`，复用现有样式和布局：

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>视频诊断系统 类型管理</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Noto+Sans+SC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="/static/styles/glass-clay.css">
    <script src="/static/vue3.global.prod.js"></script>
    <script src="/static/shared.js"></script>
    <style>
        .type-cards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 16px;
            padding: 18px;
        }
        .type-card {
            padding: 16px;
            border-top: 4px solid;
        }
        .type-card-header {
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .type-card-meta {
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 4px;
        }
        .type-card-actions {
            display: flex;
            gap: 8px;
            margin-top: 12px;
        }
        .modal-overlay {
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        .modal-box {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 24px;
            width: 90%;
            max-width: 500px;
            max-height: 80vh;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <div id="app">
        <div id="sidebar-root"></div>
        <main class="main-content">
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 18px;">
                <h1 style="font-size: 20px; font-weight: 700;">检测类型管理</h1>
                <button class="clay-button primary" @click="openDialog()">新增类型</button>
            </div>
            <div class="type-cards-grid">
                <div v-for="t in types" :key="t.key" class="glass-card type-card" :style="{ borderTopColor: t.color }">
                    <div class="type-card-header" :style="{ color: t.color }">{{ t.label }}</div>
                    <div class="type-card-meta">模型: {{ t.model_path || '未设置' }}</div>
                    <div class="type-card-meta">策略: {{ t.post_process }}</div>
                    <div class="type-card-actions">
                        <button class="clay-button" @click="openDialog(t)">编辑</button>
                        <button class="clay-button" @click="deleteType(t)">删除</button>
                        <button class="clay-button" @click="uploadModel(t)">上传模型</button>
                    </div>
                </div>
            </div>
        </main>

        <!-- 新增/编辑弹窗 -->
        <div v-if="dialog" class="modal-overlay" @click.self="dialog = null">
            <div class="modal-box">
                <h3 style="margin-bottom: 16px;">{{ dialog._existing ? '编辑类型' : '新增类型' }}</h3>
                <div class="gc-form-field">
                    <label>显示名称 *</label>
                    <input class="clay-input" v-model="dialog.label" placeholder="如：明火" />
                </div>
                <div class="gc-form-field">
                    <label>颜色 *</label>
                    <input type="color" class="clay-input" v-model="dialog.color" />
                </div>
                <div class="gc-form-field">
                    <label>CPU 模型路径</label>
                    <input class="clay-input" v-model="dialog.model_path" placeholder="如：fire_smoke.pt" />
                </div>
                <div class="gc-form-field">
                    <label>后处理策略 *</label>
                    <select class="clay-select" v-model="dialog.post_process">
                        <option value="yolo_box">yolo_box</option>
                        <option value="yolo_pose">yolo_pose</option>
                    </select>
                </div>
                <details style="margin: 16px 0;">
                    <summary style="cursor: pointer; color: var(--text-muted);">高级设置</summary>
                    <div class="gc-form-field"><label>NPU 模型路径</label><input class="clay-input" v-model="dialog.npu_model_path" /></div>
                    <div class="gc-form-field"><label>类别过滤</label><input class="clay-input" v-model="dialog.classesStr" placeholder="如：0,1" /></div>
                    <div class="gc-form-field"><label>模型置信度</label><input type="number" step="0.1" class="clay-input" v-model.number="dialog.model_confidence" /></div>
                    <div class="gc-form-field"><label>图标</label><input class="clay-input" v-model="dialog.icon" /></div>
                    <div class="gc-form-field"><label>VLM 提示词键</label><input class="clay-input" v-model="dialog.vlm_prompt_key" /></div>
                    <div class="gc-form-field"><label>巡检显示名</label><input class="clay-input" v-model="dialog.inspection_label" /></div>
                </details>
                <div style="display: flex; gap: 8px; justify-content: flex-end; margin-top: 16px;">
                    <button class="clay-button" @click="dialog = null">取消</button>
                    <button class="clay-button primary" @click="saveType">保存</button>
                </div>
            </div>
        </div>

        <!-- 上传模型弹窗 -->
        <div v-if="uploadDialog" class="modal-overlay" @click.self="uploadDialog = null">
            <div class="modal-box">
                <h3 style="margin-bottom: 16px;">上传模型 — {{ uploadDialog.label }}</h3>
                <div class="gc-form-field">
                    <label>选择文件 (.pt / .rknn)</label>
                    <input type="file" accept=".pt,.rknn" @change="onFileChange" />
                </div>
                <div style="display: flex; gap: 8px; justify-content: flex-end; margin-top: 16px;">
                    <button class="clay-button" @click="uploadDialog = null">取消</button>
                    <button class="clay-button primary" @click="doUpload">上传</button>
                </div>
            </div>
        </div>

        <div v-if="toast" :class="['toast', toast.type]">{{ toast.message }}</div>
    </div>

    <script>
        const { createApp, ref, onMounted } = Vue;
        createApp({
            setup() {
                const types = ref([]);
                const dialog = ref(null);
                const uploadDialog = ref(null);
                const toast = ref(null);
                const sidebar = ref(getSidebarContext());
                let uploadFile = null;

                function showToast(msg, type = 'success') {
                    toast.value = { message: msg, type };
                    setTimeout(() => toast.value = null, 2500);
                }

                async function loadTypes() {
                    try {
                        const data = await safeFetch('/detector/types');
                        types.value = data.types || [];
                    } catch (e) { showToast('加载类型失败', 'error'); }
                }

                function openDialog(t = null) {
                    if (t) {
                        dialog.value = { ...JSON.parse(JSON.stringify(t)), _existing: true, classesStr: (t.classes || []).join(',') };
                    } else {
                        dialog.value = {
                            label: '', color: '#888888', model_path: '', npu_model_path: '',
                            post_process: 'yolo_box', classesStr: '', model_confidence: 0.5,
                            icon: '', vlm_prompt_key: '', inspection_label: '', _existing: false
                        };
                    }
                }

                async function saveType() {
                    const d = dialog.value;
                    if (!d.label || !d.color || !d.post_process) { showToast('请填写必填项', 'error'); return; }
                    const payload = {
                        label: d.label, color: d.color, model_path: d.model_path || null,
                        npu_model_path: d.npu_model_path || null, post_process: d.post_process,
                        classes: d.classesStr ? d.classesStr.split(',').map(Number) : null,
                        model_confidence: d.model_confidence, icon: d.icon || '',
                        vlm_prompt_key: d.vlm_prompt_key || '', inspection_label: d.inspection_label || d.label
                    };
                    try {
                        const url = d._existing ? `/detector/types/${d.key}` : '/detector/types';
                        const method = d._existing ? 'PUT' : 'POST';
                        const res = await fetch(url, {
                            method, headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(payload)
                        });
                        if (res.ok) {
                            showToast('保存成功');
                            dialog.value = null;
                            loadTypes();
                        } else {
                            const err = await res.json();
                            showToast(err.error || '保存失败', 'error');
                        }
                    } catch (e) { showToast('保存失败', 'error'); }
                }

                async function deleteType(t) {
                    if (!confirm(`确定删除类型 "${t.label}"？`)) return;
                    try {
                        const res = await fetch(`/detector/types/${t.key}`, { method: 'DELETE' });
                        if (res.ok) { showToast('删除成功'); loadTypes(); }
                        else {
                            const err = await res.json();
                            showToast(err.error || '删除失败', 'error');
                        }
                    } catch (e) { showToast('删除失败', 'error'); }
                }

                function uploadModel(t) { uploadDialog.value = t; uploadFile = null; }
                function onFileChange(e) { uploadFile = e.target.files[0]; }

                async function doUpload() {
                    if (!uploadFile) { showToast('请选择文件', 'error'); return; }
                    const formData = new FormData();
                    formData.append('file', uploadFile);
                    try {
                        const res = await fetch(`/detector/types/${uploadDialog.value.key}/model`, {
                            method: 'POST', body: formData
                        });
                        if (res.ok) {
                            showToast('上传成功');
                            uploadDialog.value = null;
                            loadTypes();
                        } else {
                            const err = await res.json();
                            showToast(err.error || '上传失败', 'error');
                        }
                    } catch (e) { showToast('上传失败', 'error'); }
                }

                onMounted(() => {
                    const root = document.getElementById('sidebar-root');
                    if (root) renderSidebar(root, sidebar.value);
                    loadTypes();
                });

                return { types, dialog, uploadDialog, toast, sidebar, openDialog, saveType, deleteType, uploadModel, onFileChange, doUpload };
            }
        }).mount('#app');
    </script>
</body>
</html>
```

修改 `shared.js` 导航栏，在设置组中增加"类型管理"：

```javascript
// getSidebarContext 中增加 types 页面识别
function getSidebarContext() {
    const path = window.location.pathname;
    if (path.includes('types')) {
        return { page: 'types', tab: 'cameras', settingsExpanded: true };
    }
    ...
}

// renderSidebar 中增加类型管理链接
<a href="/types.html" class="nav-item child ${context.page === 'types' ? 'active' : ''}">类型管理</a>
```

- [ ] **Step 4: Run test to verify it passes**

访问 `/types.html`，应显示类型卡片网格，可以新增、编辑、删除、上传模型。

- [ ] **Step 5: Commit**

```bash
git add frontend/safety_detection/types.html frontend/safety_detection/shared.js
git commit -m "feat: add detection type management page"
```

---

### Task 6: 前端摄像头弹窗优化 — 手风琴、人数条件、ROI 绘制

**Files:**
- Modify: `frontend/safety_detection/settings.html`

**Interfaces:**
- Consumes: `boxCountModeToFields()`, `fieldsToBoxCountMode()`, `drawRoiOnCanvas()`
- Produces: 优化后的摄像头配置弹窗

- [ ] **Step 1: Write the failing test**

打开摄像头配置弹窗，检测类型区域应显示为手风琴模式，当前是表格平铺。

- [ ] **Step 2: Run test to verify it fails**

打开设置页 → 摄像头 → 编辑任意摄像头，检测类型配置区域是表格形式。

- [ ] **Step 3: Write minimal implementation**

在 `settings.html` 中，将检测类型配置区域从表格改为手风琴列表：

```html
<div style="font-size: 13px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin: 20px 0 12px;">检测类型配置</div>
<div class="type-accordion">
    <div v-for="t in detectionTypes" :key="t.key" class="type-accordion-item">
        <div class="type-accordion-header" @click="toggleTypeExpand(t.key)">
            <span class="type-color-dot" :style="{ backgroundColor: t.color }"></span>
            <span class="type-name" :style="{ color: t.color }">{{ t.label }}</span>
            <label class="type-enable" @click.stop>
                <input type="checkbox" v-model="cameraDialog.detection_types[t.key].enabled" /> 启用
            </label>
            <span class="type-expand-icon">{{ expandedType === t.key ? '▲' : '▼' }}</span>
        </div>
        <div v-if="expandedType === t.key" class="type-accordion-body">
            <!-- 运行参数 -->
            <div class="type-param-row">
                <div class="type-param"><label>间隔</label><input type="number" class="clay-input" v-model.number="cameraDialog.detection_types[t.key].interval" /></div>
                <div class="type-param"><label>阈值</label><input type="number" step="0.1" class="clay-input" v-model.number="cameraDialog.detection_types[t.key].threshold" /></div>
                <div class="type-param"><label>连续</label><input type="number" class="clay-input" v-model.number="cameraDialog.detection_types[t.key].consecutive_required" /></div>
                <div class="type-param"><label>冷却</label><input type="number" class="clay-input" v-model.number="cameraDialog.detection_types[t.key].cooldown" /></div>
                <div class="type-param"><label>VLM</label><input type="checkbox" v-model="cameraDialog.detection_types[t.key].use_vlm" /></div>
            </div>
            <!-- 人数条件 -->
            <div class="type-param-row">
                <div class="type-param">
                    <label>人数条件</label>
                    <select class="clay-select" v-model="cameraDialog.detection_types[t.key].box_count_mode">
                        <option value="gte">目标数 ≥ a</option>
                        <option value="lte">目标数 ≤ b</option>
                        <option value="between">a ≤ 目标数 ≤ b</option>
                        <option value="outside">目标数 < a 或 > b</option>
                    </select>
                </div>
                <div class="type-param" v-if="['gte','lte'].includes(cameraDialog.detection_types[t.key].box_count_mode)">
                    <label>{{ cameraDialog.detection_types[t.key].box_count_mode === 'gte' ? 'a' : 'b' }}</label>
                    <input type="number" class="clay-input" v-model.number="cameraDialog.detection_types[t.key].box_count_a" />
                </div>
                <template v-if="['between','outside'].includes(cameraDialog.detection_types[t.key].box_count_mode)">
                    <div class="type-param"><label>a</label><input type="number" class="clay-input" v-model.number="cameraDialog.detection_types[t.key].box_count_a" /></div>
                    <div class="type-param"><label>b</label><input type="number" class="clay-input" v-model.number="cameraDialog.detection_types[t.key].box_count_b" /></div>
                </template>
            </div>
            <!-- ROI -->
            <div class="type-param-row">
                <button class="clay-button" @click="openRoiDialog(t.key)">绘制 ROI</button>
                <span v-if="cameraDialog.detection_types[t.key].roi" style="margin-left: 8px; color: var(--text-muted);">
                    已绘制 {{ cameraDialog.detection_types[t.key].roi.length }} 个顶点
                    <button class="clay-button ghost" @click="cameraDialog.detection_types[t.key].roi = null">清除</button>
                </span>
                <label style="margin-left: 8px;">
                    <input type="checkbox" v-model="cameraDialog.detection_types[t.key].roi_invert" /> 区域外报警
                </label>
            </div>
        </div>
    </div>
</div>

<!-- ROI 绘制弹窗 -->
<div v-if="roiDialog" class="modal-overlay" @click.self="roiDialog = null">
    <div class="modal-box" style="max-width: 800px;">
        <h3 style="margin-bottom: 16px;">绘制 ROI — {{ roiDialog.dtype }}</h3>
        <div style="position: relative; display: inline-block;">
            <img :src="roiDialog.snapshotUrl" style="max-width: 100%; display: block;" @load="onSnapshotLoad" />
            <canvas ref="roiCanvas" style="position: absolute; top: 0; left: 0; cursor: crosshair;" @click="onRoiClick" @dblclick="onRoiDblClick"></canvas>
        </div>
        <div style="margin-top: 16px; display: flex; gap: 8px; justify-content: flex-end;">
            <button class="clay-button" @click="roiDialog = null">取消</button>
            <button class="clay-button" @click="clearRoi">清除</button>
            <button class="clay-button primary" @click="saveRoi">保存</button>
        </div>
    </div>
</div>
```

在 Vue setup 中新增：

```javascript
const expandedType = ref(null);
const roiDialog = ref(null);
const roiCanvas = ref(null);
let roiPoints = [];

function toggleTypeExpand(key) {
    expandedType.value = expandedType.value === key ? null : key;
}

function openRoiDialog(dtype) {
    const camId = cameraDialog.value.camera_id;
    roiDialog.value = { dtype, snapshotUrl: `/cameras/${camId}/snapshot?t=${Date.now()}` };
    roiPoints = cameraDialog.value.detection_types[dtype].roi
        ? JSON.parse(JSON.stringify(cameraDialog.value.detection_types[dtype].roi))
        : [];
}

function onSnapshotLoad(e) {
    const canvas = roiCanvas.value;
    if (!canvas) return;
    canvas.width = e.target.width;
    canvas.height = e.target.height;
    drawRoiOnCanvas(canvas, roiPoints, roiPoints.length >= 3);
}

function onRoiClick(e) {
    const canvas = roiCanvas.value;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    roiPoints.push([x, y]);
    drawRoiOnCanvas(canvas, roiPoints, false);
}

function onRoiDblClick() {
    if (roiPoints.length < 3) { showToast('至少需要 3 个顶点', 'error'); return; }
    drawRoiOnCanvas(roiCanvas.value, roiPoints, true);
}

function clearRoi() { roiPoints = []; drawRoiOnCanvas(roiCanvas.value, [], false); }

function saveRoi() {
    if (roiPoints.length < 3) { showToast('至少需要 3 个顶点', 'error'); return; }
    cameraDialog.value.detection_types[roiDialog.value.dtype].roi = roiPoints;
    roiDialog.value = null;
    showToast('ROI 已保存');
}
```

在 `saveCamera` 中，提交前将人数条件字段转换为后端格式：

```javascript
async function saveCamera() {
    const d = cameraDialog.value;
    if (!d.camera_id || !d.source) { showToast('ID 和源地址必填', 'error'); return; }
    // 转换人数条件
    for (const key of Object.keys(d.detection_types)) {
        const cfg = d.detection_types[key];
        const mode = cfg.box_count_mode || 'gte';
        const fields = boxCountModeToFields(mode, cfg.box_count_a, cfg.box_count_b);
        cfg.min_box_count = fields.min_box_count;
        cfg.max_box_count = fields.max_box_count;
        cfg.box_count_mode = fields.box_count_mode;
        delete cfg.box_count_a;
        delete cfg.box_count_b;
    }
    ...
}
```

在 `openCameraDialog` 中，加载时将后端字段转换为前端格式：

```javascript
function openCameraDialog(cam = null) {
    ...
    if (cam) {
        cameraDialog.value = {
            ...JSON.parse(JSON.stringify(cam)),
            detection_types: { ...baseTypes, ...(cam.detection_types || {}) },
            _existing: true
        };
        // 转换人数条件为前端格式
        for (const key of Object.keys(cameraDialog.value.detection_types)) {
            const cfg = cameraDialog.value.detection_types[key];
            const fc = fieldsToBoxCountMode(cfg.min_box_count, cfg.max_box_count, cfg.box_count_mode);
            cfg.box_count_mode = fc.mode;
            cfg.box_count_a = fc.a;
            cfg.box_count_b = fc.b;
        }
    }
    ...
}
```

添加 CSS：

```css
.type-accordion-item { border: 1px solid var(--border); border-radius: 8px; margin-bottom: 8px; overflow: hidden; }
.type-accordion-header { display: flex; align-items: center; gap: 12px; padding: 12px; cursor: pointer; background: var(--bg-secondary); }
.type-accordion-header:hover { background: var(--bg-hover); }
.type-color-dot { width: 12px; height: 12px; border-radius: 50%; }
.type-name { font-weight: 600; flex: 1; }
.type-enable { display: flex; align-items: center; gap: 4px; font-size: 13px; }
.type-expand-icon { color: var(--text-muted); }
.type-accordion-body { padding: 16px; border-top: 1px solid var(--border); }
.type-param-row { display: flex; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; align-items: flex-end; }
.type-param { display: flex; flex-direction: column; gap: 4px; }
.type-param label { font-size: 12px; color: var(--text-muted); }
.type-param input, .type-param select { width: 80px; padding: 6px 8px; }
```

- [ ] **Step 4: Run test to verify it passes**

打开摄像头配置弹窗，验证：
- 检测类型以手风琴形式展示
- 点击展开/收起正常
- 人数条件下拉框和输入框联动正常
- ROI 绘制弹窗可以打开、绘制、保存
- 保存摄像头配置后人数条件和 ROI 正确保存

- [ ] **Step 5: Commit**

```bash
git add frontend/safety_detection/settings.html
git commit -m "feat: redesign camera dialog with accordion, box count mode, and ROI drawing"
```

---

## Self-Review

**1. Spec coverage:**
- [x] 新增 /types.html 独立页面 — Task 5
- [x] 类型增删改查 — Task 1, 2, 5
- [x] 模型上传 — Task 1, 2, 5
- [x] 摄像头弹窗手风琴 — Task 6
- [x] ROI 绘制 — Task 4, 6
- [x] 人数条件四种模式 — Task 3, 4, 6
- [x] key 自动生成 — Task 1
- [x] 删除前检查引用 — Task 2
- [x] 导航栏更新 — Task 5

**2. Placeholder scan:**
- 无 TBD、TODO
- 所有步骤都有具体代码

**3. Type consistency:**
- `box_count_mode` 在 Task 3（后端）、Task 4（前端工具）、Task 6（前端弹窗）中一致
- `add_type` / `delete_type` / `save_model` 在 Task 1 定义，Task 2 使用，一致
- `drawRoiOnCanvas` 在 Task 4 定义，Task 6 使用，一致
