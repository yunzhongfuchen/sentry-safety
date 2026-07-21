# 模型管理 + 算法管理拆分实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将"类型管理"拆分为模型管理（只管上传模型）与算法管理（基于模型创建参数版本），摄像头配置简化为选算法 + 画 ROI + 启停。

**Architecture:** 新增 `config/models.json` 模型注册表（单 `file` 字段，上传 `.pt` 自动解析元数据）；现有 `DetectionTypeRegistry` 改造为算法注册表（`config/algorithms.json`，以 `model_key` 引用模型，`registry.get()` 动态注入 `model_path` 保持推理侧零改动）；启动时自动迁移旧配置；前端新增 models.html、改造 types.html 为 algorithms.html、简化摄像头弹窗。

**Tech Stack:** Python 3.12, FastAPI, Vue 3, pytest, ultralytics（仅上传解析时懒加载）

**Spec:** `docs/superpowers/specs/2026-07-21-model-algorithm-management-design.md`

## Global Constraints

- Python 3.12，conda 环境 `py312`
- 后端测试命令：`C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest <file> -v`
- 前端无构建步骤，直接编辑 HTML/JS
- 算法/模型 key 由后端自动生成，不暴露给用户
- ROI 坐标使用归一化格式（0~1）
- 删除模型/算法前必须检查引用，有引用返回 409
- **对 spec 的三处务实偏差**（避免无意义的全链路改名，行为与 spec 一致）：
  1. spec 的 `params` → 沿用现有 `defaults` 字段名
  2. spec 的 `vlm_prompt_key` → 沿用现有内联 `vlm_prompt` 字段
  3. spec 的 `icon` → 实际代码已废弃该字段，不实现
- 附带变更：settings.html 的"检测配置"标签页是摄像头侧参数编辑入口，与"参数全归算法"冲突，一并移除

---

### Task 1: 模型注册表 ModelRegistry

**Files:**
- Create: `backend/model_registry.py`
- Test: `tests/test_model_registry.py`

**Interfaces:**
- Consumes: 无（独立模块）
- Produces:
  - `ModelRegistry` 类：`load()`、`get(key) -> dict | None`、`all_models() -> list[str]`、`to_api_list() -> list[dict]`、`add_model(file, name, post_process, class_names) -> str`、`update_model(key, updates)`、`delete_model(key)`、`save_model_file(filename, content) -> Path`、`file_exists(key) -> bool`、`resolve_file(key) -> str | None`
  - `parse_model_metadata(path: Path) -> dict`：返回 `{"post_process": str, "class_names": dict}`，失败返回 `{}`
  - 模块级单例 `model_registry`
  - 模块属性 `CONFIG_DIR`、`MODELS_FILE`、`WEIGHTS_DIR`（测试用 monkeypatch 覆盖）

- [ ] **Step 1: Write the failing test**

```python
"""模型注册表测试"""

import pytest
import backend.model_registry as mod


@pytest.fixture
def reg(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(mod, "MODELS_FILE", tmp_path / "models.json")
    monkeypatch.setattr(mod, "WEIGHTS_DIR", tmp_path / "weights")
    r = mod.ModelRegistry()
    r.load()
    return r


class TestAddModel:
    def test_add_model_generates_key_from_filename(self, reg):
        key = reg.add_model(file="fire_smoke.pt", name="火焰烟雾", post_process="yolo_box", class_names={"0": "fire", "1": "smoke"})
        assert key == "fire_smoke"
        m = reg.get(key)
        assert m["file"] == "fire_smoke.pt"
        assert m["class_names"] == {"0": "fire", "1": "smoke"}

    def test_add_model_duplicate_filename_gets_suffix(self, reg):
        reg.add_model(file="leak.pt", name="漏液", post_process="yolo_box", class_names={})
        key2 = reg.add_model(file="leak.pt", name="漏液v2", post_process="yolo_box", class_names={})
        assert key2 == "leak_1"

    def test_persisted_to_file(self, reg, tmp_path):
        reg.add_model(file="a.pt", name="A", post_process="yolo_box", class_names={})
        r2 = mod.ModelRegistry()
        r2.load()
        assert r2.get("a") is not None


class TestUpdateDelete:
    def test_update_model_metadata(self, reg):
        key = reg.add_model(file="m.rknn", name="旧名", post_process="yolo_box", class_names={})
        reg.update_model(key, {"name": "新名", "class_names": {"0": "leak"}})
        assert reg.get(key)["name"] == "新名"
        assert reg.get(key)["class_names"] == {"0": "leak"}

    def test_delete_model(self, reg):
        key = reg.add_model(file="d.pt", name="D", post_process="yolo_box", class_names={})
        reg.delete_model(key)
        assert reg.get(key) is None

    def test_delete_unknown_raises(self, reg):
        with pytest.raises(KeyError):
            reg.delete_model("nonexistent")


class TestFileOps:
    def test_save_model_file_strips_path(self, reg, tmp_path):
        p = reg.save_model_file("../evil.pt", b"data")
        assert p.name == "evil.pt"
        assert p.parent == tmp_path / "weights"

    def test_file_exists_and_resolve(self, reg, tmp_path):
        key = reg.add_model(file="x.pt", name="X", post_process="yolo_box", class_names={})
        assert reg.file_exists(key) is False
        (tmp_path / "weights").mkdir(parents=True, exist_ok=True)
        (tmp_path / "weights" / "x.pt").write_bytes(b"fake")
        assert reg.file_exists(key) is True
        assert reg.resolve_file(key).endswith("x.pt")


class TestParseMetadata:
    def test_parse_failure_returns_empty(self, reg, tmp_path):
        bad = tmp_path / "bad.pt"
        bad.write_bytes(b"not a model")
        assert mod.parse_model_metadata(bad) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest tests/test_model_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.model_registry'`

- [ ] **Step 3: Write implementation**

创建 `backend/model_registry.py`：

```python
"""
模型注册表 — 模型文件与元数据管理

全局单例 `model_registry`，启动时从 config/models.json 加载。
每个模型条目：name / file(.pt 或 .rknn) / post_process / class_names / file_size / uploaded_at
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent.parent / "config"
MODELS_FILE = CONFIG_DIR / "models.json"
WEIGHTS_DIR = Path(__file__).parent.parent / "weights"

_TASK_TO_POST_PROCESS = {"detect": "yolo_box", "pose": "yolo_pose"}


def parse_model_metadata(path: Path) -> dict:
    """解析 .pt 模型元数据（懒加载 ultralytics），失败返回 {}"""
    if path.suffix != ".pt":
        return {}
    try:
        from ultralytics import YOLO
        model = YOLO(str(path))
        post_process = _TASK_TO_POST_PROCESS.get(getattr(model, "task", None), "yolo_box")
        names = getattr(model, "names", None) or {}
        class_names = {str(k): str(v) for k, v in dict(names).items()}
        return {"post_process": post_process, "class_names": class_names}
    except Exception as e:
        logger.warning(f"Failed to parse model metadata from {path}: {e}")
        return {}


class ModelRegistry:
    """模型注册表，全局单例"""

    def __init__(self):
        self._models: dict[str, dict[str, Any]] = {}

    def load(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if MODELS_FILE.exists():
            try:
                with open(MODELS_FILE, "r", encoding="utf-8") as f:
                    self._models = json.load(f)
            except json.JSONDecodeError:
                logger.warning("models.json corrupted, starting empty")
                self._models = {}
        logger.info(f"Model registry loaded: {list(self._models.keys())}")

    def _save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODELS_FILE, "w", encoding="utf-8") as f:
            json.dump(self._models, f, ensure_ascii=False, indent=2)

    def get(self, key: str) -> dict | None:
        return self._models.get(key)

    def all_models(self) -> list[str]:
        return list(self._models.keys())

    def to_api_list(self) -> list[dict]:
        return [dict(m, key=key) for key, m in self._models.items()]

    def _unique_key(self, base: str) -> str:
        key, i = base, 0
        while key in self._models:
            i += 1
            key = f"{base}_{i}"
        return key

    def add_model(self, file: str, name: str, post_process: str,
                  class_names: dict, file_size: int | None = None) -> str:
        key = self._unique_key(Path(file).stem)
        self._models[key] = {
            "name": name or Path(file).stem,
            "file": file,
            "post_process": post_process,
            "class_names": class_names or {},
            "file_size": file_size,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()
        return key

    def update_model(self, key: str, updates: dict) -> None:
        m = self.get(key)
        if m is None:
            raise KeyError(f"Unknown model: {key}")
        for field in ("name", "post_process", "class_names"):
            if field in updates:
                m[field] = updates[field]
        self._save()

    def delete_model(self, key: str) -> None:
        if key not in self._models:
            raise KeyError(f"Unknown model: {key}")
        entry = self._models.pop(key)
        path = WEIGHTS_DIR / entry["file"]
        if path.exists():
            path.unlink()
        self._save()

    def save_model_file(self, filename: str, content: bytes) -> Path:
        """保存模型文件到 weights/（剥离目录成分防路径穿越），重名加后缀"""
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        safe = Path(filename).name
        path = WEIGHTS_DIR / safe
        stem, suffix = path.stem, path.suffix
        i = 0
        while path.exists():
            i += 1
            path = WEIGHTS_DIR / f"{stem}_{i}{suffix}"
        path.write_bytes(content)
        return path

    def file_exists(self, key: str) -> bool:
        m = self.get(key)
        return bool(m) and (WEIGHTS_DIR / m["file"]).exists()

    def resolve_file(self, key: str) -> str | None:
        m = self.get(key)
        if not m:
            return None
        path = WEIGHTS_DIR / m["file"]
        return str(path) if path.exists() else None


model_registry = ModelRegistry()
model_registry.load()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest tests/test_model_registry.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add backend/model_registry.py tests/test_model_registry.py
git commit -m "feat: add model registry with upload-time metadata parsing"
```

---

### Task 2: 算法注册表改造（detection_registry 以 model_key 引用模型）

**Files:**
- Modify: `backend/detection_registry.py`
- Test: `tests/test_detection_registry.py`（新增测试类）

**Interfaces:**
- Consumes: `backend.model_registry.model_registry`（Task 1 的 `get()`、`resolve_file()`、`file_exists()`）
- Produces（供 Task 3/4/5 及全部现有消费方使用）:
  - `ALGORITHMS_FILE` 模块属性（`config/algorithms.json`）
  - `registry.get(dtype)` 返回的 dict 新增 `model_key` 字段，并动态注入 `model_path`（= 模型 `file`，模型缺失时为 `None`）——**推理侧 `_resolve_model_path`、`ensure_models_loaded`、gpu_scheduler `model_configs`、监控/记录页零改动的前提**
  - `add_type(type_def)`：`type_def` 必须含有效 `model_key`，否则 `ValueError`；`post_process` 从模型拷贝，不再由用户传
  - `update_type(dtype, updates)`：结构性字段变为 `label/color/model_key/classes/model_confidence/vlm_prompt/inspection_label`
  - `to_api_list()` 每项新增 `model_key` 字段（`model_path` 字段保留）
  - `get_model_keys_in_use() -> set[str]`：被算法引用的 model_key 集合（供模型删除 409 检查）

- [ ] **Step 1: Write the failing test**

在 `tests/test_detection_registry.py` 追加：

```python
class TestAlgorithmModelKey:
    @pytest.fixture
    def both(self, tmp_path, monkeypatch):
        import backend.model_registry as mmod
        import backend.detection_registry as dmod
        monkeypatch.setattr(mmod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(mmod, "MODELS_FILE", tmp_path / "models.json")
        monkeypatch.setattr(mmod, "WEIGHTS_DIR", tmp_path / "weights")
        mr = mmod.ModelRegistry()
        mr.load()
        mkey = mr.add_model(file="leak.pt", name="漏液模型", post_process="yolo_box",
                            class_names={"0": "leak"})
        monkeypatch.setattr(dmod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(dmod, "ALGORITHMS_FILE", tmp_path / "algorithms.json")
        monkeypatch.setattr(dmod, "REGISTRY_FILE", tmp_path / "detection_types.json")
        monkeypatch.setattr(dmod, "model_registry", mr)
        r = dmod.DetectionTypeRegistry()
        r.load()
        return r, mr, mkey

    def test_get_injects_model_path(self, both):
        r, mr, mkey = both
        key = r.add_type({"label": "漏液", "color": "#facc15", "model_key": mkey,
                          "classes": [0], "defaults": {"threshold": 0.5}})
        td = r.get(key)
        assert td["model_key"] == mkey
        assert td["model_path"] == "leak.pt"
        assert td["post_process"] == "yolo_box"

    def test_add_type_unknown_model_key_raises(self, both):
        r, mr, mkey = both
        with pytest.raises(ValueError, match="Unknown model"):
            r.add_type({"label": "X", "color": "#fff", "model_key": "nonexistent"})

    def test_add_type_copies_post_process_from_model(self, both):
        r, mr, mkey = both
        mr.update_model(mkey, {"post_process": "yolo_pose"})
        key = r.add_type({"label": "姿态类", "color": "#fff", "model_key": mkey})
        assert r.get(key)["post_process"] == "yolo_pose"

    def test_update_type_model_key_validated(self, both):
        r, mr, mkey = both
        key = r.add_type({"label": "漏液", "color": "#facc15", "model_key": mkey})
        with pytest.raises(ValueError, match="Unknown model"):
            r.update_type(key, {"model_key": "ghost"})

    def test_to_api_list_has_model_key_and_path(self, both):
        r, mr, mkey = both
        key = r.add_type({"label": "漏液", "color": "#facc15", "model_key": mkey})
        item = [t for t in r.to_api_list() if t["key"] == key][0]
        assert item["model_key"] == mkey
        assert item["model_path"] == "leak.pt"

    def test_get_model_keys_in_use(self, both):
        r, mr, mkey = both
        assert mkey not in r.get_model_keys_in_use()
        r.add_type({"label": "漏液", "color": "#facc15", "model_key": mkey})
        assert mkey in r.get_model_keys_in_use()

    def test_model_path_none_when_model_deleted(self, both):
        r, mr, mkey = both
        key = r.add_type({"label": "漏液", "color": "#facc15", "model_key": mkey})
        mr.delete_model(mkey)
        assert r.get(key)["model_path"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest tests/test_detection_registry.py::TestAlgorithmModelKey -v`
Expected: FAIL（`ALGORITHMS_FILE` 属性不存在 / add_type 不认识 model_key）

- [ ] **Step 3: Write implementation**

修改 `backend/detection_registry.py`：

1. 文件头部新增：

```python
from backend.model_registry import model_registry

ALGORITHMS_FILE = CONFIG_DIR / "algorithms.json"
```

2. `load()` 中把 `REGISTRY_FILE` 全部替换为 `ALGORITHMS_FILE`（共 4 处：exists 判断、open、两处日志语义无变化），`_save()` 同样改用 `ALGORITHMS_FILE`。废弃字段清理块改为：

```python
        # 动态注入 model_path（由 model_key 解析），并清理已废弃字段
        for td in self._types.values():
            td.pop("icon", None)
            td.pop("vlm_prompt_key", None)
            mkey = td.get("model_key")
            model = model_registry.get(mkey) if mkey else None
            td["model_path"] = model["file"] if model else td.get("model_path")
```

3. `get()` 改为返回浅拷贝并实时注入 `model_path`（保证模型被删后能反映为 None）：

```python
    def get(self, dtype: str) -> dict | None:
        td = self._types.get(dtype)
        if td is None:
            return None
        result = dict(td)
        model = model_registry.get(td.get("model_key") or "")
        result["model_path"] = model["file"] if model else None
        return result
```

注意：`get_types_by_model`、`get_color_bgr`、`get_defaults` 内部原来调 `self.get()` 的地方改为直接读 `self._types`（避免注入开销），`get_types_by_model` 改为按 model_key 比较。**所有写方法（`update_type` / `update_defaults` / `delete_type`）也必须把 `td = self.get(dtype)` 改为 `td = self._types.get(dtype)`**——`get()` 现在返回拷贝，在拷贝上赋值不会持久化。

```python
    def get_types_by_model(self, model_path: str) -> list[str]:
        """按模型文件名找类型（推理去重用）：先反查 model_key 再匹配"""
        mkey = None
        for k, m in model_registry._models.items():
            if m.get("file") == model_path:
                mkey = k
                break
        if mkey is None:
            return []
        return [dt for dt, td in self._types.items() if td.get("model_key") == mkey]
```

并新增：

```python
    def get_model_keys_in_use(self) -> set[str]:
        return {td.get("model_key") for td in self._types.values() if td.get("model_key")}
```

4. `add_type()`：签名字段 `model_path` 改 `model_key`，校验并拷贝 post_process：

```python
    def add_type(self, type_def: dict) -> str:
        """新增算法，自动生成唯一 key，返回 key"""
        label = type_def.get("label", "").strip()
        if not label:
            raise ValueError("label is required")
        for existing in self._types.values():
            if existing.get("label") == label:
                raise ValueError(f"label '{label}' already exists")
        mkey = type_def.get("model_key")
        model = model_registry.get(mkey or "")
        if model is None:
            raise ValueError(f"Unknown model: {mkey}")
        base = label.lower().replace(" ", "_")
        key = f"{base}_{uuid.uuid4().hex[:6]}"
        while key in self._types:
            key = f"{base}_{uuid.uuid4().hex[:6]}"
        merged = copy.deepcopy(UNIVERSAL_DEFAULTS)
        merged.update(type_def.get("defaults", {}))
        self._types[key] = {
            "label": label,
            "color": type_def.get("color", "#888888"),
            "model_key": mkey,
            "post_process": model.get("post_process", "yolo_box"),
            "classes": type_def.get("classes"),
            "model_confidence": type_def.get("model_confidence", 0.5),
            "vlm_prompt": type_def.get("vlm_prompt", ""),
            "inspection_label": type_def.get("inspection_label", label),
            "defaults": merged,
        }
        self._save(self._types)
        return key
```

5. `update_type()`：字段列表中 `model_path` 换 `model_key`，去掉 `post_process`（跟随模型），并校验：

```python
        if "model_key" in updates:
            if model_registry.get(updates["model_key"] or "") is None:
                raise ValueError(f"Unknown model: {updates['model_key']}")
            td["model_key"] = updates["model_key"]
            td["post_process"] = model_registry.get(updates["model_key"]).get("post_process", "yolo_box")
        for field in ("color", "classes", "model_confidence", "vlm_prompt", "inspection_label"):
            if field in updates:
                td[field] = updates[field]
```

6. `to_api_list()` 每项加 `"model_key": td.get("model_key")`（`model_path` 字段保留，值改为注入后的解析结果）。

7. `validate()` 中模型存在性检查改为：

```python
        for dtype, td in self._types.items():
            mkey = td.get("model_key")
            if mkey and not model_registry.file_exists(mkey):
                warnings.append(f"{dtype}: model '{mkey}' file not found in weights/")
```

8. `save_model()` 方法删除（已由 `model_registry.save_model_file` 取代）。

- [ ] **Step 4: Run tests**

Run: `C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest tests/test_detection_registry.py tests/test_model_registry.py -v`
Expected: 全部通过（注意：此任务后 `tests/test_detector_types_api.py` 等旧测试可能因接口变化失败，属预期，Task 5 修）

- [ ] **Step 5: Commit**

```bash
git add backend/detection_registry.py tests/test_detection_registry.py
git commit -m "feat: algorithm registry references models via model_key"
```

---

### Task 3: 旧配置自动迁移

**Files:**
- Modify: `backend/detection_registry.py`（加 `migrate_legacy_registry()`）
- Modify: `backend/camera_manager.py`（加载处迁移 cameras.json）
- Test: `tests/test_detection_registry.py`、`tests/test_camera_manager_fields.py`（各加一个测试类）

**Interfaces:**
- Consumes: Task 1 `model_registry`、Task 2 `ALGORITHMS_FILE`
- Produces:
  - `migrate_legacy_registry() -> bool`（detection_registry 模块级函数，`registry.load()` 开头自动调用）
  - camera_manager 加载时把 `detection_types` 段改名为 `algorithms` 并只留 `enabled/roi/roi_invert`；`CameraConfig.detection_types` 属性名不变（代码内部仍用该名，仅 JSON 段名变化，加载时兼容两个段名）

- [ ] **Step 1: Write the failing test**

`tests/test_detection_registry.py` 追加：

```python
class TestLegacyMigration:
    def test_migrates_types_to_models_and_algorithms(self, tmp_path, monkeypatch):
        import backend.model_registry as mmod
        import backend.detection_registry as dmod
        monkeypatch.setattr(mmod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(mmod, "MODELS_FILE", tmp_path / "models.json")
        monkeypatch.setattr(mmod, "WEIGHTS_DIR", tmp_path / "weights")
        monkeypatch.setattr(dmod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(dmod, "ALGORITHMS_FILE", tmp_path / "algorithms.json")
        monkeypatch.setattr(dmod, "REGISTRY_FILE", tmp_path / "detection_types.json")
        legacy = {
            "fire": {"label": "明火", "color": "#ef4444", "model_path": "fire_smoke.pt",
                     "post_process": "yolo_box", "classes": [0], "model_confidence": 0.5,
                     "vlm_prompt": "p1", "inspection_label": "明火",
                     "defaults": {"threshold": 0.6}},
            "smoke": {"label": "烟雾", "color": "#f97316", "model_path": "fire_smoke.pt",
                      "post_process": "yolo_box", "classes": [1], "model_confidence": 0.5,
                      "vlm_prompt": "p2", "inspection_label": "烟雾",
                      "defaults": {"threshold": 0.55}},
        }
        (tmp_path / "detection_types.json").write_text(
            json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
        mr = mmod.ModelRegistry()
        monkeypatch.setattr(dmod, "model_registry", mr)
        r = dmod.DetectionTypeRegistry()
        r.load()
        # 模型去重：fire/smoke 共享 fire_smoke.pt → 只有一个模型条目
        assert len(mr.all_models()) == 1
        mkey = mr.all_models()[0]
        assert mr.get(mkey)["file"] == "fire_smoke.pt"
        # 算法 key 不变，model_key 指向同一模型
        assert r.get("fire")["model_key"] == mkey
        assert r.get("smoke")["model_key"] == mkey
        assert r.get("fire")["defaults"]["threshold"] == 0.6
        # 旧文件改名 .bak
        assert not (tmp_path / "detection_types.json").exists()
        assert (tmp_path / "detection_types.json.bak").exists()
```

`tests/test_camera_manager_fields.py` 追加：

```python
class TestCameraConfigMigration:
    def test_detection_types_section_migrated(self, tmp_path, monkeypatch):
        """cameras.json 的 detection_types 段改名 algorithms，参数被剔除"""
        import json
        import backend.camera_manager as cmod
        cfg_file = tmp_path / "cameras.json"
        cfg_file.write_text(json.dumps({
            "cameras": [{
                "camera_id": "cam1", "name": "测试", "source": "rtsp://x",
                "detection_types": {
                    "fire": {"enabled": True, "threshold": 0.9, "interval": 5,
                             "roi": [[0, 0], [1, 0], [1, 1]], "roi_invert": False}
                }
            }]
        }, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(cmod, "CONFIG_FILE", cfg_file)
        cm = cmod.CameraManager()
        data = json.loads(cfg_file.read_text(encoding="utf-8"))
        cam = data["cameras"][0]
        assert "detection_types" not in cam
        assert cam["algorithms"]["fire"] == {
            "enabled": True,
            "roi": [[0, 0], [1, 0], [1, 1]],
            "roi_invert": False,
        }
```

（若 `CameraManager` 的配置文件属性名不是 `CONFIG_FILE`，以其真实属性名为准，先 `grep -n "cameras.json" backend/camera_manager.py` 确认。）

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest tests/test_detection_registry.py::TestLegacyMigration tests/test_camera_manager_fields.py::TestCameraConfigMigration -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

`backend/detection_registry.py` 新增模块级函数，并在 `load()` 第一行调用：

```python
def migrate_legacy_registry() -> bool:
    """旧 detection_types.json → models.json + algorithms.json，旧文件改名 .bak"""
    if ALGORITHMS_FILE.exists() or not REGISTRY_FILE.exists():
        return False
    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            stored = json.load(f)
    except json.JSONDecodeError:
        return False
    # 1. 按 model_path 去重生成模型条目
    key_by_path: dict[str, str] = {}
    for td in stored.values():
        mp = td.get("model_path")
        if mp and mp not in key_by_path:
            key_by_path[mp] = model_registry.add_model(
                file=mp,
                name=Path(mp).stem,
                post_process=td.get("post_process", "yolo_box"),
                class_names={},
            )
    # 2. 类型 → 算法（key 不变，model_path → model_key）
    algorithms = {}
    for dtype, td in stored.items():
        algo = {k: v for k, v in td.items() if k != "model_path"}
        algo["model_key"] = key_by_path.get(td.get("model_path"))
        algorithms[dtype] = algo
    with open(ALGORITHMS_FILE, "w", encoding="utf-8") as f:
        json.dump(algorithms, f, ensure_ascii=False, indent=2)
    # 3. 旧文件改名 .bak
    REGISTRY_FILE.rename(REGISTRY_FILE.with_suffix(".json.bak"))
    logger.info(f"Migrated {len(algorithms)} types to algorithms, {len(key_by_path)} models")
    return True
```

`backend/camera_manager.py`：找到加载 cameras.json 构造 `CameraConfig` 的位置（约 875 行 `detection_types=item.get("detection_types")` 附近），在其加载函数中对每个摄像头 dict 做段名迁移（读取后、构造前，且把迁移结果写回文件一次）：

```python
def _migrate_camera_entry(item: dict) -> dict:
    """detection_types 段 → algorithms 段，只保留 enabled/roi/roi_invert"""
    legacy = item.pop("detection_types", None)
    if legacy is None:
        return item
    keep = ("enabled", "roi", "roi_invert")
    item["algorithms"] = {
        dtype: {k: v for k, v in cfg.items() if k in keep}
        for dtype, cfg in legacy.items()
    }
    return item
```

CameraConfig 构造处改为 `detection_types=item.get("algorithms")`（Python 属性名保持 `detection_types` 不变，仅 JSON 段名变化）；加载后对全部摄像头执行过迁移则写回 cameras.json 一次。

- [ ] **Step 4: Run tests**

Run: `C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest tests/test_detection_registry.py tests/test_camera_manager_fields.py -v`
Expected: 全部通过

- [ ] **Step 5: Commit**

```bash
git add backend/detection_registry.py backend/camera_manager.py tests/
git commit -m "feat: auto-migrate legacy type registry and camera configs"
```

---

### Task 4: 模型管理 API（/models）

**Files:**
- Modify: `backend/safety_detection/api.py`
- Test: `tests/test_models_api.py`（新文件）

**Interfaces:**
- Consumes: Task 1 `model_registry`、Task 2 `registry.get_model_keys_in_use()`
- Produces:
  - `GET /models` → `{"models": [{key, name, file, post_process, class_names, file_size, uploaded_at, used_by}]}`
  - `POST /models/upload`（multipart file + 可选 name）→ `{"success", "key", "post_process", "class_names"}`
  - `PUT /models/{key}`（name/post_process/class_names）→ `{"success"}`
  - `DELETE /models/{key}` → 200 或 409

- [ ] **Step 1: Write the failing test**

```python
"""模型管理 API 测试"""

import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.safety_detection.api import router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    app.state.camera_manager = MagicMock()
    return TestClient(app)


class TestListModels:
    def test_returns_models_with_usage(self, client):
        mock_mr = MagicMock()
        mock_mr.to_api_list.return_value = [
            {"key": "leak", "name": "漏液模型", "file": "leak.pt",
             "post_process": "yolo_box", "class_names": {"0": "leak"},
             "file_size": 100, "uploaded_at": "2026-07-21"}
        ]
        mock_reg = MagicMock()
        mock_reg.get_model_keys_in_use.return_value = {"leak"}
        with patch("backend.safety_detection.api.model_registry", mock_mr), \
             patch("backend.safety_detection.api.registry", mock_reg):
            resp = client.get("/models")
            assert resp.status_code == 200
            m = resp.json()["models"][0]
            assert m["key"] == "leak"
            assert m["used_by"] == 1


class TestUploadModel:
    def test_upload_pt_parses_metadata(self, client):
        mock_mr = MagicMock()
        mock_mr.save_model_file.return_value = MagicMock(name="leak.pt", stem="leak", suffix=".pt")
        mock_mr.add_model.return_value = "leak"
        with patch("backend.safety_detection.api.model_registry", mock_mr), \
             patch("backend.safety_detection.api.parse_model_metadata",
                   return_value={"post_process": "yolo_box", "class_names": {"0": "leak"}}):
            resp = client.post("/models/upload",
                               files={"file": ("leak.pt", io.BytesIO(b"fake"), "application/octet-stream")})
            assert resp.status_code == 200
            data = resp.json()
            assert data["key"] == "leak"
            assert data["class_names"] == {"0": "leak"}

    def test_upload_rejects_bad_extension(self, client):
        resp = client.post("/models/upload",
                           files={"file": ("x.exe", io.BytesIO(b"f"), "application/octet-stream")})
        assert resp.status_code == 400


class TestDeleteModel:
    def test_delete_referenced_model_returns_409(self, client):
        mock_mr = MagicMock()
        mock_mr.get.return_value = {"name": "M", "file": "m.pt"}
        mock_reg = MagicMock()
        mock_reg.get_model_keys_in_use.return_value = {"m"}
        with patch("backend.safety_detection.api.model_registry", mock_mr), \
             patch("backend.safety_detection.api.registry", mock_reg):
            resp = client.delete("/models/m")
            assert resp.status_code == 409

    def test_delete_unreferenced_succeeds(self, client):
        mock_mr = MagicMock()
        mock_mr.get.return_value = {"name": "M", "file": "m.pt"}
        mock_reg = MagicMock()
        mock_reg.get_model_keys_in_use.return_value = set()
        with patch("backend.safety_detection.api.model_registry", mock_mr), \
             patch("backend.safety_detection.api.registry", mock_reg):
            resp = client.delete("/models/m")
            assert resp.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest tests/test_models_api.py -v`
Expected: FAIL（404，端点不存在）

- [ ] **Step 3: Write implementation**

`backend/safety_detection/api.py` 顶部 import 增加：

```python
from backend.model_registry import model_registry, parse_model_metadata
```

文件末尾追加：

```python
# ---------- 模型管理 ----------

@router.get("/models")
async def list_model_entries():
    """模型列表（含被引用算法数）"""
    in_use = registry.get_model_keys_in_use()
    models = []
    for m in model_registry.to_api_list():
        m["used_by"] = 1 if m["key"] in in_use else 0
        models.append(m)
    return {"models": models}


@router.post("/models/upload")
async def upload_model_file(file: UploadFile = File(...), name: str = ""):
    """上传模型文件并创建条目，.pt 自动解析元数据"""
    filename = file.filename or ""
    if not filename.endswith((".pt", ".rknn")):
        return JSONResponse({"error": "Only .pt and .rknn files are allowed"}, status_code=400)
    content = await file.read()
    path = model_registry.save_model_file(filename, content)
    meta = parse_model_metadata(path)
    key = model_registry.add_model(
        file=path.name,
        name=name or path.stem,
        post_process=meta.get("post_process", "yolo_box"),
        class_names=meta.get("class_names", {}),
        file_size=len(content),
    )
    entry = model_registry.get(key)
    return {"success": True, "key": key,
            "post_process": entry["post_process"],
            "class_names": entry["class_names"]}


@router.put("/models/{key}")
async def update_model_entry(key: str, data: dict):
    """更新模型名称/后处理/类别清单（.rknn 手填用）"""
    if model_registry.get(key) is None:
        return JSONResponse({"error": f"Unknown model: {key}"}, status_code=404)
    updates = {k: v for k, v in data.items() if k in ("name", "post_process", "class_names")}
    if not updates:
        return JSONResponse({"error": "No valid fields to update"}, status_code=400)
    if "post_process" in updates and updates["post_process"] not in ("yolo_box", "yolo_pose"):
        return JSONResponse({"error": "post_process must be yolo_box or yolo_pose"}, status_code=400)
    model_registry.update_model(key, updates)
    return {"success": True, "key": key}


@router.delete("/models/{key}")
async def delete_model_entry(key: str):
    """删除模型（被算法引用时 409）"""
    if model_registry.get(key) is None:
        return JSONResponse({"error": f"Unknown model: {key}"}, status_code=404)
    if key in registry.get_model_keys_in_use():
        return JSONResponse({"error": f"Model '{key}' is referenced by algorithms"}, status_code=409)
    model_registry.delete_model(key)
    return {"success": True, "key": key}
```

注意：`UploadFile`、`File`、`JSONResponse` 在 api.py 已 import，无需重复。

- [ ] **Step 4: Run tests**

Run: `C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest tests/test_models_api.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add backend/safety_detection/api.py tests/test_models_api.py
git commit -m "feat: model management API (list/upload/update/delete)"
```

---

### Task 5: 算法管理 API（/algorithms）与兼容层修复

**Files:**
- Modify: `backend/safety_detection/api.py`
- Test: `tests/test_detector_types_api.py`（修复受 Task 2 影响的用例）、`tests/test_algorithms_api.py`（新文件）

**Interfaces:**
- Consumes: Task 2 的 `registry.add_type/update_type`（model_key 版）
- Produces:
  - `GET /algorithms`、`POST /algorithms`、`PUT /algorithms/{key}`、`DELETE /algorithms/{key}`（行为同 /detector/types，字段为 `model_key`）
  - `POST /detector/types/{dtype}/model` 保留但改为：保存文件 → 创建/复用模型条目 → 算法 `model_key` 指向它
  - `/detector/types` 系列接口输出格式不变（to_api_list 已注入 model_path）

- [ ] **Step 1: Write the failing test**

`tests/test_algorithms_api.py`（新文件）：

```python
"""算法管理 API 测试"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.safety_detection.api import router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    app.state.camera_manager = MagicMock()
    app.state.camera_manager.get_camera_ids_with_type.return_value = []
    return TestClient(app)


class TestCreateAlgorithm:
    def test_create_with_model_key(self, client):
        mock_reg = MagicMock()
        mock_reg.add_type.return_value = "leak_abc123"
        mock_reg.get.return_value = {
            "label": "漏液-高灵敏", "color": "#facc15", "model_key": "leak",
            "model_path": "leak.pt", "post_process": "yolo_box",
            "classes": [0], "model_confidence": 0.5, "vlm_prompt": "",
            "inspection_label": "漏液", "defaults": {"threshold": 0.4},
        }
        with patch("backend.safety_detection.api.registry", mock_reg):
            resp = client.post("/algorithms", json={
                "label": "漏液-高灵敏", "color": "#facc15", "model_key": "leak",
                "classes": [0], "defaults": {"threshold": 0.4},
            })
            assert resp.status_code == 200
            assert resp.json()["key"] == "leak_abc123"
            # add_type 收到的 payload 含 model_key，不含 model_path
            call_arg = mock_reg.add_type.call_args[0][0]
            assert call_arg["model_key"] == "leak"
            assert "model_path" not in call_arg

    def test_create_unknown_model_returns_400(self, client):
        mock_reg = MagicMock()
        mock_reg.add_type.side_effect = ValueError("Unknown model: ghost")
        with patch("backend.safety_detection.api.registry", mock_reg):
            resp = client.post("/algorithms", json={"label": "X", "model_key": "ghost"})
            assert resp.status_code == 400


class TestLegacyUploadCompat:
    def test_legacy_upload_sets_model_key(self, client):
        """旧 /detector/types/{dtype}/model 上传改为复用模型注册表"""
        import io
        mock_reg = MagicMock()
        mock_reg.get.return_value = {"label": "明火"}
        mock_mr = MagicMock()
        saved = MagicMock()
        saved.name = "fire_smoke.pt"
        saved.stem = "fire_smoke"
        saved.suffix = ".pt"
        mock_mr.save_model_file.return_value = saved
        mock_mr.add_model.return_value = "fire_smoke"
        with patch("backend.safety_detection.api.registry", mock_reg), \
             patch("backend.safety_detection.api.model_registry", mock_mr), \
             patch("backend.safety_detection.api.parse_model_metadata", return_value={}):
            resp = client.post("/detector/types/fire/model",
                               files={"file": ("fire_smoke.pt", io.BytesIO(b"f"), "application/octet-stream")})
            assert resp.status_code == 200
            mock_reg.update_type.assert_called_once()
            updates = mock_reg.update_type.call_args[0][1]
            assert updates == {"model_key": "fire_smoke"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest tests/test_algorithms_api.py -v`
Expected: FAIL（/algorithms 404；旧上传端点行为不符）

- [ ] **Step 3: Write implementation**

`backend/safety_detection/api.py` 追加 /algorithms 系列（复用现有校验逻辑）：

```python
# ---------- 算法管理 ----------

def _algo_to_response(key: str, td: dict) -> dict:
    return {
        "key": key,
        "label": td.get("label", key),
        "color": td.get("color", "#888888"),
        "model_key": td.get("model_key"),
        "model_path": td.get("model_path"),
        "post_process": td.get("post_process", "yolo_box"),
        "classes": td.get("classes"),
        "model_confidence": td.get("model_confidence", 0.5),
        "vlm_prompt": td.get("vlm_prompt", ""),
        "inspection_label": td.get("inspection_label", td.get("label", key)),
        "defaults": td.get("defaults", {}),
    }


@router.get("/algorithms")
async def list_algorithms():
    return {"algorithms": [_algo_to_response(t["key"], t) for t in registry.to_api_list()]}


@router.post("/algorithms")
async def create_algorithm(data: dict):
    data = dict(data)
    data.pop("model_path", None)  # model_path 由 model_key 解析，不接受直传
    try:
        key = registry.add_type(data)
        return _algo_to_response(key, registry.get(key))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@router.put("/algorithms/{key}")
async def update_algorithm(key: str, data: dict):
    if registry.get(key) is None:
        return JSONResponse({"error": f"Unknown algorithm: {key}"}, status_code=404)
    structural_fields = {"label", "color", "model_key", "classes", "model_confidence",
                         "vlm_prompt", "inspection_label"}
    allowed_defaults = {"enabled", "interval", "threshold", "consecutive_required",
                        "cooldown", "use_vlm", "min_box_count", "max_box_count",
                        "box_count_mode", "static_filter", "static_diff_threshold"}
    structural_update = {k: v for k, v in data.items() if k in structural_fields}
    defaults_update = {k: v for k, v in data.items() if k not in structural_fields and k in allowed_defaults}
    if not structural_update and not defaults_update:
        return JSONResponse({"error": "No valid fields to update"}, status_code=400)
    try:
        if structural_update:
            registry.update_type(key, structural_update)
        if defaults_update:
            for k, v in defaults_update.items():
                error = _validate_default_value(k, v)
                if error:
                    return JSONResponse({"error": error}, status_code=400)
            registry.update_defaults(key, defaults_update)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"success": True, "key": key, "algorithm": _algo_to_response(key, registry.get(key))}


@router.delete("/algorithms/{key}")
async def delete_algorithm(key: str, request: Request):
    if registry.get(key) is None:
        return JSONResponse({"error": f"Unknown algorithm: {key}"}, status_code=404)
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if camera_manager is not None:
        referencing = camera_manager.get_camera_ids_with_type(key)
        if referencing:
            return JSONResponse(
                {"error": f"Algorithm '{key}' is referenced by camera '{referencing[0]}'"},
                status_code=409)
    registry.delete_type(key)
    return {"success": True, "key": key}
```

改造旧上传端点 `upload_model`（替换函数体）：

```python
@router.post("/detector/types/{dtype}/model")
async def upload_model(dtype: str, file: UploadFile = File(...)):
    """上传模型文件（兼容端点）：创建/复用模型条目并把算法指过去"""
    if registry.get(dtype) is None:
        return JSONResponse({"error": f"Unknown detection type: {dtype}"}, status_code=404)
    filename = file.filename or ""
    if not filename.endswith((".pt", ".rknn")):
        return JSONResponse({"error": "Only .pt and .rknn files are allowed"}, status_code=400)
    content = await file.read()
    path = model_registry.save_model_file(filename, content)
    # 复用同文件名的已有模型条目，否则新建
    model_key = None
    for m in model_registry.to_api_list():
        if m["file"] == path.name:
            model_key = m["key"]
            break
    if model_key is None:
        meta = parse_model_metadata(path)
        model_key = model_registry.add_model(
            file=path.name, name=path.stem,
            post_process=meta.get("post_process", "yolo_box"),
            class_names=meta.get("class_names", {}),
            file_size=len(content),
        )
    registry.update_type(dtype, {"model_key": model_key})
    return {"success": True, "model_key": model_key, "dtype": dtype}
```

同时把 `PUT /detector/types/{dtype}` 的 `structural_fields` 中 `model_path` 改为 `model_key`（与 add/update_type 新签名一致），并修复 `tests/test_detector_types_api.py` 中因 Task 2 签名变化失败的用例（`model_path` 传参改 `model_key`，并 patch `backend.safety_detection.api.model_registry`）。

- [ ] **Step 4: Run tests**

Run: `C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest tests/test_algorithms_api.py tests/test_detector_types_api.py tests/test_models_api.py -v`
Expected: 全部通过

- [ ] **Step 5: Commit**

```bash
git add backend/safety_detection/api.py tests/
git commit -m "feat: algorithm management API with legacy types compat"
```

---

### Task 6: 摄像头配置写入口径统一

**Files:**
- Modify: `backend/main_multi.py`（`/cameras/{camera_id}/config` 约 1091 行、`/cameras/batch-config` 约 1150 行）
- Test: `tests/test_camera_manager_main_camera.py` 或新增 `tests/test_camera_config_api.py`

**Interfaces:**
- Consumes: Task 3 的 cameras.json `algorithms` 段
- Produces: 两个写接口接受 `algorithms`（或旧名 `detection_types`）字段，保存前剔除 `enabled/roi/roi_invert` 之外的键

- [ ] **Step 1: Write the failing test**

```python
"""摄像头配置写接口参数剔除测试"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
import backend.main_multi as mm


class TestCameraConfigSanitize:
    def test_params_stripped_on_save(self):
        """POST /cameras/{id}/config 只保留 enabled/roi/roi_invert"""
        import backend.main_multi as mm
        sanitize = mm.sanitize_camera_algorithms
        raw = {"fire": {"enabled": True, "threshold": 0.9, "interval": 5,
                        "roi": [[0, 0], [1, 0], [1, 1]], "roi_invert": False,
                        "box_count_mode": "gte"}}
        assert sanitize(raw) == {
            "fire": {"enabled": True, "roi": [[0, 0], [1, 0], [1, 1]], "roi_invert": False}
        }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest tests/test_camera_config_api.py -v`
Expected: FAIL（`sanitize_camera_algorithms` 不存在）

- [ ] **Step 3: Write implementation**

`backend/main_multi.py` 新增模块级函数：

```python
def sanitize_camera_algorithms(raw: dict) -> dict:
    """摄像头级算法配置只保留 enabled/roi/roi_invert，参数由算法注册表统一维护"""
    keep = ("enabled", "roi", "roi_invert")
    return {
        dtype: {k: v for k, v in (cfg or {}).items() if k in keep}
        for dtype, cfg in raw.items()
    }
```

在 `/cameras/{camera_id}/config` 与 `/cameras/batch-config` 两个端点中：
- 入参取 `data.get("algorithms") or data.get("detection_types") or {}`
- 保存前过一遍 `sanitize_camera_algorithms(...)`

（保存链路仍写 `CameraConfig.detection_types` 属性，序列化到 cameras.json 时键名用 `algorithms` —— 与 Task 3 的迁移保持一致；若 camera_manager 有独立的 save/serialize 函数，在那里统一处理段名。）

- [ ] **Step 4: Run tests**

Run: `C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest tests/test_camera_config_api.py tests/test_camera_manager_fields.py tests/test_camera_manager_main_camera.py -v`
Expected: 全部通过

- [ ] **Step 5: Commit**

```bash
git add backend/main_multi.py tests/test_camera_config_api.py
git commit -m "feat: strip camera-level params, keep only enabled/roi"
```

---

### Task 7: 全量后端回归

- [ ] **Step 1: Run full test suite**

Run: `C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest tests/ -v`
Expected: 全部通过。`test_inference_engine_registry.py`、`test_main_multi_registry.py`、`test_detector_core_registry.py`、`test_config_registry.py`、`test_understander_registry.py` 是重点观察对象——它们消费 `registry.get()` 的 `model_path` 注入与 `get_types_by_model`，若有失败按新结构（model_key + 注入 model_path）修正测试夹具，**不改断言语义**。

- [ ] **Step 2: Commit（如有测试修正）**

```bash
git add tests/
git commit -m "test: adapt registry consumer tests to model_key structure"
```

---

### Task 8: 模型管理页面（models.html）

**Files:**
- Create: `frontend/safety_detection/models.html`
- Modify: `frontend/safety_detection/shared.js`（导航，约 236 行）
- Modify: `backend/main_multi.py`（页面路由，约 1385 行 `/types.html` 附近）

**Interfaces:**
- Consumes: Task 4 的 `/models` API、现有 `renderSidebar`/`getSidebarContext`/`safeFetch`（shared.js）
- Produces: 页面 `/models.html`；导航项 `模型管理`（context.page === 'models'）

- [ ] **Step 1: 创建页面**

`frontend/safety_detection/models.html` 完整内容（样式与 types.html 同套玻璃黏土体系）：

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>视频诊断系统 模型管理</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Noto+Sans+SC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="/static/styles/glass-clay.css">
    <script src="/static/vue3.global.prod.js"></script>
    <script src="/static/shared.js"></script>
    <style>
        .model-cards-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; padding: 18px; }
        .model-card { padding: 16px; }
        .model-card-header { font-size: 16px; font-weight: 700; margin-bottom: 8px; }
        .model-card-meta { font-size: 12px; color: var(--text-muted); margin-bottom: 4px; }
        .model-card-classes { display: flex; flex-wrap: wrap; gap: 4px; margin: 8px 0; }
        .class-chip { font-size: 11px; padding: 2px 8px; border-radius: 10px; background: var(--bg-elevated); color: var(--text-secondary); font-family: var(--font-mono); }
        .model-card-actions { display: flex; gap: 8px; margin-top: 12px; }
        .modal-overlay { position: fixed; inset: 0; background: rgba(15,23,42,0.45); backdrop-filter: blur(6px); display: flex; align-items: center; justify-content: center; z-index: 1000; }
        .modal-box { background: var(--bg-soft); border: 1px solid var(--glass-edge); border-radius: var(--radius-lg); box-shadow: var(--glass-shadow); padding: 24px; width: 90%; max-width: 560px; max-height: 80vh; overflow-y: auto; }
        .toast { position: fixed; top: 80px; right: 24px; padding: 12px 22px; border-radius: var(--radius-md); font-size: 14px; font-weight: 500; z-index: 2000; box-shadow: var(--glass-shadow); }
        .toast.success { background: var(--success); color: white; }
        .toast.error { background: var(--danger); color: white; }
        .form-field { display: flex; flex-direction: column; gap: 4px; margin-bottom: 12px; }
        .form-field label { font-size: 12px; font-weight: 500; color: var(--text-secondary); }
        .form-field input, .form-field select { border: none; background: var(--bg-base); color: var(--text-primary); font-size: 13px; padding: 8px 10px; border-radius: var(--radius-sm); box-shadow: var(--clay-shadow-inset); outline: none; width: 100%; box-sizing: border-box; }
        .hint { font-size: 12px; color: var(--text-muted); }
    </style>
</head>
<body>
    <div id="app" class="app-shell">
        <div id="sidebar-root"></div>
        <main class="app-main">
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 18px;">
                <h1 style="font-size: 20px; font-weight: 700;">模型管理</h1>
                <button class="clay-button primary" @click="uploadDialog = true">上传模型</button>
            </div>
            <div class="model-cards-grid">
                <div v-for="m in models" :key="m.key" class="glass-card model-card">
                    <div class="model-card-header">{{ m.name }}</div>
                    <div class="model-card-meta">文件: {{ m.file }}<span v-if="m.file_size">（{{ (m.file_size / 1048576).toFixed(1) }} MB）</span></div>
                    <div class="model-card-meta">策略: {{ m.post_process }} · 被 {{ m.used_by }} 个算法引用</div>
                    <div class="model-card-classes">
                        <span v-for="(name, id) in m.class_names" :key="id" class="class-chip">{{ id }}:{{ name }}</span>
                        <span v-if="!Object.keys(m.class_names || {}).length" class="hint">未解析到类别</span>
                    </div>
                    <div class="model-card-actions">
                        <button class="clay-button" @click="openEdit(m)">编辑</button>
                        <button class="clay-button" @click="deleteModel(m)">删除</button>
                    </div>
                </div>
            </div>
        </main>

        <!-- 上传弹窗 -->
        <div v-if="uploadDialog" class="modal-overlay" @click.self="uploadDialog = false">
            <div class="modal-box">
                <h3 style="margin-bottom: 16px;">上传模型</h3>
                <div class="form-field"><label>模型名称（默认取文件名）</label><input v-model="uploadName" placeholder="如：漏液检测模型" /></div>
                <div class="form-field">
                    <label>选择文件 (.pt / .rknn)</label>
                    <input type="file" accept=".pt,.rknn" @change="e => uploadFile = e.target.files[0]" />
                    <span class="hint">.pt 上传后自动解析类别清单，可能需要几秒钟</span>
                </div>
                <div style="display: flex; gap: 8px; justify-content: flex-end; margin-top: 16px;">
                    <button class="clay-button" @click="uploadDialog = false">取消</button>
                    <button class="clay-button primary" :disabled="uploading" @click="doUpload">{{ uploading ? '上传解析中…' : '上传' }}</button>
                </div>
            </div>
        </div>

        <!-- 编辑弹窗（主要用于 .rknn 手填元数据） -->
        <div v-if="editDialog" class="modal-overlay" @click.self="editDialog = null">
            <div class="modal-box">
                <h3 style="margin-bottom: 16px;">编辑模型 — {{ editDialog.name }}</h3>
                <div class="form-field"><label>名称</label><input v-model="editDialog.name" /></div>
                <div class="form-field">
                    <label>后处理策略</label>
                    <select v-model="editDialog.post_process">
                        <option value="yolo_box">yolo_box</option>
                        <option value="yolo_pose">yolo_pose</option>
                    </select>
                </div>
                <div class="form-field">
                    <label>类别清单（每行一条，格式 id:名称）</label>
                    <textarea v-model="editDialog.classesText" rows="5" style="border:none;background:var(--bg-base);color:var(--text-primary);font-size:13px;padding:8px 10px;border-radius:var(--radius-sm);box-shadow:var(--clay-shadow-inset);outline:none;width:100%;box-sizing:border-box;resize:vertical;" placeholder="0:fire&#10;1:smoke"></textarea>
                </div>
                <div style="display: flex; gap: 8px; justify-content: flex-end;">
                    <button class="clay-button" @click="editDialog = null">取消</button>
                    <button class="clay-button primary" @click="saveEdit">保存</button>
                </div>
            </div>
        </div>

        <div v-if="toast" :class="['toast', toast.type]">{{ toast.message }}</div>
    </div>

    <script>
        const { createApp, ref, onMounted } = Vue;
        createApp({
            setup() {
                const models = ref([]);
                const uploadDialog = ref(false);
                const uploadName = ref('');
                const uploading = ref(false);
                const editDialog = ref(null);
                const toast = ref(null);
                const sidebar = ref(getSidebarContext());
                let uploadFile = null;

                function showToast(msg, type = 'success') {
                    toast.value = { message: msg, type };
                    setTimeout(() => toast.value = null, 2500);
                }

                async function loadModels() {
                    try {
                        const data = await safeFetch('/models');
                        models.value = data.models || [];
                    } catch (e) { showToast('加载模型失败', 'error'); }
                }

                async function doUpload() {
                    if (!uploadFile) { showToast('请选择文件', 'error'); return; }
                    uploading.value = true;
                    const formData = new FormData();
                    formData.append('file', uploadFile);
                    try {
                        const res = await fetch(`/models/upload?name=${encodeURIComponent(uploadName.value)}`, {
                            method: 'POST', body: formData
                        });
                        if (res.ok) {
                            showToast('上传成功');
                            uploadDialog.value = false;
                            uploadName.value = '';
                            uploadFile = null;
                            loadModels();
                        } else {
                            const err = await res.json();
                            showToast(err.error || '上传失败', 'error');
                        }
                    } catch (e) { showToast('上传失败', 'error'); }
                    finally { uploading.value = false; }
                }

                function openEdit(m) {
                    const classesText = Object.entries(m.class_names || {})
                        .map(([id, name]) => `${id}:${name}`).join('\n');
                    editDialog.value = { key: m.key, name: m.name, post_process: m.post_process, classesText };
                }

                async function saveEdit() {
                    const d = editDialog.value;
                    const class_names = {};
                    d.classesText.split('\n').forEach(line => {
                        const m = line.trim().match(/^(\d+)[:：](.+)$/);
                        if (m) class_names[m[1]] = m[2].trim();
                    });
                    try {
                        const res = await fetch(`/models/${d.key}`, {
                            method: 'PUT', headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ name: d.name, post_process: d.post_process, class_names })
                        });
                        if (res.ok) { showToast('保存成功'); editDialog.value = null; loadModels(); }
                        else { const err = await res.json(); showToast(err.error || '保存失败', 'error'); }
                    } catch (e) { showToast('保存失败', 'error'); }
                }

                async function deleteModel(m) {
                    if (!confirm(`确定删除模型 "${m.name}"？模型文件将一并删除。`)) return;
                    try {
                        const res = await fetch(`/models/${m.key}`, { method: 'DELETE' });
                        if (res.ok) { showToast('删除成功'); loadModels(); }
                        else { const err = await res.json(); showToast(err.error || '删除失败', 'error'); }
                    } catch (e) { showToast('删除失败', 'error'); }
                }

                onMounted(() => {
                    const root = document.getElementById('sidebar-root');
                    if (root) renderSidebar(root, sidebar.value);
                    loadModels();
                });

                return { models, uploadDialog, uploadName, uploading, editDialog, toast, sidebar, doUpload, openEdit, saveEdit, deleteModel };
            }
        }).mount('#app');
    </script>
</body>
</html>
```

- [ ] **Step 2: 导航与路由**

`frontend/safety_detection/shared.js` 约 236 行，把：

```js
<a href="/types.html" class="nav-item child ${context.page === 'types' ? 'active' : ''}">类型管理</a>
```

替换为：

```js
<a href="/models.html" class="nav-item child ${context.page === 'models' ? 'active' : ''}">模型管理</a>
<a href="/algorithms.html" class="nav-item child ${context.page === 'algorithms' ? 'active' : ''}">算法管理</a>
```

`backend/main_multi.py` 在 `/types.html` 路由（约 1385 行）附近加：

```python
@app.get("/models.html")
async def models_page():
    return FileResponse(FRONTEND_DIR / "models.html")
```

（`FileResponse`/`FRONTEND_DIR` 以该文件现有写法为准，照抄 `/types.html` 路由的实现。）

models.html 中 `getSidebarContext()` 调用前设置 page：在 setup 开头 `sidebar.value.page = 'models'`（参照 types.html 现有 sidebar 用法，如 types.html 未显式设置则在 shared.js 的 `getSidebarContext` 里按路径自动识别，按其真实机制处理）。

- [ ] **Step 3: 浏览器验证**

重启服务后访问 `http://localhost:8000/models.html`：页面打开、侧边导航"模型管理"高亮、上传一个 `.pt` 测试模型后卡片显示类别 chips。

- [ ] **Step 4: Commit**

```bash
git add frontend/safety_detection/models.html frontend/safety_detection/shared.js backend/main_multi.py
git commit -m "feat: model management page with upload and metadata editing"
```

---

### Task 9: 算法管理页面（types.html → algorithms.html）

**Files:**
- Create: `frontend/safety_detection/algorithms.html`（由 types.html 改造）
- Delete: `frontend/safety_detection/types.html`（git rm）
- Modify: `backend/main_multi.py`（`/types.html` 路由改为重定向到 `/algorithms.html`，新增 `/algorithms.html` 路由）

**Interfaces:**
- Consumes: Task 5 `/algorithms` API、Task 4 `/models` API（模型下拉 + 类别勾选数据源）
- Produces: 页面 `/algorithms.html`

- [ ] **Step 1: 创建 algorithms.html**

复制 `types.html` 为 `algorithms.html`，做以下精确修改：

1. 标题："类型管理" → "算法管理"；`新增类型` 按钮文案 → `新增算法`
2. 卡片区域（原 163-172 行）替换为：

```html
<div v-for="t in types" :key="t.key" class="glass-card type-card" :style="{ borderTopColor: t.color }">
    <div class="type-card-header" :style="{ color: t.color }">{{ t.label }}</div>
    <div class="type-card-meta">模型: {{ modelName(t.model_key) }}</div>
    <div class="type-card-meta">策略: {{ t.post_process }} · 阈值 {{ t.defaults?.threshold }} · 间隔 {{ t.defaults?.interval }}s · 冷却 {{ t.defaults?.cooldown }}s</div>
    <div class="type-card-actions">
        <button class="clay-button" @click="openDialog(t)">编辑</button>
        <button class="clay-button" @click="deleteType(t)">删除</button>
    </div>
</div>
```

（删除"上传模型"按钮。）

3. 弹窗"基本信息"区（原 184-195 行）中，把"模型路径"输入框和"后处理策略"下拉两个 field 替换为：

```html
<div class="type-card-field">
    <label>模型 *</label>
    <select v-model="dialog.model_key" @change="onModelChange">
        <option value="" disabled>请选择模型</option>
        <option v-for="m in models" :key="m.key" :value="m.key">{{ m.name }}（{{ m.file }}）</option>
    </select>
</div>
```

4. "高级参数"区中"类别过滤"文本输入（`classesStr`）替换为勾选组：

```html
<div class="type-card-field">
    <label>类别过滤（不勾 = 不过滤）</label>
    <div class="type-card-checkboxes" style="flex-wrap: wrap;">
        <label v-for="c in availableClasses" :key="c.id" class="type-card-checkbox">
            <input type="checkbox" :value="c.id" v-model="dialog.classesChecked" /> {{ c.id }}:{{ c.name }}
        </label>
        <span v-if="!availableClasses.length" style="font-size: 12px; color: var(--text-muted);">该模型无类别清单，保存后不过滤</span>
    </div>
</div>
```

5. script 部分修改：
   - `setup()` 中新增：`const models = ref([]);`、`const availableClasses = ref([]);`
   - `loadTypes()` 的 `safeFetch('/detector/types')` 改为 `safeFetch('/algorithms')`，取值 `data.algorithms`；新增：

```js
async function loadModels() {
    try {
        const data = await safeFetch('/models');
        models.value = data.models || [];
    } catch (e) { /* 模型列表失败不阻塞页面 */ }
}

function modelName(key) {
    const m = models.value.find(x => x.key === key);
    return m ? m.name : (key || '未设置');
}

function onModelChange() {
    const m = models.value.find(x => x.key === dialog.value.model_key);
    availableClasses.value = m
        ? Object.entries(m.class_names || {}).map(([id, name]) => ({ id: Number(id), name }))
        : [];
    if (dialog.value) dialog.value.classesChecked = [];
}
```

   - `openDialog(t)`：`copy.classesStr = ...` 一行改为 `copy.classesChecked = (t.classes || []).slice();`，新增分支里 `classesStr: ''` 改为 `classesChecked: []`、`model_key: ''`（替换原 `model_path: ''`）；编辑分支调用 `onModelChange()` 前先把 `availableClasses` 按当前模型填充（直接复用 onModelChange 但保留已勾选项：先存 checked 再调用后恢复）。
   - `saveType()`：payload 构造改为：

```js
const payload = {
    label: d.label, color: d.color, model_key: d.model_key,
    classes: d.classesChecked && d.classesChecked.length ? d.classesChecked.map(Number) : null,
    model_confidence: d.model_confidence,
    vlm_prompt: d.vlm_prompt || '', inspection_label: d.inspection_label || d.label
};
```

   校验改为 `if (!d.label || !d.color || !d.model_key)`；url 改为 `/algorithms` 与 `/algorithms/${d.key}`。
   - 删除 `uploadDialog`、`uploadModel`、`onFileChange`、`doUpload` 及上传弹窗模板（原 255-268 行）与 `fileInput`。
   - `onMounted` 中 `loadTypes()` 后加 `loadModels()`；return 中去掉 upload 相关，加上 `models, availableClasses, modelName, onModelChange`。

6. `backend/main_multi.py`：`/types.html` 路由改为 `RedirectResponse('/algorithms.html')`，新增 `/algorithms.html` 的 `FileResponse` 路由。

- [ ] **Step 2: 浏览器验证**

访问 `/algorithms.html`：算法卡片显示模型名与参数摘要；新增算法弹窗中选模型后类别变为勾选项；保存后卡片出现。访问 `/types.html` 自动跳转 `/algorithms.html`。

- [ ] **Step 3: Commit**

```bash
git add frontend/safety_detection/algorithms.html backend/main_multi.py
git rm frontend/safety_detection/types.html
git commit -m "feat: algorithm management page replaces type management"
```

---

### Task 10: settings.html 摄像头弹窗简化 + 移除检测配置标签页

**Files:**
- Modify: `frontend/safety_detection/settings.html`

**Interfaces:**
- Consumes: 现有 `cameraDialog.detection_types`（现在只含 enabled/roi/roi_invert）
- Produces: 弹窗中每个算法只显示 `[启用开关] [展开▼]`，展开只含 ROI 区

- [ ] **Step 1: 简化类型配置弹窗**

`settings.html` 约 395-437 行（`typeConfigDialog` 展开区）：
- **删除**：间隔/阈值/连续/冷却四个 `type-param`（395-398 行）、人数条件下拉及 a/b 输入（404-418 行）、`static_filter` 勾选（约 431 行）
- **保留**：ROI 区（绘制按钮 / 顶点数 / 清除 / 区域外报警勾选，约 420-427 行）

保存逻辑（约 712-721 行）中 box_count 字段转换代码（`boxCountModeToFields`、删除 box_count_a/b 等）整段删除，改为在提交前净化：

```js
const keep = ['enabled', 'roi', 'roi_invert'];
for (const key of Object.keys(d.detection_types)) {
    const cfg = d.detection_types[key];
    d.detection_types[key] = Object.fromEntries(
        Object.entries(cfg).filter(([k]) => keep.includes(k)));
}
```

弹窗初始化（约 667-672、699-704 行）中的 `fieldsToBoxCountMode` 转换块同样删除。

- [ ] **Step 2: 移除"检测配置"标签页**

- 删除标签页导航项与整个 `default_detection_types` 设置面板（约 216-246 行区域）
- 删除批量配置弹窗（batchDialog，约 475-495 行区域）及其相关函数；摄像头列表的"批量配置"入口按钮一并删除
- 摄像头弹窗初始化中 `settings.value.default_detection_types` 种子来源（约 657-663、695-696 行）改为直接用 `defaultDetectionTypes()`（shared.js 提供，来自 `/detector/types` 兼容接口），并对每个 key 只保留 `{ enabled, roi: null, roi_invert: false }` 结构

- [ ] **Step 3: 浏览器验证**

打开摄像头编辑弹窗：每个算法只有启停开关和 ROI 绘制；保存后 `cameras.json` 中该摄像头 `algorithms` 段只含 enabled/roi/roi_invert。设置页只剩"摄像头"和"系统设置"两个标签。

- [ ] **Step 4: Commit**

```bash
git add frontend/safety_detection/settings.html
git commit -m "feat: simplify camera dialog to enable toggle + ROI only"
```

---

### Task 11: 端到端冒烟验证

- [ ] **Step 1: 迁移验证**

备份 `config/cameras.json`、`config/detection_types.json` 后重启服务：
- `config/models.json`、`config/algorithms.json` 自动生成，`detection_types.json.bak` 存在
- `cameras.json` 中无 `detection_types` 段，ROI 保留
- 日志出现 `Migrated N types to algorithms, M models`

- [ ] **Step 2: 页面流程验证（Playwright 或手动）**

完整走一遍：模型管理上传模型 → 算法管理新建算法（选模型、勾类别、设参数）→ 摄像头弹窗选算法 + 画 ROI → 监控页出现该算法开关且画面检测正常 → 产生一条告警，告警卡片显示算法 label。

- [ ] **Step 3: 全量测试最终回归**

Run: `C:/Users/12800/miniconda3/envs/py312/python.exe -m pytest tests/ -v`
Expected: 全部通过

- [ ] **Step 4: Commit（如有遗留修正）**

```bash
git add -A
git commit -m "chore: e2e fixes for model/algorithm split"
```
