# 检测类型注册表框架 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将视频诊断系统的 6 种检测类型从硬编码改造为 JSON 配置驱动的注册表框架，新增检测类型只需编辑配置 + 放模型文件，不改代码。

**Architecture:** 新增 `backend/detection_registry.py` 作为全局单例注册表，启动时从 `config/detection_types.json` 加载类型定义（首次自动生成）。推理引擎、检测核心、GPU 调度器、VLM 理解器、前端均从注册表读取类型信息，消除所有硬编码的 if/elif 分发链。后处理策略模式（`yolo_box` / `yolo_pose`）解耦推理与后处理。

**Tech Stack:** Python 3.12, FastAPI, ultralytics YOLO, OpenCV, Vue.js (CDN), pytest

**设计文档:** `docs/superpowers/specs/2026-07-13-detection-type-registry-design.md`

## Global Constraints

- Python 3.12, conda 环境 `py312`
- 不新增任何第三方依赖
- 所有现有 API 端点保持兼容，新端点为纯新增
- `cameras.json` 格式向后兼容，新字段（`roi`、`roi_invert`、`min_box_count`、`max_box_count`）均可选
- 注册表首次启动自动从内置默认值生成 `config/detection_types.json`，已有文件以文件为准并自动补全缺失字段
- hex 颜色字段在后端自动转 BGR tuple
- 前端 `DETECTION_TYPES` 改为动态加载，API 调不通时 fallback 到内置 6 个默认类型
- 每个 Task 的测试必须先写、先跑失败、再实现、再跑通过

---

## File Structure

### 新增文件

| 文件                                             | 职责                                                               |
| ------------------------------------------------ | ------------------------------------------------------------------ |
| `backend/detection_registry.py`                | `DetectionTypeRegistry` 单例类 + 后处理策略函数 + 默认注册表数据 |
| `tests/test_detection_registry.py`             | 注册表单元测试                                                     |
| `tests/test_detection_registry_integration.py` | 注册表与 config/inference_engine 集成测试                          |

### 改造文件

| 文件                                          | 改动范围                                                                                                                                                                                       |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend/inference_engine.py`               | 删除`MODEL_PATHS`、6 个 `_load_xxx_model`、6 个 `_detect_xxx`；`ensure_models_loaded` 和 `detect` 改为注册表驱动；新增 `_run_model`、`_process_yolo_box`、`_process_yolo_pose` |
| `backend/config.py`                         | `DEFAULT_TYPE_CONFIG` 和 `DEFAULT_GLOBAL_SETTINGS.display_detection_types` 从注册表动态生成；`apply_camera_globals` 迭代注册表而非硬编码 key                                             |
| `backend/safety_detection/detector_core.py` | `_annotate_frame` 颜色/标签从注册表读取；`_get_due_types` 前置冷却检查；`_handle_standard_detection` 新增 ROI 过滤和 box_count 判断                                                      |
| `backend/main_multi.py`                     | GPU scheduler`model_configs` 从注册表生成；`_resolve_model_path` 调用改为注册表                                                                                                            |
| `backend/understander.py`                   | `_build_inspection_prompt` 的 `type_desc` 从注册表 `inspection_label` 读取                                                                                                               |
| `backend/safety_detection/api.py`           | 新增`GET /detector/types`、`GET /detector/types/{dtype}`、`PUT /detector/types/{dtype}`                                                                                                  |
| `frontend/safety_detection/shared.js`       | `DETECTION_TYPES` 改为动态加载 + fallback；`defaultDetectionTypes()` 从 API 数据生成                                                                                                       |
| `frontend/safety_detection/monitor.html`    | 类型开关从动态`DETECTION_TYPES` 渲染（现有逻辑已是 `v-for`，无需改模板）                                                                                                                   |
| `frontend/safety_detection/settings.html`   | 同上，已是`v-for`，模板不变；JS 初始化改为 await 动态加载                                                                                                                                    |
| `frontend/safety_detection/records.html`    | 同上                                                                                                                                                                                           |

---

### Task 1: 检测类型注册表核心模块

**Files:**

- Create: `backend/detection_registry.py`
- Create: `tests/test_detection_registry.py`

**Interfaces:**

- Consumes: 无外部依赖（纯数据模块）
- Produces:
  - `registry` — 全局单例 `DetectionTypeRegistry` 实例
  - `registry.load()` → `None`（加载/重载注册表）
  - `registry.get(dtype: str)` → `dict | None`（类型定义，未注册返回 None）
  - `registry.all_types()` → `list[str]`
  - `registry.get_types_by_model(model_path: str)` → `list[str]`
  - `registry.get_color_bgr(dtype: str)` → `tuple[int, int, int]`
  - `registry.get_defaults(dtype: str)` → `dict`
  - `registry.merge_camera_config(dtype: str, overrides: dict)` → `dict`
  - `registry.validate()` → `list[str]`（警告消息列表）
  - `registry.to_api_list()` → `list[dict]`（前端 API 返回格式）
  - `DEFAULT_DETECTION_TYPE_REGISTRY` — 内置默认注册表 dict
  - `hex_to_bgr(hex_color: str)` → `tuple[int, int, int]`

- [ ] **Step 1: Write failing tests for `hex_to_bgr`**

```python
# tests/test_detection_registry.py
import pytest


def test_hex_to_bgr_standard():
    from backend.detection_registry import hex_to_bgr
    assert hex_to_bgr("#ef4444") == (68, 68, 239)


def test_hex_to_bgr_without_hash():
    from backend.detection_registry import hex_to_bgr
    assert hex_to_bgr("ef4444") == (68, 68, 239)


def test_hex_to_bgr_white():
    from backend.detection_registry import hex_to_bgr
    assert hex_to_bgr("#ffffff") == (255, 255, 255)


def test_hex_to_bgr_black():
    from backend.detection_registry import hex_to_bgr
    assert hex_to_bgr("#000000") == (0, 0, 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_detection_registry.py::test_hex_to_bgr_standard -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create `backend/detection_registry.py` with `hex_to_bgr` + `DEFAULT_DETECTION_TYPE_REGISTRY` + `DetectionTypeRegistry` class**

```python
# backend/detection_registry.py
"""
检测类型注册表 — 配置驱动的类型定义管理

全局单例 `registry`，启动时从 config/detection_types.json 加载。
首次启动自动从 DEFAULT_DETECTION_TYPE_REGISTRY 生成配置文件。
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent.parent / "config"
REGISTRY_FILE = CONFIG_DIR / "detection_types.json"


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """'#ef4444' → (68, 68, 239)"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


DEFAULT_DETECTION_TYPE_REGISTRY: dict[str, dict[str, Any]] = {
    "fire": {
        "label": "明火",
        "color": "#ef4444",
        "icon": "flame",
        "model_path": "fire_smoke.pt",
        "npu_model_path": "fire_smoke.rknn",
        "post_process": "yolo_box",
        "classes": [0],
        "model_confidence": 0.5,
        "vlm_prompt_key": "fire_review",
        "inspection_label": "明火",
        "defaults": {
            "enabled": False,
            "interval": 1,
            "threshold": 0.6,
            "consecutive_required": 3,
            "cooldown": 60,
            "use_vlm": False,
            "min_box_count": 1,
            "max_box_count": None,
        },
    },
    "smoke": {
        "label": "烟雾",
        "color": "#f97316",
        "icon": "cloud",
        "model_path": "fire_smoke.pt",
        "npu_model_path": "fire_smoke.rknn",
        "post_process": "yolo_box",
        "classes": [1],
        "model_confidence": 0.5,
        "vlm_prompt_key": "smoke_review",
        "inspection_label": "烟雾",
        "defaults": {
            "enabled": False,
            "interval": 1,
            "threshold": 0.55,
            "consecutive_required": 3,
            "cooldown": 60,
            "use_vlm": False,
            "min_box_count": 1,
            "max_box_count": None,
        },
    },
    "uniform": {
        "label": "工服",
        "color": "#22c55e",
        "icon": "shirt",
        "model_path": "uniform.pt",
        "npu_model_path": "uniform.rknn",
        "post_process": "yolo_box",
        "classes": [1],
        "model_confidence": 0.5,
        "vlm_prompt_key": "uniform_review",
        "inspection_label": "未穿工服",
        "defaults": {
            "enabled": False,
            "interval": 1,
            "threshold": 0.5,
            "consecutive_required": 3,
            "cooldown": 60,
            "use_vlm": False,
            "min_box_count": 1,
            "max_box_count": None,
        },
    },
    "mask": {
        "label": "口罩",
        "color": "#0ea5e9",
        "icon": "shield",
        "model_path": "mask.pt",
        "npu_model_path": "mask.rknn",
        "post_process": "yolo_box",
        "classes": [1],
        "model_confidence": 0.5,
        "vlm_prompt_key": "mask_review",
        "inspection_label": "未戴口罩",
        "defaults": {
            "enabled": False,
            "interval": 1,
            "threshold": 0.5,
            "consecutive_required": 3,
            "cooldown": 60,
            "use_vlm": False,
            "min_box_count": 1,
            "max_box_count": None,
        },
    },
    "cigarette": {
        "label": "吸烟",
        "color": "#a855f7",
        "icon": "cigarette",
        "model_path": "cigarette.pt",
        "npu_model_path": "cigarette.rknn",
        "post_process": "yolo_box",
        "classes": [0],
        "model_confidence": 0.5,
        "vlm_prompt_key": "cigarette_review",
        "inspection_label": "吸烟",
        "defaults": {
            "enabled": False,
            "interval": 1,
            "threshold": 0.5,
            "consecutive_required": 3,
            "cooldown": 60,
            "use_vlm": False,
            "min_box_count": 1,
            "max_box_count": None,
        },
    },
    "sleep": {
        "label": "睡岗",
        "color": "#eab308",
        "icon": "moon",
        "model_path": "yolov8n-pose.pt",
        "npu_model_path": None,
        "post_process": "yolo_pose",
        "classes": None,
        "model_confidence": 0.1,
        "vlm_prompt_key": "sleep_review",
        "inspection_label": "睡岗/打盹",
        "defaults": {
            "enabled": False,
            "interval": 60,
            "threshold": 0.7,
            "consecutive_required": 3,
            "cooldown": 60,
            "use_vlm": False,
            "min_box_count": 1,
            "max_box_count": None,
        },
    },
}


class DetectionTypeRegistry:
    """检测类型注册表，全局单例"""

    def __init__(self):
        self._types: dict[str, dict[str, Any]] = {}

    def load(self) -> None:
        """加载注册表：文件存在则读取并补全缺失字段，不存在则从默认值生成"""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        if REGISTRY_FILE.exists():
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                stored = json.load(f)
            merged = json.loads(json.dumps(DEFAULT_DETECTION_TYPE_REGISTRY))
            for dtype, type_def in stored.items():
                if dtype in merged:
                    merged[dtype].update(type_def)
                    for key, val in DEFAULT_DETECTION_TYPE_REGISTRY[dtype]["defaults"].items():
                        merged[dtype]["defaults"].setdefault(key, val)
                else:
                    merged[dtype] = type_def
            if merged != stored:
                self._save(merged)
            self._types = merged
        else:
            self._types = json.loads(json.dumps(DEFAULT_DETECTION_TYPE_REGISTRY))
            self._save(self._types)

        logger.info(f"Detection registry loaded: {list(self._types.keys())}")

    def _save(self, data: dict) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get(self, dtype: str) -> dict:
        if dtype not in self._types:
            return None
        return self._types[dtype]

    def all_types(self) -> list[str]:
        return list(self._types.keys())

    def get_types_by_model(self, model_path: str) -> list[str]:
        return [dt for dt, td in self._types.items() if td["model_path"] == model_path]

    def get_color_bgr(self, dtype: str) -> tuple[int, int, int]:
        return hex_to_bgr(self.get(dtype)["color"])

    def get_defaults(self, dtype: str) -> dict:
        return dict(self.get(dtype)["defaults"])

    def merge_camera_config(self, dtype: str, overrides: dict) -> dict:
        """合并摄像头级覆盖到注册表默认值"""
        result = self.get_defaults(dtype)
        for key, val in overrides.items():
            if key in result or key in ("roi", "roi_invert"):
                result[key] = val
        return result

    def validate(self) -> list[str]:
        """校验注册表（模型文件是否存在等），返回警告列表"""
        warnings = []
        models_dir = Path(__file__).parent.parent / "models"
        weights_dir = Path(__file__).parent.parent / "weights"
        for dtype, td in self._types.items():
            mp = td.get("model_path")
            if mp:
                found = (models_dir / mp).exists() or (weights_dir / mp).exists()
                if not found:
                    warnings.append(f"{dtype}: model file '{mp}' not found in models/ or weights/")
            if td.get("post_process") not in ("yolo_box", "yolo_pose"):
                warnings.append(f"{dtype}: unknown post_process '{td.get('post_process')}'")
        return warnings

    def to_api_list(self) -> list[dict]:
        """返回前端 API 格式的类型列表"""
        result = []
        for key, td in self._types.items():
            result.append({
                "key": key,
                "label": td["label"],
                "color": td["color"],
                "icon": td.get("icon", ""),
                "post_process": td["post_process"],
                "defaults": dict(td["defaults"]),
            })
        return result

    def update_defaults(self, dtype: str, new_defaults: dict) -> None:
        """更新类型的运行参数默认值并持久化"""
        td = self.get(dtype)
        allowed = set(DEFAULT_DETECTION_TYPE_REGISTRY.get(dtype, {}).get("defaults", {}).keys())
        if not allowed:
            allowed = set(td["defaults"].keys())
        for key, val in new_defaults.items():
            if key in allowed:
                td["defaults"][key] = val
        self._save(self._types)


registry = DetectionTypeRegistry()
```

- [ ] **Step 4: Run `hex_to_bgr` tests to verify they pass**

Run: `python -m pytest tests/test_detection_registry.py -k "hex_to_bgr" -v`
Expected: 4 PASSED

- [ ] **Step 5: Write failing tests for registry core methods**

Append to `tests/test_detection_registry.py`:

```python
import json


class TestDetectionTypeRegistry:
    """注册表核心功能测试"""

    def _make_registry(self, tmp_path, monkeypatch, data=None):
        import backend.detection_registry as mod
        monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(mod, "REGISTRY_FILE", tmp_path / "detection_types.json")
        r = mod.DetectionTypeRegistry()
        if data is not None:
            (tmp_path / "detection_types.json").write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
        r.load()
        return r

    def test_load_generates_file_when_missing(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        assert (tmp_path / "detection_types.json").exists()
        assert "fire" in r.all_types()
        assert "sleep" in r.all_types()

    def test_load_preserves_existing_and_backfills(self, tmp_path, monkeypatch):
        partial = {"fire": {"label": "自定义火焰", "color": "#ff0000", "model_path": "custom.pt",
                            "post_process": "yolo_box", "classes": [0], "model_confidence": 0.5,
                            "vlm_prompt_key": "fire_review", "inspection_label": "火",
                            "defaults": {"enabled": True, "threshold": 0.9}}}
        r = self._make_registry(tmp_path, monkeypatch, data=partial)
        fire = r.get("fire")
        assert fire["label"] == "自定义火焰"
        assert fire["defaults"]["enabled"] is True
        assert fire["defaults"]["threshold"] == 0.9
        # backfilled fields
        assert "cooldown" in fire["defaults"]
        assert "min_box_count" in fire["defaults"]

    def test_get_unknown_returns_none(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        assert r.get("unknown_type") is None

    def test_all_types_returns_six(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        assert len(r.all_types()) == 6
        assert set(r.all_types()) == {"fire", "smoke", "uniform", "mask", "cigarette", "sleep"}

    def test_get_types_by_model_shared(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        shared = r.get_types_by_model("fire_smoke.pt")
        assert set(shared) == {"fire", "smoke"}

    def test_get_types_by_model_unique(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        assert r.get_types_by_model("mask.pt") == ["mask"]

    def test_get_color_bgr(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        assert r.get_color_bgr("fire") == (68, 68, 239)

    def test_get_defaults(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        d = r.get_defaults("sleep")
        assert d["interval"] == 60
        assert d["threshold"] == 0.7
        assert d["min_box_count"] == 1

    def test_merge_camera_config(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        merged = r.merge_camera_config("fire", {"enabled": True, "threshold": 0.8,
                                                  "roi": [[0.1, 0.1], [0.9, 0.9]]})
        assert merged["enabled"] is True
        assert merged["threshold"] == 0.8
        assert merged["roi"] == [[0.1, 0.1], [0.9, 0.9]]
        # inherited defaults
        assert merged["cooldown"] == 60
        assert merged["consecutive_required"] == 3

    def test_to_api_list(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        api = r.to_api_list()
        assert len(api) == 6
        fire_entry = next(e for e in api if e["key"] == "fire")
        assert fire_entry["label"] == "明火"
        assert fire_entry["color"] == "#ef4444"
        assert "defaults" in fire_entry
        # structural fields not exposed
        assert "model_path" not in fire_entry

    def test_validate_warns_missing_model(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        warnings = r.validate()
        # models don't exist in test env, so should have warnings
        assert len(warnings) > 0
        assert any("fire_smoke.pt" in w for w in warnings)

    def test_update_defaults_persists(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        r.update_defaults("fire", {"threshold": 0.99, "cooldown": 120})
        assert r.get_defaults("fire")["threshold"] == 0.99
        assert r.get_defaults("fire")["cooldown"] == 120
        # verify persistence
        with open(tmp_path / "detection_types.json", "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["fire"]["defaults"]["threshold"] == 0.99

    def test_update_defaults_ignores_structural_fields(self, tmp_path, monkeypatch):
        r = self._make_registry(tmp_path, monkeypatch)
        r.update_defaults("fire", {"model_path": "hacked.pt", "threshold": 0.1})
        assert r.get("fire")["model_path"] != "hacked.pt"
        assert r.get_defaults("fire")["threshold"] == 0.1
```

- [ ] **Step 6: Run all registry tests to verify they fail**

Run: `python -m pytest tests/test_detection_registry.py -v`
Expected: `hex_to_bgr` tests PASS, `TestDetectionTypeRegistry` tests PASS (implementation already in Step 3)

- [ ] **Step 7: Run all registry tests to verify they pass**

Run: `python -m pytest tests/test_detection_registry.py -v`
Expected: all PASSED

- [ ] **Step 8: Run full test suite for regression**

Run: `python -m pytest tests/ -v`
Expected: all existing tests still PASS, new tests PASS

- [ ] **Step 9: Commit**

```bash
git add backend/detection_registry.py tests/test_detection_registry.py
git commit -m "feat: add DetectionTypeRegistry core module with JSON config support"
```

---

### Task 2: 推理引擎改造（inference_engine.py）

**目标**：把 `inference_engine.py` 从硬编码的 6 类型分支改为注册表驱动，消除重复代码。

**改造前**：~930 行，6 个 `_load_xxx_model` + 6 个 `_detect_xxx` + `MODEL_PATHS` + `TARGET_CLASSES` 常量。

**改造后**：~400 行，1 个通用 `_load_model` + 2 个策略函数 `_process_yolo_box` / `_process_yolo_pose` + 注册表驱动的 `detect()`。

**Files:**

- Modify: `backend/inference_engine.py`
- Test: `tests/test_inference_engine_registry.py` (新建)
- Depends on: Task 1 (`backend/detection_registry.py`)

**Interfaces:**

- Consumes: `from backend.detection_registry import registry` — `registry.get(dtype)`, `registry.all_types()`, `registry.get_types_by_model(model_path)`
- Produces:
  - `SafetyDetector.detect(frame, detection_types, core_id=0) -> Dict[str, dict]` — 返回格式不变
  - `SafetyDetector.ensure_models_loaded(detection_types, device=None) -> None` — 接口不变
  - `SafetyDetector.get_model_status() -> List[Dict[str, Any]]` — 返回格式不变
  - `SafetyDetector.release()` / `SafetyDetector.loaded_models` — 不变
  - `_resolve_model_path(dtype, use_npu)` — 改为从注册表读 model_path/npu_model_path
  - `detect_npu_cores()`, `detect_best_device()` — 不动

**不动的代码**：

- `detect_npu_cores()`、`detect_best_device()`、`_device_fallback_order()` — 与检测类型无关
- `_preprocess()`、`_postprocess_rknn()`、`_nms_boxes()` — NPU 后处理通用逻辑
- `_detect_persons()` — 内部方法，非检测类型
- `release()`、`loaded_models` 属性 — 已是通用实现

- [ ] **Step 1: Write failing tests for registry-driven inference engine**

Create `tests/test_inference_engine_registry.py`:

```python
"""
inference_engine 注册表驱动改造测试
不依赖真实模型和 GPU/NPU，通过 mock 验证注册表驱动逻辑
"""
import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


@pytest.fixture
def fake_registry(tmp_path, monkeypatch):
    """创建一个指向 tmp_path 的测试注册表"""
    import backend.detection_registry as reg_mod
    monkeypatch.setattr(reg_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(reg_mod, "REGISTRY_FILE", tmp_path / "detection_types.json")
    reg_mod.registry.load()
    return reg_mod.registry


@pytest.fixture
def detector():
    """创建 SafetyDetector 实例（CPU 模式，不加载真实模型）"""
    from backend.inference_engine import SafetyDetector
    return SafetyDetector(npu_cores=0, device="cpu")


class TestResolveModelPath:
    """_resolve_model_path 改为注册表驱动后的测试"""

    def test_resolve_reads_from_registry(self, fake_registry, tmp_path, monkeypatch):
        """_resolve_model_path 从注册表读取 model_path 字段"""
        from backend.inference_engine import _resolve_model_path
        # 创建一个假模型文件让路径解析成功
        model_file = tmp_path / "fire_smoke.pt"
        model_file.touch()
        # 注册表里 fire 的 model_path 是 "fire_smoke.pt"
        # _resolve_model_path 应搜索到 tmp_path 下的文件
        import backend.inference_engine as ie_mod
        monkeypatch.setattr(ie_mod, "WEIGHTS_DIR", tmp_path)
        result = _resolve_model_path("fire", use_npu=False)
        assert result is not None
        assert "fire_smoke" in result

    def test_resolve_npu_path(self, fake_registry, tmp_path, monkeypatch):
        """NPU 模型路径从注册表 npu_model_path 读取"""
        from backend.inference_engine import _resolve_model_path
        rknn_file = tmp_path / "fire_smoke.rknn"
        rknn_file.touch()
        import backend.inference_engine as ie_mod
        monkeypatch.setattr(ie_mod, "WEIGHTS_DIR", tmp_path)
        result = _resolve_model_path("fire", use_npu=True)
        assert result is not None
        assert "fire_smoke.rknn" in result

    def test_resolve_returns_none_for_missing(self, fake_registry):
        """模型文件不存在时返回 None"""
        from backend.inference_engine import _resolve_model_path
        result = _resolve_model_path("fire", use_npu=False)
        # 文件不存在于任何候选路径，应返回 None
        assert result is None


class TestEnsureModelsLoaded:
    """ensure_models_loaded 注册表驱动（共享模型去重）"""

    def test_shared_model_loaded_once(self, detector, fake_registry, monkeypatch):
        """fire 和 smoke 共享 model_path，只加载一次"""
        load_calls = []

        def mock_load_model(self_, model_key, device):
            load_calls.append(model_key)

        monkeypatch.setattr(
            type(detector), "_load_model", mock_load_model
        )
        detector.ensure_models_loaded(["fire", "smoke"])
        # fire_smoke.pt 对应的 model_key 只出现一次
        assert len(load_calls) == 1

    def test_different_models_loaded_separately(self, detector, fake_registry, monkeypatch):
        """不同 model_path 的类型分别加载"""
        load_calls = []

        def mock_load_model(self_, model_key, device):
            load_calls.append(model_key)

        monkeypatch.setattr(
            type(detector), "_load_model", mock_load_model
        )
        detector.ensure_models_loaded(["fire", "mask", "sleep"])
        # fire_smoke.pt, mask.pt, yolov8n-pose.pt → 3 次
        assert len(load_calls) == 3


class TestDetectDispatch:
    """detect() 注册表驱动分发"""

    def test_detect_uses_registry_dispatch(self, detector, fake_registry, monkeypatch):
        """detect() 按注册表遍历类型，调用对应策略函数"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # mock _run_model 返回空 boxes
        monkeypatch.setattr(
            type(detector), "_run_model",
            lambda self_, model_key, frame_, type_def, core_id: []
        )

        results = detector.detect(frame, ["fire", "smoke", "mask"])
        assert "fire" in results
        assert "smoke" in results
        assert "mask" in results
        # 每个结果都是标准格式
        for dtype in ["fire", "smoke", "mask"]:
            assert "detected" in results[dtype]
            assert "boxes" in results[dtype]
            assert "scores" in results[dtype]

    def test_detect_shared_model_runs_once(self, detector, fake_registry, monkeypatch):
        """共享模型只推理一次"""
        run_calls = []

        def mock_run_model(self_, model_key, frame_, type_def, core_id):
            run_calls.append(model_key)
            return []

        monkeypatch.setattr(type(detector), "_run_model", mock_run_model)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.detect(frame, ["fire", "smoke"])
        # fire_smoke.pt 只运行一次
        assert len(run_calls) == 1

    def test_detect_yolo_box_filters_by_class(self, detector, fake_registry, monkeypatch):
        """yolo_box 策略按 classes 过滤"""
        mock_boxes = [
            {"xyxy": [10, 10, 50, 50], "class_id": 0, "confidence": 0.9},  # fire
            {"xyxy": [60, 60, 100, 100], "class_id": 1, "confidence": 0.8},  # smoke
        ]

        monkeypatch.setattr(
            type(detector), "_run_model",
            lambda self_, model_key, frame_, type_def, core_id: mock_boxes
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame, ["fire", "smoke"])

        assert results["fire"]["detected"] is True
        assert len(results["fire"]["boxes"]) == 1
        assert results["fire"]["boxes"][0] == [10, 10, 50, 50]

        assert results["smoke"]["detected"] is True
        assert len(results["smoke"]["boxes"]) == 1
        assert results["smoke"]["boxes"][0] == [60, 60, 100, 100]


class TestGetModelStatus:
    """get_model_status 从注册表读取类型列表"""

    def test_status_lists_all_registry_types(self, detector, fake_registry):
        """get_model_status 返回注册表中所有类型"""
        status = detector.get_model_status()
        type_names = {s["type"] for s in status}
        assert type_names == {"fire", "smoke", "uniform", "mask", "cigarette", "sleep"}

    def test_status_unloaded(self, detector, fake_registry):
        """未加载模型时所有类型 loaded=False"""
        status = detector.get_model_status()
        for s in status:
            assert s["loaded"] is False


class TestProcessYoloBox:
    """_process_yolo_box 后处理函数"""

    def test_filters_by_classes(self, fake_registry):
        from backend.inference_engine import _process_yolo_box
        raw_boxes = [
            {"xyxy": [10, 10, 50, 50], "class_id": 0, "confidence": 0.9},
            {"xyxy": [60, 60, 100, 100], "class_id": 1, "confidence": 0.8},
            {"xyxy": [110, 110, 150, 150], "class_id": 2, "confidence": 0.7},
        ]
        type_def = {"classes": [0, 2], "post_process": "yolo_box"}
        result = _process_yolo_box(raw_boxes, type_def)
        assert result["detected"] is True
        assert len(result["boxes"]) == 2
        assert len(result["scores"]) == 2

    def test_no_classes_filter(self, fake_registry):
        """classes=None 时不过滤"""
        from backend.inference_engine import _process_yolo_box
        raw_boxes = [
            {"xyxy": [10, 10, 50, 50], "class_id": 0, "confidence": 0.9},
        ]
        type_def = {"classes": None, "post_process": "yolo_box"}
        result = _process_yolo_box(raw_boxes, type_def)
        assert result["detected"] is True
        assert len(result["boxes"]) == 1

    def test_empty_boxes(self, fake_registry):
        from backend.inference_engine import _process_yolo_box
        result = _process_yolo_box([], {"classes": [0], "post_process": "yolo_box"})
        assert result["detected"] is False
        assert result["boxes"] == []
        assert result["scores"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_inference_engine_registry.py -v`
Expected: FAIL — `_process_yolo_box` not found, `_run_model` not found, `ensure_models_loaded` still uses if/elif chain

- [ ] **Step 3: Delete hardcoded constants and rewrite `_resolve_model_path`**

In `backend/inference_engine.py`:

1. **删除** `MODEL_PATHS` 字典（原 47-126 行）
2. **删除** `MASK_TARGET_CLASSES`、`CIGARETTE_TARGET_CLASSES`、`UNIFORM_TARGET_CLASSES` 三个常量（原 128-132 行）
3. **新增** 注册表导入
4. **改写** `_resolve_model_path` 从注册表读取模型文件名

```python
# ---- 在文件顶部 imports 之后、detect_npu_cores 之前 ----

from backend.detection_registry import registry

# 删除原 MODEL_PATHS 字典（整个 47-126 行）
# 删除原 MASK_TARGET_CLASSES、CIGARETTE_TARGET_CLASSES、UNIFORM_TARGET_CLASSES（128-132 行）


def _resolve_model_path(dtype: str, use_npu: bool) -> Optional[str]:
    """解析模型路径：优先环境变量，其次按注册表文件名在标准目录中查找"""
    type_def = registry.get(dtype)
    filename = type_def.get("npu_model_path") if use_npu else type_def.get("model_path")
    if filename is None:
        return None

    env_key = f"{dtype.upper()}_RKNN_MODEL" if use_npu else f"{dtype.upper()}_MODEL"
    env_path = os.getenv(env_key)
    if env_path and os.path.exists(env_path):
        return env_path

    candidates = [
        str(WEIGHTS_DIR / filename),
        str(PROJECT_ROOT / filename),
        f"models/{filename}",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None
```

- [ ] **Step 4: 添加两个后处理策略函数**

在 `_resolve_model_path` 之后、`SafetyDetector` 类之前添加：

```python
def _process_yolo_box(raw_boxes: list, type_def: dict) -> dict:
    """yolo_box 后处理：按 classes 过滤框"""
    result = {"detected": False, "boxes": [], "scores": []}
    target_classes = type_def.get("classes")
    for box in raw_boxes:
        if target_classes is not None and box.get("class_id") not in target_classes:
            continue
        result["boxes"].append(box["xyxy"])
        result["scores"].append(box["confidence"])
    result["detected"] = len(result["boxes"]) > 0
    if result["scores"]:
        result["max_confidence"] = max(result["scores"])
    else:
        result["max_confidence"] = 0.0
    return result


def _process_yolo_pose(raw_output, type_def: dict, frame: np.ndarray) -> dict:
    """yolo_pose 后处理：姿态模型 + sleep_detect 分析"""
    result = {"detected": False, "boxes": [], "scores": [], "subjects": [], "count": 0}
    try:
        from safety_detection.sleep_detect import process_frame
        model = raw_output  # yolo_pose 时 raw_output 就是模型实例
        subjects = process_frame(model, frame, conf=type_def.get("model_confidence", 0.1))
        for s in subjects:
            result["boxes"].append(s["box"])
            result["scores"].append(s.get("sleep_confidence", 0))
            result["subjects"].append({
                "box": s["box"],
                "score": s.get("score", 0),
                "sleep_confidence": s.get("sleep_confidence", 0),
                "posture": s.get("posture_label", ""),
                "keypoints": s.get("keypoints"),
                "sleeping": s.get("sleeping", False),
            })
            if s.get("sleeping"):
                result["detected"] = True
                result["count"] += 1
        result["max_confidence"] = max(result["scores"]) if result["scores"] else 0.0
    except Exception as e:
        logger.error(f"Pose post-process error: {e}")
    return result


POST_PROCESSORS = {
    "yolo_box": _process_yolo_box,
    "yolo_pose": _process_yolo_pose,
}
```

- [ ] **Step 5: 改写 `SafetyDetector` 核心方法**

替换 `ensure_models_loaded`、6 个 `_load_xxx_model`、`detect`、6 个 `_detect_xxx`：

```python
    def ensure_models_loaded(self, detection_types: List[str], device: str = None) -> None:
        """懒加载指定类型所需的模型，共享 model_path 的类型只加载一次"""
        if device is None:
            device = self.device
        loaded_paths = set()
        with self._model_lock:
            for dtype in detection_types:
                type_def = registry.get(dtype)
                if type_def is None:
                    logger.warning(f"Unknown detection type: {dtype}")
                    continue
                model_key = type_def["model_path"]
                if model_key in loaded_paths:
                    continue
                loaded_paths.add(model_key)
                self._load_model(model_key, device)

    def _load_model(self, model_key: str, device: str) -> None:
        """通用模型加载：npu / gpu / cpu 三分支"""
        if device == "npu" and self._npu_cores > 0:
            if model_key not in self._npu_models:
                # 找到使用此模型的第一个类型来解析路径
                dtypes = registry.get_types_by_model(model_key)
                if not dtypes:
                    return
                path = _resolve_model_path(dtypes[0], use_npu=True)
                if path and RKNN_AVAILABLE:
                    self._npu_models[model_key] = {}
                    for core_id in range(self._npu_cores):
                        rknn = RKNNLite(verbose=False)
                        ret = rknn.load_rknn(path)
                        if ret != 0:
                            logger.error(f"Failed to load {model_key} RKNN for core {core_id}")
                            continue
                        ret = rknn.init_runtime(core_mask=self.CORE_MASKS[core_id])
                        if ret != 0:
                            logger.error(f"Failed to init {model_key} RKNN runtime for core {core_id}")
                            continue
                        self._npu_models[model_key][core_id] = rknn
                        logger.info(f"{model_key} RKNN loaded on core {core_id}")
                else:
                    logger.warning(f"{model_key} RKNN not found, falling back to CPU")
        elif device == "gpu":
            if model_key not in self._cpu_models:
                dtypes = registry.get_types_by_model(model_key)
                if not dtypes:
                    return
                path = _resolve_model_path(dtypes[0], use_npu=False)
                if path and ULTRALYTICS_AVAILABLE:
                    model = YOLO(path)
                    try:
                        model = model.to("cuda")
                        logger.info(f"{model_key} GPU model loaded from {path}")
                    except Exception as e:
                        logger.warning(f"Failed to move {model_key} to GPU: {e}")
                    self._cpu_models[model_key] = model
                else:
                    logger.warning(f"{model_key} GPU model not found")
        else:
            if model_key not in self._cpu_models:
                dtypes = registry.get_types_by_model(model_key)
                if not dtypes:
                    return
                path = _resolve_model_path(dtypes[0], use_npu=False)
                if path and ULTRALYTICS_AVAILABLE:
                    self._cpu_models[model_key] = YOLO(path)
                    logger.info(f"{model_key} CPU model loaded from {path}")
                else:
                    logger.warning(f"{model_key} model not found")

    def _run_model(self, model_key: str, frame: np.ndarray,
                   type_def: dict, core_id: int = 0):
        """执行模型推理，返回原始检测结果"""
        model = None
        use_npu = False
        with self._model_lock:
            if model_key in self._npu_models and core_id in self._npu_models[model_key]:
                model = self._npu_models[model_key][core_id]
                use_npu = True
            elif model_key in self._cpu_models:
                model = self._cpu_models[model_key]

        if model is None:
            logger.warning(f"Model {model_key} not loaded")
            return [] if type_def["post_process"] == "yolo_box" else model

        # yolo_pose 直接返回模型实例（由 _process_yolo_pose 自行调用 process_frame）
        if type_def["post_process"] == "yolo_pose":
            return model

        # yolo_box：统一推理流程
        try:
            if use_npu:
                input_frame = self._preprocess(frame)
                outputs = model.inference(inputs=[input_frame])
                conf = type_def.get("model_confidence", 0.5)
                return self._postprocess_rknn(outputs, frame.shape[:2], conf_threshold=conf)
            else:
                conf = type_def.get("model_confidence", 0.5)
                pred = model.predict(frame, conf=conf, verbose=False)
                boxes = []
                if pred and pred[0].boxes is not None:
                    for b in pred[0].boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        cls = int(b.cls[0])
                        score = float(b.conf[0])
                        boxes.append({"xyxy": [x1, y1, x2, y2], "class_id": cls, "confidence": score})
                return boxes
        except Exception as e:
            logger.error(f"Model {model_key} inference error: {e}")
            return []

    def detect(self, frame: np.ndarray, detection_types: List[str],
               core_id: int = 0) -> Dict[str, dict]:
        """
        对单帧执行多类型检测（注册表驱动）

        共享 model_path 的类型只推理一次，各自按 classes 过滤。
        """
        results: Dict[str, dict] = {}
        processed_models = set()

        for dtype in detection_types:
            type_def = registry.get(dtype)
            if type_def is None:
                logger.warning(f"Unknown detection type: {dtype}")
                continue

            model_key = type_def["model_path"]
            if model_key in processed_models:
                continue

            raw_output = self._run_model(model_key, frame, type_def, core_id)

            # 对共享此模型的所有请求类型执行后处理
            for related_dtype in registry.get_types_by_model(model_key):
                if related_dtype not in detection_types:
                    continue
                related_def = registry.get(related_dtype)
                strategy = related_def["post_process"]
                processor = POST_PROCESSORS.get(strategy)
                if processor is None:
                    logger.warning(f"Unknown post_process strategy: {strategy}")
                    continue
                if strategy == "yolo_pose":
                    results[related_dtype] = processor(raw_output, related_def, frame)
                else:
                    results[related_dtype] = processor(raw_output, related_def)

            processed_models.add(model_key)

        return results
```

- [ ] **Step 6: 改写 `get_model_status` 从注册表读取类型列表**

```python
    def get_model_status(self) -> List[Dict[str, Any]]:
        """返回模型状态详情（注册表驱动）"""
        status = []
        with self._model_lock:
            for dtype in registry.all_types():
                type_def = registry.get(dtype)
                model_key = type_def["model_path"]
                # 检查是否已加载
                is_loaded = model_key in self._cpu_models or model_key in self._npu_models
                if is_loaded:
                    if model_key in self._cpu_models:
                        model = self._cpu_models[model_key]
                        model_device = getattr(model, "device", None)
                        if model_device is not None:
                            device_type = str(model_device).split(":")[0]
                        else:
                            device_type = "cpu"
                        if device_type == "cuda":
                            entry = {"type": dtype, "backend": "gpu", "device": "cuda", "loaded": True}
                        else:
                            entry = {"type": dtype, "backend": "cpu", "device": "pytorch", "loaded": True}
                    else:
                        core_map = self._npu_models.get(model_key, {})
                        entry = {"type": dtype, "backend": "npu", "device": "rk3588",
                                 "cores": len(core_map), "loaded": True}
                else:
                    entry = {"type": dtype, "backend": self.device,
                             "device": self.device, "loaded": False}
                status.append(entry)
        return status
```

- [ ] **Step 7: 更新 `loaded_models` 属性**

```python
    @property
    def loaded_models(self) -> List[str]:
        """返回已加载的模型列表（model_key，不含内部 person 模型）"""
        models = []
        with self._model_lock:
            models.extend([k for k in self._cpu_models.keys() if k != "person"])
            models.extend(self._npu_models.keys())
        return list(dict.fromkeys(models))
```

（逻辑不变，只是 key 从 dtype 变成了 model_key，但 `person` 仍是内部模型需排除）

- [ ] **Step 8: 删除旧的硬编码方法**

删除以下方法（由通用 `_load_model` + `_run_model` + 策略函数替代）：

- `_load_fire_smoke_model`（原 255-299 行）
- `_load_uniform_model`（原 301-345 行）
- `_load_person_model`（原 347-389 行，**保留**，person 是内部模型不在注册表中）
- `_load_mask_model`（原 391-435 行）
- `_load_cigarette_model`（原 437-479 行）
- `_load_sleep_model`（原 481-526 行）
- `_detect_fire_smoke`（原 558-612 行）
- `_detect_uniform`（原 654-693 行）
- `_detect_mask`（原 695-733 行）
- `_detect_cigarette`（原 735-773 行）
- `_detect_sleep`（原 775-801 行）

**注意保留**：`_detect_persons`（614-652 行）— 内部人员检测，不属于注册表管理的检测类型。

`_load_person_model` 也保留，因为 person 模型是 sleep 检测的依赖，不由注册表管理。如果 `ensure_models_loaded` 遇到 `dtype == "sleep"`，仍需在加载 pose 模型前确保 person 模型已加载（保留原有调用链）。

- [ ] **Step 9: Run new tests**

Run: `python -m pytest tests/test_inference_engine_registry.py -v`
Expected: all PASS

- [ ] **Step 10: Run full test suite for regression**

Run: `python -m pytest tests/ -v`
Expected: all existing tests still PASS, new tests PASS

注意：如果现有测试中有直接引用 `MODEL_PATHS`、`MASK_TARGET_CLASSES` 等已删除常量的，需要更新这些测试。

- [ ] **Step 11: Commit**

```bash
git add backend/inference_engine.py tests/test_inference_engine_registry.py
git commit -m "refactor: registry-driven inference engine, eliminate 6x load/detect methods"
```

---

### Task 3: 配置模块改造（config.py）

**目标**：把 `config.py` 中硬编码的 `DEFAULT_TYPE_CONFIG` 和 `DEFAULT_GLOBAL_SETTINGS.display_detection_types` 改为从注册表动态生成，让新增检测类型自动出现在配置默认值中。

**Files:**

- Modify: `backend/config.py`
- Test: `tests/test_config_registry.py` (新建)
- Depends on: Task 1 (`backend/detection_registry.py`)

**Interfaces:**

- Consumes: `from backend.detection_registry import registry` — `registry.all_types()`, `registry.get_defaults(dtype)`
- Produces:
  - `DEFAULT_TYPE_CONFIG` — 改为函数 `get_default_type_config() -> dict`，运行时从注册表生成
  - `DEFAULT_GLOBAL_SETTINGS` — `display_detection_types` 从注册表动态生成
  - `DEFAULT_CAMERA_GLOBALS` — `detection_types` 从注册表动态生成
  - `apply_camera_globals(cam_config, globals_data)` — 遍历注册表类型而非硬编码
  - `load_camera_configs()` — 迁移逻辑使用注册表默认值

**改造策略**：

`DEFAULT_TYPE_CONFIG` 目前是模块级常量，在 `import backend.config` 时就会执行。但注册表需要在 `registry.load()` 之后才能使用。有两种方案：

1. 改为函数调用（`get_default_type_config()`），每次需要时动态生成
2. 保留常量但在模块加载时从注册表生成

选择方案 1：改为函数。原因是注册表可能在运行时被 reload，函数调用总是获取最新值。同时保留一个模块级 `DEFAULT_TYPE_CONFIG` 变量作为向后兼容（延迟初始化）。

- [ ] **Step 1: Write failing tests**

Create `tests/test_config_registry.py`:

```python
"""
config.py 注册表驱动改造测试
验证 DEFAULT_TYPE_CONFIG 和 display_detection_types 从注册表动态生成
"""
import json
import pytest


@pytest.fixture
def setup_registry(tmp_path, monkeypatch):
    """初始化测试注册表"""
    import backend.detection_registry as reg_mod
    monkeypatch.setattr(reg_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(reg_mod, "REGISTRY_FILE", tmp_path / "detection_types.json")
    reg_mod.registry.load()
    return reg_mod.registry


class TestGetDefaultTypeConfig:
    """get_default_type_config() 从注册表动态生成"""

    def test_returns_all_types(self, setup_registry):
        from backend.config import get_default_type_config
        dtc = get_default_type_config()
        assert set(dtc.keys()) == {"fire", "smoke", "uniform", "mask", "cigarette", "sleep"}

    def test_values_match_registry_defaults(self, setup_registry):
        from backend.config import get_default_type_config
        dtc = get_default_type_config()
        assert dtc["fire"]["threshold"] == 0.6
        assert dtc["fire"]["interval"] == 1
        assert dtc["fire"]["cooldown"] == 60
        assert dtc["sleep"]["interval"] == 60
        assert dtc["sleep"]["threshold"] == 0.7

    def test_includes_all_default_fields(self, setup_registry):
        from backend.config import get_default_type_config
        dtc = get_default_type_config()
        required_fields = {"enabled", "interval", "threshold", "consecutive_required", "cooldown", "use_vlm"}
        for dtype, cfg in dtc.items():
            for field in required_fields:
                assert field in cfg, f"{dtype} missing field: {field}"

    def test_excludes_structural_fields(self, setup_registry):
        """不包含 model_path 等结构性字段"""
        from backend.config import get_default_type_config
        dtc = get_default_type_config()
        structural = {"model_path", "npu_model_path", "post_process", "classes", "model_confidence"}
        for dtype, cfg in dtc.items():
            for field in structural:
                assert field not in cfg, f"{dtype} should not contain {field}"


class TestDefaultGlobalSettings:
    """display_detection_types 从注册表动态生成"""

    def test_display_types_matches_registry(self, setup_registry):
        from backend.config import get_default_global_settings
        settings = get_default_global_settings()
        ddt = settings["display_detection_types"]
        assert set(ddt.keys()) == {"fire", "smoke", "uniform", "mask", "cigarette", "sleep"}
        # 所有类型默认显示
        for dtype, enabled in ddt.items():
            assert enabled is True


class TestApplyCameraGlobals:
    """apply_camera_globals 使用注册表类型"""

    def test_fills_all_registry_types(self, setup_registry, tmp_path, monkeypatch):
        """空摄像头配置应填充注册表中所有类型的默认值"""
        monkeypatch.setattr("backend.config.CONFIG_DIR", tmp_path)
        from backend.config import apply_camera_globals
        cam = {"camera_id": "test", "source": "rtsp://test"}
        result = apply_camera_globals(cam)
        dt = result["detection_types"]
        assert set(dt.keys()) == {"fire", "smoke", "uniform", "mask", "cigarette", "sleep"}

    def test_preserves_camera_overrides(self, setup_registry, tmp_path, monkeypatch):
        """摄像头级配置覆盖注册表默认值"""
        monkeypatch.setattr("backend.config.CONFIG_DIR", tmp_path)
        from backend.config import apply_camera_globals
        cam = {
            "camera_id": "test",
            "detection_types": {
                "fire": {"enabled": True, "threshold": 0.99},
            },
        }
        result = apply_camera_globals(cam)
        assert result["detection_types"]["fire"]["enabled"] is True
        assert result["detection_types"]["fire"]["threshold"] == 0.99
        # 缺失字段从注册表默认值补全
        assert result["detection_types"]["fire"]["cooldown"] == 60


class TestLoadCameraConfigs:
    """load_camera_configs 迁移逻辑使用注册表默认值"""

    def test_migration_uses_registry_defaults(self, setup_registry, tmp_path, monkeypatch):
        """旧配置迁移时使用注册表默认值"""
        cameras_file = tmp_path / "cameras.json"
        old_cam = {
            "camera_id": "old",
            "source": "rtsp://old",
            # 没有 detection_types 字段 → 应从注册表填充
        }
        cameras_file.write_text(
            json.dumps({"cameras": [old_cam]}, ensure_ascii=False),
            encoding="utf-8",
        )
        monkeypatch.setattr("backend.config.CAMERAS_CONFIG_FILE", cameras_file)
        from backend.config import load_camera_configs
        cameras = load_camera_configs()
        cam = cameras[0]
        assert "detection_types" in cam
        # 所有注册表类型都应存在
        assert set(cam["detection_types"].keys()) >= {"fire", "smoke", "uniform", "mask", "cigarette", "sleep"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config_registry.py -v`
Expected: FAIL — `get_default_type_config` 和 `get_default_global_settings` 不存在

- [ ] **Step 3: 添加注册表驱动的函数**

在 `backend/config.py` 中，在原 `DEFAULT_TYPE_CONFIG` 位置：

```python
from backend.detection_registry import registry


def get_default_type_config() -> dict:
    """从注册表动态生成默认检测类型配置（运行参数部分）"""
    dtc = {}
    for dtype in registry.all_types():
        defaults = registry.get_defaults(dtype)
        dtc[dtype] = {
            "enabled": defaults.get("enabled", False),
            "interval": defaults.get("interval", 1),
            "threshold": defaults.get("threshold", 0.5),
            "consecutive_required": defaults.get("consecutive_required", 3),
            "cooldown": defaults.get("cooldown", 60),
            "use_vlm": defaults.get("use_vlm", False),
            "min_box_count": defaults.get("min_box_count"),
            "max_box_count": defaults.get("max_box_count"),
        }
    return dtc


# 向后兼容：模块级变量，首次访问时从注册表生成
# 注意：如果在 registry.load() 之前就 import 了 config，
# 这里会 fallback 到硬编码默认值
try:
    DEFAULT_TYPE_CONFIG = get_default_type_config()
except Exception:
    DEFAULT_TYPE_CONFIG = {
        "fire": {"enabled": False, "interval": 1, "threshold": 0.6, "consecutive_required": 3, "cooldown": 60, "use_vlm": False},
        "smoke": {"enabled": False, "interval": 1, "threshold": 0.55, "consecutive_required": 3, "cooldown": 60, "use_vlm": False},
        "uniform": {"enabled": False, "interval": 1, "threshold": 0.5, "consecutive_required": 3, "cooldown": 60, "use_vlm": False},
        "mask": {"enabled": False, "interval": 1, "threshold": 0.5, "consecutive_required": 3, "cooldown": 60, "use_vlm": False},
        "cigarette": {"enabled": False, "interval": 1, "threshold": 0.5, "consecutive_required": 3, "cooldown": 60, "use_vlm": False},
        "sleep": {"enabled": False, "interval": 60, "threshold": 0.7, "consecutive_required": 3, "cooldown": 60, "use_vlm": False},
    }
```

- [ ] **Step 4: 改写 `DEFAULT_GLOBAL_SETTINGS` 的 `display_detection_types`**

```python
def get_default_global_settings() -> dict:
    """动态生成全局默认设置，display_detection_types 从注册表读取"""
    try:
        ddt = {dtype: True for dtype in registry.all_types()}
    except Exception:
        ddt = {"fire": True, "smoke": True, "uniform": True, "mask": True, "cigarette": True, "sleep": True}
    return {
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
        "display_detection_types": ddt,
        "display_detection_interval": 1.0,
        "save_image_timestamp": True,
    }


# 向后兼容
try:
    DEFAULT_GLOBAL_SETTINGS = get_default_global_settings()
except Exception:
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
        "display_detection_types": {"fire": True, "smoke": True, "uniform": True, "mask": True, "cigarette": True, "sleep": True},
        "display_detection_interval": 1.0,
        "save_image_timestamp": True,
    }
```

- [ ] **Step 5: 改写 `DEFAULT_CAMERA_GLOBALS`**

```python
try:
    DEFAULT_CAMERA_GLOBALS = {
        "width": 640,
        "height": 480,
        "fps": 15,
        "source_type": "auto",
        "detection_types": get_default_type_config(),
    }
except Exception:
    DEFAULT_CAMERA_GLOBALS = {
        "width": 640,
        "height": 480,
        "fps": 15,
        "source_type": "auto",
        "detection_types": dict(DEFAULT_TYPE_CONFIG),
    }
```

- [ ] **Step 6: 改写 `apply_camera_globals` 使用注册表**

```python
def apply_camera_globals(cam_config: dict, globals_data: dict = None) -> dict:
    """将全局默认值应用到摄像头配置（不覆盖已有值）"""
    if globals_data is None:
        globals_data = load_camera_globals()

    result = dict(cam_config)

    for key in ("width", "height", "fps", "source_type"):
        if result.get(key) is None:
            result[key] = globals_data.get(key, DEFAULT_CAMERA_GLOBALS.get(key))

    dt = result.get("detection_types")
    default_dtc = get_default_type_config()
    if not dt:
        result["detection_types"] = {
            k: dict(v) for k, v in globals_data.get("detection_types", default_dtc).items()
        }
    else:
        merged_dt = {}
        global_dt = globals_data.get("detection_types", default_dtc)
        for dtype, default_cfg in default_dtc.items():
            cam_cfg = dt.get(dtype, {})
            merged_cfg = dict(default_cfg)
            global_cfg = global_dt.get(dtype, {})
            for k, v in global_cfg.items():
                merged_cfg[k] = v
            for k, v in cam_cfg.items():
                merged_cfg[k] = v
            merged_dt[dtype] = merged_cfg
        result["detection_types"] = merged_dt

    return result
```

- [ ] **Step 7: 改写 `load_camera_configs` 迁移逻辑**

只改一行：旧格式迁移时使用 `get_default_type_config()` 代替 `DEFAULT_TYPE_CONFIG`：

```python
    for cam in cameras:
        if "detection_types" not in cam:
            cam["detection_types"] = get_default_type_config()  # 改这里
            if cam.get("detection_enabled") is False:
                for dtype in cam["detection_types"]:
                    cam["detection_types"][dtype]["enabled"] = False
            migrated = True

        for dtype, cfg in cam.get("detection_types", {}).items():
            if "use_vlm" not in cfg:
                cfg["use_vlm"] = False
                migrated = True

        for dtype, cfg in cam.get("detection_types", {}).items():
            if "level" in cfg:
                del cfg["level"]
                migrated = True
            if "cooldown" not in cfg:
                cfg["cooldown"] = get_default_type_config().get(dtype, {}).get("cooldown", 3)  # 改这里
                migrated = True
```

- [ ] **Step 8: 改写 `load_camera_globals` 使用注册表**

```python
def load_camera_globals() -> dict:
    """加载摄像头全局默认参数，不存在则创建"""
    globals_file = CONFIG_DIR / "camera_globals.json"
    if globals_file.exists():
        try:
            with open(globals_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            merged = dict(DEFAULT_CAMERA_GLOBALS)
            merged.update(data)
            if "detection_types" in data:
                merged_dt = get_default_type_config()  # 改这里
                for k, v in data["detection_types"].items():
                    if isinstance(v, dict):
                        merged_dt[k] = {**merged_dt.get(k, {}), **v}
                merged["detection_types"] = merged_dt
            return merged
        except Exception:
            pass
    save_camera_globals(DEFAULT_CAMERA_GLOBALS)
    return dict(DEFAULT_CAMERA_GLOBALS)
```

- [ ] **Step 9: Run tests**

Run: `python -m pytest tests/test_config_registry.py -v`
Expected: all PASS

- [ ] **Step 10: Run full test suite for regression**

Run: `python -m pytest tests/ -v`
Expected: all existing tests still PASS, new tests PASS

现有 `test_config_defaults.py` 中引用 `DEFAULT_TYPE_CONFIG` 的测试应仍然通过（向后兼容变量保留）。

- [ ] **Step 11: Commit**

```bash
git add backend/config.py tests/test_config_registry.py
git commit -m "refactor: config.py reads detection type defaults from registry"
```

---

### Task 4: detector_core.py 改造

**Files:**

- Modify: `backend/safety_detection/detector_core.py:28-44` — `TypeSchedule` 新增字段
- Modify: `backend/safety_detection/detector_core.py:281-370` — `_annotate_frame` 改用注册表
- Modify: `backend/safety_detection/detector_core.py:430-434` — `_get_due_types` 前置冷却
- Modify: `backend/safety_detection/detector_core.py:440-512` — `_handle_standard_detection` 新增 ROI 过滤 + box_count
- Test: `tests/test_detector_core_registry.py`

**Interfaces:**

- Consumes: `backend/detection_registry.py` (Task 1) — `registry.get(dtype)` 返回类型定义, `registry.get_color_bgr(dtype)` 返回 BGR 元组, `registry.all_types()` 返回所有类型 key
- Produces: 无新公开接口，改造内部方法

- [ ] **Step 1: 给 TypeSchedule 添加新字段**

在 `backend/safety_detection/detector_core.py` 的 `TypeSchedule` dataclass 中新增 `roi`、`roi_invert`、`min_box_count`、`max_box_count` 字段：

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
    use_vlm: bool = False
    externally_managed: bool = False
    roi: list = None
    roi_invert: bool = False
    min_box_count: int = None
    max_box_count: int = None
```

注意：`roi` 是归一化坐标的多边形顶点数组（来自摄像头级配置），`min_box_count`/`max_box_count` 可来自注册表 defaults 或摄像头级覆盖。

- [ ] **Step 2: 添加 filter_by_roi 函数**

在 `detector_core.py` 顶部（`TypeSchedule` 之后、`DetectionStrategy` 之前）添加 ROI 过滤函数：

```python
def filter_by_roi(result: dict, roi: list, roi_invert: bool,
                  frame_width: int, frame_height: int) -> dict:
    if not roi:
        return result

    polygon = np.array([
        [int(x * frame_width), int(y * frame_height)]
        for x, y in roi
    ], dtype=np.int32)

    filtered_boxes, filtered_scores = [], []
    filtered_subjects = []
    subjects = result.get("subjects", [])

    for i, (box, score) in enumerate(zip(result.get("boxes", []), result.get("scores", []))):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        inside = cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0
        keep = inside if not roi_invert else not inside
        if keep:
            filtered_boxes.append(box)
            filtered_scores.append(score)
            if i < len(subjects):
                filtered_subjects.append(subjects[i])

    out = {
        **result,
        "boxes": filtered_boxes,
        "scores": filtered_scores,
        "detected": len(filtered_boxes) > 0,
    }
    if subjects:
        out["subjects"] = filtered_subjects
    return out
```

- [ ] **Step 3: 添加 check_box_count 函数**

紧跟 `filter_by_roi` 之后添加 box_count 判断函数：

```python
def check_box_count(result: dict, min_box_count: int = None,
                    max_box_count: int = None) -> dict:
    box_count = len(result.get("boxes", []))

    if min_box_count is not None and box_count < min_box_count:
        return {**result, "detected": False}

    if max_box_count is not None and box_count > max_box_count:
        return {**result, "detected": False}

    if max_box_count is not None and box_count <= max_box_count:
        return {**result, "detected": True}

    return result
```

注意 `max_box_count` 的特殊逻辑：当设置了 `max_box_count` 且框数量在范围内，即使原始 `detected=False`（0 个框），也应该设 `detected=True`（离岗场景：0 人就报警）。

- [ ] **Step 4: 改造 _annotate_frame 使用注册表**

将 `_annotate_frame` 中的硬编码 `type_colors` 和 `type_labels` 替换为注册表查询，骨架绘制改用 `post_process == "yolo_pose"` 判断：

```python
@staticmethod
def _annotate_frame(frame: np.ndarray, results: Dict[str, dict],
                    camera_id: str = "", due_types: list = None) -> np.ndarray:
    """在帧上绘制检测框和标签，返回标注后的帧副本"""
    from backend.detection_registry import registry

    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for dtype, result in results.items():
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        if not boxes:
            continue

        type_def = registry.get(dtype)
        base_color = registry.get_color_bgr(dtype) if type_def else (0, 255, 0)
        base_label = type_def.get("label", dtype) if type_def else dtype
        is_pose = type_def.get("post_process") == "yolo_pose" if type_def else False

        for i, box in enumerate(boxes):
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = map(int, box[:4])
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            if is_pose:
                subjects = result.get("subjects", [])
                is_sleeping = subjects[i].get("sleeping", False) if i < len(subjects) else False
                color = base_color if is_sleeping else (255, 255, 0)
                label = base_label if is_sleeping else "person"
            else:
                color = base_color
                label = base_label

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            conf = scores[i] if i < len(scores) else 0.0
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if is_pose:
            skeleton = [
                (0, 1), (0, 2), (1, 3), (2, 4),
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (5, 11), (6, 12), (11, 12),
                (11, 13), (13, 15), (12, 14), (14, 16),
            ]
            subjects = result.get("subjects", [])
            for subj in subjects:
                kpts = subj.get("keypoints")
                if kpts is None or len(kpts) < 17:
                    continue
                is_sleeping = subj.get("sleeping", False)
                sk_color = base_color if is_sleeping else (255, 255, 0)
                for a, b in skeleton:
                    if a < len(kpts) and b < len(kpts):
                        xa, ya, ca = kpts[a]
                        xb, yb, cb = kpts[b]
                        if ca > 0.4 and cb > 0.4:
                            pt_a = (int(xa), int(ya))
                            pt_b = (int(xb), int(yb))
                            cv2.line(annotated, pt_a, pt_b, sk_color, 2)
                for idx, (kx, ky, kc) in enumerate(kpts[:17]):
                    if kc > 0.4:
                        cv2.circle(annotated, (int(kx), int(ky)), 3, sk_color, -1)

    return annotated
```

关键变化：

- `type_colors`/`type_labels` 硬编码删除，改用 `registry.get_color_bgr(dtype)` 和 `type_def["label"]`
- `dtype == "sleep"` 判断改为 `is_pose = type_def.get("post_process") == "yolo_pose"`
- 未注册的类型 fallback 到绿色 `(0, 255, 0)` 和 `dtype` 本身作为标签

- [ ] **Step 5: 改造 _get_due_types 前置冷却检查**

将冷却检查提前到 `_get_due_types`，冷却中的类型不进入推理。采用不加锁的内部方法避免死锁：

```python
def _is_in_cooldown_unlocked(self, camera_id: str, dtype: str, now: float) -> bool:
    last = self._cooldowns.get(camera_id, {}).get(dtype, 0)
    schedule = self._schedules.get(camera_id, {}).get(dtype)
    cooldown = schedule.cooldown if schedule else 3.0
    return now - last < cooldown

def _get_due_types(self, camera_id: str, now: float) -> List[str]:
    """获取当前到期的检测类型（跳过冷却中和外部调度器管理的类型）"""
    with self._lock:
        schedules = self._schedules.get(camera_id, {})
        due = []
        for dtype, s in schedules.items():
            if s.externally_managed:
                continue
            if not s.is_due(now):
                continue
            if self._is_in_cooldown_unlocked(camera_id, dtype, now):
                continue
            due.append(dtype)
        return due

def is_in_cooldown(self, camera_id: str, dtype: str, now: float) -> bool:
    with self._lock:
        return self._is_in_cooldown_unlocked(camera_id, dtype, now)
```

同时删除 `_handle_standard_detection` 中的冷却检查（477-480 行），因为冷却已在 `_get_due_types` 阶段拦截。

- [ ] **Step 6: 改造 _handle_standard_detection 添加 ROI 过滤和 box_count**

在 `_handle_standard_detection` 开头（threshold 检查之前）添加 ROI 过滤和 box_count 判断：

```python
def _handle_standard_detection(
    self, camera_id: str, dtype: str, frame: np.ndarray,
    result: dict, schedule: TypeSchedule
) -> None:
    # ROI 过滤
    if schedule.roi:
        h, w = frame.shape[:2]
        result = filter_by_roi(result, schedule.roi, schedule.roi_invert, w, h)

    # box_count 判断
    if schedule.min_box_count is not None or schedule.max_box_count is not None:
        result = check_box_count(result, schedule.min_box_count, schedule.max_box_count)

    detected = result.get("detected", False)
    max_conf = max(result.get("scores", [0]) or [0])

    if not detected or max_conf < schedule.threshold:
        if not detected and result.get("boxes"):
            logger.warning(f"{camera_id} {dtype} has boxes but detected=False, resetting count")
        elif detected and max_conf < schedule.threshold:
            logger.info(f"{camera_id} {dtype} blocked by threshold: conf={max_conf:.2f} < threshold={schedule.threshold}")
        if self.camera_manager is not None:
            self.camera_manager.clear_detection_frames(camera_id, dtype)
        schedule.consecutive_count = 0
        return

    schedule.consecutive_count += 1
    logger.info(f"{camera_id} {dtype} consecutive={schedule.consecutive_count}/{schedule.consecutive_required} conf={max_conf:.2f}")

    if self.camera_manager is not None:
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

    now = time.time()
    # 冷却检查已移至 _get_due_types，此处删除

    logger.info(f"{camera_id} {dtype} TRIGGERING alarm (conf={max_conf:.2f})")
    self._cooldowns[camera_id][dtype] = now

    result["level"] = "small_model_alarm"
    if not result.get("reason"):
        result["reason"] = f"检测到 {dtype}，置信度 {max_conf:.2f}"

    self._alert_states[camera_id][dtype] = {"active": True, "time": now, "level": "small_model_alarm"}
    result["detection_frames"] = (
        self.camera_manager.get_detection_frames(camera_id, dtype)
        if self.camera_manager is not None
        else []
    )
    if schedule.use_vlm:
        result["pending_vlm_review"] = True
        vlm_frames = result["detection_frames"][-MAX_VLM_REVIEW_FRAMES:]
        self._submit_vlm_review(camera_id, dtype, vlm_frames, schedule, result)
    if self.trigger_callback:
        try:
            self.trigger_callback(camera_id, dtype, frame, result)
        except Exception as e:
            logger.error(f"Trigger callback error: {e}")

    if self.camera_manager is not None:
        self.camera_manager.clear_detection_frames(camera_id, dtype)
```

关键变化：

- 在 threshold 检查之前插入 ROI 过滤和 box_count 判断
- 删除原有 477-480 行的 `is_in_cooldown` 检查（已前置到 `_get_due_types`）
- 注意 `max_box_count` 场景下 `check_box_count` 可能把 `detected` 从 False 改为 True（离岗检测：0 人 → 报警），此时 `max_conf` 为 0，需要跳过 threshold 检查。修改 threshold 判断：

```python
    # 对 max_box_count 模式（如离岗：0人报警），0 个框也算检测到，跳过 threshold
    has_max_box_trigger = (schedule.max_box_count is not None
                          and len(result.get("boxes", [])) <= schedule.max_box_count
                          and detected)

    if not detected or (max_conf < schedule.threshold and not has_max_box_trigger):
```

- [ ] **Step 7: 写测试文件 tests/test_detector_core_registry.py**

```python
"""detector_core.py 注册表改造测试"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass


# ---------- filter_by_roi ----------

class TestFilterByRoi:
    def setup_method(self):
        from backend.safety_detection.detector_core import filter_by_roi
        self.filter_by_roi = filter_by_roi

    def test_no_roi_returns_unchanged(self):
        result = {"detected": True, "boxes": [[10, 10, 50, 50]], "scores": [0.9]}
        out = self.filter_by_roi(result, None, False, 640, 480)
        assert out is result

    def test_box_inside_roi_kept(self):
        result = {"detected": True, "boxes": [[100, 100, 200, 200]], "scores": [0.9]}
        roi = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        out = self.filter_by_roi(result, roi, False, 640, 480)
        assert len(out["boxes"]) == 1
        assert out["detected"] is True

    def test_box_outside_roi_removed(self):
        result = {"detected": True, "boxes": [[500, 400, 600, 460]], "scores": [0.9]}
        roi = [[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]]
        out = self.filter_by_roi(result, roi, False, 640, 480)
        assert len(out["boxes"]) == 0
        assert out["detected"] is False

    def test_roi_invert_keeps_outside(self):
        result = {"detected": True, "boxes": [[500, 400, 600, 460]], "scores": [0.9]}
        roi = [[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]]
        out = self.filter_by_roi(result, roi, True, 640, 480)
        assert len(out["boxes"]) == 1

    def test_subjects_synced_with_boxes(self):
        result = {
            "detected": True,
            "boxes": [[10, 10, 50, 50], [500, 400, 600, 460]],
            "scores": [0.9, 0.8],
            "subjects": [{"sleeping": True}, {"sleeping": False}],
        }
        roi = [[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]]
        out = self.filter_by_roi(result, roi, False, 640, 480)
        assert len(out["subjects"]) == 1
        assert out["subjects"][0]["sleeping"] is True


# ---------- check_box_count ----------

class TestCheckBoxCount:
    def setup_method(self):
        from backend.safety_detection.detector_core import check_box_count
        self.check_box_count = check_box_count

    def test_no_limits_unchanged(self):
        result = {"detected": True, "boxes": [[1, 1, 2, 2]], "scores": [0.9]}
        out = self.check_box_count(result)
        assert out["detected"] is True

    def test_min_box_count_blocks(self):
        result = {"detected": True, "boxes": [[1, 1, 2, 2]], "scores": [0.9]}
        out = self.check_box_count(result, min_box_count=3)
        assert out["detected"] is False

    def test_min_box_count_passes(self):
        result = {"detected": True, "boxes": [[1, 1, 2, 2], [3, 3, 4, 4], [5, 5, 6, 6]], "scores": [0.9, 0.8, 0.7]}
        out = self.check_box_count(result, min_box_count=3)
        assert out["detected"] is True

    def test_max_box_count_zero_person_absent(self):
        """离岗检测：0 人 → detected=True"""
        result = {"detected": False, "boxes": [], "scores": []}
        out = self.check_box_count(result, max_box_count=0)
        assert out["detected"] is True

    def test_max_box_count_exceeded(self):
        """人数超限场景"""
        result = {"detected": True, "boxes": [[1, 1, 2, 2]] * 5, "scores": [0.9] * 5}
        out = self.check_box_count(result, max_box_count=3)
        assert out["detected"] is False

    def test_min_and_max_range(self):
        """区间检测：min=1, max=3 → 0 人或 ≥4 人报警"""
        result_0 = {"detected": False, "boxes": [], "scores": []}
        assert self.check_box_count(result_0, min_box_count=1, max_box_count=3)["detected"] is False

        result_2 = {"detected": True, "boxes": [[1, 1, 2, 2]] * 2, "scores": [0.9] * 2}
        assert self.check_box_count(result_2, min_box_count=1, max_box_count=3)["detected"] is True


# ---------- _annotate_frame 使用注册表 ----------

class TestAnnotateFrameRegistry:
    def test_uses_registry_color(self):
        from backend.safety_detection.detector_core import MultiDetector

        mock_registry = MagicMock()
        mock_registry.get.return_value = {
            "label": "明火", "color": "#ef4444", "post_process": "yolo_box"
        }
        mock_registry.get_color_bgr.return_value = (68, 68, 239)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = {"fire": {"detected": True, "boxes": [[100, 100, 200, 200]], "scores": [0.9]}}

        with patch("backend.detection_registry.registry", mock_registry):
            annotated = MultiDetector._annotate_frame(frame, results)
            assert annotated is not frame
            mock_registry.get_color_bgr.assert_called_with("fire")

    def test_unknown_type_fallback(self):
        from backend.safety_detection.detector_core import MultiDetector

        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = {"unknown_type": {"detected": True, "boxes": [[10, 10, 50, 50]], "scores": [0.8]}}

        with patch("backend.detection_registry.registry", mock_registry):
            annotated = MultiDetector._annotate_frame(frame, results)
            assert annotated is not frame

    def test_pose_type_draws_skeleton(self):
        from backend.safety_detection.detector_core import MultiDetector

        mock_registry = MagicMock()
        mock_registry.get.return_value = {
            "label": "睡岗", "color": "#eab308", "post_process": "yolo_pose"
        }
        mock_registry.get_color_bgr.return_value = (8, 179, 234)

        kpts = [(100 + i * 10, 100 + i * 5, 0.9) for i in range(17)]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = {
            "sleep": {
                "detected": True,
                "boxes": [[50, 50, 250, 350]],
                "scores": [0.85],
                "subjects": [{"sleeping": True, "keypoints": kpts}],
            }
        }

        with patch("backend.detection_registry.registry", mock_registry):
            annotated = MultiDetector._annotate_frame(frame, results)
            assert not np.array_equal(annotated, frame)


# ---------- _get_due_types 冷却前置 ----------

class TestGetDueTypesCooldown:
    def test_cooldown_type_excluded(self):
        from backend.safety_detection.detector_core import MultiDetector, TypeSchedule
        import time

        md = MultiDetector.__new__(MultiDetector)
        md._lock = __import__("threading").RLock()
        md._schedules = {}
        md._cooldowns = {}

        now = time.time()
        s = TypeSchedule(dtype="fire", enabled=True, interval=1, threshold=0.5, cooldown=60)
        s.last_run = 0
        md._schedules["cam1"] = {"fire": s}
        md._cooldowns["cam1"] = {"fire": now - 10}

        due = md._get_due_types("cam1", now)
        assert "fire" not in due

    def test_non_cooldown_type_included(self):
        from backend.safety_detection.detector_core import MultiDetector, TypeSchedule
        import time

        md = MultiDetector.__new__(MultiDetector)
        md._lock = __import__("threading").RLock()
        md._schedules = {}
        md._cooldowns = {}

        now = time.time()
        s = TypeSchedule(dtype="fire", enabled=True, interval=1, threshold=0.5, cooldown=60)
        s.last_run = 0
        md._schedules["cam1"] = {"fire": s}
        md._cooldowns["cam1"] = {"fire": now - 120}

        due = md._get_due_types("cam1", now)
        assert "fire" in due


# ---------- _handle_standard_detection ROI + box_count ----------

class TestHandleDetectionRoiBoxCount:
    def _make_detector(self):
        from backend.safety_detection.detector_core import MultiDetector, TypeSchedule
        import threading
        from collections import defaultdict

        md = MultiDetector.__new__(MultiDetector)
        md._lock = threading.RLock()
        md._schedules = {}
        md._cooldowns = defaultdict(dict)
        md._alert_states = defaultdict(dict)
        md._latest_results = {}
        md.camera_manager = None
        md.trigger_callback = None
        md.vlm_queue = None
        return md

    def test_roi_filters_before_threshold(self):
        from backend.safety_detection.detector_core import TypeSchedule

        md = self._make_detector()
        s = TypeSchedule(
            dtype="fire", enabled=True, interval=1, threshold=0.5, cooldown=60,
            roi=[[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]],
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = {"detected": True, "boxes": [[300, 300, 400, 400]], "scores": [0.9]}

        md._handle_standard_detection("cam1", "fire", frame, result, s)
        assert s.consecutive_count == 0

    def test_box_count_min_blocks_single_box(self):
        from backend.safety_detection.detector_core import TypeSchedule

        md = self._make_detector()
        s = TypeSchedule(
            dtype="person", enabled=True, interval=1, threshold=0.3, cooldown=60,
            min_box_count=5,
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = {"detected": True, "boxes": [[100, 100, 200, 200]], "scores": [0.9]}

        md._handle_standard_detection("cam1", "person", frame, result, s)
        assert s.consecutive_count == 0
```

- [ ] **Step 8: 确保 _lock 是 RLock**

检查 `MultiDetector.__init__` 中 `self._lock` 的类型。如果是 `threading.Lock()`，改为 `threading.RLock()`，因为 `_get_due_types` 现在在持有 `_lock` 的情况下调用 `_is_in_cooldown_unlocked`。如果保持原始的 `is_in_cooldown`（也加锁），就会死锁。

在 `__init__` 中找到 `self._lock = threading.Lock()` 改为 `self._lock = threading.RLock()`。

- [ ] **Step 9: Run tests**

Run: `python -m pytest tests/test_detector_core_registry.py -v`
Expected: all PASS

- [ ] **Step 10: Run full test suite for regression**

Run: `python -m pytest tests/ -v`
Expected: all existing tests still PASS, new tests PASS

- [ ] **Step 11: Commit**

```bash
git add backend/safety_detection/detector_core.py tests/test_detector_core_registry.py
git commit -m "refactor: detector_core uses registry for colors/labels, adds ROI filter and box_count"
```

---

### Task 5: main_multi.py 改造

**Files:**

- Modify: `backend/main_multi.py:227-277` — `_convert_ultralytics_result` 改用注册表判断 pose 类型
- Modify: `backend/main_multi.py:416` — 删除硬编码 import（`MASK_TARGET_CLASSES` 等）
- Modify: `backend/main_multi.py:436-452` — `model_configs` 从注册表动态生成
- Modify: `backend/main_multi.py:1141-1142` — `DEFAULT_TYPE_CONFIG` 引用改为注册表
- Test: `tests/test_main_multi_registry.py`

**Interfaces:**

- Consumes: `backend/detection_registry.py` (Task 1) — `registry.all_types()`, `registry.get(dtype)`, `registry.get_defaults(dtype)`
- Consumes: `backend/inference_engine.py` (Task 2) — 改造后的 `_resolve_model_path(dtype, use_npu)` 从注册表读模型路径
- Produces: 无新公开接口，改造内部启动逻辑

- [ ] **Step 1: 改造 _convert_ultralytics_result**

将 `dtype == "sleep"` 硬编码判断改为注册表的 `post_process` 查询：

```python
def _convert_ultralytics_result(dtype: str, result) -> Optional[dict]:
    """将 ultralytics Results 转换为 SafetyDetector 风格的 dict"""
    from detection_registry import registry

    type_def = registry.get(dtype)
    is_pose = type_def.get("post_process") == "yolo_pose" if type_def else False

    if result is None or result.boxes is None or len(result.boxes) == 0:
        if is_pose:
            return {"detected": False, "boxes": [], "scores": [], "subjects": [], "count": 0}
        return {"detected": False, "boxes": [], "scores": [], "max_confidence": 0.0}

    boxes = []
    scores = []
    for b in result.boxes:
        boxes.append(list(map(int, b.xyxy[0])))
        scores.append(float(b.conf[0]))

    if is_pose:
        subjects = []
        detected = False
        count = 0
        if result.keypoints is not None and result.keypoints.data is not None:
            for i in range(len(result.boxes)):
                bbox = result.boxes.xyxy[i].cpu().numpy()
                kp = result.keypoints.data[i].cpu().numpy()
                if len(kp) >= 17:
                    from safety_detection.sleep_detect import analyze_sleep
                    info = analyze_sleep(kp, bbox)
                    subjects.append({
                        "box": bbox.tolist(),
                        "score": float(result.boxes.conf[i]),
                        "sleeping": info["is_sleeping"],
                        "posture_label": info["posture_label"],
                        "sleep_confidence": info["sleep_confidence"],
                        "keypoints": kp,
                    })
                    if info["is_sleeping"]:
                        detected = True
                        count += 1
        return {
            "detected": detected,
            "boxes": boxes,
            "scores": scores,
            "subjects": subjects,
            "count": count,
            "max_confidence": max(scores) if scores else 0.0,
        }

    max_conf = max(scores) if scores else 0.0
    return {
        "detected": len(boxes) > 0,
        "boxes": boxes,
        "scores": scores,
        "max_confidence": max_conf,
    }
```

- [ ] **Step 2: 改造 GPU scheduler model_configs 生成**

将硬编码的 `model_configs` 块（436-452 行）改为从注册表动态生成：

```python
    if use_gpu_scheduler and device == "gpu":
        try:
            from gpu_scheduler import ModelConfig, GPUDynamicScheduler
            from inference_engine import _resolve_model_path
            from detection_registry import registry

            def _gpu_on_result(cam_id: str, dtype: str, result):
                # ... 保持不变 ...

            model_configs = {}
            seen_models = set()
            for dtype in registry.all_types():
                type_def = registry.get(dtype)
                model_file = type_def["model_path"]
                model_path = _resolve_model_path(dtype, use_npu=False)
                if not model_path:
                    continue
                classes = type_def.get("classes")
                confidence = type_def.get("model_confidence", 0.5)
                model_configs[dtype] = ModelConfig(
                    model_path, dtype, device="cuda",
                    confidence=confidence,
                    classes=classes,
                )
```

关键变化：

- 删除 `MASK_TARGET_CLASSES, CIGARETTE_TARGET_CLASSES, UNIFORM_TARGET_CLASSES` import
- 6 个 if/else 块替换为单个循环
- `classes` 和 `confidence` 来自注册表

- [ ] **Step 3: 改造恢复默认值中的 DEFAULT_TYPE_CONFIG 引用**

在 `restore_camera_defaults` 端点（约 1141 行）：

```python
    # 原来：
    restored["detection_types"] = {
        k: dict(v) for k, v in camera_globals.get("detection_types", app_config.DEFAULT_TYPE_CONFIG).items()
    }

    # 改为：
    from detection_registry import registry
    fallback_dt = {dtype: registry.get_defaults(dtype) for dtype in registry.all_types()}
    restored["detection_types"] = {
        k: dict(v) for k, v in camera_globals.get("detection_types", fallback_dt).items()
    }
```

- [ ] **Step 4: 写测试文件 tests/test_main_multi_registry.py**

```python
"""main_multi.py 注册表改造测试"""

import pytest
from unittest.mock import patch, MagicMock


class TestConvertUltralyticsResult:
    def test_yolo_box_empty_result(self):
        from backend.main_multi import _convert_ultralytics_result

        mock_registry = MagicMock()
        mock_registry.get.return_value = {"post_process": "yolo_box"}

        with patch("backend.main_multi.registry", mock_registry):
            out = _convert_ultralytics_result("fire", None)
            assert out["detected"] is False
            assert "subjects" not in out

    def test_yolo_pose_empty_result(self):
        from backend.main_multi import _convert_ultralytics_result

        mock_registry = MagicMock()
        mock_registry.get.return_value = {"post_process": "yolo_pose"}

        with patch("backend.main_multi.registry", mock_registry):
            out = _convert_ultralytics_result("sleep", None)
            assert out["detected"] is False
            assert out["subjects"] == []

    def test_unknown_type_treated_as_box(self):
        from backend.main_multi import _convert_ultralytics_result

        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        with patch("backend.main_multi.registry", mock_registry):
            out = _convert_ultralytics_result("new_type", None)
            assert out["detected"] is False
            assert "subjects" not in out


class TestGpuModelConfigsFromRegistry:
    def test_model_configs_built_from_registry(self):
        """验证 model_configs 可以从注册表动态构建"""
        from backend.detection_registry import registry

        mock_reg = MagicMock()
        mock_reg.all_types.return_value = ["fire", "smoke", "mask"]
        mock_reg.get.side_effect = lambda dtype: {
            "fire": {"model_path": "fire_smoke.pt", "classes": [0], "model_confidence": 0.5},
            "smoke": {"model_path": "fire_smoke.pt", "classes": [1], "model_confidence": 0.5},
            "mask": {"model_path": "mask.pt", "classes": [1], "model_confidence": 0.5},
        }[dtype]

        configs = {}
        for dtype in mock_reg.all_types():
            type_def = mock_reg.get(dtype)
            configs[dtype] = {
                "model_path": type_def["model_path"],
                "classes": type_def.get("classes"),
                "confidence": type_def.get("model_confidence", 0.5),
            }

        assert len(configs) == 3
        assert configs["fire"]["classes"] == [0]
        assert configs["smoke"]["classes"] == [1]
        assert configs["fire"]["model_path"] == configs["smoke"]["model_path"]
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_main_multi_registry.py -v`
Expected: all PASS

- [ ] **Step 6: Run full test suite for regression**

Run: `python -m pytest tests/ -v`
Expected: all existing tests still PASS

- [ ] **Step 7: Commit**

```bash
git add backend/main_multi.py tests/test_main_multi_registry.py
git commit -m "refactor: main_multi.py builds GPU model_configs from registry"
```

---

### Task 6: understander.py 改造

**Files:**

- Modify: `backend/understander.py:297-318` — `_build_inspection_prompt` 的 `type_desc` 从注册表读取
- Test: `tests/test_understander_registry.py`

**Interfaces:**

- Consumes: `backend/detection_registry.py` (Task 1) — `registry.get(dtype)` 返回的 `inspection_label` 字段
- Produces: 无新公开接口

**说明：** `_FALLBACK_PROMPT_TEMPLATES` 保持不变（兜底模板），不从注册表读取。只改造 `_build_inspection_prompt` 中的 `type_desc` 硬编码字典。

- [ ] **Step 1: 改造 _build_inspection_prompt**

将硬编码 `type_desc` 替换为注册表查询：

```python
def _build_inspection_prompt(self, extra_context: dict) -> str:
    """动态构建巡检 prompt"""
    from detection_registry import registry

    types = extra_context.get("enabled_types", [])
    checks = []
    for t in types:
        type_def = registry.get(t)
        desc = type_def.get("inspection_label", t) if type_def else t
        checks.append(f"- {desc}")
    checks_str = "\n".join(checks)
    detections_json = "\n".join(
        [f'    "{t}": {{"detected": true/false, "confidence": 0.0-1.0, "reason": "判断理由"}}' for t in types]
    )

    template = _get_prompt_template("inspection")
    return template.format(
        enabled_types_desc=checks_str,
        detections_json=detections_json,
    )
```

关键变化：

- 删除硬编码 `type_desc` 字典
- 从 `registry.get(t)["inspection_label"]` 读取中文描述
- 未注册类型 fallback 到 `t`（类型 key 本身）

- [ ] **Step 2: 写测试文件 tests/test_understander_registry.py**

```python
"""understander.py 注册表改造测试"""

import pytest
from unittest.mock import patch, MagicMock


class TestBuildInspectionPromptRegistry:
    def test_uses_registry_inspection_label(self):
        from backend.understander import VideoUnderstander

        mock_registry = MagicMock()
        mock_registry.get.side_effect = lambda dtype: {
            "fire": {"inspection_label": "明火"},
            "smoke": {"inspection_label": "烟雾"},
        }.get(dtype)

        vu = VideoUnderstander.__new__(VideoUnderstander)

        with patch("backend.understander.registry", mock_registry):
            prompt = vu._build_inspection_prompt({"enabled_types": ["fire", "smoke"]})
            assert "明火" in prompt
            assert "烟雾" in prompt

    def test_unknown_type_uses_key_as_fallback(self):
        from backend.understander import VideoUnderstander

        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        vu = VideoUnderstander.__new__(VideoUnderstander)

        with patch("backend.understander.registry", mock_registry):
            prompt = vu._build_inspection_prompt({"enabled_types": ["new_type"]})
            assert "new_type" in prompt

    def test_new_type_with_inspection_label(self):
        from backend.understander import VideoUnderstander

        mock_registry = MagicMock()
        mock_registry.get.return_value = {"inspection_label": "未戴安全帽"}

        vu = VideoUnderstander.__new__(VideoUnderstander)

        with patch("backend.understander.registry", mock_registry):
            prompt = vu._build_inspection_prompt({"enabled_types": ["helmet"]})
            assert "未戴安全帽" in prompt
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_understander_registry.py -v`
Expected: all PASS

- [ ] **Step 4: Run full test suite for regression**

Run: `python -m pytest tests/ -v`
Expected: all existing tests still PASS

- [ ] **Step 5: Commit**

```bash
git add backend/understander.py tests/test_understander_registry.py
git commit -m "refactor: understander reads inspection_label from registry"
```

---

### Task 7: api.py — 新增 /detector/types CRUD 端点

**Files:**

- Modify: `backend/safety_detection/api.py` — 新增 GET/PUT `/detector/types` 端点
- Test: `tests/test_detector_types_api.py`

**Interfaces:**

- Consumes: `backend/detection_registry.py` (Task 1) — `registry.to_api_list()`, `registry.get(dtype)`, `registry.update_defaults(dtype, overrides)`, `registry.validate()`
- Produces: API 端点
  - `GET /detector/types` — 返回 `{"types": [...]}`
  - `GET /detector/types/{dtype}` — 返回单个类型定义
  - `PUT /detector/types/{dtype}` — 更新 defaults 运行参数

- [ ] **Step 1: 实现 GET /detector/types 端点**

在 `backend/safety_detection/api.py` 新增：

```python
from backend.detection_registry import registry


@router.get("/detector/types")
async def list_detection_types():
    """获取所有检测类型定义"""
    return {"types": registry.to_api_list()}


@router.get("/detector/types/{dtype}")
async def get_detection_type(dtype: str):
    """获取单个检测类型定义"""
    type_def = registry.get(dtype)
    if type_def is None:
        return JSONResponse({"error": f"Unknown detection type: {dtype}"}, status_code=404)
    return {
        "key": dtype,
        "label": type_def.get("label", dtype),
        "color": type_def.get("color", "#888888"),
        "icon": type_def.get("icon", ""),
        "post_process": type_def.get("post_process", "yolo_box"),
        "defaults": type_def.get("defaults", {}),
    }
```

- [ ] **Step 2: 实现 PUT /detector/types/{dtype} 端点**

第一期只允许修改 `defaults` 中的运行参数，不允许修改 `model_path`、`post_process` 等结构性字段：

```python
@router.put("/detector/types/{dtype}")
async def update_detection_type(dtype: str, data: dict):
    """更新检测类型的默认运行参数"""
    type_def = registry.get(dtype)
    if type_def is None:
        return JSONResponse({"error": f"Unknown detection type: {dtype}"}, status_code=404)

    allowed_keys = {"enabled", "interval", "threshold", "consecutive_required",
                    "cooldown", "use_vlm", "min_box_count", "max_box_count"}
    defaults_update = {k: v for k, v in data.items() if k in allowed_keys}

    if not defaults_update:
        return JSONResponse({"error": "No valid fields to update"}, status_code=400)

    registry.update_defaults(dtype, defaults_update)
    return {"success": True, "dtype": dtype, "defaults": registry.get_defaults(dtype)}
```

- [ ] **Step 3: 写测试文件 tests/test_detector_types_api.py**

```python
"""detector types API 端点测试"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.safety_detection.api import router
from fastapi import FastAPI


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestListDetectionTypes:
    def test_returns_types_list(self, client):
        mock_registry = MagicMock()
        mock_registry.to_api_list.return_value = [
            {"key": "fire", "label": "明火", "color": "#ef4444", "icon": "flame",
             "post_process": "yolo_box", "defaults": {"enabled": False}},
        ]

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.get("/detector/types")
            assert resp.status_code == 200
            data = resp.json()
            assert "types" in data
            assert len(data["types"]) == 1
            assert data["types"][0]["key"] == "fire"


class TestGetDetectionType:
    def test_existing_type(self, client):
        mock_registry = MagicMock()
        mock_registry.get.return_value = {
            "label": "明火", "color": "#ef4444", "icon": "flame",
            "post_process": "yolo_box", "defaults": {"enabled": False},
        }

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.get("/detector/types/fire")
            assert resp.status_code == 200
            assert resp.json()["key"] == "fire"

    def test_unknown_type_404(self, client):
        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.get("/detector/types/nonexistent")
            assert resp.status_code == 404


class TestUpdateDetectionType:
    def test_update_defaults(self, client):
        mock_registry = MagicMock()
        mock_registry.get.return_value = {"defaults": {"enabled": False, "threshold": 0.5}}
        mock_registry.get_defaults.return_value = {"enabled": True, "threshold": 0.8}

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.put("/detector/types/fire", json={"enabled": True, "threshold": 0.8})
            assert resp.status_code == 200
            assert resp.json()["success"] is True
            mock_registry.update_defaults.assert_called_once()

    def test_structural_fields_ignored(self, client):
        mock_registry = MagicMock()
        mock_registry.get.return_value = {"defaults": {"enabled": False}}

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.put("/detector/types/fire", json={"model_path": "evil.pt"})
            assert resp.status_code == 400

    def test_unknown_type_404(self, client):
        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        with patch("backend.safety_detection.api.registry", mock_registry):
            resp = client.put("/detector/types/nope", json={"enabled": True})
            assert resp.status_code == 404
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_detector_types_api.py -v`
Expected: all PASS

- [ ] **Step 5: Run full test suite for regression**

Run: `python -m pytest tests/ -v`
Expected: all existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add backend/safety_detection/api.py tests/test_detector_types_api.py
git commit -m "feat: add GET/PUT /detector/types API endpoints"
```

---

### Task 8: 前端改造 — shared.js + HTML 页面

**Files:**

- Modify: `frontend/safety_detection/shared.js:65-131` — `DETECTION_TYPES` 和 `defaultDetectionTypes()` 改为动态加载
- Modify: `frontend/safety_detection/settings.html` — 检测类型配置表单从动态 `DETECTION_TYPES` 渲染
- Modify: `frontend/safety_detection/monitor.html` — 类型筛选按钮从动态 `DETECTION_TYPES` 渲染
- Modify: `frontend/safety_detection/records.html` — 类型筛选下拉从动态 `DETECTION_TYPES` 渲染

**Interfaces:**

- Consumes: `GET /detector/types` (Task 7) — 返回 `{"types": [{"key", "label", "color", "icon", "defaults"}]}`
- Produces: 前端全局 `DETECTION_TYPES` 数组和 `defaultDetectionTypes()` 函数，自动响应注册表变化

- [ ] **Step 1: shared.js — 添加动态加载逻辑**

将 `DETECTION_TYPES` 从硬编码常量改为可更新的变量，添加 `loadDetectionTypes()` 异步加载函数，保留内置 fallback：

```javascript
// 内置默认值（API 不可用时的 fallback）
const _BUILTIN_DETECTION_TYPES = [
    { key: 'fire', label: '明火', color: '#ef4444' },
    { key: 'smoke', label: '烟雾', color: '#f97316' },
    { key: 'uniform', label: '工服', color: '#22c55e' },
    { key: 'mask', label: '口罩', color: '#0ea5e9' },
    { key: 'cigarette', label: '吸烟', color: '#a855f7' },
    { key: 'sleep', label: '睡岗', color: '#eab308' },
];

let DETECTION_TYPES = [..._BUILTIN_DETECTION_TYPES];

async function loadDetectionTypes() {
    try {
        const resp = await fetch('/detector/types');
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        if (data.types && data.types.length > 0) {
            DETECTION_TYPES = data.types.map(t => ({
                key: t.key,
                label: t.label,
                color: t.color,
                icon: t.icon || '',
                defaults: t.defaults || {},
            }));
        }
    } catch (e) {
        console.warn('Failed to load detection types from API, using builtin defaults:', e.message);
    }
    return DETECTION_TYPES;
}
```

- [ ] **Step 2: shared.js — 改造 defaultDetectionTypes()**

从动态加载的 `DETECTION_TYPES` 生成默认配置，不再硬编码：

```javascript
function defaultDetectionTypes() {
    const result = {};
    for (const t of DETECTION_TYPES) {
        if (t.defaults) {
            result[t.key] = { ...t.defaults };
        } else {
            result[t.key] = {
                enabled: false, interval: 1, threshold: 0.5,
                consecutive_required: 3, cooldown: 60, use_vlm: false,
                min_box_count: 1, max_box_count: null,
            };
        }
    }
    return result;
}
```

- [ ] **Step 3: settings.html — 在 Vue mounted 中调用 loadDetectionTypes()**

在 `settings.html` 的 Vue `mounted` 钩子（或 `created`）中，页面初始化时先调用 `await loadDetectionTypes()`，再执行原有的数据加载逻辑。已有的 `v-for="t in DETECTION_TYPES"` 循环不需要改动，因为 `DETECTION_TYPES` 已经是动态数组。

如果 settings.html 的 Vue data 中持有 `detectionTypes` 的本地副本（如 `detectionTypes: DETECTION_TYPES`），需要在 `loadDetectionTypes()` 完成后同步更新：

```javascript
async mounted() {
    await loadDetectionTypes();
    this.detectionTypes = DETECTION_TYPES;
    // ... 原有初始化逻辑 ...
}
```

- [ ] **Step 4: monitor.html — 在初始化中调用 loadDetectionTypes()**

在 `monitor.html` 的 Vue 初始化中添加：

```javascript
async mounted() {
    await loadDetectionTypes();
    this.detectionTypes = DETECTION_TYPES;
    // ... 原有初始化逻辑 ...
}
```

如果 monitor.html 中 `detectionTypes` 是在 `data()` 中从 `DETECTION_TYPES` 初始化的，需要在 `mounted` 中重新赋值以获取动态加载后的值。

- [ ] **Step 5: records.html — 在初始化中调用 loadDetectionTypes()**

与 monitor.html 同理，在 records.html 的 Vue 初始化中：

```javascript
async mounted() {
    await loadDetectionTypes();
    // 如果有本地 detectionTypes 引用，同步更新
    // ... 原有初始化逻辑 ...
}
```

- [ ] **Step 6: 手动验证**

启动开发服务器后在浏览器中验证：

1. 打开 `/monitor` 页面 → 检测类型筛选按钮应显示所有注册表中的类型
2. 打开 `/settings` 页面 → 检测类型配置列表应包含所有注册表类型
3. 打开 `/records` 页面 → 类型筛选下拉应包含所有注册表类型
4. 在 `config/detection_types.json` 中新增一个测试类型（如 `helmet`），重启后端 → 前端应自动显示新类型
5. 停止后端 API → 前端页面仍然正常显示 6 个内置默认类型（fallback 生效）

- [ ] **Step 7: Commit**

```bash
git add frontend/safety_detection/shared.js frontend/safety_detection/settings.html frontend/safety_detection/monitor.html frontend/safety_detection/records.html
git commit -m "feat: frontend loads detection types dynamically from /detector/types API"
```
