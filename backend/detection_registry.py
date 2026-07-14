"""
检测类型注册表 — 配置驱动的类型定义管理

全局单例 `registry`，启动时从 config/detection_types.json 加载。
首次启动自动从 DEFAULT_DETECTION_TYPE_REGISTRY 生成配置文件。
"""

import copy
import json
import logging
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent.parent / "config"
REGISTRY_FILE = CONFIG_DIR / "detection_types.json"
PROJECT_ROOT = Path(__file__).parent.parent


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """'#ef4444' or 'ef4444' → (68, 68, 239)"""
    if not isinstance(hex_color, str):
        raise ValueError(f"hex_color must be a string, got {type(hex_color).__name__}")
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"hex_color must be 6 hex digits optionally preceded by '#', got {hex_color!r}")
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        raise ValueError(f"hex_color must contain valid hex digits, got {hex_color!r}")
    return (b, g, r)


UNIVERSAL_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "interval": 1,
    "threshold": 0.5,
    "consecutive_required": 3,
    "cooldown": 60,
    "use_vlm": False,
    "min_box_count": 1,
    "max_box_count": None,
}


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
            try:
                with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                    stored = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Registry file corrupted, regenerating from defaults")
                self._types = copy.deepcopy(DEFAULT_DETECTION_TYPE_REGISTRY)
                self._save(self._types)
                logger.info(f"Detection registry loaded: {list(self._types.keys())}")
                return
            merged = copy.deepcopy(DEFAULT_DETECTION_TYPE_REGISTRY)
            for dtype, type_def in stored.items():
                if dtype in merged:
                    merged[dtype].update(type_def)
                    base_defaults = DEFAULT_DETECTION_TYPE_REGISTRY[dtype]["defaults"]
                else:
                    merged[dtype] = type_def
                    base_defaults = UNIVERSAL_DEFAULTS
                merged[dtype].setdefault("defaults", {})
                for key, val in base_defaults.items():
                    merged[dtype]["defaults"].setdefault(key, val)
            if merged != stored:
                self._save(merged)
            self._types = merged
        else:
            self._types = copy.deepcopy(DEFAULT_DETECTION_TYPE_REGISTRY)
            self._save(self._types)

        logger.info(f"Detection registry loaded: {list(self._types.keys())}")

    def _save(self, data: dict) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get(self, dtype: str) -> dict | None:
        if dtype not in self._types:
            return None
        return self._types[dtype]

    def all_types(self) -> list[str]:
        return list(self._types.keys())

    def get_types_by_model(self, model_path: str) -> list[str]:
        return [dt for dt, td in self._types.items() if td.get("model_path") == model_path]

    def get_color_bgr(self, dtype: str) -> tuple[int, int, int]:
        td = self.get(dtype)
        if td is None:
            return (0, 255, 0)
        return hex_to_bgr(td["color"])

    def get_defaults(self, dtype: str) -> dict:
        td = self.get(dtype)
        if td is None:
            return {}
        return dict(td["defaults"])

    def merge_camera_config(self, dtype: str, overrides: dict) -> dict:
        """合并摄像头级覆盖到注册表默认值"""
        if self.get(dtype) is None:
            return dict(overrides)
        defaults = self.get_defaults(dtype)
        result = dict(defaults)
        for key, val in overrides.items():
            if key in result or key in ("roi", "roi_invert"):
                result[key] = val
        return result

    def validate(self) -> list[str]:
        """校验注册表（模型文件是否存在等），返回警告列表"""
        warnings = []
        models_dir = PROJECT_ROOT / "models"
        weights_dir = PROJECT_ROOT / "weights"
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
                "label": td.get("label", key),
                "color": td.get("color", "#888888"),
                "icon": td.get("icon", ""),
                "model_path": td.get("model_path"),
                "npu_model_path": td.get("npu_model_path"),
                "post_process": td.get("post_process", "yolo_box"),
                "classes": td.get("classes"),
                "model_confidence": td.get("model_confidence", 0.5),
                "vlm_prompt_key": td.get("vlm_prompt_key", ""),
                "inspection_label": td.get("inspection_label", td.get("label", key)),
                "defaults": dict(td.get("defaults", {})),
            })
        return result

    def update_defaults(self, dtype: str, new_defaults: dict) -> None:
        """更新类型的运行参数默认值并持久化"""
        td = self.get(dtype)
        if td is None:
            return
        allowed = set(DEFAULT_DETECTION_TYPE_REGISTRY.get(dtype, {}).get("defaults", {}).keys())
        if not allowed:
            allowed = set(td["defaults"].keys())
        for key, val in new_defaults.items():
            if key in allowed:
                td["defaults"][key] = val
        self._save(self._types)

    def add_type(self, type_def: dict) -> str:
        """新增检测类型，自动生成唯一 key，返回 key"""
        label = type_def.get("label", "").strip()
        if not label:
            raise ValueError("label is required")
        for existing in self._types.values():
            if existing.get("label") == label:
                raise ValueError(f"label '{label}' already exists")
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


registry = DetectionTypeRegistry()
registry.load()
