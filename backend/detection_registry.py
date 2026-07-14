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

    def get(self, dtype: str) -> dict | None:
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
