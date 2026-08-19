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

from backend.model_registry import model_registry

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent.parent / "config"
REGISTRY_FILE = CONFIG_DIR / "detection_types.json"
ALGORITHMS_FILE = CONFIG_DIR / "algorithms.json"
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
    "box_count_mode": None,
    "static_filter": False,
    "static_diff_threshold": 0.02,
}


FIRE_REVIEW_PROMPT = """你正在复核一个工业安全监控系统的火焰检测结果。
请仔细查看图片，判断画面中是否真的有明火。
注意排除以下误判情况：
- 红色灯光、红色物体反光
- 夕阳、晚霞
- 橙色安全帽或衣服

请以 JSON 格式返回：
{"confirmed": true/false, "confidence": 0.0-1.0, "reason": "判断理由"}"""

SMOKE_REVIEW_PROMPT = """你正在复核一个工业安全监控系统的烟雾检测结果。
请仔细查看图片，判断画面中是否真的有烟雾。
注意排除以下误判情况：
- 水蒸气、雾气
- 灰尘扬起
- 白色墙壁反光

请以 JSON 格式返回：
{"confirmed": true/false, "confidence": 0.0-1.0, "reason": "判断理由"}"""

MASK_REVIEW_PROMPT = """你正在复核一个工业安全监控系统的口罩佩戴检测结果。
请仔细查看图片，判断画面中是否真的有未佩戴口罩的人员。
注意排除以下情况：
- 人员正在喝水或用餐（暂时摘下）
- 人员手持物品遮挡面部
- 距离太远看不清

请以 JSON 格式返回：
{"confirmed": true/false, "confidence": 0.0-1.0, "reason": "判断理由"}"""

CIGARETTE_REVIEW_PROMPT = """你正在复核一个工业安全监控系统的吸烟行为检测结果。
请仔细查看图片，判断画面中是否真的有人正在吸烟。
注意排除以下情况：
- 手持笔、筷子等细长物体
- 人员只是在摸嘴或吃东西
- 画面模糊无法确认

请以 JSON 格式返回：
{"confirmed": true/false, "confidence": 0.0-1.0, "reason": "判断理由"}"""

UNIFORM_REVIEW_PROMPT = """你正在复核一个工业安全监控系统的工服/反光背心检测结果。
请仔细查看图片，判断画面中是否真的有未穿工服或反光背心的人员。
注意：
- 不同岗位工服颜色可能不同
- 只需判断是否有"未穿"的情况

请以 JSON 格式返回：
{"confirmed": true/false, "confidence": 0.0-1.0, "reason": "判断理由"}"""

SLEEP_REVIEW_PROMPT = """你正在复核一个工业安全监控系统的睡岗/打盹检测结果。
请仔细查看图片，判断画面中是否真的有人正在睡岗或打盹。
注意排除以下情况：
- 人员只是低头看手机或文件
- 人员闭目休息但时间很短
- 画面模糊无法确认

请以 JSON 格式返回：
{"confirmed": true/false, "confidence": 0.0-1.0, "reason": "判断理由"}"""


DEFAULT_DETECTION_TYPE_REGISTRY: dict[str, dict[str, Any]] = {
    "fire": {
        "label": "明火",
        "color": "#ef4444",
        "model_path": "fire_smoke.pt",
        "post_process": "yolo_box",
        "classes": [0],
        "model_confidence": 0.5,
        "vlm_prompt": FIRE_REVIEW_PROMPT,
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
            "box_count_mode": None,
            "static_filter": True,
            "static_diff_threshold": 0.02,
        },
    },
    "smoke": {
        "label": "烟雾",
        "color": "#f97316",
        "model_path": "fire_smoke.pt",
        "post_process": "yolo_box",
        "classes": [1],
        "model_confidence": 0.5,
        "vlm_prompt": SMOKE_REVIEW_PROMPT,
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
            "box_count_mode": None,
            "static_filter": True,
            "static_diff_threshold": 0.02,
        },
    },
    "uniform": {
        "label": "工服",
        "color": "#22c55e",
        "model_path": "uniform.pt",
        "post_process": "yolo_box",
        "classes": [1],
        "model_confidence": 0.5,
        "vlm_prompt": UNIFORM_REVIEW_PROMPT,
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
            "box_count_mode": None,
            "static_filter": False,
            "static_diff_threshold": 0.02,
        },
    },
    "mask": {
        "label": "口罩",
        "color": "#0ea5e9",
        "model_path": "mask.pt",
        "post_process": "yolo_box",
        "classes": [1],
        "model_confidence": 0.5,
        "vlm_prompt": MASK_REVIEW_PROMPT,
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
            "box_count_mode": None,
            "static_filter": False,
            "static_diff_threshold": 0.02,
        },
    },
    "cigarette": {
        "label": "吸烟",
        "color": "#a855f7",
        "model_path": "cigarette.pt",
        "post_process": "yolo_box",
        "classes": [0],
        "model_confidence": 0.5,
        "vlm_prompt": CIGARETTE_REVIEW_PROMPT,
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
            "box_count_mode": None,
            "static_filter": False,
            "static_diff_threshold": 0.02,
        },
    },
    "sleep": {
        "label": "睡岗",
        "color": "#eab308",
        "model_path": "yolov8n-pose.pt",
        "post_process": "yolo_pose",
        "classes": None,
        "model_confidence": 0.1,
        "vlm_prompt": SLEEP_REVIEW_PROMPT,
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
            "box_count_mode": None,
            "static_filter": False,
            "static_diff_threshold": 0.02,
        },
    },
}


def _migrate_type_dicts(stored: dict) -> dict:
    """类型 dict（model_path 版）→ 算法 dict（model_key 版），按 model_path 去重注册模型"""
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
    algorithms = {}
    for dtype, td in stored.items():
        algo = {k: v for k, v in td.items() if k != "model_path"}
        algo["model_key"] = key_by_path.get(td.get("model_path"))
        algorithms[dtype] = algo
    return algorithms


def migrate_legacy_registry() -> bool:
    """旧 detection_types.json → models.json + algorithms.json（旧文件改名 .bak）。

    algorithms.json 已存在则跳过；无旧文件时从内置默认类型迁移（全新安装播种）。
    """
    if ALGORITHMS_FILE.exists():
        return False
    rename_legacy = False
    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                stored = json.load(f)
        except json.JSONDecodeError:
            return False
        rename_legacy = True
    else:
        stored = copy.deepcopy(DEFAULT_DETECTION_TYPE_REGISTRY)
    algorithms = _migrate_type_dicts(stored)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(ALGORITHMS_FILE, "w", encoding="utf-8") as f:
        json.dump(algorithms, f, ensure_ascii=False, indent=2)
    if rename_legacy:
        REGISTRY_FILE.rename(REGISTRY_FILE.with_suffix(".json.bak"))
    logger.info(f"Migrated {len(algorithms)} types to algorithms, models seeded")
    return True


class DetectionTypeRegistry:
    """检测类型注册表，全局单例"""

    def __init__(self):
        self._types: dict[str, dict[str, Any]] = {}

    def load(self) -> None:
        """加载注册表：文件存在则读取并补全缺失字段，不存在则从默认值生成"""
        migrate_legacy_registry()
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        if ALGORITHMS_FILE.exists():
            try:
                with open(ALGORITHMS_FILE, "r", encoding="utf-8") as f:
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

        # 动态注入 model_path（由 model_key 解析），并清理已废弃字段
        for td in self._types.values():
            td.pop("icon", None)
            td.pop("vlm_prompt_key", None)
            mkey = td.get("model_key")
            model = model_registry.get(mkey) if mkey else None
            td["model_path"] = model["file"] if model else td.get("model_path")

        logger.info(f"Detection registry loaded: {list(self._types.keys())}")

    def _save(self, data: dict) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(ALGORITHMS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get(self, dtype: str) -> dict | None:
        td = self._types.get(dtype)
        if td is None:
            return None
        result = dict(td)
        model = model_registry.get(td.get("model_key") or "")
        result["model_path"] = model["file"] if model else None
        return result

    def all_types(self) -> list[str]:
        return list(self._types.keys())

    def get_types_by_model(self, model_key: str) -> list[str]:
        """按 model_key 找类型（推理去重用）"""
        return [dt for dt, td in self._types.items() if td.get("model_key") == model_key]

    def get_model_keys_in_use(self) -> set[str]:
        return {td.get("model_key") for td in self._types.values() if td.get("model_key")}

    def get_model_usage_counts(self) -> dict[str, int]:
        """每个模型被多少个算法引用（模型管理页 used_by 展示用）"""
        counts: dict[str, int] = {}
        for td in self._types.values():
            mk = td.get("model_key")
            if mk:
                counts[mk] = counts.get(mk, 0) + 1
        return counts

    def get_color_bgr(self, dtype: str) -> tuple[int, int, int]:
        td = self._types.get(dtype)
        if td is None:
            return (0, 255, 0)
        return hex_to_bgr(td["color"])

    def get_defaults(self, dtype: str) -> dict:
        td = self._types.get(dtype)
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
        for dtype, td in self._types.items():
            mkey = td.get("model_key")
            if mkey and not model_registry.file_exists(mkey):
                warnings.append(f"{dtype}: model '{mkey}' file not found in weights/")
            if td.get("post_process") not in ("yolo_box", "yolo_pose"):
                warnings.append(f"{dtype}: unknown post_process '{td.get('post_process')}'")
        return warnings

    def to_api_list(self) -> list[dict]:
        """返回前端 API 格式的类型列表"""
        result = []
        for key in self._types:
            td = self.get(key)
            result.append({
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
                "alarm_description": td.get("alarm_description", ""),
                "defaults": dict(td.get("defaults", {})),
            })
        return result

    def update_defaults(self, dtype: str, new_defaults: dict) -> None:
        """更新类型的运行参数默认值并持久化"""
        td = self._types.get(dtype)
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
            "alarm_description": type_def.get("alarm_description", ""),
            "defaults": merged,
        }
        self._save(self._types)
        return key

    def update_type(self, dtype: str, updates: dict) -> None:
        """更新检测类型的结构性字段（不更新 defaults）"""
        td = self._types.get(dtype)
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
        if "model_key" in updates:
            if model_registry.get(updates["model_key"] or "") is None:
                raise ValueError(f"Unknown model: {updates['model_key']}")
            td["model_key"] = updates["model_key"]
            td["post_process"] = model_registry.get(updates["model_key"]).get("post_process", "yolo_box")
        for field in ("color", "classes", "model_confidence", "vlm_prompt", "inspection_label", "alarm_description"):
            if field in updates:
                td[field] = updates[field]
        self._save(self._types)

    def delete_type(self, dtype: str) -> None:
        """删除检测类型"""
        if dtype not in self._types:
            raise KeyError(f"Unknown detection type: {dtype}")
        del self._types[dtype]
        self._save(self._types)


registry = DetectionTypeRegistry()
registry.load()
