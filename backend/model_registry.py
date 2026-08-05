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

    def replace_model_file(self, key: str, filename: str, content: bytes) -> Path:
        """更换模型文件，保留 key 与 name，更新 file / file_size / uploaded_at / 元数据"""
        m = self.get(key)
        if m is None:
            raise KeyError(f"Unknown model: {key}")
        old_file = m.get("file")
        path = self.save_model_file(filename, content)
        meta = parse_model_metadata(path)
        m["file"] = path.name
        m["file_size"] = len(content)
        m["uploaded_at"] = datetime.now(timezone.utc).isoformat()
        m["post_process"] = meta.get("post_process", m.get("post_process", "yolo_box"))
        m["class_names"] = meta.get("class_names", {})
        self._save()
        if old_file and old_file != path.name:
            old_path = WEIGHTS_DIR / old_file
            if old_path.exists():
                old_path.unlink()
        return path


model_registry = ModelRegistry()
model_registry.load()
