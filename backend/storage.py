import json
import os
import logging
import base64
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
RECORDS_FILE = DATA_DIR / "records.json"
FRAMES_DIR = DATA_DIR / "frames"


def ensure_dirs():
    """确保数据目录存在"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def _frame_path(record_id: str, kind: str, index: int = 0) -> Path:
    """获取帧图片文件路径
    kind: 'snapshot' | 'frame'
    """
    if kind == "snapshot":
        return FRAMES_DIR / f"{record_id}_snapshot.jpg"
    return FRAMES_DIR / f"{record_id}_frame_{index:03d}.jpg"


def save_image(record_id: str, kind: str, b64_data: str, index: int = 0) -> str:
    """将 base64 图片保存为文件，返回相对路径"""
    ensure_dirs()
    path = _frame_path(record_id, kind, index)
    try:
        path.write_bytes(base64.b64decode(b64_data))
        return str(path.relative_to(DATA_DIR))
    except Exception as e:
        logger.error(f"Failed to save image {path}: {e}")
        return ""


def load_image_b64(record_id: str, kind: str, index: int = 0) -> Optional[str]:
    """按需从文件加载图片为 base64"""
    path = _frame_path(record_id, kind, index)
    if not path.exists():
        return None
    try:
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to load image {path}: {e}")
        return None


def get_frame_count(record_id: str) -> int:
    """获取某条记录的帧数"""
    count = 0
    while _frame_path(record_id, "frame", count).exists():
        count += 1
    return count


def load_records() -> List[Dict]:
    """加载记录元数据（不含图片数据）"""
    ensure_dirs()
    if not RECORDS_FILE.exists():
        return []
    try:
        with open(RECORDS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} records metadata")
        return data
    except Exception as e:
        logger.error(f"Failed to load records: {e}")
        return []


def save_records(records: List[Dict]):
    """保存记录元数据（不含图片数据）"""
    ensure_dirs()
    try:
        with open(RECORDS_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(records)} records metadata")
    except Exception as e:
        logger.error(f"Failed to save records: {e}")


def delete_record_images(record_id: str):
    """删除某条记录的所有图片文件"""
    snapshot = _frame_path(record_id, "snapshot")
    if snapshot.exists():
        snapshot.unlink()
    i = 0
    while True:
        p = _frame_path(record_id, "frame", i)
        if not p.exists():
            break
        p.unlink()
        i += 1
