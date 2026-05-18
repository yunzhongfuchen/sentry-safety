"""
高性能存储模块 - 优化图片加载和分页查询
"""

import json
import base64
import logging
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
RECORDS_FILE = DATA_DIR / "records.json"
FRAMES_DIR = DATA_DIR / "frames"

CONFINED_RECORDS_FILE = DATA_DIR / "confined_records.json"
CONFINED_FRAMES_DIR = DATA_DIR / "confined_frames"

# 图片缓存 (LRU)
class LRUCache:
    def __init__(self, max_size: int = 50):
        self._cache: Dict[str, tuple] = {}
        self._max_size = max_size
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if key in self._cache:
                value, _ = self._cache[key]
                # 更新访问时间
                self._cache[key] = (value, time.time())
                return value
            return None
    
    def put(self, key: str, value: str):
        with self._lock:
            if len(self._cache) >= self._max_size:
                # 移除最旧的
                oldest = min(self._cache.items(), key=lambda x: x[1][1])
                del self._cache[oldest[0]]
            self._cache[key] = (value, time.time())
    
    def clear(self):
        with self._lock:
            self._cache.clear()

# 全局图片缓存
_image_cache = LRUCache(max_size=100)


def ensure_dirs():
    """确保数据目录存在"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def _frame_path(record_id: str, kind: str, index: int = 0) -> Path:
    """获取帧图片文件路径"""
    if kind == "snapshot":
        return FRAMES_DIR / f"{record_id}_snapshot.jpg"
    return FRAMES_DIR / f"{record_id}_frame_{index:03d}.jpg"


def save_image(record_id: str, kind: str, b64_data: str, index: int = 0) -> str:
    """保存 base64 图片为文件"""
    ensure_dirs()
    path = _frame_path(record_id, kind, index)
    try:
        path.write_bytes(base64.b64decode(b64_data))
        # 同时存入缓存
        cache_key = f"{record_id}_{kind}_{index}"
        _image_cache.put(cache_key, b64_data)
        return str(path.relative_to(DATA_DIR))
    except Exception as e:
        logger.error(f"Failed to save image {path}: {e}")
        return ""


def load_image_b64(record_id: str, kind: str, index: int = 0) -> Optional[str]:
    """加载图片为 base64 (带缓存)"""
    cache_key = f"{record_id}_{kind}_{index}"
    
    # 先查缓存
    cached = _image_cache.get(cache_key)
    if cached:
        return cached
    
    # 从文件加载
    path = _frame_path(record_id, kind, index)
    if not path.exists():
        return None
    
    try:
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        # 存入缓存
        _image_cache.put(cache_key, data)
        return data
    except Exception as e:
        logger.error(f"Failed to load image {path}: {e}")
        return None


def load_image_b64_batch(record_id: str, kind: str, count: int) -> List[str]:
    """批量加载图片"""
    results = []
    for i in range(count):
        b64 = load_image_b64(record_id, kind, i)
        if b64:
            results.append(b64)
    return results


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
    """保存记录元数据"""
    ensure_dirs()
    try:
        with open(RECORDS_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
        logger.info(f"Saved {len(records)} records metadata")
    except Exception as e:
        logger.error(f"Failed to save records: {e}")


def delete_record_images(record_id: str):
    """删除某条记录的所有图片文件"""
    # 清除缓存
    for key in list(_image_cache._cache.keys()):
        if key.startswith(record_id):
            del _image_cache._cache[key]
    
    # 删除文件
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


def get_records_paginated(
    page: int = 1,
    page_size: int = 20,
    camera_id: Optional[str] = None,
    level: Optional[str] = None,
    dtype: Optional[str] = None,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
) -> Tuple[List[Dict], int]:
    """
    分页查询告警记录（不含图片）

    Args:
        level: "P0" | "P1" | None
        dtype: "fire" | "smoke" | "uniform" | "mask" | "cigarette" | "sleep" | None
        status: "alerted" | "pending" | "confirmed" | "rejected" | "false_positive" | None

    Returns:
        (当前页记录列表, 总记录数)
    """
    records = load_records()

    # 过滤
    filtered = records
    if camera_id:
        filtered = [r for r in filtered if r.get("camera_id") == camera_id]
    if level:
        filtered = [r for r in filtered if r.get("level") == level]
    if dtype:
        filtered = [r for r in filtered if r.get("detection_type") == dtype]
    if status:
        filtered = [r for r in filtered if r.get("status") == status]

    total = len(filtered)

    # 分页
    start = (page - 1) * page_size
    end = start + page_size
    page_records = filtered[start:end]

    # 只返回元数据，不包含图片
    result = []
    for r in page_records:
        item = {
            "id": r.get("id"),
            "camera_id": r.get("camera_id", "unknown"),
            "time": r.get("time"),
            "detection_type": r.get("detection_type"),
            "level": r.get("level"),
            "status": r.get("status"),
            "confidence": r.get("confidence"),
            "reason": r.get("reason"),
            "small_model": r.get("small_model"),
            "vlm_review": r.get("vlm_review"),
            "source": r.get("source", "small_model"),
            "frame_count": r.get("frame_count", 0),
            "has_snapshot": _frame_path(r.get("id", ""), "snapshot").exists()
        }
        result.append(item)

    return result, total


def get_record_detail(record_id: str, include_frames: bool = True) -> Optional[Dict]:
    """
    获取单条记录详情
    
    Args:
        record_id: 记录ID
        include_frames: 是否包含帧图片（默认True，可通过参数控制）
    """
    records = load_records()
    
    meta = None
    for r in records:
        if r.get("id") == record_id:
            meta = dict(r)
            break
    
    if meta is None:
        return None
    
    # 异步加载图片（使用线程池）
    def load_images():
        # 加载快照
        snapshot = load_image_b64(record_id, "snapshot")
        if snapshot:
            meta["snapshot"] = snapshot
        
        # 加载帧（可选）
        if include_frames:
            frame_count = meta.get("frame_count", 0)
            # 限制最大帧数，避免内存溢出
            max_frames = min(frame_count, 30)
            frames = load_image_b64_batch(record_id, "frame", max_frames)
            meta["frames"] = frames
            meta["frame_count_actual"] = len(frames)
    
    # 在非异步环境直接加载
    load_images()
    
    return meta


def get_record_summary() -> Dict:
    """获取记录统计摘要（按类型和级别聚合）"""
    records = load_records()

    total = len(records)
    by_level = {"P0": 0, "P1": 0}
    by_type = {}
    by_status = {}

    for r in records:
        level = r.get("level", "P1")
        by_level[level] = by_level.get(level, 0) + 1

        dtype = r.get("detection_type", "unknown")
        by_type[dtype] = by_type.get(dtype, 0) + 1

        status = r.get("status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1

    # 按摄像头分组统计
    camera_stats = {}
    for r in records:
        cam_id = r.get("camera_id", "unknown")
        camera_stats[cam_id] = camera_stats.get(cam_id, 0) + 1

    return {
        "total": total,
        "by_level": by_level,
        "by_type": by_type,
        "by_status": by_status,
        "by_camera": camera_stats,
    }


# ----------------------------------------------------------------------
# 存储清理
# ----------------------------------------------------------------------

def get_storage_size_mb(data_dir: Path = DATA_DIR) -> float:
    """获取数据目录占用空间（MB）"""
    total = 0
    for p in data_dir.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


def cleanup_old_records(target_mb: float, data_dir: Path = DATA_DIR) -> int:
    """
    按存储空间清理：删除最旧的记录直到空间低于 target_mb
    返回删除的记录数
    """
    size_mb = get_storage_size_mb(data_dir)
    if size_mb <= target_mb:
        return 0

    records = load_records()
    if not records:
        return 0

    # 按时间升序排列
    sorted_records = sorted(records, key=lambda r: r.get("time", ""))
    removed = 0

    while size_mb > target_mb and sorted_records:
        oldest = sorted_records.pop(0)
        rid = oldest.get("id")
        if rid:
            delete_record_images(rid)
        records = [r for r in records if r.get("id") != rid]
        removed += 1
        size_mb = get_storage_size_mb(data_dir)

    save_records(records)
    logger.info(f"Cleaned up {removed} old records, storage now {size_mb:.1f} MB")
    return removed


def cleanup_oldest_records(ratio: float = 0.2, data_dir: Path = DATA_DIR) -> int:
    """
    紧急清理：删除最旧 ratio 比例（默认 20%）的记录
    返回删除的记录数
    """
    records = load_records()
    if not records:
        return 0

    remove_count = max(1, int(len(records) * ratio))
    sorted_records = sorted(records, key=lambda r: r.get("time", ""))
    to_remove = sorted_records[:remove_count]

    removed_ids = {r.get("id") for r in to_remove}
    for rid in removed_ids:
        if rid:
            delete_record_images(rid)

    remaining = [r for r in records if r.get("id") not in removed_ids]
    save_records(remaining)
    logger.warning(f"Emergency cleanup: removed {remove_count} oldest records")
    return remove_count


def storage_cleanup_loop(
    max_records: int = 100,
    max_storage_mb: float = 500.0,
    memory_threshold_percent: float = 80.0,
    emergency_cleanup_ratio: float = 0.2,
    interval_seconds: float = 3600.0,
):
    """
    后台存储清理循环（每小时检查一次）
    可单独作为线程启动
    """
    import psutil

    while True:
        time.sleep(interval_seconds)
        try:
            records = load_records()
            triggered = False

            # 1. 按记录数清理
            if len(records) > max_records:
                logger.info(f"Record count {len(records)} > max {max_records}, triggering ratio cleanup")
                triggered = True

            # 2. 按存储空间清理（0 表示无限制）
            size_mb = get_storage_size_mb()
            if max_storage_mb > 0 and size_mb > max_storage_mb:
                logger.info(f"Storage {size_mb:.1f}MB > limit {max_storage_mb}MB, triggering ratio cleanup")
                triggered = True

            # 3. 按内存阈值紧急清理
            mem = psutil.virtual_memory()
            if mem.percent > memory_threshold_percent:
                logger.warning(f"Memory usage {mem.percent}% > threshold {memory_threshold_percent}%, emergency cleanup")
                triggered = True

            if triggered:
                cleanup_oldest_records(emergency_cleanup_ratio)

        except Exception as e:
            logger.error(f"Storage cleanup loop error: {e}")


# 兼容性：保持与原 storage 模块相同的接口
save_records_sync = save_records
load_records_sync = load_records


# ==================== 有限空间独立记录存储 ====================

def ensure_confined_dirs():
    """确保有限空间数据目录存在"""
    CONFINED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def save_confined_records(records: List[Dict]) -> None:
    """保存有限空间记录"""
    ensure_dirs()
    try:
        with open(CONFINED_RECORDS_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
        logger.info(f"Saved {len(records)} confined space records")
    except Exception as e:
        logger.error(f"Failed to save confined records: {e}")


def load_confined_records() -> List[Dict]:
    """加载有限空间记录"""
    if not CONFINED_RECORDS_FILE.exists():
        return []
    try:
        with open(CONFINED_RECORDS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"Loaded {len(data)} confined space records")
            return data
    except Exception as e:
        logger.error(f"Failed to load confined records: {e}")
        return []


def save_confined_image(record_id: str, kind: str, b64_data: str, index: int = 0) -> str:
    """保存有限空间图片"""
    ensure_confined_dirs()
    try:
        img_data = base64.b64decode(b64_data)
        if kind == "snapshot":
            path = CONFINED_FRAMES_DIR / f"{record_id}_snapshot.jpg"
        else:
            path = CONFINED_FRAMES_DIR / f"{record_id}_frame_{index:03d}.jpg"
        with open(path, "wb") as f:
            f.write(img_data)
        return str(path)
    except Exception as e:
        logger.error(f"Failed to save confined image: {e}")
        return ""


def load_confined_image_b64(record_id: str, kind: str, index: int = 0) -> Optional[str]:
    """加载有限空间图片为 base64"""
    cache_key = f"confined_{record_id}_{kind}_{index}"
    cached = _image_cache.get(cache_key)
    if cached:
        return cached

    if kind == "snapshot":
        path = CONFINED_FRAMES_DIR / f"{record_id}_snapshot.jpg"
    else:
        path = CONFINED_FRAMES_DIR / f"{record_id}_frame_{index:03d}.jpg"

    if not path.exists():
        return None

    try:
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        _image_cache.put(cache_key, data)
        return data
    except Exception as e:
        logger.error(f"Failed to load confined image {path}: {e}")
        return None


def load_confined_image_b64_batch(record_id: str, kind: str, count: int) -> List[str]:
    """批量加载有限空间图片"""
    results = []
    for i in range(count):
        b64 = load_confined_image_b64(record_id, kind, i)
        if b64:
            results.append(b64)
    return results


def get_confined_frame_count(record_id: str) -> int:
    """获取有限空间某条记录的帧数"""
    count = 0
    while (CONFINED_FRAMES_DIR / f"{record_id}_frame_{count:03d}.jpg").exists():
        count += 1
    return count


def delete_confined_record_images(record_id: str) -> None:
    """删除有限空间记录的图片"""
    try:
        # 清除缓存
        for key in list(_image_cache._cache.keys()):
            if key.startswith(f"confined_{record_id}"):
                del _image_cache._cache[key]

        snapshot = CONFINED_FRAMES_DIR / f"{record_id}_snapshot.jpg"
        if snapshot.exists():
            snapshot.unlink()
        i = 0
        while True:
            p = CONFINED_FRAMES_DIR / f"{record_id}_frame_{i:03d}.jpg"
            if not p.exists():
                break
            p.unlink()
            i += 1
    except Exception as e:
        logger.error(f"Failed to delete confined images: {e}")


def get_confined_records_paginated(
    page: int = 1,
    size: int = 20,
    zone_id: Optional[str] = None,
    event_type: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> Tuple[List[Dict], int]:
    """分页查询有限空间记录"""
    records = load_confined_records()

    # 筛选
    if zone_id:
        records = [r for r in records if r.get("zone_id") == zone_id]
    if event_type:
        records = [r for r in records if r.get("event_type") == event_type]
    if start_time:
        records = [r for r in records if r.get("timestamp", "") >= start_time]
    if end_time:
        records = [r for r in records if r.get("timestamp", "") <= end_time]

    # 按时间倒序
    records.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    total = len(records)
    start = (page - 1) * size
    end = start + size
    return records[start:end], total
