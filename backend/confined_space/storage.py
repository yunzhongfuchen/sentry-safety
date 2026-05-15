"""
有限空间独立记录存储封装
提供记录增删改查、图片保存、统计等功能
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import performance_storage as storage

logger = logging.getLogger(__name__)

_records_lock = threading.Lock()
_records_dirty = threading.Event()
_detection_records: List[dict] = []


def init() -> None:
    """初始化：加载历史记录"""
    global _detection_records
    _detection_records = storage.load_confined_records()
    logger.info(f"Loaded {len(_detection_records)} confined space records")


def get_all_records() -> List[dict]:
    """获取所有记录"""
    with _records_lock:
        return list(_detection_records)


def get_records_paginated(
    page: int = 1,
    size: int = 20,
    zone_id: Optional[str] = None,
    event_type: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> dict:
    """分页查询记录"""
    records, total = storage.get_confined_records_paginated(
        page=page, size=size, zone_id=zone_id, event_type=event_type,
        start_time=start_time, end_time=end_time,
    )
    return {
        "records": records,
        "total": total,
        "page": page,
        "size": size,
        "pages": (total + size - 1) // size,
    }


def get_stats() -> dict:
    """获取记录统计"""
    with _records_lock:
        records = list(_detection_records)

    today = datetime.now().strftime("%Y-%m-%d")
    today_records = [r for r in records if r.get("timestamp", "").startswith(today)]

    enter_count = sum(1 for r in today_records if r.get("event_type") == "enter")
    leave_count = sum(1 for r in today_records if r.get("event_type") == "leave")
    other_count = sum(1 for r in today_records if r.get("event_type") == "other")
    alert_count = sum(1 for r in today_records if r.get("alert", False))

    return {
        "today_enter": enter_count,
        "today_leave": leave_count,
        "today_other": other_count,
        "today_alert": alert_count,
        "total_records": len(records),
    }


def add_record(record: dict) -> None:
    """添加一条记录"""
    global _detection_records
    with _records_lock:
        _detection_records.insert(0, record)
        max_records = 1000  # 默认上限
        if len(_detection_records) > max_records:
            for old in _detection_records[max_records:]:
                storage.delete_confined_record_images(old.get("id", ""))
            _detection_records = _detection_records[:max_records]
    _records_dirty.set()


def save_frame(record_id: str, b64_data: str, index: int) -> str:
    """保存帧图片"""
    return storage.save_confined_image(record_id, "frame", b64_data, index)


def update_record(record_id: str, updates: dict) -> bool:
    """更新已有记录的字段"""
    global _detection_records
    with _records_lock:
        for i, r in enumerate(_detection_records):
            if r.get("event_id") == record_id or r.get("id") == record_id:
                _detection_records[i] = {**r, **updates}
                _records_dirty.set()
                return True
    return False


def load_frames(record_id: str, count: int = 30) -> List[str]:
    """加载帧序列 base64"""
    frame_count = storage.get_confined_frame_count(record_id)
    max_frames = min(frame_count, count)
    return storage.load_confined_image_b64_batch(record_id, "frame", max_frames)


def get_record_by_id(record_id: str) -> Optional[dict]:
    """根据 ID 获取单条记录"""
    with _records_lock:
        for r in _detection_records:
            if r.get("event_id") == record_id or r.get("id") == record_id:
                return dict(r)
    return None


def clear_records() -> None:
    """清空所有有限空间记录及关联图片"""
    global _detection_records
    with _records_lock:
        _detection_records = []

    # 删除所有关联图片
    try:
        import shutil
        import performance_storage as ps
        if ps.CONFINED_FRAMES_DIR.exists():
            shutil.rmtree(ps.CONFINED_FRAMES_DIR)
            ps.CONFINED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to clear confined frames: {e}")

    # 保存空记录列表并触发后台保存
    storage.save_confined_records([])
    _records_dirty.set()


def _saver_loop() -> None:
    """后台保存线程"""
    while True:
        _records_dirty.wait()
        _records_dirty.clear()
        time.sleep(1)
        try:
            with _records_lock:
                data = list(_detection_records)
            storage.save_confined_records(data)
        except Exception as e:
            logger.error(f"Failed to save confined records: {e}")


def start_saver() -> None:
    """启动后台保存线程"""
    threading.Thread(target=_saver_loop, daemon=True).start()
