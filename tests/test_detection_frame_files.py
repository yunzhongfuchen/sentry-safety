import time
import glob
import os
from pathlib import Path
import numpy as np
import pytest
from backend import main_multi, storage


DATA_FRAMES_DIR = Path(storage.FRAMES_DIR)


def _cleanup_test_frames(prefix: str):
    for p in DATA_FRAMES_DIR.glob(f"{prefix}*"):
        try:
            p.unlink()
        except Exception:
            pass


def test_on_trigger_writes_detection_frame_files():
    """端到端验证：on_trigger 接收 detection_frames 后真正把帧文件写入磁盘。"""
    main_multi._save_executor = None
    main_multi._global_settings = {
        "frame_quality": 60,
        "save_image_timestamp": True,
        "max_records": 100,
        "emergency_cleanup_ratio": 0.2,
        "snapshot_quality": 70,
    }

    camera_id = f"test_cam_{int(time.time() * 1000) % 100000}"
    dtype = "fire"
    prefix = f"{camera_id}_{dtype}_"
    _cleanup_test_frames(prefix)

    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    detection_frames = [
        (time.time(), b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 50),
        (time.time() + 1, b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 50),
    ]
    result = {
        "detected": True,
        "scores": [0.9],
        "level": "small_model_alarm",
        "detection_frames": detection_frames,
    }

    try:
        main_multi.on_trigger(camera_id, dtype, frame, result)
        # 等待同步保存完成
        time.sleep(0.1)

        snapshot = list(DATA_FRAMES_DIR.glob(f"{prefix}*_snapshot.jpg"))
        frame_files = sorted(DATA_FRAMES_DIR.glob(f"{prefix}*_frame_*.jpg"))

        assert len(snapshot) == 1, "应生成一张快照"
        assert len(frame_files) == len(detection_frames), "应写入与 detection_frames 数量一致的帧文件"
        assert frame_files[0].name.endswith("_frame_000.jpg")
        assert frame_files[1].name.endswith("_frame_001.jpg")
        for f in frame_files:
            assert f.stat().st_size > 0
    finally:
        _cleanup_test_frames(prefix)


def test_storage_save_image_accepts_bytes_and_b64():
    """storage.save_image 同时接受原始 JPEG 字节和 base64 字符串。"""
    record_id = f"test_storage_{int(time.time() * 1000) % 100000}"
    raw = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 50
    _cleanup_test_frames(record_id)

    try:
        path1 = storage.save_image(record_id, "frame", raw, 0)
        assert path1
        assert (storage.DATA_DIR / path1).exists()
        assert (storage.DATA_DIR / path1).read_bytes() == raw

        import base64
        b64 = base64.b64encode(raw).decode("utf-8")
        path2 = storage.save_image(record_id, "frame", b64, 1)
        assert path2
        assert (storage.DATA_DIR / path2).read_bytes() == raw
    finally:
        _cleanup_test_frames(record_id)
