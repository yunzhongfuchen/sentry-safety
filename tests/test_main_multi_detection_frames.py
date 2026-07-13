import time
import numpy as np
import pytest
from unittest.mock import Mock, patch
from backend import main_multi


def test_on_trigger_uses_detection_frames():
    # Force synchronous save path to avoid async ThreadPoolExecutor race.
    main_multi._save_executor = None
    main_multi._global_settings = {
        "frame_quality": 60,
        "save_image_timestamp": True,
        "max_records": 100,
        "emergency_cleanup_ratio": 0.2,
    }
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    detection_frames = [(time.time(), b"f1"), (time.time(), b"f2")]
    result = {
        "detected": True,
        "scores": [0.9],
        "level": "small_model_alarm",
        "detection_frames": detection_frames,
    }
    with patch.object(main_multi.storage, "save_image") as mock_save:
        main_multi.on_trigger("cam1", "fire", frame, result)
        # snapshot + 2 detection frames
        assert mock_save.call_count == 3


def test_save_image_timestamp_false_does_not_draw_on_snapshot():
    main_multi._global_settings = {
        "frame_quality": 60,
        "save_image_timestamp": False,
        "max_records": 100,
        "emergency_cleanup_ratio": 0.2,
    }
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    result = {
        "detected": True,
        "scores": [0.9],
        "level": "small_model_alarm",
        "detection_frames": [],
    }
    with patch.object(main_multi, "draw_timestamp_on_frame") as mock_draw_ts:
        with patch.object(main_multi.storage, "save_image"):
            main_multi.on_trigger("cam1", "fire", frame, result)
        mock_draw_ts.assert_not_called()


def test_save_detection_frames_async_writes_bytes():
    frames = [(time.time(), b"f1"), (time.time(), b"f2")]
    with patch.object(main_multi.storage, "save_image") as mock_save:
        main_multi._save_detection_frames_async("rec1", frames)
        assert mock_save.call_count == 2


def test_save_detection_frames_async_exception_logged():
    frames = [(time.time(), b"f1")]
    with patch.object(main_multi.storage, "save_image", side_effect=RuntimeError("disk full")) as mock_save:
        with patch.object(main_multi, "logger") as mock_logger:
            main_multi._save_detection_frames_async("rec1", frames)
            mock_save.assert_called_once()
            mock_logger.error.assert_called_once()
