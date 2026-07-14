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

        with patch("backend.safety_detection.detector_core.registry", mock_registry):
            annotated = MultiDetector._annotate_frame(frame, results)
            assert annotated is not frame
            mock_registry.get_color_bgr.assert_called_with("fire")

    def test_unknown_type_fallback(self):
        from backend.safety_detection.detector_core import MultiDetector

        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = {"unknown_type": {"detected": True, "boxes": [[10, 10, 50, 50]], "scores": [0.8]}}

        with patch("backend.safety_detection.detector_core.registry", mock_registry):
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

        with patch("backend.safety_detection.detector_core.registry", mock_registry):
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
