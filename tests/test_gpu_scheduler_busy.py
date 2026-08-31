import threading
import time
from unittest.mock import MagicMock, patch
import numpy as np
from backend.gpu_scheduler import GPUDynamicScheduler, ModelConfig


def test_scheduler_starts_not_busy():
    cm = MagicMock()
    cm._cameras = {}
    cm.get_latest_frame = MagicMock(return_value=None)

    with patch("backend.gpu_scheduler.YOLO"):
        scheduler = GPUDynamicScheduler(
            cm,
            {"fire": ModelConfig("dummy.pt", "fire", device="cpu")},
            num_queues=1,
            interval=0.1,
            warmup=False,
        )
        assert scheduler._busy is False
        scheduler.start()
        scheduler.stop()


def test_scheduler_sets_busy_during_inference():
    cm = MagicMock()
    cm._cameras = {}
    cm.get_latest_frame = MagicMock(return_value=None)

    with patch("backend.gpu_scheduler.YOLO"):
        scheduler = GPUDynamicScheduler(
            cm,
            {"fire": ModelConfig("dummy.pt", "fire", device="cpu")},
            num_queues=1,
            interval=0.1,
            warmup=False,
        )

        block_event = threading.Event()

        def blocking_infer(tasks):
            block_event.wait()

        scheduler._collect_due_frames = MagicMock(
            return_value={
                "fire": [("cam1", np.zeros((10, 10, 3), dtype=np.uint8))],
            }
        )
        scheduler._infer_batch = MagicMock(side_effect=blocking_infer)

        scheduler.start()
        try:
            busy = False
            for _ in range(100):
                if scheduler._busy:
                    busy = True
                    break
                time.sleep(0.01)
            assert busy, "Expected _busy to be True during inference"
        finally:
            block_event.set()
            scheduler.stop()


def test_collect_due_frames_skips_cooldown_types():
    """_collect_due_frames 应跳过处于冷却期的类型"""
    cm = MagicMock()
    cam_state = MagicMock()
    cam_state.config.enabled = True
    cam_state.config.detection_enabled = True
    cam_state.config.detection_types = {"fire": {"enabled": True, "interval": 0.1}}
    cm._cameras = {"cam1": cam_state}
    cm.get_latest_frame = MagicMock(return_value=np.zeros((10, 10, 3), dtype=np.uint8))

    with patch("backend.gpu_scheduler.YOLO"):
        scheduler = GPUDynamicScheduler(
            cm,
            {"fire": ModelConfig("dummy.pt", "fire", device="cpu")},
            num_queues=1,
            interval=0.1,
            warmup=False,
            cooldown_checker=lambda cam_id, dtype, now: True,
        )



def test_collect_due_frames_uses_due_checker_and_new_frame():
    cm = MagicMock()
    cam_state = MagicMock()
    cam_state.config.enabled = True
    cam_state.config.detection_enabled = True
    cam_state.config.detection_types = {"fire": {"enabled": True, "interval": 300}}
    cm._cameras = {"cam1": cam_state}
    cm.get_latest_frame.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    cm.get_frame_seq.return_value = 7

    with patch("backend.gpu_scheduler.YOLO"):
        scheduler = GPUDynamicScheduler(
            cm,
            {"fire": ModelConfig("dummy.pt", "fire", device="cpu")},
            num_queues=1,
            interval=0.1,
            warmup=False,
            due_checker=lambda cam_id, dtype, now: True,
        )

        assert scheduler._collect_due_frames(time.time())["fire"]
        scheduler.last_frame_seq[("cam1", "fire")] = 7
        assert scheduler._collect_due_frames(time.time()) == {}
