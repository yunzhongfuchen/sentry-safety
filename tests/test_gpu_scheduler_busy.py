from unittest.mock import MagicMock, patch
import numpy as np
from backend.gpu_scheduler import GPUDynamicScheduler, ModelConfig


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
        assert scheduler._busy is False
        scheduler.start()
        scheduler.stop()
