import time
import numpy as np
from unittest.mock import MagicMock, patch
from backend.gpu_scheduler import GPUDynamicScheduler, ModelConfig


def _cleanup_scheduler(scheduler):
    """Stop queue workers without joining the scheduler thread (which may not be started)."""
    scheduler.running = False
    for w in scheduler.queues.values():
        w.stop()


def test_scheduler_busy_flag_initialized_false():
    """_busy flag should start as False."""
    cm = MagicMock()
    cm._cameras = {
        "cam_01": MagicMock(
            config=MagicMock(enabled=True, detection_enabled=True, detection_types={
                "helmet": {"enabled": True, "interval": 1.0}
            })
        )
    }
    cm.get_camera_ids = MagicMock(return_value=["cam_01"])
    cm.request_frame = MagicMock(return_value=np.zeros((480, 640, 3), dtype=np.uint8))

    model_configs = {
        "helmet": ModelConfig(model_path="dummy.pt", detection_type="helmet", device="cpu")
    }

    with patch("backend.inference_backend.YOLO"):
        scheduler = GPUDynamicScheduler(cm, model_configs, num_queues=1, interval=0.1, warmup=False)
        assert scheduler._busy is False
        assert scheduler.MAX_FRAME_AGE == 0.5
        _cleanup_scheduler(scheduler)


def test_collect_due_frames_uses_request_frame():
    """_collect_due_frames should call camera_manager.request_frame."""
    cm = MagicMock()
    cm._cameras = {
        "cam_01": MagicMock(
            config=MagicMock(enabled=True, detection_enabled=True, detection_types={
                "helmet": {"enabled": True, "interval": 1.0}
            })
        )
    }
    cm.request_frame = MagicMock(return_value=np.zeros((480, 640, 3), dtype=np.uint8))

    model_configs = {
        "helmet": ModelConfig(model_path="dummy.pt", detection_type="helmet", device="cpu")
    }

    with patch("backend.inference_backend.YOLO"):
        scheduler = GPUDynamicScheduler(cm, model_configs, num_queues=1, interval=0.1, warmup=False)
        now = time.time()
        tasks = scheduler._collect_due_frames(now)

        assert "helmet" in tasks
        assert len(tasks["helmet"]) == 1
        assert tasks["helmet"][0][0] == "cam_01"
        cm.request_frame.assert_called_once_with("cam_01", timeout=1.0, store_history=True)
        _cleanup_scheduler(scheduler)


def test_collect_due_frames_drops_none_frames():
    """_collect_due_frames should skip cameras that return None."""
    cm = MagicMock()
    cm._cameras = {
        "cam_01": MagicMock(
            config=MagicMock(enabled=True, detection_enabled=True, detection_types={
                "helmet": {"enabled": True, "interval": 1.0}
            })
        )
    }
    cm.request_frame = MagicMock(return_value=None)

    model_configs = {
        "helmet": ModelConfig(model_path="dummy.pt", detection_type="helmet", device="cpu")
    }

    with patch("backend.inference_backend.YOLO"):
        scheduler = GPUDynamicScheduler(cm, model_configs, num_queues=1, interval=0.1, warmup=False)
        now = time.time()
        tasks = scheduler._collect_due_frames(now)
        assert tasks == {}
        _cleanup_scheduler(scheduler)


def test_collect_due_frames_drops_old_frames():
    """_collect_due_frames should drop frames older than MAX_FRAME_AGE."""
    cm = MagicMock()
    cm._cameras = {
        "cam_01": MagicMock(
            config=MagicMock(enabled=True, detection_enabled=True, detection_types={
                "helmet": {"enabled": True, "interval": 1.0}
            })
        )
    }

    call_times = []

    def slow_request_frame(*args, **kwargs):
        call_times.append(time.time())
        time.sleep(0.6)  # Simulate slow frame acquisition
        return np.zeros((480, 640, 3), dtype=np.uint8)

    cm.request_frame = MagicMock(side_effect=slow_request_frame)

    model_configs = {
        "helmet": ModelConfig(model_path="dummy.pt", detection_type="helmet", device="cpu")
    }

    with patch("backend.inference_backend.YOLO"):
        scheduler = GPUDynamicScheduler(cm, model_configs, num_queues=1, interval=0.1, warmup=False)
        now = time.time()
        tasks = scheduler._collect_due_frames(now)
        # Because request_frame sleeps 0.6s > MAX_FRAME_AGE (0.5s), frame should be dropped
        assert tasks == {}
        _cleanup_scheduler(scheduler)


def test_collect_due_frames_respects_interval():
    """_collect_due_frames should not include cameras whose interval has not elapsed."""
    cm = MagicMock()
    cm._cameras = {
        "cam_01": MagicMock(
            config=MagicMock(enabled=True, detection_enabled=True, detection_types={
                "helmet": {"enabled": True, "interval": 5.0}
            })
        )
    }
    cm.request_frame = MagicMock(return_value=np.zeros((480, 640, 3), dtype=np.uint8))

    model_configs = {
        "helmet": ModelConfig(model_path="dummy.pt", detection_type="helmet", device="cpu")
    }

    with patch("backend.inference_backend.YOLO"):
        scheduler = GPUDynamicScheduler(cm, model_configs, num_queues=1, interval=0.1, warmup=False)
        # Set last_infer to now so interval has not elapsed
        now = time.time()
        scheduler.last_infer[("cam_01", "helmet")] = now
        tasks = scheduler._collect_due_frames(now)
        assert tasks == {}
        _cleanup_scheduler(scheduler)
