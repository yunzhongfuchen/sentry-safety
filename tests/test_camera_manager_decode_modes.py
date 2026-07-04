import pytest
from backend.camera_manager import DecodeMode, CameraConfig, CameraState


def test_decode_mode_enum_values():
    assert DecodeMode.CONTINUOUS.value == "continuous"
    assert DecodeMode.SCHEDULED.value == "scheduled"


def test_camera_state_defaults_to_scheduled():
    cfg = CameraConfig(camera_id="cam_01", source="0")
    state = CameraState(config=cfg)
    assert state.decode_mode == DecodeMode.SCHEDULED
    assert state.frame_request_event is not None
    assert state.frame_ready_event is not None
    assert state.current_scheduled_frame is None
