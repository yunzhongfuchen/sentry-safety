from backend.camera_manager import CameraConfig, CameraState


def test_camera_state_has_last_decode_time():
    cfg = CameraConfig(camera_id="cam_01", source="0")
    state = CameraState(config=cfg)
    assert hasattr(state, "last_decode_time")
    assert state.last_decode_time == 0.0


def test_camera_state_has_decode_queued():
    cfg = CameraConfig(camera_id="cam_01", source="0")
    state = CameraState(config=cfg)
    assert hasattr(state, "decode_queued")
    assert state.decode_queued is False
