from backend.camera_manager import CameraConfig, CameraManager


def test_frame_seq_starts_zero_and_getter():
    cm = CameraManager()
    assert cm.get_frame_seq("nonexistent") == -1

    cm.register_camera(CameraConfig(camera_id="cam_01", source="0"))
    assert cm.get_frame_seq("cam_01") == 0
