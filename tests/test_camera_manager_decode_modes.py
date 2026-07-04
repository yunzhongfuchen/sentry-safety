import threading
import time
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
from backend.camera_manager import CameraManager, CameraConfig, CameraState, DecodeMode


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


def test_set_decode_mode_changes_state():
    cm = CameraManager()
    cfg = CameraConfig(camera_id="cam_01", source="0")
    cm.register_camera(cfg)
    cm.set_decode_mode("cam_01", DecodeMode.CONTINUOUS)
    assert cm._cameras["cam_01"].decode_mode == DecodeMode.CONTINUOUS
    cm.set_decode_mode("cam_01", DecodeMode.SCHEDULED)
    assert cm._cameras["cam_01"].decode_mode == DecodeMode.SCHEDULED


def test_set_decode_mode_returns_false_for_unknown_camera():
    cm = CameraManager()
    assert cm.set_decode_mode("nonexistent", DecodeMode.CONTINUOUS) is False


def test_set_decode_mode_clears_events_when_switching_to_scheduled():
    cm = CameraManager()
    cfg = CameraConfig(camera_id="cam_01", source="0")
    cm.register_camera(cfg)
    state = cm._cameras["cam_01"]
    # Simulate a pending request
    state.frame_request_event.set()
    state.frame_ready_event.set()
    cm.set_decode_mode("cam_01", DecodeMode.SCHEDULED)
    assert not state.frame_request_event.is_set()
    assert not state.frame_ready_event.is_set()


class TestConnectAndStreamModes:
    """Test _connect_and_stream behavior in CONTINUOUS vs SCHEDULED modes."""

    @patch("backend.camera_manager.cv2.VideoCapture")
    def test_continuous_mode_updates_current_frame(self, mock_cap_class):
        """CONTINUOUS mode: frame loop updates current_frame and limits to ~25 FPS."""
        cm = CameraManager()
        cfg = CameraConfig(camera_id="cam_01", source="0")
        cm.register_camera(cfg)
        cm.set_decode_mode("cam_01", DecodeMode.CONTINUOUS)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        call_count = 0

        def mock_read():
            nonlocal call_count
            call_count += 1
            return True, frame

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 0,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_POS_FRAMES: 0,
        }.get(prop, 0)
        mock_cap.read.side_effect = mock_read
        mock_cap_class.return_value = mock_cap

        # Start stream in a thread so we can stop it after a few frames
        stream_thread = threading.Thread(target=cm._connect_and_stream, args=("cam_01",))
        cm._cameras["cam_01"].running = True
        stream_thread.start()

        # Let it run for a few frames, then stop
        time.sleep(0.3)
        with cm._lock:
            cm._cameras["cam_01"].running = False
        stream_thread.join(timeout=2.0)
        assert not stream_thread.is_alive()

        state = cm._cameras["cam_01"]
        assert call_count >= 3
        assert state.frame_count >= 3
        assert state.current_frame is not None

    @patch("backend.camera_manager.cv2.VideoCapture")
    def test_scheduled_mode_waits_for_request(self, mock_cap_class):
        """SCHEDULED mode: thread waits for frame_request_event, decodes one frame, sets frame_ready_event."""
        cm = CameraManager()
        cfg = CameraConfig(camera_id="cam_01", source="0")
        cm.register_camera(cfg)
        cm.set_decode_mode("cam_01", DecodeMode.SCHEDULED)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 0,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_POS_FRAMES: 0,
        }.get(prop, 0)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        decoded_count = 0

        def mock_read():
            nonlocal decoded_count
            decoded_count += 1
            return True, frame.copy()

        mock_cap.read = mock_read
        mock_cap_class.return_value = mock_cap

        state = cm._cameras["cam_01"]

        # Start _connect_and_stream in a thread
        stream_thread = threading.Thread(target=cm._connect_and_stream, args=("cam_01",))
        state.running = True
        stream_thread.start()

        # Request a frame
        time.sleep(0.1)  # Let thread reach wait
        state.frame_request_event.set()
        ready = state.frame_ready_event.wait(timeout=2.0)
        assert ready, "frame_ready_event should be set after decoding"

        with state.lock:
            assert state.current_scheduled_frame is not None

        # Stop
        with cm._lock:
            state.running = False
        state.frame_request_event.set()  # Wake it up if still waiting
        stream_thread.join(timeout=2.0)
        assert not stream_thread.is_alive()
        assert decoded_count >= 1

    @patch("backend.camera_manager.cv2.VideoCapture")
    def test_scheduled_mode_no_decode_without_request(self, mock_cap_class):
        """SCHEDULED mode: without frame_request_event, no frames should be decoded."""
        cm = CameraManager()
        cfg = CameraConfig(camera_id="cam_01", source="0")
        cm.register_camera(cfg)
        cm.set_decode_mode("cam_01", DecodeMode.SCHEDULED)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 0,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_POS_FRAMES: 0,
        }.get(prop, 0)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        decoded_count = 0

        def mock_read():
            nonlocal decoded_count
            decoded_count += 1
            return True, frame.copy()

        mock_cap.read = mock_read
        mock_cap_class.return_value = mock_cap

        state = cm._cameras["cam_01"]
        state.running = True

        stream_thread = threading.Thread(target=cm._connect_and_stream, args=("cam_01",))
        stream_thread.start()

        # Wait a bit without setting frame_request_event
        time.sleep(0.3)

        # Stop
        with cm._lock:
            state.running = False
        state.frame_request_event.set()  # Wake it up
        stream_thread.join(timeout=2.0)
        assert not stream_thread.is_alive()

        # Should not have decoded any frames without a request
        assert decoded_count == 0


def test_request_frame_continuous_returns_current_frame():
    cm = CameraManager()
    cfg = CameraConfig(camera_id="cam_01", source="0")
    cm.register_camera(cfg)
    cm.set_decode_mode("cam_01", DecodeMode.CONTINUOUS)

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    cm._cameras["cam_01"].current_frame = dummy

    frame = cm.request_frame("cam_01")
    assert frame is not None
    assert frame.shape == dummy.shape


def test_request_frame_scheduled_triggers_decode_and_appends_history():
    cm = CameraManager()
    cfg = CameraConfig(camera_id="cam_01", source="0")
    cm.register_camera(cfg)

    # 模拟 SCHEDULED 模式下解码线程已准备好帧
    state = cm._cameras["cam_01"]
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    state.current_scheduled_frame = dummy

    # 先 set event，再 request，模拟线程已解完帧
    state.frame_ready_event.set()

    frame = cm.request_frame("cam_01", timeout=0.1)
    assert frame is not None
    assert len(state.frame_history) == 1


def test_camera_manager_set_main_camera():
    cm = CameraManager()
    cm.register_camera(CameraConfig(camera_id="old", source="0"))
    cm.register_camera(CameraConfig(camera_id="new", source="0"))

    cm.set_main_camera("old")
    assert cm._cameras["old"].decode_mode == DecodeMode.CONTINUOUS

    cm.set_main_camera("new")
    assert cm._cameras["old"].decode_mode == DecodeMode.SCHEDULED
    assert cm._cameras["new"].decode_mode == DecodeMode.CONTINUOUS
    assert cm.get_main_camera() == "new"


def test_set_main_camera_returns_old_main():
    cm = CameraManager()
    cm.register_camera(CameraConfig(camera_id="cam_a", source="0"))
    cm.register_camera(CameraConfig(camera_id="cam_b", source="0"))

    old = cm.set_main_camera("cam_a")
    assert old is None

    old = cm.set_main_camera("cam_b")
    assert old == "cam_a"


def test_set_main_camera_to_none_clears_main():
    cm = CameraManager()
    cm.register_camera(CameraConfig(camera_id="cam_1", source="0"))

    cm.set_main_camera("cam_1")
    assert cm.get_main_camera() == "cam_1"

    cm.set_main_camera(None)
    assert cm.get_main_camera() is None
    assert cm._cameras["cam_1"].decode_mode == DecodeMode.SCHEDULED


def test_set_main_camera_unknown_id_warns_and_clears():
    cm = CameraManager()
    cm.register_camera(CameraConfig(camera_id="cam_1", source="0"))
    cm.set_main_camera("cam_1")

    old = cm.set_main_camera("unknown")
    assert old == "cam_1"
    assert cm.get_main_camera() is None
    assert cm._cameras["cam_1"].decode_mode == DecodeMode.SCHEDULED
