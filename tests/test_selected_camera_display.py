from unittest.mock import MagicMock, patch

import numpy as np

from backend.main_multi import SelectedCameraDisplay


@patch("backend.main_multi.DisplayDetectionWorker")
@patch("backend.main_multi.cv2.VideoCapture")
def test_selected_camera_display_reader_uses_private_capture(mock_videocap, _mock_worker_cls):
    frame = np.ones((16, 16, 3), dtype=np.uint8)
    cap = MagicMock()
    cap.isOpened.return_value = True
    mock_videocap.return_value = cap

    camera_manager = MagicMock()
    state = MagicMock()
    state.config.source = "0"
    state.config.width = 640
    state.config.height = 480
    state.config.fps = 25
    camera_manager._cameras = {"cam_01": state}
    stream_server = MagicMock()

    display = SelectedCameraDisplay(camera_manager, stream_server, npu_cores=0, device="cpu")
    display._running = True
    display.set_selected_camera("cam_01")

    def stop_after_first_read():
        display._running = False
        return True, frame.copy()

    cap.read.side_effect = stop_after_first_read

    display._reader_loop()

    mock_videocap.assert_called()
    assert np.array_equal(display._latest_frame, frame)


@patch("backend.main_multi.DisplayDetectionWorker")
def test_selected_camera_display_switch_clears_cached_frames_and_overlay(_mock_worker_cls):
    camera_manager = MagicMock()
    stream_server = MagicMock()
    display = SelectedCameraDisplay(camera_manager, stream_server, npu_cores=0, device="cpu")

    display._latest_frame = np.ones((8, 8, 3), dtype=np.uint8)
    display._last_detection_results = {"fire": {"boxes": [[1, 1, 2, 2]]}}
    display._overlay_expires_at = 123.0

    display.set_selected_camera("cam_02")

    assert display._selected_camera_id == "cam_02"
    assert display._latest_frame is None
    assert display._last_detection_results == {}
    assert display._overlay_expires_at == 0.0


@patch("backend.main_multi.DisplayDetectionWorker")
def test_selected_camera_display_clears_overlay_when_all_types_disabled(_mock_worker_cls):
    camera_manager = MagicMock()
    stream_server = MagicMock()
    display = SelectedCameraDisplay(camera_manager, stream_server, npu_cores=0, device="cpu")

    display._last_detection_results = {"fire": {"boxes": [[1, 1, 2, 2]]}}
    display._overlay_expires_at = 123.0

    display.set_display_types({"fire": False})

    assert display._last_detection_results == {}
    assert display._overlay_expires_at == 0.0


@patch("backend.main_multi.DisplayDetectionWorker")
@patch("backend.main_multi.MultiDetector._annotate_frame")
def test_display_loop_reuses_detection_results_on_latest_frame(mock_annotate, _mock_worker_cls):
    camera_manager = MagicMock()
    state = MagicMock()
    state.config.width = 640
    state.config.height = 480
    camera_manager._cameras = {"cam_01": state}
    stream_server = MagicMock()

    display = SelectedCameraDisplay(camera_manager, stream_server, npu_cores=0, device="cpu")
    raw = np.zeros((8, 8, 3), dtype=np.uint8)
    annotated = np.full((8, 8, 3), 255, dtype=np.uint8)
    detection_results = {"fire": {"boxes": [[1, 1, 2, 2]]}}

    display._selected_camera_id = "cam_01"
    display._latest_frame = raw
    display._last_detection_results = detection_results
    display._display_types = {"fire": True}
    display._running = True

    mock_annotate.return_value = annotated

    def stop_after_first_update(*args, **kwargs):
        display._running = False

    stream_server.update_frame.side_effect = stop_after_first_update

    display._display_loop()

    annotate_args = mock_annotate.call_args.args
    assert np.array_equal(annotate_args[0], raw)
    assert annotate_args[1] == detection_results
    pushed = stream_server.update_frame.call_args.args[1]
    assert np.array_equal(pushed, annotated)


@patch("backend.main_multi.DisplayDetectionWorker")
@patch("backend.main_multi.MultiDetector._annotate_frame")
def test_display_loop_uses_raw_frame_when_no_detection_results(mock_annotate, _mock_worker_cls):
    camera_manager = MagicMock()
    state = MagicMock()
    state.config.width = 640
    state.config.height = 480
    camera_manager._cameras = {"cam_01": state}
    stream_server = MagicMock()

    display = SelectedCameraDisplay(camera_manager, stream_server, npu_cores=0, device="cpu")
    raw = np.zeros((8, 8, 3), dtype=np.uint8)

    display._selected_camera_id = "cam_01"
    display._latest_frame = raw
    display._last_detection_results = {}
    display._display_types = {"fire": True}
    display._running = True

    def stop_after_first_update(*args, **kwargs):
        display._running = False

    stream_server.update_frame.side_effect = stop_after_first_update

    display._display_loop()

    mock_annotate.assert_not_called()
    pushed = stream_server.update_frame.call_args.args[1]
    assert np.array_equal(pushed, raw)


def test_clamp_display_interval_bounds():
    assert SelectedCameraDisplay._clamp_display_interval(0.05) == 0.1
    assert SelectedCameraDisplay._clamp_display_interval(20.0) == 10.0
    assert SelectedCameraDisplay._clamp_display_interval("abc") == 1.0
    assert SelectedCameraDisplay._clamp_display_interval(5.0) == 5.0
    assert SelectedCameraDisplay._clamp_display_interval(None) == 1.0


@patch("backend.main_multi.DisplayDetectionWorker")
def test_selected_camera_display_uses_configurable_interval(_mock_worker_cls):
    camera_manager = MagicMock()
    stream_server = MagicMock()
    display = SelectedCameraDisplay(
        camera_manager, stream_server, npu_cores=0, device="cpu", display_interval=0.5
    )
    assert display._display_interval == 0.5


@patch("backend.main_multi.DisplayDetectionWorker")
def test_set_display_config_updates_interval_and_types(_mock_worker_cls):
    camera_manager = MagicMock()
    stream_server = MagicMock()
    display = SelectedCameraDisplay(camera_manager, stream_server, npu_cores=0, device="cpu")

    display.set_display_config({"fire": True, "smoke": False}, 2.5)
    assert display._display_types == {"fire": True, "smoke": False}
    assert display._display_interval == 2.5

    # 只更新类型时，频率保持不变
    display.set_display_config({"fire": False, "smoke": True})
    assert display._display_types == {"fire": False, "smoke": True}
    assert display._display_interval == 2.5

    # 只更新频率时，类型保持不变
    display.set_display_config(display_interval=0.8)
    assert display._display_types == {"fire": False, "smoke": True}
    assert display._display_interval == 0.8


@patch("backend.main_multi.DisplayDetectionWorker")
def test_set_display_config_interval_is_clamped(_mock_worker_cls):
    camera_manager = MagicMock()
    stream_server = MagicMock()
    display = SelectedCameraDisplay(camera_manager, stream_server, npu_cores=0, device="cpu")

    display.set_display_config(display_interval=0.05)
    assert display._display_interval == 0.1

    display.set_display_config(display_interval=20.0)
    assert display._display_interval == 10.0


@patch("backend.main_multi.DisplayDetectionWorker")
def test_detect_loop_submits_without_blocking(_mock_worker_cls):
    camera_manager = MagicMock()
    stream_server = MagicMock()
    display = SelectedCameraDisplay(camera_manager, stream_server, npu_cores=0, device="cpu")

    worker = MagicMock()
    worker.submit.return_value = True
    display._detection_worker = worker
    display._latest_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    display._selected_camera_id = "cam_01"
    display._display_types = {"fire": True}
    display._display_interval = 0.001
    display._running = True

    def stop_after_submit(*args, **kwargs):
        display._running = False
        return True

    worker.submit.side_effect = stop_after_submit

    display._detect_loop()

    worker.submit.assert_called_once()
    worker.detect.assert_not_called()
    submitted_frame, submitted_types = worker.submit.call_args.args
    assert np.array_equal(submitted_frame, display._latest_frame)
    assert submitted_types == ["fire"]


@patch("backend.main_multi.DisplayDetectionWorker")
def test_detect_loop_skips_when_worker_busy(_mock_worker_cls):
    camera_manager = MagicMock()
    stream_server = MagicMock()
    display = SelectedCameraDisplay(camera_manager, stream_server, npu_cores=0, device="cpu")

    worker = MagicMock()
    worker.submit.return_value = False
    display._detection_worker = worker
    display._latest_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    display._selected_camera_id = "cam_01"
    display._display_types = {"fire": True}
    display._display_interval = 0.001
    display._running = True

    call_count = [0]

    def count_submits(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] >= 2:
            display._running = False
        return False

    worker.submit.side_effect = count_submits

    display._detect_loop()

    assert worker.submit.call_count == 2
    worker.detect.assert_not_called()


@patch("backend.main_multi.DisplayDetectionWorker")
def test_result_loop_updates_detection_results(_mock_worker_cls):
    camera_manager = MagicMock()
    stream_server = MagicMock()
    display = SelectedCameraDisplay(camera_manager, stream_server, npu_cores=0, device="cpu")

    worker = MagicMock()
    results = [{"fire": {"boxes": [[1, 1, 2, 2]]}}, None]

    def side_effect(*args, **kwargs):
        if not results:
            display._running = False
            return None
        return results.pop(0)

    worker.get_result.side_effect = side_effect
    display._detection_worker = worker
    display._display_types = {"fire": True}
    display._running = True

    display._result_loop()

    assert display._last_detection_results == {"fire": {"boxes": [[1, 1, 2, 2]]}}


@patch("backend.main_multi.DisplayDetectionWorker")
def test_result_loop_filters_disabled_types(_mock_worker_cls):
    camera_manager = MagicMock()
    stream_server = MagicMock()
    display = SelectedCameraDisplay(camera_manager, stream_server, npu_cores=0, device="cpu")

    worker = MagicMock()

    def side_effect(*args, **kwargs):
        display._running = False
        return {"fire": {"boxes": [[1, 1, 2, 2]]}, "smoke": {"boxes": [[3, 3, 4, 4]]}}

    worker.get_result.side_effect = side_effect
    display._detection_worker = worker
    display._display_types = {"fire": True, "smoke": False}
    display._running = True

    display._result_loop()

    assert display._last_detection_results == {"fire": {"boxes": [[1, 1, 2, 2]]}}


@patch("backend.main_multi.DisplayDetectionWorker")
def test_display_detection_worker_submit_and_get_result(_mock_worker_cls):
    from backend.display_detection_worker import DisplayDetectionWorker

    worker = DisplayDetectionWorker(npu_cores=0, device="cpu")
    worker._input_queue = MagicMock()
    worker._output_queue = MagicMock()

    worker._input_queue.put_nowait.return_value = None
    assert worker.submit(np.zeros((8, 8, 3), dtype=np.uint8), ["fire"]) is True
    worker._input_queue.put_nowait.assert_called_once()

    worker._input_queue.put_nowait.side_effect = Exception("full")
    assert worker.submit(np.zeros((8, 8, 3), dtype=np.uint8), ["fire"]) is False

    worker._output_queue.get.side_effect = Exception("empty")
    assert worker.get_result(timeout=0.1) is None
