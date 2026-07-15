"""流故障防护修复测试：reopen 退避、竞态防护、主画面切换节流"""
import threading
import time
from unittest.mock import MagicMock, patch

from backend.camera_manager import CameraManager, CameraConfig
from backend import main_multi


class TestReopenBackoff:
    """_reopen_capture 指数退避"""

    def _make_manager(self):
        cm = CameraManager()
        cm.register_camera(CameraConfig(camera_id="cam_01", source="0"))
        state = cm._cameras["cam_01"]
        state.running = True
        state.cap = MagicMock()
        return cm, state

    def test_reopen_sleeps_with_exponential_backoff(self):
        """连续 reopen 的退避时间应为 1s→2s→4s"""
        cm, state = self._make_manager()
        sleep_times = []
        with patch.object(cm, "_open_capture"), \
             patch("backend.camera_manager.time.sleep", side_effect=lambda s: sleep_times.append(s)):
            for _ in range(3):
                cm._reopen_capture("cam_01")
        assert sleep_times == [1.0, 2.0, 4.0]

    def test_reopen_backoff_capped_at_30s(self):
        """退避上限 30s"""
        cm, state = self._make_manager()
        state.reconnect_attempts = 10
        sleep_times = []
        with patch.object(cm, "_open_capture"), \
             patch("backend.camera_manager.time.sleep", side_effect=lambda s: sleep_times.append(s)):
            cm._reopen_capture("cam_01")
        assert sleep_times == [30.0]


class TestReopenRaceGuard:
    """stop 之后旧 reader 线程的 reopen 必须放弃"""

    def test_reopen_skipped_when_not_running(self):
        cm = CameraManager()
        cm.register_camera(CameraConfig(camera_id="cam_01", source="0"))
        state = cm._cameras["cam_01"]
        state.running = False  # 已停止
        state.cap = MagicMock()
        attempts_before = state.reconnect_attempts

        with patch.object(cm, "_open_capture") as mock_open:
            cm._reopen_capture("cam_01")

        mock_open.assert_not_called()
        assert state.reconnect_attempts == attempts_before  # 未递增，说明直接返回
        state.cap.release.assert_not_called()  # 不应 release

    def test_reopen_abandoned_during_backoff(self):
        """退避期间被 stop，reopen 应放弃且不调用 _open_capture"""
        cm = CameraManager()
        cm.register_camera(CameraConfig(camera_id="cam_01", source="0"))
        state = cm._cameras["cam_01"]
        state.running = True
        state.cap = MagicMock()

        def stop_during_sleep(_):
            state.running = False

        with patch.object(cm, "_open_capture") as mock_open, \
             patch("backend.camera_manager.time.sleep", side_effect=stop_during_sleep):
            cm._reopen_capture("cam_01")

        mock_open.assert_not_called()


class TestMainCameraThrottle:
    """set_main_camera 500ms 节流"""

    def setup_method(self):
        # 每个测试前重置节流状态
        with main_multi._main_switch_lock:
            if main_multi._main_switch_timer is not None:
                main_multi._main_switch_timer.cancel()
                main_multi._main_switch_timer = None
            main_multi._main_switch_pending = main_multi._MAIN_SWITCH_UNSET

    def teardown_method(self):
        self.setup_method()

    def test_first_switch_executes_immediately(self):
        with patch.object(main_multi, "_do_set_main_camera") as mock_do:
            main_multi.set_main_camera("cam_01")
        mock_do.assert_called_once_with("cam_01")

    def test_rapid_switches_coalesce_to_latest(self):
        """节流窗口内连续切换只保留最新目标"""
        with patch.object(main_multi, "_do_set_main_camera") as mock_do:
            main_multi.set_main_camera("cam_01")  # 立即执行
            main_multi.set_main_camera("cam_02")  # pending
            main_multi.set_main_camera("cam_03")  # pending（覆盖 cam_02）
            assert mock_do.call_count == 1
            # 触发节流窗口结束
            main_multi._end_main_switch_throttle()
            assert mock_do.call_count == 2
            mock_do.assert_called_with("cam_03")

    def test_pending_none_switch_not_lost(self):
        """pending 目标为 None（取消主画面）也应被保留执行"""
        with patch.object(main_multi, "_do_set_main_camera") as mock_do:
            main_multi.set_main_camera("cam_01")
            main_multi.set_main_camera(None)
            main_multi._end_main_switch_throttle()
            assert mock_do.call_count == 2
            assert mock_do.call_args_list[1].args[0] is None
