"""开机自启（Windows 计划任务）模块与接口测试"""

from unittest.mock import patch, MagicMock

import backend.autostart as autostart


def _ok_run(args, **_):
    return MagicMock(returncode=0, stdout="成功".encode("gbk"), stderr=b"")


def _fail_run(args, **_):
    return MagicMock(returncode=1, stdout="错误: 拒绝访问。".encode("gbk"), stderr=b"")


class TestAutostartModule:
    def test_unsupported_on_non_windows(self):
        with patch.object(autostart.sys, "platform", "linux"):
            assert autostart.supported() is False
            assert autostart.is_enabled() is False
            ok, msg = autostart.enable()
            assert ok is False and "不支持" in msg
            ok, msg = autostart.disable()
            assert ok is False and "不支持" in msg

    def test_is_enabled_true_when_task_exists(self):
        with patch.object(autostart.sys, "platform", "win32"), \
             patch.object(autostart.subprocess, "run", side_effect=_ok_run):
            assert autostart.is_enabled() is True

    def test_is_enabled_false_when_task_missing(self):
        with patch.object(autostart.sys, "platform", "win32"), \
             patch.object(autostart.subprocess, "run", side_effect=_fail_run):
            assert autostart.is_enabled() is False

    def test_enable_registers_onstart_task(self, tmp_path):
        calls = []
        launcher = tmp_path / "autostart_run.bat"

        def tracking_run(args, **_):
            calls.append(args)
            return MagicMock(returncode=0, stdout=b"OK", stderr=b"")

        with patch.object(autostart.sys, "platform", "win32"), \
             patch.object(autostart, "_is_admin", return_value=True), \
             patch.object(autostart, "LAUNCHER_PATH", launcher), \
             patch.object(autostart.subprocess, "run", side_effect=tracking_run):
            ok, msg = autostart.enable()

        assert ok is True and "无需登录" in msg
        assert len(calls) == 1
        assert "ONSTART" in calls[0]
        # 启动器内嵌当前 python 路径与项目目录
        content = launcher.read_text(encoding="gbk")
        assert autostart.sys.executable in content
        assert "backend\\main_multi.py" in content

    def test_enable_elevates_via_uac_when_not_admin(self, tmp_path):
        """非管理员：走 UAC 弹窗提权注册，用户点"是"后任务存在即成功"""
        with patch.object(autostart.sys, "platform", "win32"), \
             patch.object(autostart, "_is_admin", return_value=False), \
             patch.object(autostart, "LAUNCHER_PATH", tmp_path / "autostart_run.bat"), \
             patch.object(autostart, "_elevate_and_create", return_value=True) as mock_elevate:
            ok, msg = autostart.enable()
        assert ok is True and "无需登录" in msg
        mock_elevate.assert_called_once()

    def test_enable_falls_back_to_onlogon_when_uac_cancelled(self, tmp_path):
        """UAC 弹窗被取消后降级为登录时启动"""
        calls = []

        def run_side_effect(args, **_):
            calls.append(args)
            return MagicMock(returncode=0, stdout=b"OK", stderr=b"")

        with patch.object(autostart.sys, "platform", "win32"), \
             patch.object(autostart, "_is_admin", return_value=False), \
             patch.object(autostart, "LAUNCHER_PATH", tmp_path / "autostart_run.bat"), \
             patch.object(autostart, "_elevate_and_create", return_value=False), \
             patch.object(autostart.subprocess, "run", side_effect=run_side_effect):
            ok, msg = autostart.enable()

        assert ok is True and "登录时运行" in msg
        assert len(calls) == 1
        assert "ONLOGON" in calls[0]

    def test_enable_reports_failure_when_all_paths_fail(self, tmp_path):
        with patch.object(autostart.sys, "platform", "win32"), \
             patch.object(autostart, "_is_admin", return_value=False), \
             patch.object(autostart, "LAUNCHER_PATH", tmp_path / "autostart_run.bat"), \
             patch.object(autostart, "_elevate_and_create", return_value=False), \
             patch.object(autostart.subprocess, "run", side_effect=_fail_run):
            ok, msg = autostart.enable()
        assert ok is False and "失败" in msg

    def test_elevate_and_create_uses_powershell_runas(self, tmp_path):
        """UAC 提权通过 PowerShell Start-Process -Verb RunAs 拉起注册脚本"""
        register_bat = tmp_path / "autostart_register.bat"
        ps_calls = []

        def run_side_effect(args, **_):
            if args[0] == "powershell":
                ps_calls.append(args)
                return MagicMock(returncode=0, stdout=b"", stderr=b"")
            # schtasks /Query：任务已存在
            return MagicMock(returncode=0, stdout=b"OK", stderr=b"")

        with patch.object(autostart.sys, "platform", "win32"), \
             patch.object(autostart, "LAUNCHER_PATH", tmp_path / "autostart_run.bat"), \
             patch.object(autostart, "REGISTER_BAT_PATH", register_bat), \
             patch.object(autostart.subprocess, "run", side_effect=run_side_effect):
            assert autostart._elevate_and_create() is True

        assert len(ps_calls) == 1
        ps_cmd = ps_calls[0][-1]
        assert "-Verb RunAs" in ps_cmd and "-Wait" in ps_cmd
        # 注册脚本已自清理
        assert not register_bat.exists()

    def test_elevate_and_create_returns_false_when_uac_cancelled(self, tmp_path):
        """用户取消 UAC：注册脚本未执行，任务查询不到 → False"""

        def run_side_effect(args, **_):
            if args[0] == "powershell":
                return MagicMock(returncode=1, stdout=b"", stderr="操作已被用户取消".encode("gbk"))
            return MagicMock(returncode=1, stdout=b"", stderr=b"")

        with patch.object(autostart.sys, "platform", "win32"), \
             patch.object(autostart, "LAUNCHER_PATH", tmp_path / "autostart_run.bat"), \
             patch.object(autostart, "REGISTER_BAT_PATH", tmp_path / "autostart_register.bat"), \
             patch.object(autostart.subprocess, "run", side_effect=run_side_effect):
            assert autostart._elevate_and_create() is False

    def test_disable_deletes_task_and_launcher(self, tmp_path):
        launcher = tmp_path / "autostart_run.bat"
        launcher.write_text("x", encoding="gbk")
        with patch.object(autostart.sys, "platform", "win32"), \
             patch.object(autostart, "LAUNCHER_PATH", launcher), \
             patch.object(autostart.subprocess, "run", side_effect=_ok_run):
            ok, msg = autostart.disable()
        assert ok is True
        assert not launcher.exists()

    def test_disable_elevates_when_direct_delete_denied(self, tmp_path):
        """直接删除被拒且非管理员 → 走 UAC 提权删除，成功后任务不存在"""
        ps_called = []

        def run_side_effect(args, **_):
            if args[0] == "powershell":
                ps_called.append(args)
                return MagicMock(returncode=0, stdout=b"", stderr=b"")
            if "/Delete" in args:
                return _fail_run(args)
            # 提权删除后 Query 查不到任务
            return _fail_run(args)

        with patch.object(autostart.sys, "platform", "win32"), \
             patch.object(autostart, "_is_admin", return_value=False), \
             patch.object(autostart, "LAUNCHER_PATH", tmp_path / "autostart_run.bat"), \
             patch.object(autostart, "REGISTER_BAT_PATH", tmp_path / "autostart_register.bat"), \
             patch.object(autostart.subprocess, "run", side_effect=run_side_effect):
            ok, msg = autostart.disable()
        assert ok is True
        assert ps_called, "应当触发了 UAC 提权删除"


class TestAutostartEndpoints:
    def test_get_status(self):
        from fastapi.testclient import TestClient
        from backend.main_multi import app

        with patch("backend.autostart.supported", return_value=True), \
             patch("backend.autostart.is_enabled", return_value=False):
            client = TestClient(app)
            resp = client.get("/autostart")
        assert resp.status_code == 200
        assert resp.json() == {"supported": True, "enabled": False}

    def test_post_enable_success(self):
        from fastapi.testclient import TestClient
        from backend.main_multi import app

        with patch("backend.autostart.supported", return_value=True), \
             patch("backend.autostart.enable", return_value=(True, "已开启开机自启")) as mock_enable, \
             patch("backend.autostart.is_enabled", return_value=True):
            client = TestClient(app)
            resp = client.post("/autostart", json={"enabled": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True and data["enabled"] is True
        mock_enable.assert_called_once()

    def test_post_disable_success(self):
        from fastapi.testclient import TestClient
        from backend.main_multi import app

        with patch("backend.autostart.supported", return_value=True), \
             patch("backend.autostart.disable", return_value=(True, "已关闭开机自启")) as mock_disable, \
             patch("backend.autostart.is_enabled", return_value=False):
            client = TestClient(app)
            resp = client.post("/autostart", json={"enabled": False})
        assert resp.status_code == 200
        mock_disable.assert_called_once()

    def test_post_unsupported_returns_400(self):
        from fastapi.testclient import TestClient
        from backend.main_multi import app

        with patch("backend.autostart.supported", return_value=False):
            client = TestClient(app)
            resp = client.post("/autostart", json={"enabled": True})
        assert resp.status_code == 400
        assert resp.json()["ok"] is False

    def test_post_enable_failure_returns_400_with_message(self):
        from fastapi.testclient import TestClient
        from backend.main_multi import app

        with patch("backend.autostart.supported", return_value=True), \
             patch("backend.autostart.enable", return_value=(False, "注册计划任务失败：拒绝访问")):
            client = TestClient(app)
            resp = client.post("/autostart", json={"enabled": True})
        assert resp.status_code == 400
        assert "拒绝访问" in resp.json()["error"]
