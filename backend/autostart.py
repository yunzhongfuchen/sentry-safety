# -*- coding: utf-8 -*-
"""Windows 开机自启管理（计划任务方式）

- 优先注册"系统启动时"任务（ONSTART，无需登录，需管理员权限）
- 权限不足时降级为"用户登录时"任务（ONLOGON，普通用户可注册）
- 启动器脚本 autostart_run.bat 在开启时生成，内嵌当前后端进程的 python 路径，
  避免 SYSTEM 账户环境下 %USERPROFILE% 指向错误导致找不到解释器
"""

import subprocess
import sys
from pathlib import Path
from typing import Tuple

TASK_NAME = "SentrySafetyDetection"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER_PATH = PROJECT_ROOT / "autostart_run.bat"
REGISTER_BAT_PATH = PROJECT_ROOT / "autostart_register.bat"


def supported() -> bool:
    return sys.platform == "win32"


def _is_admin() -> bool:
    """当前进程是否具有管理员权限"""
    if not supported():
        return False
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _run(args: list) -> Tuple[int, str]:
    """执行 schtasks，返回 (退出码, 输出)。中文系统输出为 GBK。"""
    proc = subprocess.run(args, capture_output=True, timeout=15)
    out = (proc.stdout or b"").decode("gbk", errors="replace")
    err = (proc.stderr or b"").decode("gbk", errors="replace")
    return proc.returncode, (out + err).strip()


def is_enabled() -> bool:
    if not supported():
        return False
    code, _ = _run(["schtasks", "/Query", "/TN", TASK_NAME])
    return code == 0


def _write_launcher() -> None:
    content = (
        "@echo off\r\n"
        f'cd /d "{PROJECT_ROOT}"\r\n'
        "if not exist logs mkdir logs\r\n"
        f'"{sys.executable}" backend\\main_multi.py >> logs\\main_multi.log 2>&1\r\n'
    )
    LAUNCHER_PATH.write_text(content, encoding="gbk")


def _create_onstart_task() -> Tuple[int, str]:
    """直接注册"系统启动时"任务（当前进程有管理员权限时才能成功）"""
    return _run([
        "schtasks", "/Create", "/TN", TASK_NAME,
        "/TR", f'"{LAUNCHER_PATH}"',
        "/SC", "ONSTART", "/RL", "HIGHEST", "/F",
    ])


def _elevate_and_create() -> bool:
    """触发 UAC 授权弹窗，以管理员权限注册 ONSTART 任务。

    把注册命令写入临时 bat，用 PowerShell Start-Process -Verb RunAs 拉起：
    用户在 UAC 弹窗点"是"则 bat 以管理员身份执行；点"否"/取消则不执行。
    通过事后查询任务是否存在来判定结果（runas 方式拿不到子进程退出码）。
    """
    register_cmd = (
        f'@echo off\r\nschtasks /Create /TN {TASK_NAME} '
        f'/TR "\\"{LAUNCHER_PATH}\\"" /SC ONSTART /RL HIGHEST /F\r\n'
    )
    try:
        REGISTER_BAT_PATH.write_text(register_cmd, encoding="gbk")
    except OSError:
        return False
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f'Start-Process -FilePath "{REGISTER_BAT_PATH}" -Verb RunAs -Wait'],
            capture_output=True, timeout=120,
        )
    except Exception:
        return False
    finally:
        try:
            REGISTER_BAT_PATH.unlink(missing_ok=True)
        except OSError:
            pass
    return is_enabled()


def enable() -> Tuple[bool, str]:
    if not supported():
        return False, "当前系统不支持开机自启（仅支持 Windows）"
    try:
        _write_launcher()
    except OSError as e:
        return False, f"生成启动脚本失败：{e}"

    # 1. 当前进程有管理员权限：直接注册，无弹窗
    if _is_admin():
        code, out = _create_onstart_task()
        if code == 0:
            return True, "已开启开机自启（系统启动时运行，无需登录）"
        return False, f"注册计划任务失败：{out}"

    # 2. 无管理员权限：弹 UAC 授权框，用户同意后以管理员身份注册
    if _elevate_and_create():
        return True, "已开启开机自启（系统启动时运行，无需登录）"

    # 3. 用户取消授权或提权失败：降级为当前用户登录时启动（无需管理员）
    code, out = _run([
        "schtasks", "/Create", "/TN", TASK_NAME,
        "/TR", f'"{LAUNCHER_PATH}"',
        "/SC", "ONLOGON", "/F",
    ])
    if code == 0:
        return True, "已开启自启（当前用户登录时运行；管理员授权已取消，如需免登录自启请重试并在弹窗中点击『是』）"
    return False, f"注册计划任务失败：{out}"


def disable() -> Tuple[bool, str]:
    if not supported():
        return False, "当前系统不支持开机自启（仅支持 Windows）"
    code, out = _run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"])
    if code != 0 and not _is_admin():
        # 系统级任务删除需要管理员权限，走 UAC 弹窗
        register_cmd = f'@echo off\r\nschtasks /Delete /TN {TASK_NAME} /F\r\n'
        try:
            REGISTER_BAT_PATH.write_text(register_cmd, encoding="gbk")
            subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f'Start-Process -FilePath "{REGISTER_BAT_PATH}" -Verb RunAs -Wait'],
                capture_output=True, timeout=120,
            )
        except Exception:
            pass
        finally:
            try:
                REGISTER_BAT_PATH.unlink(missing_ok=True)
            except OSError:
                pass
        if is_enabled():
            return False, "删除计划任务失败：需要管理员权限，请在弹窗中点击『是』"
        code = 0
    if code == 0:
        try:
            LAUNCHER_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        return True, "已关闭开机自启"
    return False, f"删除计划任务失败：{out}"
