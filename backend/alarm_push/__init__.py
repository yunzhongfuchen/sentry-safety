"""报警推送模块：通道抽象 + 管理器

新增推送通道（飞书/钉钉等）只需：
1. 新建文件继承 AlarmPushChannel，实现 send_created / send_reviewed
2. 在 main_multi.init_components 中根据配置追加到 PushManager 的 channels
"""

from backend.alarm_push.base import AlarmPushChannel
from backend.alarm_push.manager import PushManager

__all__ = ["AlarmPushChannel", "PushManager"]
