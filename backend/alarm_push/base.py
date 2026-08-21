"""报警推送通道抽象基类"""

from abc import ABC, abstractmethod
from typing import Optional


class AlarmPushChannel(ABC):
    """报警推送通道基类。

    每个通道负责把一条告警记录推送到一个外部系统（webhook / 飞书 / 钉钉 ...）。
    实现约定：失败只记日志、返回 False，绝不抛异常（不能影响检测主流程）。
    """

    name: str = "base"

    @abstractmethod
    def send_created(self, record: dict, snapshot_b64: Optional[str], frames_b64: list) -> bool:
        """推送 alarm.created 报文（含图片），成功返回 True"""

    @abstractmethod
    def send_reviewed(self, record: dict) -> bool:
        """推送 alarm.reviewed 报文（不含图片），成功返回 True"""
