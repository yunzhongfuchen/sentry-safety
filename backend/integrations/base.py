"""
外部推送通道抽象基类
"""

from abc import ABC, abstractmethod
from typing import Optional


class AlarmPushChannel(ABC):
    """报警推送通道基类。
    每个通道负责把一条告警记录推送到一个外部系统（国经 / 飞书 / 钉钉 / 其它定制系统）。
    实现约定：失败只记日志、返回 False，绝不抛异常（不能影响检测主流程）。

    types：通道订阅的检测类型列表；None 表示全部类型（含以后新增）。
    """

    name: str = "base"
    types = None

    def accepts(self, record: dict) -> bool:
        """判断该通道是否订阅了这条报警的检测类型"""
        if self.types is None:
            return True
        return record.get("detection_type") in self.types

    @abstractmethod
    def send_created(self, record: dict, snapshot_b64: Optional[str], frames_b64: list) -> bool:
        """推送告警产生报文（含快照及检测帧）"""

    @abstractmethod
    def send_reviewed(self, record: dict) -> bool:
        """推送告警复核/状态更新报文（不含图片）"""
