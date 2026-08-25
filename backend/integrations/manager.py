"""
外部集成管理器：持有所有启用的推送通道，统一分发
"""

import logging
from typing import List, Optional

from backend.integrations.base import AlarmPushChannel

logger = logging.getLogger(__name__)


class IntegrationManager:
    """管理所有外部推送通道，收到告警时广播给每个启用的通道。
    单通道失败不影响其他通道，也不向上抛异常。
    """

    def __init__(self, channels: Optional[List[AlarmPushChannel]] = None):
        self.channels: List[AlarmPushChannel] = list(channels or [])

    @property
    def enabled(self) -> bool:
        return len(self.channels) > 0

    def push_created(self, record: dict, snapshot_b64: Optional[str], frames_b64: list) -> None:
        for ch in self.channels:
            try:
                ch.send_created(record, snapshot_b64, frames_b64)
            except Exception as e:
                logger.warning(f"Integration channel '{ch.name}' created push failed: {e}")

    def push_reviewed(self, record: dict) -> None:
        for ch in self.channels:
            try:
                ch.send_reviewed(record)
            except Exception as e:
                logger.warning(f"Integration channel '{ch.name}' reviewed push failed: {e}")
