"""
外部系统集成层 (Integrations Layer)
统一对外暴露初始化工厂、推送管理器与 API 聚合路由
"""

import logging
from typing import Callable, Dict, Optional

from backend.integrations.base import AlarmPushChannel
from backend.integrations.manager import IntegrationManager
from backend.integrations.router import router as integrations_router
from backend.integrations.guojing.channel import GuojingWebhookChannel

logger = logging.getLogger(__name__)


def init_integration_manager(
    global_settings: Dict,
    log: Optional[Callable[[str, str], None]] = None,
) -> IntegrationManager:
    """根据全局配置初始化所有启用的外部推送通道"""
    channels = []

    # 1. 国经公司 webhook 推送通道
    guojing_url = global_settings.get("alarm_push_webhook_url", "").strip()
    if guojing_url:
        channels.append(GuojingWebhookChannel(guojing_url, log=log))
        if log:
            log(f"Integration channel [Guojing Webhook] enabled: {guojing_url}", "info")
        else:
            logger.info(f"Integration channel [Guojing Webhook] enabled: {guojing_url}")

    # 后续若有其它公司的推送通道（如 飞书/钉钉/某公司定制推送），在此追加即可

    return IntegrationManager(channels)


__all__ = [
    "AlarmPushChannel",
    "IntegrationManager",
    "integrations_router",
    "init_integration_manager",
    "GuojingWebhookChannel",
]
