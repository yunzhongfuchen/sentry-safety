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
from backend.integrations.dingtalk.channel import DingTalkChannel
from backend.integrations.feishu.channel import FeishuChannel

logger = logging.getLogger(__name__)

_CHANNEL_CLASSES = {
    "dingtalk": DingTalkChannel,
    "feishu": FeishuChannel,
}


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

    # 2. 群机器人推送通道（卡片列表，每张卡可选平台并订阅不同检测类型）
    #    统一键 push_channels（带 platform 字段）；兼容旧键 dingtalk_channels（视为钉钉卡）
    push_cards = global_settings.get("push_channels")
    if push_cards is None:
        push_cards = [
            {**card, "platform": "dingtalk"}
            for card in (global_settings.get("dingtalk_channels") or [])
        ]
    for card in push_cards:
        url = (card.get("webhook_url") or "").strip()
        if not url:
            continue
        channel_cls = _CHANNEL_CLASSES.get(card.get("platform", "dingtalk"))
        if channel_cls is None:
            logger.warning(f"Unknown push channel platform: {card.get('platform')}, skipped")
            continue
        types = None if card.get("all_types", True) else (card.get("types") or [])
        channels.append(channel_cls(
            url,
            secret=(card.get("secret") or "").strip(),
            snapshot_base_url=(card.get("base_url") or "").strip(),
            types=types,
            log=log,
        ))
        card_name = card.get("name") or url[-24:]
        if log:
            log(f"Integration channel [{channel_cls.name}:{card_name}] enabled, types={'all' if types is None else types}", "info")
        else:
            logger.info(f"Integration channel [{channel_cls.name}:{card_name}] enabled, types={'all' if types is None else types}")

    # 旧版单通道配置兼容：未配置卡片列表但填了旧字段时，按"全部类型"建一张钉钉卡
    legacy_url = global_settings.get("dingtalk_webhook_url", "").strip()
    if not push_cards and legacy_url:
        channels.append(DingTalkChannel(
            legacy_url,
            secret=global_settings.get("dingtalk_secret", "").strip(),
            snapshot_base_url=global_settings.get("dingtalk_base_url", "").strip(),
            log=log,
        ))
        if log:
            log(f"Integration channel [DingTalk] enabled: {legacy_url}", "info")
        else:
            logger.info(f"Integration channel [DingTalk] enabled: {legacy_url}")

    # 后续若有其它公司的推送通道（如 飞书/某公司定制推送），在此追加即可

    return IntegrationManager(channels)


__all__ = [
    "AlarmPushChannel",
    "IntegrationManager",
    "integrations_router",
    "init_integration_manager",
    "GuojingWebhookChannel",
    "DingTalkChannel",
    "FeishuChannel",
]
