"""
报警推送模块（向后兼容重定向至 backend.integrations）
"""

from backend.integrations.base import AlarmPushChannel
from backend.integrations.manager import IntegrationManager as PushManager
from backend.integrations.guojing.channel import GuojingWebhookChannel as WebhookChannel

__all__ = ["AlarmPushChannel", "PushManager", "WebhookChannel"]
