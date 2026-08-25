"""
国经公司 (Guojing) 外部对接适配器
"""

from backend.integrations.guojing.channel import GuojingWebhookChannel, WebhookChannel

__all__ = ["GuojingWebhookChannel", "WebhookChannel"]
