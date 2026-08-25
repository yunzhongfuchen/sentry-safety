"""
国经公司 HTTP Webhook 告警推送通道实现
"""

import json
import logging
import time
from typing import Optional

import requests

from backend.detection_registry import registry
from backend.integrations.base import AlarmPushChannel

logger = logging.getLogger(__name__)


class GuojingWebhookChannel(AlarmPushChannel):
    """国经公司 webhook 通道：把告警报文以 JSON POST 发送到国经平台接收接口。

    报文结构规范：
    外层信封 {event, event_id, sent_at, data}，data 为报警业务数据。
    """

    name = "guojing_webhook"

    def __init__(self, url: str, timeout: float = 5.0, log=None):
        self.url = url
        self.timeout = timeout
        self._log = log or (lambda msg, level="info": getattr(logger, level, logger.info)(msg))

    def build_event(
        self,
        event: str,
        record: dict,
        snapshot_b64: Optional[str],
        frames_b64: list,
    ) -> dict:
        """组装国经平台协议报文。alarm.created 带图片，alarm.reviewed 不带。"""
        type_def = registry.get(record.get("detection_type", "")) or {}
        data = {
            "id": record.get("id"),
            "camera_id": record.get("camera_id"),
            "detection_type": record.get("detection_type"),
            "detection_label": type_def.get("label", record.get("detection_type")),
            "level": record.get("level"),
            "status": record.get("status"),
            "time": record.get("time"),
            "confidence": record.get("confidence"),
            "reason": record.get("reason"),
            "small_model": record.get("small_model"),
            "vlm_review": record.get("vlm_review"),
            "source": record.get("source", "small_model"),
            "frame_count": record.get("frame_count", 0),
        }
        if event == "alarm.created":
            data["snapshot"] = snapshot_b64 or ""
            data["frames"] = frames_b64

        return {
            "event": event,
            "event_id": f"{record.get('id')}_{event.split('.')[-1]}",
            "sent_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "data": data,
        }

    def _post(self, payload: dict) -> bool:
        event_id = payload.get("event_id", "?")
        try:
            resp = requests.post(
                self.url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8"},
                timeout=self.timeout,
            )
        except Exception as e:
            self._log(f"Alarm push {event_id} to {self.url} failed: {e}", "error")
            return False

        if resp.status_code != 200:
            self._log(f"Alarm push {event_id} to {self.url} failed: HTTP {resp.status_code}", "error")
            return False

        try:
            code = resp.json().get("code")
        except ValueError:
            code = None
        if code != 0:
            self._log(f"Alarm push {event_id} to {self.url} rejected: body={resp.text[:200]}", "error")
            return False

        self._log(f"Alarm push {event_id} to {self.url} succeeded", "info")
        return True

    def send_created(self, record: dict, snapshot_b64: Optional[str], frames_b64: list) -> bool:
        return self._post(self.build_event("alarm.created", record, snapshot_b64, frames_b64))

    def send_reviewed(self, record: dict) -> bool:
        return self._post(self.build_event("alarm.reviewed", record, None, []))


# 别名兼容
WebhookChannel = GuojingWebhookChannel
