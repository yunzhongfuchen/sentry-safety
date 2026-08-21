"""HTTP webhook 推送通道：POST JSON 报文到外部接收接口"""

import json
import logging
import time
from typing import Optional

import requests

from backend.alarm_push.base import AlarmPushChannel
from backend.detection_registry import registry

logger = logging.getLogger(__name__)


class WebhookChannel(AlarmPushChannel):
    """通用 webhook 通道：把告警报文以 JSON POST 到配置的 URL。

    报文结构遵循 docs/api/alarm_push_api.md：
    信封 {event, event_id, sent_at, data}，data 为完整报警记录。
    """

    name = "webhook"

    def __init__(self, url: str, timeout: float = 5.0, log=None):
        self.url = url
        self.timeout = timeout
        # 日志回调，签名 log(msg, level)；默认走 logging，由上层注入 log_message 以进前端日志面板
        self._log = log or (lambda msg, level="info": getattr(logger, level, logger.info)(msg))

    # ------------------------------------------------------------------
    # 报文组装
    # ------------------------------------------------------------------
    def build_event(self, event: str, record: dict,
                    snapshot_b64: Optional[str], frames_b64: list) -> dict:
        """组装信封 + 数据两层报文。created 带图片，reviewed 不带。"""
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

    # ------------------------------------------------------------------
    # 推送
    # ------------------------------------------------------------------
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

        # 文档约定：成功 = HTTP 200 且 body code == 0
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
