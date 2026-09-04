"""
钉钉自定义机器人告警推送通道实现

机器人文档：消息类型 markdown；安全设置支持"加签"（timestamp + HMAC-SHA256 签名）。
成功判定：HTTP 200 且响应体 errcode == 0。
"""

import base64
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
from typing import Optional

import requests

from backend.detection_registry import registry
from backend.integrations.base import AlarmPushChannel

logger = logging.getLogger(__name__)


class DingTalkChannel(AlarmPushChannel):
    """钉钉群机器人通道：把告警以 markdown 卡片消息推送到钉钉群。

    - secret 非空时在 webhook 地址后附加 timestamp/sign（对应机器人"加签"安全设置）
    - snapshot_base_url 非空时在消息中嵌入报警快照图片链接
    """

    name = "dingtalk"

    def __init__(
        self,
        webhook_url: str,
        secret: str = "",
        snapshot_base_url: str = "",
        types=None,
        timeout: float = 5.0,
        log=None,
    ):
        self.webhook_url = webhook_url
        self.secret = secret
        self.snapshot_base_url = snapshot_base_url.rstrip("/")
        self.types = types  # None=全部类型；列表=只推送勾选的检测类型
        self.timeout = timeout
        self._log = log or (lambda msg, level="info": getattr(logger, level, logger.info)(msg))

    def _signed_url(self) -> str:
        """加签：timestamp\nsecret 的 HMAC-SHA256，base64 后 urlencode 拼到 webhook 上"""
        if not self.secret:
            return self.webhook_url
        timestamp = str(round(time.time() * 1000))
        string_to_sign = f"{timestamp}\n{self.secret}"
        sign = urllib.parse.quote_plus(
            base64.b64encode(
                hmac.new(self.secret.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha256).digest()
            )
        )
        sep = "&" if "?" in self.webhook_url else "?"
        return f"{self.webhook_url}{sep}timestamp={timestamp}&sign={sign}"

    def _snapshot_link(self, record: dict) -> str:
        if not self.snapshot_base_url:
            return ""
        record_id = urllib.parse.quote(str(record.get("id", "")), safe="")
        return f"{self.snapshot_base_url}/cvApi/open/api/cv/warning/image/{record_id}"

    def build_message(self, event: str, record: dict) -> dict:
        """组装钉钉 markdown 消息。alarm.created 可附快照图，alarm.reviewed 为复核结论。"""
        type_def = registry.get(record.get("detection_type", "")) or {}
        label = type_def.get("label", record.get("detection_type"))
        confidence = record.get("confidence") or 0

        lines = [
            f"**报警类型**：{label}",
            f"**摄像头**：{record.get('camera_id')}",
            f"**时间**：{record.get('time')}",
            f"**置信度**：{confidence:.2f}",
        ]

        if event == "alarm.reviewed":
            review = record.get("vlm_review") or {}
            title = f"【报警复核】{label}"
            conclusion = "确认报警" if review.get("confirmed") else "判定误报"
            lines.insert(0, f"### {title}")
            lines.append(f"**复核结论**：{conclusion}")
            if review.get("confidence") is not None:
                lines.append(f"**复核置信度**：{review['confidence']:.2f}")
            if review.get("reason"):
                lines.append(f"**复核说明**：{review['reason']}")
        else:
            title = f"【报警】{label}"
            lines.insert(0, f"### {title}")
            if record.get("reason"):
                lines.append(f"**原因**：{record['reason']}")
            image_url = self._snapshot_link(record)
            if image_url:
                lines.append(f"![snapshot]({image_url})")

        return {"msgtype": "markdown", "markdown": {"title": title, "text": "\n\n".join(lines)}}

    def _post(self, payload: dict) -> bool:
        title = payload.get("markdown", {}).get("title", "?")
        try:
            resp = requests.post(
                self._signed_url(),
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8"},
                timeout=self.timeout,
            )
        except Exception as e:
            self._log(f"DingTalk push '{title}' failed: {e}", "error")
            return False

        if resp.status_code != 200:
            self._log(f"DingTalk push '{title}' failed: HTTP {resp.status_code}", "error")
            return False

        try:
            errcode = resp.json().get("errcode")
        except ValueError:
            errcode = None
        if errcode != 0:
            self._log(f"DingTalk push '{title}' rejected: body={resp.text[:200]}", "error")
            return False

        self._log(f"DingTalk push '{title}' succeeded", "info")
        return True

    def send_created(self, record: dict, snapshot_b64: Optional[str], frames_b64: list) -> bool:
        return self._post(self.build_message("alarm.created", record))

    def send_reviewed(self, record: dict) -> bool:
        return self._post(self.build_message("alarm.reviewed", record))
