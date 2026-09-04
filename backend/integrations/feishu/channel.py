"""
飞书自定义机器人告警推送通道实现

机器人文档：消息类型 interactive 卡片（lark_md 语法）；安全设置支持"签名校验"。
签名算法：HMAC-SHA256(key=f"{timestamp}\\n{secret}", 空消息) 后 base64，放入请求体。
成功判定：HTTP 200 且响应体 code == 0。

注意：飞书卡片图片仅支持 img_key（需企业应用上传接口），不支持外链图片，
因此报警快照以 lark_md 链接形式附在消息中，点击跳转查看。
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


class FeishuChannel(AlarmPushChannel):
    """飞书群机器人通道：把告警以 interactive 卡片消息推送到飞书群。

    - secret 非空时在请求体附带 timestamp/sign（对应机器人"签名校验"安全设置）
    - snapshot_base_url 非空时在消息中附报警快照的查看链接（飞书不支持外链图嵌入）
    """

    name = "feishu"

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

    def _sign_fields(self) -> dict:
        """签名字段：timestamp(秒) + sign，未配置密钥时返回空 dict"""
        if not self.secret:
            return {}
        timestamp = str(int(time.time()))
        string_to_sign = f"{timestamp}\n{self.secret}"
        sign = base64.b64encode(
            hmac.new(string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
        ).decode("utf-8")
        return {"timestamp": timestamp, "sign": sign}

    def _snapshot_link(self, record: dict) -> str:
        if not self.snapshot_base_url:
            return ""
        record_id = urllib.parse.quote(str(record.get("id", "")), safe="")
        return f"{self.snapshot_base_url}/cvApi/open/api/cv/warning/image/{record_id}"

    def build_message(self, event: str, record: dict) -> dict:
        """组装飞书 interactive 卡片。alarm.created 可附快照链接，alarm.reviewed 为复核结论。"""
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
            template = "blue"
            conclusion = "确认报警" if review.get("confirmed") else "判定误报"
            lines.append(f"**复核结论**：{conclusion}")
            if review.get("confidence") is not None:
                lines.append(f"**复核置信度**：{review['confidence']:.2f}")
            if review.get("reason"):
                lines.append(f"**复核说明**：{review['reason']}")
        else:
            title = f"【报警】{label}"
            template = "red"
            if record.get("reason"):
                lines.append(f"**原因**：{record['reason']}")
            image_url = self._snapshot_link(record)
            if image_url:
                lines.append(f"[查看报警快照]({image_url})")

        return {
            **self._sign_fields(),
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": template,
                },
                "elements": [
                    {"tag": "div", "text": {"tag": "lark_md", "content": "\n".join(lines)}}
                ],
            },
        }

    def _post(self, payload: dict) -> bool:
        title = payload.get("card", {}).get("header", {}).get("title", {}).get("content", "?")
        try:
            resp = requests.post(
                self.webhook_url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8"},
                timeout=self.timeout,
            )
        except Exception as e:
            self._log(f"Feishu push '{title}' failed: {e}", "error")
            return False

        if resp.status_code != 200:
            self._log(f"Feishu push '{title}' failed: HTTP {resp.status_code}", "error")
            return False

        try:
            body = resp.json()
            code = body.get("code", body.get("StatusCode"))
        except ValueError:
            code = None
        if code != 0:
            self._log(f"Feishu push '{title}' rejected: body={resp.text[:200]}", "error")
            return False

        self._log(f"Feishu push '{title}' succeeded", "info")
        return True

    def send_created(self, record: dict, snapshot_b64: Optional[str], frames_b64: list) -> bool:
        return self._post(self.build_message("alarm.created", record))

    def send_reviewed(self, record: dict) -> bool:
        return self._post(self.build_message("alarm.reviewed", record))
