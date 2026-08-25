"""报警推送模块测试"""

import base64
import json
from unittest.mock import patch, MagicMock

import pytest

from backend.integrations.base import AlarmPushChannel
from backend.integrations.guojing.channel import GuojingWebhookChannel as WebhookChannel
from backend.integrations.manager import IntegrationManager as PushManager


def _make_record(**overrides):
    record = {
        "id": "3_fire_1784538005159",
        "camera_id": "3",
        "detection_type": "fire",
        "level": "small_model_alarm",
        "status": "pending",
        "time": "2026-07-30 14:23:15",
        "confidence": 0.876,
        "reason": "检测到明火异常",
        "small_model": {"detected": True, "confidence": 0.876, "boxes": [[120, 45, 380, 420]]},
        "vlm_review": None,
        "source": "small_model",
        "frame_count": 2,
    }
    record.update(overrides)
    return record


class TestBuildEvent:
    """报文组装"""

    def test_created_event_structure(self):
        """created 报文：信封 + 全量 data + 图片"""
        record = _make_record()
        with patch("backend.integrations.guojing.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            ch = WebhookChannel("http://x/y")
            payload = ch.build_event("alarm.created", record, "SNAP", ["F1", "F2"])

        assert payload["event"] == "alarm.created"
        assert payload["event_id"] == "3_fire_1784538005159_created"
        assert "sent_at" in payload

        data = payload["data"]
        assert data["id"] == "3_fire_1784538005159"
        assert data["detection_label"] == "明火"
        assert data["vlm_review"] is None
        assert data["snapshot"] == "SNAP"
        assert data["frames"] == ["F1", "F2"]
        assert data["frame_count"] == 2

    def test_reviewed_event_without_images(self):
        """reviewed 报文：不带图片，vlm_review 有值"""
        record = _make_record(
            level="vlm_alarm",
            reason="[VLM 确认] 真的有明火",
            vlm_review={"confirmed": True, "confidence": 0.92, "reason": "真的有明火"},
        )
        with patch("backend.integrations.guojing.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            ch = WebhookChannel("http://x/y")
            payload = ch.build_event("alarm.reviewed", record, None, [])

        assert payload["event"] == "alarm.reviewed"
        assert payload["event_id"] == "3_fire_1784538005159_reviewed"
        data = payload["data"]
        assert data["vlm_review"]["confirmed"] is True
        assert "snapshot" not in data
        assert "frames" not in data
        assert data["frame_count"] == 2


class TestWebhookSend:
    """webhook 通道 HTTP 推送"""

    def test_send_success_returns_true(self):
        ch = WebhookChannel("http://x/y")
        with patch("backend.integrations.guojing.channel.requests.post") as mock_post, \
             patch("backend.integrations.guojing.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            resp = MagicMock(status_code=200)
            resp.json.return_value = {"code": 0}
            mock_post.return_value = resp
            ok = ch.send_created(_make_record(), "SNAP", ["F1"])
        assert ok is True
        mock_post.assert_called_once()
        body = mock_post.call_args.kwargs["data"]
        parsed = json.loads(body)
        assert parsed["event"] == "alarm.created"

    def test_send_http_200_but_code_nonzero_is_failure(self):
        """HTTP 200 但 body code != 0 视为失败（按文档约定）"""
        ch = WebhookChannel("http://x/y")
        with patch("backend.integrations.guojing.channel.requests.post") as mock_post, \
             patch("backend.integrations.guojing.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            resp = MagicMock(status_code=200)
            resp.json.return_value = {"code": 500}
            resp.text = '{"code":500,"msg":null}'
            mock_post.return_value = resp
            ok = ch.send_created(_make_record(), None, [])
        assert ok is False

    def test_send_http_error_returns_false_no_raise(self):
        ch = WebhookChannel("http://x/y")
        with patch("backend.integrations.guojing.channel.requests.post") as mock_post, \
             patch("backend.integrations.guojing.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            mock_post.return_value = MagicMock(status_code=500)
            ok = ch.send_created(_make_record(), None, [])
        assert ok is False

    def test_send_network_error_returns_false_no_raise(self):
        ch = WebhookChannel("http://x/y")
        with patch("backend.integrations.guojing.channel.requests.post") as mock_post, \
             patch("backend.integrations.guojing.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            mock_post.side_effect = ConnectionError("refused")
            ok = ch.send_created(_make_record(), None, [])
        assert ok is False

    def test_success_and_failure_are_logged(self):
        """成功/失败都通过注入的 log 回调记录，带 event_id"""
        logs = []
        ch = WebhookChannel("http://x/y", log=lambda msg, level="info": logs.append((level, msg)))
        with patch("backend.integrations.guojing.channel.requests.post") as mock_post, \
             patch("backend.integrations.guojing.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            ok_resp = MagicMock(status_code=200)
            ok_resp.json.return_value = {"code": 0}
            mock_post.return_value = ok_resp
            ch.send_created(_make_record(), None, [])
            mock_post.side_effect = ConnectionError("refused")
            ch.send_created(_make_record(), None, [])
        levels = [lv for lv, _ in logs]
        assert "info" in levels and "error" in levels
        assert any("testcam" not in m and "_created" in m for _, m in logs)


class TestPushManager:
    """PushManager 遍历所有通道，单通道失败不影响其他"""

    def test_push_calls_all_channels(self):
        ch1 = MagicMock(spec=AlarmPushChannel)
        ch2 = MagicMock(spec=AlarmPushChannel)
        mgr = PushManager([ch1, ch2])
        mgr.push_created(_make_record(), "SNAP", ["F1"])
        ch1.send_created.assert_called_once()
        ch2.send_created.assert_called_once()

    def test_failing_channel_does_not_block_others(self):
        ch1 = MagicMock(spec=AlarmPushChannel)
        ch1.send_created.side_effect = RuntimeError("boom")
        ch2 = MagicMock(spec=AlarmPushChannel)
        mgr = PushManager([ch1, ch2])
        mgr.push_created(_make_record(), None, [])  # 不抛异常
        ch2.send_created.assert_called_once()

    def test_push_reviewed(self):
        ch1 = MagicMock(spec=AlarmPushChannel)
        mgr = PushManager([ch1])
        mgr.push_reviewed(_make_record())
        ch1.send_reviewed.assert_called_once()
