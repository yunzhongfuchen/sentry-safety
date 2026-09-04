"""飞书推送通道测试"""

import base64
import hashlib
import hmac
import json
from unittest.mock import patch, MagicMock

from backend.integrations.feishu.channel import FeishuChannel


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
        "small_model": {"detected": True, "confidence": 0.876},
        "vlm_review": None,
        "source": "small_model",
        "frame_count": 2,
    }
    record.update(overrides)
    return record


class TestBuildMessage:
    def test_created_card_structure(self):
        with patch("backend.integrations.feishu.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            ch = FeishuChannel("https://open.feishu.cn/open-apis/bot/v2/hook/x")
            payload = ch.build_message("alarm.created", _make_record())

        assert payload["msg_type"] == "interactive"
        card = payload["card"]
        assert card["header"]["title"]["content"] == "【报警】明火"
        assert card["header"]["template"] == "red"
        text = card["elements"][0]["text"]["content"]
        assert "**报警类型**：明火" in text
        assert "**置信度**：0.88" in text
        assert "查看报警快照" not in text  # 未配置 base_url

    def test_created_card_with_snapshot_link(self):
        with patch("backend.integrations.feishu.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            ch = FeishuChannel(
                "https://open.feishu.cn/open-apis/bot/v2/hook/x",
                snapshot_base_url="http://192.168.1.10:8111/",
            )
            payload = ch.build_message("alarm.created", _make_record())

        text = payload["card"]["elements"][0]["text"]["content"]
        assert "[查看报警快照](http://192.168.1.10:8111/cvApi/open/api/cv/warning/image/3_fire_1784538005159)" in text

    def test_reviewed_card_structure(self):
        record = _make_record(
            level="vlm_alarm",
            vlm_review={"confirmed": False, "confidence": 0.8, "reason": "只是灯光反光"},
        )
        with patch("backend.integrations.feishu.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            ch = FeishuChannel("https://open.feishu.cn/open-apis/bot/v2/hook/x")
            payload = ch.build_message("alarm.reviewed", record)

        card = payload["card"]
        assert card["header"]["title"]["content"] == "【报警复核】明火"
        assert card["header"]["template"] == "blue"
        text = card["elements"][0]["text"]["content"]
        assert "**复核结论**：判定误报" in text
        assert "**复核置信度**：0.80" in text


class TestSign:
    def test_no_secret_no_sign_fields(self):
        ch = FeishuChannel("https://x/hook")
        assert ch._sign_fields() == {}

    def test_sign_matches_feishu_algorithm(self):
        """按飞书官方算法独立计算签名，与通道产出一致"""
        secret = "abc123"
        with patch("backend.integrations.feishu.channel.time") as mock_time:
            mock_time.time.return_value = 1700000000.5
            ch = FeishuChannel("https://x/hook", secret=secret)
            fields = ch._sign_fields()

        timestamp = "1700000000"
        string_to_sign = f"{timestamp}\n{secret}"
        expected = base64.b64encode(
            hmac.new(string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
        ).decode("utf-8")
        assert fields == {"timestamp": timestamp, "sign": expected}

    def test_signed_message_carries_sign_in_body(self):
        with patch("backend.integrations.feishu.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            ch = FeishuChannel("https://x/hook", secret="abc123")
            payload = ch.build_message("alarm.created", _make_record())
        assert "timestamp" in payload and "sign" in payload


class TestFeishuSend:
    def test_send_success(self):
        ch = FeishuChannel("https://x/hook")
        with patch("backend.integrations.feishu.channel.requests.post") as mock_post, \
             patch("backend.integrations.feishu.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            resp = MagicMock(status_code=200)
            resp.json.return_value = {"code": 0, "msg": "success"}
            mock_post.return_value = resp
            ok = ch.send_created(_make_record(), None, [])
        assert ok is True
        body = json.loads(mock_post.call_args.kwargs["data"])
        assert body["msg_type"] == "interactive"

    def test_code_nonzero_is_failure(self):
        ch = FeishuChannel("https://x/hook")
        with patch("backend.integrations.feishu.channel.requests.post") as mock_post, \
             patch("backend.integrations.feishu.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            resp = MagicMock(status_code=200)
            resp.json.return_value = {"code": 19021, "msg": "sign not match"}
            resp.text = '{"code":19021,"msg":"sign not match"}'
            mock_post.return_value = resp
            ok = ch.send_created(_make_record(), None, [])
        assert ok is False

    def test_network_error_returns_false_no_raise(self):
        ch = FeishuChannel("https://x/hook")
        with patch("backend.integrations.feishu.channel.requests.post") as mock_post, \
             patch("backend.integrations.feishu.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            mock_post.side_effect = ConnectionError("refused")
            ok = ch.send_reviewed(_make_record())
        assert ok is False


class TestPlatformDispatch:
    """init_integration_manager 按 platform 字段分派通道类型"""

    def test_feishu_card_creates_feishu_channel(self):
        from backend.integrations import init_integration_manager

        mgr = init_integration_manager({
            "push_channels": [
                {"platform": "feishu", "name": "飞书群", "webhook_url": "https://x/feishu",
                 "secret": "S1", "all_types": False, "types": ["fire"]},
                {"platform": "dingtalk", "name": "钉钉群", "webhook_url": "https://x/ding"},
            ]
        })
        assert len(mgr.channels) == 2
        assert isinstance(mgr.channels[0], FeishuChannel)
        assert mgr.channels[0].types == ["fire"]
        assert mgr.channels[0].secret == "S1"
        assert mgr.channels[1].name == "dingtalk"

    def test_dingtalk_channels_key_treated_as_dingtalk(self):
        """旧键 dingtalk_channels 的卡片不带 platform 字段，按钉钉处理"""
        from backend.integrations import init_integration_manager

        mgr = init_integration_manager({
            "dingtalk_channels": [{"webhook_url": "https://x/ding", "all_types": True}]
        })
        assert len(mgr.channels) == 1
        assert mgr.channels[0].name == "dingtalk"

    def test_unknown_platform_skipped(self):
        from backend.integrations import init_integration_manager

        mgr = init_integration_manager({
            "push_channels": [{"platform": "wecom", "webhook_url": "https://x/wecom"}]
        })
        assert not mgr.enabled


class TestTestEndpointPlatform:
    def test_feishu_platform_test_message(self):
        from fastapi.testclient import TestClient
        from backend.main_multi import app

        client = TestClient(app)
        with patch("backend.integrations.feishu.channel.requests.post") as mock_post:
            resp_ok = MagicMock(status_code=200)
            resp_ok.json.return_value = {"code": 0, "msg": "success"}
            mock_post.return_value = resp_ok
            resp = client.post("/settings/dingtalk/test", json={
                "webhook_url": "https://open.feishu.cn/open-apis/bot/v2/hook/x",
                "secret": "abc123",
                "platform": "feishu",
            })
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "error": None}
        body = json.loads(mock_post.call_args.kwargs["data"])
        assert body["msg_type"] == "interactive"
        assert "测试消息" in body["card"]["header"]["title"]["content"]
        assert "sign" in body  # 飞书签名在请求体

    def test_unknown_platform_returns_400(self):
        from fastapi.testclient import TestClient
        from backend.main_multi import app

        client = TestClient(app)
        resp = client.post("/settings/dingtalk/test", json={
            "webhook_url": "https://x/hook", "platform": "wecom",
        })
        assert resp.status_code == 400
        assert "不支持的平台" in resp.json()["error"]
