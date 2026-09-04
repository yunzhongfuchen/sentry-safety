"""钉钉推送通道测试"""

import json
from unittest.mock import patch, MagicMock

from backend.integrations.dingtalk.channel import DingTalkChannel


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
    def test_created_message_structure(self):
        with patch("backend.integrations.dingtalk.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            ch = DingTalkChannel("https://oapi.dingtalk.com/robot/send?access_token=x")
            payload = ch.build_message("alarm.created", _make_record())

        assert payload["msgtype"] == "markdown"
        md = payload["markdown"]
        assert md["title"] == "【报警】明火"
        assert "**报警类型**：明火" in md["text"]
        assert "**摄像头**：3" in md["text"]
        assert "**置信度**：0.88" in md["text"]
        assert "**原因**：检测到明火异常" in md["text"]
        # 未配置 base_url 时不带图
        assert "![snapshot]" not in md["text"]

    def test_created_message_with_snapshot_link(self):
        with patch("backend.integrations.dingtalk.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            ch = DingTalkChannel(
                "https://oapi.dingtalk.com/robot/send?access_token=x",
                snapshot_base_url="http://192.168.1.10:8111/",
            )
            payload = ch.build_message("alarm.created", _make_record())

        text = payload["markdown"]["text"]
        assert "![snapshot](http://192.168.1.10:8111/cvApi/open/api/cv/warning/image/3_fire_1784538005159)" in text

    def test_reviewed_message_structure(self):
        record = _make_record(
            level="vlm_alarm",
            vlm_review={"confirmed": True, "confidence": 0.92, "reason": "真的有明火"},
        )
        with patch("backend.integrations.dingtalk.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            ch = DingTalkChannel("https://oapi.dingtalk.com/robot/send?access_token=x")
            payload = ch.build_message("alarm.reviewed", record)

        md = payload["markdown"]
        assert md["title"] == "【报警复核】明火"
        assert "**复核结论**：确认报警" in md["text"]
        assert "**复核置信度**：0.92" in md["text"]
        assert "**复核说明**：真的有明火" in md["text"]
        assert "![snapshot]" not in md["text"]

    def test_reviewed_message_false_alarm(self):
        record = _make_record(
            level="false_alarm",
            vlm_review={"confirmed": False, "confidence": 0.8, "reason": "只是灯光反光"},
        )
        with patch("backend.integrations.dingtalk.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            ch = DingTalkChannel("https://oapi.dingtalk.com/robot/send?access_token=x")
            payload = ch.build_message("alarm.reviewed", record)

        assert "**复核结论**：判定误报" in payload["markdown"]["text"]


class TestSign:
    def test_no_secret_returns_plain_url(self):
        ch = DingTalkChannel("https://oapi.dingtalk.com/robot/send?access_token=x")
        assert ch._signed_url() == "https://oapi.dingtalk.com/robot/send?access_token=x"

    def test_secret_appends_timestamp_and_sign(self):
        ch = DingTalkChannel("https://oapi.dingtalk.com/robot/send?access_token=x", secret="SECabc")
        url = ch._signed_url()
        assert "timestamp=" in url
        assert "sign=" in url
        assert url.startswith("https://oapi.dingtalk.com/robot/send?access_token=x&")

    def test_sign_matches_dingtalk_algorithm(self):
        """按钉钉官方算法独立计算签名，与通道产出一致"""
        import base64, hashlib, hmac, urllib.parse

        secret = "SECabc"
        with patch("backend.integrations.dingtalk.channel.time") as mock_time:
            mock_time.time.return_value = 1700000000.0
            ch = DingTalkChannel("https://x/hook", secret=secret)
            url = ch._signed_url()

        timestamp = str(round(1700000000.0 * 1000))
        string_to_sign = f"{timestamp}\n{secret}"
        expected_sign = urllib.parse.quote_plus(
            base64.b64encode(hmac.new(secret.encode(), string_to_sign.encode(), hashlib.sha256).digest())
        )
        assert url == f"https://x/hook?timestamp={timestamp}&sign={expected_sign}"


class TestDingTalkSend:
    def test_send_success(self):
        ch = DingTalkChannel("https://x/hook")
        with patch("backend.integrations.dingtalk.channel.requests.post") as mock_post, \
             patch("backend.integrations.dingtalk.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            resp = MagicMock(status_code=200)
            resp.json.return_value = {"errcode": 0, "errmsg": "ok"}
            mock_post.return_value = resp
            ok = ch.send_created(_make_record(), None, [])
        assert ok is True
        body = json.loads(mock_post.call_args.kwargs["data"])
        assert body["msgtype"] == "markdown"

    def test_errcode_nonzero_is_failure(self):
        ch = DingTalkChannel("https://x/hook")
        with patch("backend.integrations.dingtalk.channel.requests.post") as mock_post, \
             patch("backend.integrations.dingtalk.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            resp = MagicMock(status_code=200)
            resp.json.return_value = {"errcode": 310000, "errmsg": "keywords not in content"}
            resp.text = '{"errcode":310000}'
            mock_post.return_value = resp
            ok = ch.send_created(_make_record(), None, [])
        assert ok is False

    def test_network_error_returns_false_no_raise(self):
        ch = DingTalkChannel("https://x/hook")
        with patch("backend.integrations.dingtalk.channel.requests.post") as mock_post, \
             patch("backend.integrations.dingtalk.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            mock_post.side_effect = ConnectionError("refused")
            ok = ch.send_reviewed(_make_record())
        assert ok is False

    def test_success_and_failure_are_logged(self):
        logs = []
        ch = DingTalkChannel("https://x/hook", log=lambda msg, level="info": logs.append((level, msg)))
        with patch("backend.integrations.dingtalk.channel.requests.post") as mock_post, \
             patch("backend.integrations.dingtalk.channel.registry") as mock_reg:
            mock_reg.get.return_value = {"label": "明火"}
            ok_resp = MagicMock(status_code=200)
            ok_resp.json.return_value = {"errcode": 0}
            mock_post.return_value = ok_resp
            ch.send_created(_make_record(), None, [])
            mock_post.side_effect = ConnectionError("refused")
            ch.send_created(_make_record(), None, [])
        levels = [lv for lv, _ in logs]
        assert "info" in levels and "error" in levels


class TestInitIntegrationManager:
    def test_dingtalk_channel_registered_when_url_set(self):
        from backend.integrations import init_integration_manager

        mgr = init_integration_manager({
            "alarm_push_webhook_url": "",
            "dingtalk_webhook_url": "https://x/hook",
            "dingtalk_secret": "SECabc",
            "dingtalk_base_url": "http://192.168.1.10:8111",
        })
        assert mgr.enabled
        assert len(mgr.channels) == 1
        ch = mgr.channels[0]
        assert isinstance(ch, DingTalkChannel)
        assert ch.secret == "SECabc"
        assert ch.snapshot_base_url == "http://192.168.1.10:8111"

    def test_both_channels_registered(self):
        from backend.integrations import init_integration_manager

        mgr = init_integration_manager({
            "alarm_push_webhook_url": "http://guojing/hook",
            "dingtalk_webhook_url": "https://x/hook",
        })
        assert len(mgr.channels) == 2
        names = {c.name for c in mgr.channels}
        assert names == {"guojing_webhook", "dingtalk"}

    def test_no_channels_when_urls_empty(self):
        from backend.integrations import init_integration_manager

        mgr = init_integration_manager({})
        assert not mgr.enabled


class TestCardListChannels:
    """dingtalk_channels 卡片列表初始化与类型订阅"""

    def test_init_from_card_list(self):
        from backend.integrations import init_integration_manager

        mgr = init_integration_manager({
            "dingtalk_channels": [
                {"name": "消防群", "webhook_url": "https://x/hook1", "secret": "S1",
                 "base_url": "http://b1", "all_types": False, "types": ["fire", "smoke"]},
                {"name": "全员群", "webhook_url": "https://x/hook2", "all_types": True},
                {"name": "空地址卡", "webhook_url": "", "all_types": True},
            ]
        })
        assert len(mgr.channels) == 2
        ch1, ch2 = mgr.channels
        assert ch1.types == ["fire", "smoke"]
        assert ch1.secret == "S1"
        assert ch1.snapshot_base_url == "http://b1"
        assert ch2.types is None  # all_types=True → 全部类型

    def test_legacy_fields_fallback_when_no_cards(self):
        from backend.integrations import init_integration_manager

        mgr = init_integration_manager({
            "dingtalk_webhook_url": "https://x/legacy",
            "dingtalk_secret": "SEClegacy",
        })
        assert len(mgr.channels) == 1
        assert mgr.channels[0].secret == "SEClegacy"
        assert mgr.channels[0].types is None

    def test_card_list_takes_precedence_over_legacy(self):
        from backend.integrations import init_integration_manager

        mgr = init_integration_manager({
            "dingtalk_channels": [{"webhook_url": "https://x/card", "all_types": True}],
            "dingtalk_webhook_url": "https://x/legacy",
        })
        urls = [c.webhook_url for c in mgr.channels]
        assert urls == ["https://x/card"]


class TestTypeFiltering:
    """通道按检测类型订阅过滤"""

    def _record(self, dtype="fire"):
        return {"id": "r1", "detection_type": dtype}

    def test_all_types_channel_accepts_everything(self):
        ch = DingTalkChannel("https://x/hook", types=None)
        assert ch.accepts(self._record("fire"))
        assert ch.accepts(self._record("mask"))

    def test_types_list_filters(self):
        ch = DingTalkChannel("https://x/hook", types=["fire", "smoke"])
        assert ch.accepts(self._record("fire"))
        assert not ch.accepts(self._record("mask"))

    def test_manager_skips_unsubscribed_channel(self):
        from backend.integrations.manager import IntegrationManager

        ch_fire = MagicMock()
        ch_fire.name = "fire_only"
        ch_fire.accepts = lambda r: r.get("detection_type") == "fire"
        ch_all = MagicMock()
        ch_all.name = "all"
        ch_all.accepts = lambda r: True

        mgr = IntegrationManager([ch_fire, ch_all])
        mgr.push_created(self._record("mask"), None, [])
        ch_fire.send_created.assert_not_called()
        ch_all.send_created.assert_called_once()

        mgr.push_reviewed(self._record("mask"))
        ch_fire.send_reviewed.assert_not_called()
        ch_all.send_reviewed.assert_called_once()


class TestDingtalkTestEndpoint:
    """POST /settings/dingtalk/test 测试消息接口"""

    def test_empty_webhook_returns_400(self):
        from fastapi.testclient import TestClient
        from backend.main_multi import app

        client = TestClient(app)
        resp = client.post("/settings/dingtalk/test", json={"webhook_url": "  "})
        assert resp.status_code == 400
        assert resp.json()["ok"] is False

    def test_success_returns_ok(self):
        from fastapi.testclient import TestClient
        from backend.main_multi import app

        client = TestClient(app)
        with patch("backend.integrations.dingtalk.channel.requests.post") as mock_post:
            resp_ok = MagicMock(status_code=200)
            resp_ok.json.return_value = {"errcode": 0, "errmsg": "ok"}
            mock_post.return_value = resp_ok
            resp = client.post("/settings/dingtalk/test", json={
                "webhook_url": "https://oapi.dingtalk.com/robot/send?access_token=x",
                "secret": "SECabc",
            })
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "error": None}
        body = json.loads(mock_post.call_args.kwargs["data"])
        assert body["msgtype"] == "markdown"
        assert "测试消息" in body["markdown"]["title"]
        # 加签参数已拼到请求 URL 上
        assert "sign=" in mock_post.call_args.args[0]

    def test_dingtalk_reject_returns_error_message(self):
        from fastapi.testclient import TestClient
        from backend.main_multi import app

        client = TestClient(app)
        with patch("backend.integrations.dingtalk.channel.requests.post") as mock_post:
            resp_bad = MagicMock(status_code=200)
            resp_bad.json.return_value = {"errcode": 310000, "errmsg": "keywords not in content"}
            resp_bad.text = '{"errcode":310000,"errmsg":"keywords not in content"}'
            mock_post.return_value = resp_bad
            resp = client.post("/settings/dingtalk/test", json={
                "webhook_url": "https://oapi.dingtalk.com/robot/send?access_token=x",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert data["error"]


class TestSettingsHotReload:
    """保存 dingtalk_* 配置后热重建推送管理器"""

    def test_dingtalk_settings_trigger_push_manager_reload(self):
        import backend.main_multi as mm
        from fastapi.testclient import TestClient

        original_manager = mm.push_manager
        try:
            with patch.object(mm.app_config, "save_global_settings"), \
                 patch.object(mm.app_config, "load_global_settings", return_value={
                     "dingtalk_webhook_url": "https://x/hook",
                     "dingtalk_secret": "SECabc",
                 }):
                client = TestClient(mm.app)
                resp = client.post("/settings", json={"dingtalk_webhook_url": "https://x/hook"})
                assert resp.status_code == 200
                assert mm.push_manager is not original_manager
                assert len(mm.push_manager.channels) == 1
                assert mm.push_manager.channels[0].name == "dingtalk"

                # 清空地址后再次保存，通道被移除
                with patch.object(mm.app_config, "load_global_settings", return_value={}):
                    resp = client.post("/settings", json={"dingtalk_webhook_url": ""})
                    assert resp.status_code == 200
                    assert not mm.push_manager.enabled
        finally:
            mm.push_manager = original_manager

    def test_unrelated_settings_do_not_reload(self):
        import backend.main_multi as mm
        from fastapi.testclient import TestClient

        original_manager = mm.push_manager
        try:
            with patch.object(mm.app_config, "save_global_settings"), \
                 patch.object(mm.app_config, "load_global_settings", return_value={}):
                client = TestClient(mm.app)
                resp = client.post("/settings", json={"snapshot_quality": 80})
                assert resp.status_code == 200
                assert mm.push_manager is original_manager
        finally:
            mm.push_manager = original_manager
