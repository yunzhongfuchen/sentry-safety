from unittest.mock import patch, MagicMock
from backend.integrations.base import AlarmPushChannel
from backend.integrations.manager import IntegrationManager
from backend.integrations.guojing.channel import GuojingWebhookChannel

def test_guojing_channel_build_event():
    channel = GuojingWebhookChannel("http://test.local/hook")
    record = {
        "id": "cam01_fire_1000",
        "camera_id": "cam01",
        "detection_type": "fire",
        "level": "small_model_alarm",
        "status": "pending",
        "time": "2026-08-25 10:00:00",
        "confidence": 0.9,
        "reason": "fire detected",
        "small_model": {"boxes": []},
        "vlm_review": None,
        "source": "small_model",
        "frame_count": 1,
    }
    with patch("backend.integrations.guojing.channel.registry") as mock_reg:
        mock_reg.get.return_value = {"label": "明火"}
        event = channel.build_event("alarm.created", record, "b64_snap", ["b64_f1"])
        assert event["event"] == "alarm.created"
        assert event["data"]["snapshot"] == "b64_snap"
        assert event["data"]["detection_label"] == "明火"

def test_integration_manager_broadcast():
    ch1 = MagicMock(spec=AlarmPushChannel)
    ch2 = MagicMock(spec=AlarmPushChannel)
    mgr = IntegrationManager([ch1, ch2])
    assert mgr.enabled is True

    record = {"id": "1"}
    mgr.push_created(record, None, [])
    ch1.send_created.assert_called_once_with(record, None, [])
    ch2.send_created.assert_called_once_with(record, None, [])
