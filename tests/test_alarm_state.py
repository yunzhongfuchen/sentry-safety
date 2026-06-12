import time

def test_create_record_initial_state():
    from backend.alarm_state import create_record
    result = {"detected": True, "max_confidence": 0.87, "boxes": [[1, 2, 3, 4]], "reason": "检测到 fire"}
    record = create_record("cam01", "fire", result)
    assert record["camera_id"] == "cam01"
    assert record["detection_type"] == "fire"
    assert record["level"] == "small_model_alarm"
    assert record["status"] == "pending"
    assert record["small_model"]["detected"] is True
    assert record["vlm_review"] is None

def test_vlm_review_does_not_change_status():
    from backend.alarm_state import create_record, apply_vlm_review
    record = create_record("cam01", "fire", {"detected": True, "max_confidence": 0.8})
    apply_vlm_review(record, {"confirmed": True, "confidence": 0.9, "reason": "有火焰"})
    assert record["level"] == "vlm_alarm"
    assert record["status"] == "pending"

    apply_vlm_review(record, {"confirmed": False, "confidence": 0.9, "reason": "无火焰"})
    assert record["level"] == "vlm_ignore"
    assert record["status"] == "pending"

def test_human_confirm_changes_status():
    from backend.alarm_state import create_record, confirm_alarm, confirm_false_positive
    record = create_record("cam01", "fire", {"detected": True, "max_confidence": 0.8})
    confirm_alarm(record)
    assert record["status"] == "confirmed"

    record2 = create_record("cam01", "fire", {"detected": True, "max_confidence": 0.8})
    confirm_false_positive(record2)
    assert record2["status"] == "false_positive"
