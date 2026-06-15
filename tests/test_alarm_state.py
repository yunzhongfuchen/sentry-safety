from backend.alarm_state import create_record, apply_vlm_review, confirm_alarm, confirm_false_positive


def test_create_record_initial_state():
    result = {"detected": True, "max_confidence": 0.87, "boxes": [[1, 2, 3, 4]], "reason": "检测到 fire"}
    record = create_record("cam01", "fire", result)
    assert record["camera_id"] == "cam01"
    assert record["detection_type"] == "fire"
    assert record["level"] == "small_model_alarm"
    assert record["status"] == "pending"
    assert record["small_model"]["detected"] is True
    assert record["vlm_review"] is None


def test_create_record_with_record_id_and_now():
    now = 1718000000.123
    result = {"detected": True, "max_confidence": 0.87, "boxes": [[1, 2, 3, 4]], "reason": "检测到 fire"}
    record = create_record("cam01", "fire", result, record_id="custom_id", now=now)
    assert record["id"] == "custom_id"
    assert record["time"] == "2024-06-10 14:13:20"
    assert record["confidence"] == 0.87
    assert record["small_model"]["confidence"] == 0.87


def test_create_record_without_record_id_uses_now():
    now = 1718000000.123
    result = {"detected": True, "max_confidence": 0.87}
    record = create_record("cam01", "fire", result, now=now)
    assert record["id"] == "cam01_fire_1718000000123"
    assert record["time"] == "2024-06-10 14:13:20"


def test_create_record_confidence_fallback():
    result = {"detected": True, "confidence": 0.65}
    record = create_record("cam01", "fire", result)
    assert record["confidence"] == 0.65
    assert record["small_model"]["confidence"] == 0.65


def test_vlm_review_does_not_change_status():
    record = create_record("cam01", "fire", {"detected": True, "max_confidence": 0.8})
    apply_vlm_review(record, {"confirmed": True, "confidence": 0.9, "reason": "有火焰"})
    assert record["level"] == "vlm_alarm"
    assert record["status"] == "pending"

    apply_vlm_review(record, {"confirmed": False, "confidence": 0.9, "reason": "无火焰"})
    assert record["level"] == "vlm_ignore"
    assert record["status"] == "pending"


def test_vlm_review_state_flip():
    record = create_record("cam01", "fire", {"detected": True, "max_confidence": 0.8})
    apply_vlm_review(record, {"confirmed": True, "confidence": 0.9, "reason": "有火焰"})
    assert record["level"] == "vlm_alarm"
    assert record["reason"] == "[VLM 确认] 有火焰"

    apply_vlm_review(record, {"confirmed": False, "confidence": 0.1, "reason": "无火焰"})
    assert record["level"] == "vlm_ignore"
    assert record["reason"] == "[VLM 已排除] 无火焰"


def test_vlm_review_missing_reason():
    record = create_record("cam01", "fire", {"detected": True, "max_confidence": 0.8})
    apply_vlm_review(record, {"confirmed": True, "confidence": 0.9})
    assert record["reason"] == "[VLM 确认] 复核通过"

    apply_vlm_review(record, {"confirmed": False, "confidence": 0.1})
    assert record["reason"] == "[VLM 已排除] 复核未通过"


def test_human_confirm_changes_status():
    record = create_record("cam01", "fire", {"detected": True, "max_confidence": 0.8})
    confirm_alarm(record)
    assert record["status"] == "confirmed"
    assert record["level"] == "small_model_alarm"

    record2 = create_record("cam01", "fire", {"detected": True, "max_confidence": 0.8})
    confirm_false_positive(record2)
    assert record2["status"] == "false_positive"
    assert record2["level"] == "small_model_alarm"
