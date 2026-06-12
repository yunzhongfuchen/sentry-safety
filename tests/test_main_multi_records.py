"""Tests for alarm_state integration equivalence (Task 4).

on_trigger and on_vlm_result are closures inside init_components(),
so they cannot be directly imported for unit testing.
These tests verify the equivalent logic via alarm_state functions.
"""
import pytest
import time

from backend.alarm_state import create_record, apply_vlm_review, confirm_alarm, confirm_false_positive


def test_create_record_basic_fields():
    """create_record 应生成包含所有必要字段的记录"""
    record = create_record("cam01", "fire", {"detected": True, "max_confidence": 0.87})
    assert record["camera_id"] == "cam01"
    assert record["detection_type"] == "fire"
    assert record["level"] == "small_model_alarm"
    assert record["status"] == "pending"
    assert record["confidence"] == 0.87
    assert record["small_model"]["detected"] is True
    assert record["small_model"]["confidence"] == 0.87
    assert record["vlm_review"] is None
    assert record["frame_count"] == 0


def test_create_record_with_custom_id_and_time():
    """create_record 应支持自定义 record_id 和 now"""
    now = 1234567890.0
    record = create_record("cam02", "smoke", {"detected": False}, record_id="custom_id", now=now)
    assert record["id"] == "custom_id"
    assert record["time"] == time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))


def test_apply_vlm_review_confirmed():
    """VLM 确认后 level 应变为 vlm_alarm，reason 应更新"""
    record = create_record("cam01", "fire", {"detected": True, "max_confidence": 0.87})
    apply_vlm_review(record, {"confirmed": True, "confidence": 0.9, "reason": "有火焰"})
    assert record["level"] == "vlm_alarm"
    assert record["status"] == "pending"  # status 不应改变
    assert record["vlm_review"]["confirmed"] is True
    assert record["vlm_review"]["confidence"] == 0.9
    assert record["reason"] == "[VLM 确认] 有火焰"


def test_apply_vlm_review_rejected():
    """VLM 排除后 level 应变为 vlm_ignore"""
    record = create_record("cam01", "fire", {"detected": True, "max_confidence": 0.87})
    apply_vlm_review(record, {"confirmed": False, "confidence": 0.2, "reason": "只是烟雾"})
    assert record["level"] == "vlm_ignore"
    assert record["status"] == "pending"  # status 不应改变
    assert record["vlm_review"]["confirmed"] is False
    assert record["reason"] == "[VLM 已排除] 只是烟雾"


def test_apply_vlm_review_no_reason():
    """VLM 结果无 reason 时应使用默认文案"""
    record = create_record("cam01", "fire", {"detected": True, "max_confidence": 0.87})
    apply_vlm_review(record, {"confirmed": True, "confidence": 0.9})
    assert record["reason"] == "[VLM 确认] 复核通过"

    record2 = create_record("cam02", "smoke", {"detected": True, "max_confidence": 0.65})
    apply_vlm_review(record2, {"confirmed": False, "confidence": 0.1})
    assert record2["reason"] == "[VLM 已排除] 复核未通过"


def test_confirm_alarm():
    """人工确认后 status 应变为 confirmed"""
    record = create_record("cam01", "fire", {"detected": True, "max_confidence": 0.87})
    confirm_alarm(record)
    assert record["status"] == "confirmed"
    assert record["level"] == "small_model_alarm"  # level 不应改变


def test_confirm_false_positive():
    """人工标记误报后 status 应变为 false_positive"""
    record = create_record("cam01", "fire", {"detected": True, "max_confidence": 0.87})
    confirm_false_positive(record)
    assert record["status"] == "false_positive"
    assert record["level"] == "small_model_alarm"  # level 不应改变


def test_full_lifecycle_small_model_to_vlm_to_human():
    """完整生命周期：小模型触发 -> VLM 确认 -> 人工确认"""
    record = create_record("cam01", "fire", {"detected": True, "max_confidence": 0.87})
    assert record["level"] == "small_model_alarm"
    assert record["status"] == "pending"

    apply_vlm_review(record, {"confirmed": True, "confidence": 0.95, "reason": "明显火焰"})
    assert record["level"] == "vlm_alarm"
    assert record["status"] == "pending"

    confirm_alarm(record)
    assert record["level"] == "vlm_alarm"
    assert record["status"] == "confirmed"


def test_full_lifecycle_small_model_to_vlm_reject_to_human_fp():
    """完整生命周期：小模型触发 -> VLM 排除 -> 人工标记误报"""
    record = create_record("cam01", "smoke", {"detected": True, "max_confidence": 0.65})
    assert record["level"] == "small_model_alarm"
    assert record["status"] == "pending"

    apply_vlm_review(record, {"confirmed": False, "confidence": 0.1, "reason": "只是水蒸气"})
    assert record["level"] == "vlm_ignore"
    assert record["status"] == "pending"

    confirm_false_positive(record)
    assert record["level"] == "vlm_ignore"
    assert record["status"] == "false_positive"
