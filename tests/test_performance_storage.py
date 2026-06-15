import pytest
import backend.performance_storage as ps


def test_summary_uses_status_and_level():
    ps.save_records([
        {"id": "1", "status": "pending", "level": "small_model_alarm", "detection_type": "fire", "camera_id": "cam01"},
        {"id": "2", "status": "confirmed", "level": "vlm_alarm", "detection_type": "mask", "camera_id": "cam01"},
        {"id": "3", "status": "false_positive", "level": "vlm_ignore", "detection_type": "smoke", "camera_id": "cam02"},
    ])
    summary = ps.get_record_summary()
    assert summary["total"] == 3
    assert summary["by_status"]["pending"] == 1
    assert summary["by_status"]["confirmed"] == 1
    assert summary["by_status"]["false_positive"] == 1
    assert summary["by_level"]["small_model_alarm"] == 1
    assert summary["by_level"]["vlm_alarm"] == 1
    assert summary["by_level"]["vlm_ignore"] == 1
    assert summary["by_type"]["fire"] == 1
    assert summary["by_type"]["mask"] == 1
    assert summary["by_type"]["smoke"] == 1
    assert summary["camera_stats"]["cam01"] == 2
    assert summary["camera_stats"]["cam02"] == 1


def test_summary_empty_records():
    ps.save_records([])
    summary = ps.get_record_summary()
    assert summary["total"] == 0
    assert summary["by_status"] == {}
    assert summary["by_type"] == {}
    assert summary["by_level"] == {}
    assert summary["camera_stats"] == {}


def test_summary_missing_fields():
    ps.save_records([
        {"id": "1", "detection_type": "fire", "camera_id": "cam01"},
        {"id": "2", "status": "pending", "camera_id": "cam01"},
    ])
    summary = ps.get_record_summary()
    assert summary["total"] == 2
    assert summary["by_status"]["unknown"] == 1
    assert summary["by_status"]["pending"] == 1
    assert summary["by_level"]["unknown"] == 2
