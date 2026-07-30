import numpy as np
import pytest
import backend.performance_storage as ps


@pytest.fixture(autouse=True)
def isolate_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "RECORDS_FILE", tmp_path / "records.json")
    monkeypatch.setattr(ps, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ps, "_records_cache", None)
    yield
    monkeypatch.setattr(ps, "_records_cache", None)


@pytest.fixture
def bypass_records_cache(monkeypatch):
    """防止 save_records 写入缓存导致 load_records 直接返回内存对象"""
    monkeypatch.setattr(ps, "_records_cache", None)
    original_save = ps.save_records

    def save_without_cache(records):
        original_save(records)
        monkeypatch.setattr(ps, "_records_cache", None)

    monkeypatch.setattr(ps, "save_records", save_without_cache)
    yield


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


def test_save_records_serializes_numpy_types(tmp_path, bypass_records_cache):
    """回归测试：numpy.float32 等类型不应导致保存失败"""
    records = [
        {
            "id": "1",
            "camera_id": "cam01",
            "detection_type": "fire",
            "confidence": np.float32(0.876),
            "small_model": {"boxes": np.array([[1, 2, 3, 4]], dtype=np.float32)},
            "frame_count": np.int64(3),
        }
    ]
    ps.save_records(records)

    loaded = ps.load_records()
    assert len(loaded) == 1
    assert loaded[0]["confidence"] == pytest.approx(0.876)
    assert loaded[0]["small_model"]["boxes"] == [[1.0, 2.0, 3.0, 4.0]]
    assert loaded[0]["frame_count"] == 3


def test_cleanup_orphan_images_removes_only_orphans(tmp_path):
    """孤儿图片应被删除，有元数据的图片和非图片文件应保留"""
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "orphan1_snapshot.jpg").write_bytes(b"x")
    (frames / "orphan1_frame_000.jpg").write_bytes(b"x")
    (frames / "known1_snapshot.jpg").write_bytes(b"x")
    (frames / "not_an_image.txt").write_bytes(b"x")
    ps.save_records([{"id": "known1", "time": "2026-07-30T00:00:00"}])

    removed = ps.cleanup_orphan_images(data_dir=tmp_path, max_age_seconds=0)

    assert removed == 2
    assert not (frames / "orphan1_snapshot.jpg").exists()
    assert not (frames / "orphan1_frame_000.jpg").exists()
    assert (frames / "known1_snapshot.jpg").exists()
    assert (frames / "not_an_image.txt").exists()


def test_cleanup_orphan_images_keeps_recent_files(tmp_path):
    """修改时间不足 max_age_seconds 的孤儿文件应保留（避免误删正在写入的图片）"""
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "new1_snapshot.jpg").write_bytes(b"x")
    ps.save_records([])

    removed = ps.cleanup_orphan_images(data_dir=tmp_path, max_age_seconds=3600)

    assert removed == 0
    assert (frames / "new1_snapshot.jpg").exists()
