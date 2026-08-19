"""摄像头配置写接口参数剔除测试"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
import backend.main_multi as mm


class TestCameraConfigSanitize:
    def test_params_stripped_on_save(self):
        """POST /cameras/{id}/config 只保留 enabled/roi/roi_invert"""
        import backend.main_multi as mm
        sanitize = mm.sanitize_camera_algorithms
        raw = {"fire": {"enabled": True, "threshold": 0.9, "interval": 5,
                        "roi": [[0, 0], [1, 0], [1, 1]], "roi_invert": False,
                        "box_count_mode": "gte"}}
        assert sanitize(raw) == {
            "fire": {"enabled": True, "roi": [[0, 0], [1, 0], [1, 1]], "roi_invert": False}
        }


class TestCameraConfigEndpoints:
    def _make_client(self):
        return TestClient(mm.app)

    def _patched_globals(self, saved: list):
        mock_cm = MagicMock()
        mock_cm._cameras = {}
        mock_md = MagicMock()
        mock_cfg = MagicMock()
        mock_cfg.load_camera_configs.return_value = [
            {"camera_id": "cam_01", "algorithms": {"fire": {"enabled": False}}}
        ]

        def fake_save(cameras):
            saved.append(cameras)
            return True

        mock_cfg.save_camera_configs.side_effect = fake_save
        return (
            patch.object(mm, "camera_manager", mock_cm),
            patch.object(mm, "multi_detector", mock_md),
            patch.object(mm, "app_config", mock_cfg),
        )

    def test_single_config_accepts_algorithms_and_strips_params(self):
        """POST /cameras/{id}/config 接受 algorithms 键，保存时剔除参数"""
        saved = []
        patches = self._patched_globals(saved)
        with patches[0], patches[1], patches[2]:
            client = self._make_client()
            resp = client.post("/cameras/cam_01/config", json={
                "algorithms": {"fire": {"enabled": True, "threshold": 0.9,
                                        "roi": [[0, 0]], "roi_invert": True}}
            })
        assert resp.status_code == 200
        assert saved, "save_camera_configs 未被调用"
        cam = saved[0][0]
        assert "detection_types" not in cam
        assert cam["algorithms"] == {
            "fire": {"enabled": True, "roi": [[0, 0]], "roi_invert": True}
        }

    def test_single_config_accepts_legacy_detection_types_key(self):
        """POST /cameras/{id}/config 兼容旧 detection_types 键"""
        saved = []
        patches = self._patched_globals(saved)
        with patches[0], patches[1], patches[2]:
            client = self._make_client()
            resp = client.post("/cameras/cam_01/config", json={
                "detection_types": {"smoke": {"enabled": True, "interval": 5}}
            })
        assert resp.status_code == 200
        assert saved, "save_camera_configs 未被调用"
        cam = saved[0][0]
        assert "detection_types" not in cam
        assert cam["algorithms"] == {"smoke": {"enabled": True}}

    def test_batch_config_strips_params(self):
        """POST /cameras/batch-config 保存时剔除参数"""
        saved = []
        patches = self._patched_globals(saved)
        with patches[0], patches[1], patches[2]:
            client = self._make_client()
            resp = client.post("/cameras/batch-config", json={
                "camera_ids": ["cam_01"],
                "algorithms": {"fire": {"enabled": True, "threshold": 0.9,
                                        "roi": [[1, 1]]}}
            })
        assert resp.status_code == 200
        assert saved, "save_camera_configs 未被调用"
        cam = saved[0][0]
        assert "detection_types" not in cam
        assert cam["algorithms"] == {"fire": {"enabled": True, "roi": [[1, 1]]}}


class TestCameraApiErrors:
    """摄像头写接口的错误响应：不存在的摄像头必须 404，且响应带说明"""

    def _patched_globals(self, saved: list, runtime_ids=()):
        mock_cm = MagicMock()
        mock_cm._cameras = {
            cid: MagicMock(config=MagicMock()) for cid in runtime_ids
        }
        mock_md = MagicMock()
        mock_cfg = MagicMock()
        mock_cfg.load_camera_configs.return_value = [
            {"camera_id": "cam_01", "algorithms": {"fire": {"enabled": False}}}
        ]

        def fake_save(cameras):
            saved.append(cameras)
            return True

        mock_cfg.save_camera_configs.side_effect = fake_save
        return mock_cm, mock_md, (
            patch.object(mm, "camera_manager", mock_cm),
            patch.object(mm, "multi_detector", mock_md),
            patch.object(mm, "app_config", mock_cfg),
        )

    def test_config_returns_404_for_unknown_camera(self):
        """POST /cameras/{id}/config 摄像头不存在 → 404 且不触碰检测器"""
        saved = []
        mock_cm, mock_md, patches = self._patched_globals(saved)
        with patches[0], patches[1], patches[2]:
            client = TestClient(mm.app)
            resp = client.post("/cameras/ghost/config", json={
                "detection_types": {"fire": {"enabled": True}}
            })
        assert resp.status_code == 404
        assert resp.json()["error"] == "Camera not found"
        mock_md.update_camera_config.assert_not_called()
        assert not saved, "不存在的摄像头不应持久化"

    def test_enable_returns_404_for_unknown_camera(self):
        """POST /cameras/{id}/enable 摄像头不存在 → 404"""
        saved = []
        _, _, patches = self._patched_globals(saved)
        with patches[0], patches[1], patches[2]:
            client = TestClient(mm.app)
            resp = client.post("/cameras/ghost/enable")
        assert resp.status_code == 404
        assert resp.json()["error"] == "Camera not found"

    def test_disable_returns_404_for_unknown_camera(self):
        """POST /cameras/{id}/disable 摄像头不存在 → 404"""
        saved = []
        _, _, patches = self._patched_globals(saved)
        with patches[0], patches[1], patches[2]:
            client = TestClient(mm.app)
            resp = client.post("/cameras/ghost/disable")
        assert resp.status_code == 404
        assert resp.json()["error"] == "Camera not found"

    def test_batch_config_reports_not_found_and_updates_runtime(self):
        """POST /cameras/batch-config 响应区分 updated/not_found，且同步运行时配置"""
        saved = []
        mock_cm, _, patches = self._patched_globals(saved, runtime_ids=("cam_01",))
        with patches[0], patches[1], patches[2]:
            client = TestClient(mm.app)
            resp = client.post("/cameras/batch-config", json={
                "camera_ids": ["cam_01", "ghost"],
                "algorithms": {"fire": {"enabled": True}}
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["updated"] == ["cam_01"]
        assert body["not_found"] == ["ghost"]
        # 运行时配置同步更新，避免后续 save_camera_configs 用旧值覆盖
        runtime_cfg = mock_cm._cameras["cam_01"].config
        assert runtime_cfg.detection_types == {"fire": {"enabled": True}}

    def test_source_returns_404_for_unknown_camera(self):
        """POST /cameras/{id}/source 摄像头不存在 → 404"""
        saved = []
        mock_cm, _, patches = self._patched_globals(saved)
        with patches[0], patches[1], patches[2]:
            client = TestClient(mm.app)
            resp = client.post("/cameras/ghost/source", json={"source": "rtsp://x"})
        assert resp.status_code == 404
        assert resp.json()["error"] == "Camera not found"
        mock_cm.set_camera_source.assert_not_called()

    def test_add_requires_camera_id(self):
        """POST /cameras/add 缺 camera_id → 400 带说明"""
        saved = []
        _, _, patches = self._patched_globals(saved)
        with patches[0], patches[1], patches[2]:
            client = TestClient(mm.app)
            resp = client.post("/cameras/add", json={"source": "rtsp://x"})
        assert resp.status_code == 400
        assert resp.json()["error"] == "camera_id is required"

    def test_add_requires_source(self):
        """POST /cameras/add 缺 source → 400 带说明"""
        saved = []
        _, _, patches = self._patched_globals(saved)
        with patches[0], patches[1], patches[2]:
            client = TestClient(mm.app)
            resp = client.post("/cameras/add", json={"camera_id": "cam_02"})
        assert resp.status_code == 400
        assert resp.json()["error"] == "source is required"
