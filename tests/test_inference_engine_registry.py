"""
inference_engine 注册表驱动改造测试
不依赖真实模型和 GPU/NPU，通过 mock 验证注册表驱动逻辑
"""
import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


@pytest.fixture
def fake_registry(tmp_path, monkeypatch):
    """创建一个指向 tmp_path 的测试注册表"""
    import backend.detection_registry as reg_mod
    monkeypatch.setattr(reg_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(reg_mod, "REGISTRY_FILE", tmp_path / "detection_types.json")
    reg_mod.registry.load()
    return reg_mod.registry


@pytest.fixture
def detector():
    """创建 SafetyDetector 实例（CPU 模式，不加载真实模型）"""
    from backend.inference_engine import SafetyDetector
    return SafetyDetector(npu_cores=0, device="cpu")


class TestResolveModelPath:
    """_resolve_model_path 改为注册表驱动后的测试"""

    def test_resolve_reads_from_registry(self, fake_registry, tmp_path, monkeypatch):
        """_resolve_model_path 从注册表读取 model_path 字段"""
        from backend.inference_engine import _resolve_model_path
        # 创建一个假模型文件让路径解析成功
        model_file = tmp_path / "fire_smoke.pt"
        model_file.touch()
        # 注册表里 fire 的 model_path 是 "fire_smoke.pt"
        # _resolve_model_path 应搜索到 tmp_path 下的文件
        import backend.inference_engine as ie_mod
        monkeypatch.setattr(ie_mod, "WEIGHTS_DIR", tmp_path)
        result = _resolve_model_path("fire", use_npu=False)
        assert result is not None
        assert "fire_smoke" in result

    def test_resolve_npu_path(self, fake_registry, tmp_path, monkeypatch):
        """NPU 模型路径从注册表 npu_model_path 读取"""
        from backend.inference_engine import _resolve_model_path
        rknn_file = tmp_path / "fire_smoke.rknn"
        rknn_file.touch()
        import backend.inference_engine as ie_mod
        monkeypatch.setattr(ie_mod, "WEIGHTS_DIR", tmp_path)
        result = _resolve_model_path("fire", use_npu=True)
        assert result is not None
        assert "fire_smoke.rknn" in result

    def test_resolve_returns_none_for_missing(self, fake_registry, tmp_path, monkeypatch):
        """模型文件不存在时返回 None"""
        from backend.inference_engine import _resolve_model_path
        import backend.inference_engine as ie_mod
        monkeypatch.setattr(ie_mod, "WEIGHTS_DIR", tmp_path / "weights")
        monkeypatch.setattr(ie_mod, "PROJECT_ROOT", tmp_path)
        result = _resolve_model_path("fire", use_npu=False)
        # 文件不存在于任何候选路径，应返回 None
        assert result is None


class TestEnsureModelsLoaded:
    """ensure_models_loaded 注册表驱动（共享模型去重）"""

    def test_shared_model_loaded_once(self, detector, fake_registry, monkeypatch):
        """fire 和 smoke 共享 model_path，只加载一次"""
        load_calls = []

        def mock_load_model(self_, model_key, device):
            load_calls.append(model_key)

        monkeypatch.setattr(
            type(detector), "_load_model", mock_load_model
        )
        detector.ensure_models_loaded(["fire", "smoke"])
        # fire_smoke.pt 对应的 model_key 只出现一次
        assert len(load_calls) == 1

    def test_different_models_loaded_separately(self, detector, fake_registry, monkeypatch):
        """不同 model_path 的类型分别加载"""
        load_calls = []

        def mock_load_model(self_, model_key, device):
            load_calls.append(model_key)

        monkeypatch.setattr(
            type(detector), "_load_model", mock_load_model
        )
        detector.ensure_models_loaded(["fire", "mask", "sleep"])
        # fire_smoke.pt, mask.pt, yolov8n-pose.pt → 3 次
        assert len(load_calls) == 3


class TestDetectDispatch:
    """detect() 注册表驱动分发"""

    def test_detect_uses_registry_dispatch(self, detector, fake_registry, monkeypatch):
        """detect() 按注册表遍历类型，调用对应策略函数"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # mock _run_model 返回空 boxes
        monkeypatch.setattr(
            type(detector), "_run_model",
            lambda self_, model_key, frame_, type_def, core_id: []
        )

        results = detector.detect(frame, ["fire", "smoke", "mask"])
        assert "fire" in results
        assert "smoke" in results
        assert "mask" in results
        # 每个结果都是标准格式
        for dtype in ["fire", "smoke", "mask"]:
            assert "detected" in results[dtype]
            assert "boxes" in results[dtype]
            assert "scores" in results[dtype]

    def test_detect_shared_model_runs_once(self, detector, fake_registry, monkeypatch):
        """共享模型只推理一次"""
        run_calls = []

        def mock_run_model(self_, model_key, frame_, type_def, core_id):
            run_calls.append(model_key)
            return []

        monkeypatch.setattr(type(detector), "_run_model", mock_run_model)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.detect(frame, ["fire", "smoke"])
        # fire_smoke.pt 只运行一次
        assert len(run_calls) == 1

    def test_detect_yolo_box_filters_by_class(self, detector, fake_registry, monkeypatch):
        """yolo_box 策略按 classes 过滤"""
        mock_boxes = [
            {"xyxy": [10, 10, 50, 50], "class_id": 0, "confidence": 0.9},  # fire
            {"xyxy": [60, 60, 100, 100], "class_id": 1, "confidence": 0.8},  # smoke
        ]

        monkeypatch.setattr(
            type(detector), "_run_model",
            lambda self_, model_key, frame_, type_def, core_id: mock_boxes
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame, ["fire", "smoke"])

        assert results["fire"]["detected"] is True
        assert len(results["fire"]["boxes"]) == 1
        assert results["fire"]["boxes"][0] == [10, 10, 50, 50]

        assert results["smoke"]["detected"] is True
        assert len(results["smoke"]["boxes"]) == 1
        assert results["smoke"]["boxes"][0] == [60, 60, 100, 100]


class TestGetModelStatus:
    """get_model_status 从注册表读取类型列表"""

    def test_status_lists_all_registry_types(self, detector, fake_registry):
        """get_model_status 返回注册表中所有类型"""
        status = detector.get_model_status()
        type_names = {s["type"] for s in status}
        assert type_names == {"fire", "smoke", "uniform", "mask", "cigarette", "sleep"}

    def test_status_unloaded(self, detector, fake_registry):
        """未加载模型时所有类型 loaded=False"""
        status = detector.get_model_status()
        for s in status:
            assert s["loaded"] is False


class TestProcessYoloBox:
    """_process_yolo_box 后处理函数"""

    def test_filters_by_classes(self, fake_registry):
        from backend.inference_engine import _process_yolo_box
        raw_boxes = [
            {"xyxy": [10, 10, 50, 50], "class_id": 0, "confidence": 0.9},
            {"xyxy": [60, 60, 100, 100], "class_id": 1, "confidence": 0.8},
            {"xyxy": [110, 110, 150, 150], "class_id": 2, "confidence": 0.7},
        ]
        type_def = {"classes": [0, 2], "post_process": "yolo_box"}
        result = _process_yolo_box(raw_boxes, type_def)
        assert result["detected"] is True
        assert len(result["boxes"]) == 2
        assert len(result["scores"]) == 2

    def test_no_classes_filter(self, fake_registry):
        """classes=None 时不过滤"""
        from backend.inference_engine import _process_yolo_box
        raw_boxes = [
            {"xyxy": [10, 10, 50, 50], "class_id": 0, "confidence": 0.9},
        ]
        type_def = {"classes": None, "post_process": "yolo_box"}
        result = _process_yolo_box(raw_boxes, type_def)
        assert result["detected"] is True
        assert len(result["boxes"]) == 1

    def test_empty_boxes(self, fake_registry):
        from backend.inference_engine import _process_yolo_box
        result = _process_yolo_box([], {"classes": [0], "post_process": "yolo_box"})
        assert result["detected"] is False
        assert result["boxes"] == []
        assert result["scores"] == []
