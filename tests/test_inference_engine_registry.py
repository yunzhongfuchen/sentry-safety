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
    """创建一个指向 tmp_path 的测试注册表（播种 6 种内置算法）"""
    import backend.model_registry as model_mod
    import backend.detection_registry as reg_mod

    # 同时 monkeypatch model_registry 和 detection_registry 的路径
    monkeypatch.setattr(model_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(model_mod, "MODELS_FILE", tmp_path / "models.json")
    monkeypatch.setattr(model_mod, "WEIGHTS_DIR", tmp_path)

    monkeypatch.setattr(reg_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(reg_mod, "REGISTRY_FILE", tmp_path / "detection_types.json")
    monkeypatch.setattr(reg_mod, "ALGORITHMS_FILE", tmp_path / "algorithms.json")
    monkeypatch.setattr(reg_mod, "PROJECT_ROOT", tmp_path)

    # 播种模型和算法
    model_mod.model_registry._models = {}
    seeded = reg_mod._migrate_type_dicts(reg_mod.DEFAULT_DETECTION_TYPE_REGISTRY)
    (tmp_path / "algorithms.json").write_text(
        json.dumps(seeded, ensure_ascii=False), encoding="utf-8"
    )

    # 先加载 model_registry,再加载 detection_registry
    model_mod.model_registry.load()
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
        result = _resolve_model_path("fire", "fire_smoke.pt", use_npu=False)
        assert result is not None
        assert "fire_smoke" in result

    def test_resolve_npu_path(self, fake_registry, tmp_path, monkeypatch):
        """NPU 环境同样从注册表 model_path 解析（不再有独立 npu 路径字段）"""
        from backend.inference_engine import _resolve_model_path
        model_file = tmp_path / "fire_smoke.pt"
        model_file.touch()
        import backend.inference_engine as ie_mod
        monkeypatch.setattr(ie_mod, "WEIGHTS_DIR", tmp_path)
        result = _resolve_model_path("fire", "fire_smoke.pt", use_npu=True)
        assert result is not None
        assert "fire_smoke.pt" in result

    def test_resolve_picks_requested_model_path(self, fake_registry, tmp_path, monkeypatch):
        """同一 dtype 含多个 model_path 时按传入的 model_path 精确解析"""
        from backend.inference_engine import _resolve_model_path
        import backend.inference_engine as ie_mod

        class _FakeReg:
            def get(self, dtype):
                if dtype == "multi":
                    return {
                        "models": [
                            {"model_path": "a.pt"},
                            {"model_path": "b.pt"},
                        ]
                    }
                return fake_registry.get(dtype)

        monkeypatch.setattr(ie_mod, "registry", _FakeReg())
        monkeypatch.setattr(ie_mod, "WEIGHTS_DIR", tmp_path)

        (tmp_path / "a.pt").touch()
        (tmp_path / "b.pt").touch()

        assert _resolve_model_path("multi", "a.pt") == str(tmp_path / "a.pt")
        assert _resolve_model_path("multi", "b.pt") == str(tmp_path / "b.pt")

    def test_resolve_rejects_unowned_model_path(self, fake_registry, tmp_path, monkeypatch):
        """请求不属于该 dtype 的 model_path 时返回 None"""
        from backend.inference_engine import _resolve_model_path
        import backend.inference_engine as ie_mod

        class _FakeReg:
            def get(self, dtype):
                if dtype == "multi":
                    return {"models": [{"model_path": "a.pt"}]}
                return fake_registry.get(dtype)

        monkeypatch.setattr(ie_mod, "registry", _FakeReg())
        monkeypatch.setattr(ie_mod, "WEIGHTS_DIR", tmp_path)
        (tmp_path / "a.pt").touch()
        (tmp_path / "b.pt").touch()

        assert _resolve_model_path("multi", "b.pt") is None

    def test_resolve_returns_none_for_missing(self, fake_registry, tmp_path, monkeypatch):
        """模型文件不存在时返回 None"""
        from backend.inference_engine import _resolve_model_path
        import backend.inference_engine as ie_mod
        monkeypatch.setattr(ie_mod, "WEIGHTS_DIR", tmp_path / "weights")
        monkeypatch.setattr(ie_mod, "PROJECT_ROOT", tmp_path)
        result = _resolve_model_path("fire", "fire_smoke.pt", use_npu=False)
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
        # fire_smoke.pt 对应的 model_path 只出现一次
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
        # fire_smoke.pt, mask.pt, yolov8n-pose.pt -> 3 次
        assert len(load_calls) == 3


class TestDetectDispatch:
    """detect() 注册表驱动分发"""

    def test_detect_uses_registry_dispatch(self, detector, fake_registry, monkeypatch):
        """detect() 按注册表遍历类型，调用对应策略函数"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # mock _run_model 返回空 boxes
        monkeypatch.setattr(
            type(detector), "_run_model",
            lambda self_, model_path, frame_, conf, is_pose, core_id: []
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

        def mock_run_model(self_, model_path, frame_, conf, is_pose, core_id):
            run_calls.append(model_path)
            return []

        monkeypatch.setattr(type(detector), "_run_model", mock_run_model)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.detect(frame, ["fire", "smoke"])
        # fire_smoke.pt 只运行一次
        assert len(run_calls) == 1

    def test_detect_relation_filters_by_class(self, detector, fake_registry, monkeypatch):
        """yolo_relation 策略按规则 classes 过滤，返回的 boxes 仅含规则指定类别"""
        import copy
        from backend import inference_engine as ie_mod

        # 临时把 fire/smoke 规则改成 overlap 自身，使 evaluate_rule 返回匹配框，便于断言过滤
        original_get = fake_registry.get
        fire_key = original_get("fire")["models"][0]["model_key"]
        smoke_key = original_get("smoke")["models"][0]["model_key"]

        def patched_get(dtype):
            td = original_get(dtype)
            if dtype == "fire":
                td = copy.deepcopy(td)
                td["rule"] = {"groups": [{"conditions": [
                    {"left": {"model_key": fire_key, "classes": [0]},
                     "op": "overlap",
                     "right": {"model_key": fire_key, "classes": [0]},
                     "iou": 0.001}
                ]}]}
            elif dtype == "smoke":
                td = copy.deepcopy(td)
                td["rule"] = {"groups": [{"conditions": [
                    {"left": {"model_key": smoke_key, "classes": [1]},
                     "op": "overlap",
                     "right": {"model_key": smoke_key, "classes": [1]},
                     "iou": 0.001}
                ]}]}
            return td

        monkeypatch.setattr(ie_mod.registry, "get", patched_get)

        mock_boxes = [
            {"xyxy": [10, 10, 50, 50], "class_id": 0, "confidence": 0.9},  # fire
            {"xyxy": [60, 60, 100, 100], "class_id": 1, "confidence": 0.8},  # smoke
        ]

        monkeypatch.setattr(
            type(detector), "_run_model",
            lambda self_, model_path, frame_, conf, is_pose, core_id: mock_boxes
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame, ["fire", "smoke"])

        assert results["fire"]["detected"] is True
        assert results["smoke"]["detected"] is True
        # 仅保留对应类别的框
        assert results["fire"]["boxes"] == [[10, 10, 50, 50]]
        assert results["smoke"]["boxes"] == [[60, 60, 100, 100]]


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
