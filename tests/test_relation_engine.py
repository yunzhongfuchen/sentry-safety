import numpy as np
from backend.inference_engine import SafetyDetector


class _FakeRegistry:
    def __init__(self, types):
        self._types = types

    def get(self, dtype):
        return self._types.get(dtype)

    def all_types(self):
        return list(self._types.keys())

    def get_types_by_model(self, model_key):
        return [dt for dt, td in self._types.items() if td.get("model_key") == model_key]


def _make_detector(monkeypatch, types, raw_map, calls):
    det = SafetyDetector(npu_cores=0, device="cpu")
    monkeypatch.setattr("backend.inference_engine.registry", _FakeRegistry(types))

    def fake_run(model_path, frame, conf, is_pose, core_id=0):
        calls.append(model_path)
        return raw_map[model_path]

    monkeypatch.setattr(det, "_run_model", fake_run)
    return det


def test_shared_model_inferred_once(monkeypatch):
    weld_rule = {
        "groups": [
            {
                "conditions": [
                    {
                        "left": {"model_key": "chem", "classes": [1]},
                        "op": "overlap",
                        "right": {"model_key": "chem", "classes": [2]},
                        "iou": 0.001,
                    }
                ]
            }
        ]
    }
    smoke_rule = {
        "groups": [
            {
                "conditions": [
                    {
                        "left": {"model_key": "chem", "classes": [8]},
                        "op": "exists",
                    }
                ]
            }
        ]
    }
    types = {
        "algo_weld": {
            "post_process": "yolo_relation",
            "models": [{"model_key": "chem", "model_path": "chem.pt", "model_confidence": 0.3}],
            "rule": weld_rule,
        },
        "algo_smoke": {
            "post_process": "yolo_relation",
            "models": [{"model_key": "chem", "model_path": "chem.pt", "model_confidence": 0.5}],
            "rule": smoke_rule,
        },
    }
    raw = [
        {"xyxy": [0, 0, 100, 200], "class_id": 1, "confidence": 0.9},
        {"xyxy": [50, 50, 80, 80], "class_id": 2, "confidence": 0.8},
    ]
    calls = []
    det = _make_detector(monkeypatch, types, {"chem.pt": raw}, calls)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = det.detect(frame, ["algo_weld", "algo_smoke"], camera_id="c1", frame_seq=1)
    assert calls == ["chem.pt"]  # 同帧同模型只推理一次
    assert results["algo_weld"]["detected"] is True
    assert results["algo_smoke"]["detected"] is False


def test_frame_seq_cache(monkeypatch):
    smoke_rule = {
        "groups": [
            {
                "conditions": [
                    {
                        "left": {"model_key": "chem", "classes": [8]},
                        "op": "exists",
                    }
                ]
            }
        ]
    }
    types = {
        "algo_smoke": {
            "post_process": "yolo_relation",
            "models": [{"model_key": "chem", "model_path": "chem.pt", "model_confidence": 0.5}],
            "rule": smoke_rule,
        }
    }
    calls = []
    det = _make_detector(monkeypatch, types, {"chem.pt": []}, calls)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det.detect(frame, ["algo_smoke"], camera_id="c1", frame_seq=7)
    det.detect(frame, ["algo_smoke"], camera_id="c1", frame_seq=7)  # 同帧 -> 缓存命中
    det.detect(frame, ["algo_smoke"], camera_id="c1", frame_seq=8)  # 新帧 -> 重新推理
    assert calls == ["chem.pt", "chem.pt"]


def test_roi_prefilter(monkeypatch):
    smoke_rule = {
        "groups": [
            {
                "conditions": [
                    {
                        "left": {"model_key": "chem", "classes": [8]},
                        "op": "exists",
                    }
                ]
            }
        ]
    }
    types = {
        "algo_smoke": {
            "post_process": "yolo_relation",
            "models": [{"model_key": "chem", "model_path": "chem.pt", "model_confidence": 0.5}],
            "rule": smoke_rule,
        }
    }
    raw = [{"xyxy": [590, 430, 630, 470], "class_id": 8, "confidence": 0.9}]  # 右下角，ROI 外
    calls = []
    det = _make_detector(monkeypatch, types, {"chem.pt": raw}, calls)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    roi = [[(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]]  # 左上四分之一
    results = det.detect(
        frame,
        ["algo_smoke"],
        camera_id="c1",
        frame_seq=1,
        roi_map={"algo_smoke": (roi, False)},
    )
    assert results["algo_smoke"]["detected"] is False
    assert results["algo_smoke"].get("roi_applied") is True
