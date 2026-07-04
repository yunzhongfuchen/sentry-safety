"""
推理后端抽象接口
用于隔离 YOLO/CUDA 与后续 Sophon NPU 实现
"""

from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np


try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class InferenceBackend(ABC):
    """推理后端抽象"""

    @abstractmethod
    def predict_batch(self, frames: List[np.ndarray], dtype: str) -> List[Any]:
        """对一批帧进行推理，返回与输入顺序一致的结果列表"""
        ...


class YoloCudaBackend(InferenceBackend):
    """包装现有 Ultralytics YOLO，兼容 CUDA/CPU"""

    def __init__(self, model_path: str, device: str = "cuda",
                 confidence: float = 0.5, classes: List[int] = None):
        if YOLO is None:
            raise RuntimeError("ultralytics is required for YoloCudaBackend")
        self.model = YOLO(model_path)
        self.device = device
        self.confidence = confidence
        self.classes = classes if classes is not None else [0]

    def predict_batch(self, frames: List[np.ndarray], dtype: str) -> List[Any]:
        return self.model(
            frames,
            conf=self.confidence,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
