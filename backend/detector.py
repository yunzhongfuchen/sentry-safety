import logging

from typing import List
import numpy as np
from ultralytics import YOLO
import config

logger = logging.getLogger(__name__)


class PersonDetector:
    """人体检测器，使用YOLOv8检测人体"""

    def __init__(self):
        self.model = None
        self.model_path = config.YOLO_MODEL
        self.confidence = config.DETECTION_CONFIDENCE
        self.device = config.DETECTION_DEVICE

    def load_model(self):
        """加载YOLOv8模型"""
        if self.model is None:
            logger.info(f"Loading YOLOv8 model: {self.model_path}")
            self.model = YOLO(self.model_path)
        return self.model

    def detect(self, frame: np.ndarray) -> bool:
        """
        检测画面中是否有人体

        Args:
            frame: OpenCV读取的帧 (BGR格式)

        Returns:
            True - 检测到人体, False - 未检测到人体
        """
        if self.model is None:
            self.load_model()

        # YOLOv8 person class ID is 0
        results = self.model(
            frame,
            conf=self.confidence,
            classes=[0],
            device=self.device,
            verbose=False
        )

        for result in results:
            if len(result.boxes) > 0:
                logger.debug(f"Detected {len(result.boxes)} person(s)")
                return True

        return False

    def detect_with_boxes(self, frame: np.ndarray) -> List[dict]:
        """
        检测人体并返回边界框信息

        Args:
            frame: OpenCV读取的帧

        Returns:
            边界框列表 [{'xyxy': [x1,y1,x2,y2], 'confidence': float}, ...]
        """
        if self.model is None:
            self.load_model()

        results = self.model(
            frame,
            conf=self.confidence,
            classes=[0],
            device=self.device,
            verbose=False
        )

        boxes = []
        for result in results:
            for box in result.boxes:
                boxes.append({
                    'xyxy': box.xyxy[0].tolist(),
                    'confidence': float(box.conf[0])
                })

        return boxes