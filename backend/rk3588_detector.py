"""
RK3588 NPU 加速检测模块
支持 RKNN 模型推理，利用 RK3588 的 3 个 NPU 核心进行并行推理
"""

import logging
import os
import threading
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# RKNN 运行时导入标记
RKNN_AVAILABLE = False
try:
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
    logger.info("RKNNLite imported successfully")
except ImportError:
    logger.warning("RKNNLite not available, falling back to CPU mode")


@dataclass
class RKNNConfig:
    """RKNN 模型配置"""
    model_path: str
    target_platform: str = "rk3588"
    core_mask: int = RKNNLite.NPU_CORE_0 if RKNN_AVAILABLE else 0
    # 输入尺寸
    input_size: Tuple[int, int] = (640, 640)
    # 置信度阈值
    conf_threshold: float = 0.5
    nms_threshold: float = 0.45
    # 检测类别 (person = 0)
    class_filter: List[int] = None
    
    def __post_init__(self):
        if self.class_filter is None:
            self.class_filter = [0]  # 只检测人


class RK3588Detector:
    """
    RK3588 NPU 检测器
    支持单模型多核心调度，实现负载均衡
    """
    
    # NPU 核心掩码
    CORE_MASKS = [
        RKNNLite.NPU_CORE_0 if RKNN_AVAILABLE else 0,
        RKNNLite.NPU_CORE_1 if RKNN_AVAILABLE else 1,
        RKNNLite.NPU_CORE_2 if RKNN_AVAILABLE else 2,
    ]
    
    def __init__(self, config: RKNNConfig = None):
        self.config = config or RKNNConfig(
            model_path=self._get_default_model_path()
        )
        self._rknn_instances: Dict[int, RKNNLite] = {}
        self._lock = threading.RLock()
        self._initialized = False
        self._next_core = 0
        
    def _get_default_model_path(self) -> str:
        """获取默认模型路径"""
        # 优先使用 RKNN 模型
        model_dir = Path(__file__).parent.parent / "models"
        rknn_path = model_dir / "yolov8n.rknn"
        if rknn_path.exists():
            return str(rknn_path)
        return str(model_dir / "yolov8n.pt")
    
    def init(self) -> bool:
        """初始化 RKNN 运行时"""
        if not RKNN_AVAILABLE:
            logger.warning("RKNN not available, init skipped")
            return False
        
        if self._initialized:
            return True
        
        model_path = self.config.model_path
        if not os.path.exists(model_path):
            logger.error(f"RKNN model not found: {model_path}")
            return False
        
        try:
            # 为每个 NPU 核心创建 RKNN 实例
            for i, core_mask in enumerate(self.CORE_MASKS):
                rknn = RKNNLite(verbose=False)
                
                # 加载模型
                ret = rknn.load_rknn(model_path)
                if ret != 0:
                    logger.error(f"Failed to load RKNN model for core {i}")
                    continue
                
                # 初始化运行时
                ret = rknn.init_runtime(core_mask=core_mask)
                if ret != 0:
                    logger.error(f"Failed to init runtime for core {i}")
                    continue
                
                self._rknn_instances[i] = rknn
                logger.info(f"RKNN core {i} initialized")
            
            if self._rknn_instances:
                self._initialized = True
                logger.info(f"RK3588Detector initialized with {len(self._rknn_instances)} cores")
                return True
            else:
                logger.error("No RKNN cores initialized")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize RKNN: {e}")
            return False
    
    def release(self):
        """释放 RKNN 资源"""
        with self._lock:
            for core_id, rknn in self._rknn_instances.items():
                try:
                    rknn.release()
                    logger.info(f"RKNN core {core_id} released")
                except Exception as e:
                    logger.error(f"Error releasing core {core_id}: {e}")
            self._rknn_instances.clear()
            self._initialized = False
    
    def detect(self, frame: np.ndarray, core_id: Optional[int] = None) -> List[dict]:
        """
        执行目标检测
        
        Args:
            frame: 输入图像 (BGR格式)
            core_id: 指定NPU核心，None则自动轮询
            
        Returns:
            检测结果列表 [{'xyxy': [x1,y1,x2,y2], 'confidence': float, 'class_id': int}, ...]
        """
        if not self._initialized or not RKNN_AVAILABLE:
            return []
        
        # 选择 NPU 核心
        if core_id is None:
            core_id = self._get_next_core()
        
        if core_id not in self._rknn_instances:
            logger.error(f"Invalid core_id: {core_id}")
            return []
        
        rknn = self._rknn_instances[core_id]
        
        # 预处理
        input_frame = self._preprocess(frame)
        
        # 推理
        try:
            outputs = rknn.inference(inputs=[input_frame])
            
            # 后处理
            results = self._postprocess(outputs, frame.shape[:2])
            return results
            
        except Exception as e:
            logger.error(f"Inference error on core {core_id}: {e}")
            return []
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[dict]]:
        """
        批量检测，自动分配到多个 NPU 核心并行处理
        
        Args:
            frames: 帧列表
            
        Returns:
            每帧的检测结果
        """
        if not self._initialized:
            return [[] for _ in frames]
        
        results = []
        for i, frame in enumerate(frames):
            core_id = i % len(self._rknn_instances)
            result = self.detect(frame, core_id=core_id)
            results.append(result)
        
        return results
    
    def _get_next_core(self) -> int:
        """轮询选择下一个 NPU 核心"""
        with self._lock:
            core_id = self._next_core
            self._next_core = (self._next_core + 1) % len(self._rknn_instances)
            return core_id
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        图像预处理
        RKNN YOLOv8 输入格式: RGB, 640x640, uint8
        """
        # 调整大小
        input_w, input_h = self.config.input_size
        resized = cv2.resize(frame, (input_w, input_h))
        
        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        return rgb
    
    def _postprocess(self, outputs, orig_shape: Tuple[int, int]) -> List[dict]:
        """
        后处理 RKNN 输出
        
        RKNN YOLOv8 输出格式需要转换，这里简化处理
        实际实现需根据具体 RKNN 模型输出格式调整
        """
        results = []
        
        # TODO: 根据实际 RKNN 模型输出格式实现
        # 这里提供一个通用的 YOLO 后处理框架
        
        orig_h, orig_w = orig_shape
        input_w, input_h = self.config.input_size
        
        # 计算缩放比例
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h
        
        # 解析输出 (需要根据实际模型格式调整)
        # 简化示例：假设输出为 [batch, num_boxes, 6] (x, y, w, h, conf, cls)
        if len(outputs) > 0:
            predictions = outputs[0]
            
            # 过滤置信度
            for pred in predictions:
                if len(pred) < 6:
                    continue
                    
                x, y, w, h, conf, cls = pred[:6]
                
                if conf < self.config.conf_threshold:
                    continue
                
                cls_id = int(cls)
                if self.config.class_filter and cls_id not in self.config.class_filter:
                    continue
                
                # 转换到原图坐标
                x1 = int((x - w/2) * scale_x)
                y1 = int((y - h/2) * scale_y)
                x2 = int((x + w/2) * scale_x)
                y2 = int((y + h/2) * scale_y)
                
                results.append({
                    'xyxy': [max(0, x1), max(0, y1), min(orig_w, x2), min(orig_h, y2)],
                    'confidence': float(conf),
                    'class_id': cls_id
                })
        
        # NMS 去重
        results = self._nms(results)
        
        return results
    
    def _nms(self, boxes: List[dict]) -> List[dict]:
        """非极大值抑制"""
        if not boxes:
            return []
        
        # 按置信度排序
        boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while boxes:
            best = boxes[0]
            keep.append(best)
            
            # 计算 IoU
            boxes = [b for b in boxes[1:] if self._iou(best['xyxy'], b['xyxy']) < self.config.nms_threshold]
        
        return keep
    
    def _iou(self, box1: List[int], box2: List[int]) -> float:
        """计算 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter / (area1 + area2 - inter + 1e-6)
    
    def get_status(self) -> dict:
        """获取检测器状态"""
        return {
            "initialized": self._initialized,
            "rknn_available": RKNN_AVAILABLE,
            "active_cores": len(self._rknn_instances),
            "model_path": self.config.model_path,
        }


class HybridDetector:
    """
    混合检测器 - 优先使用 RKNN NPU，不可用时回退到 CPU
    """
    
    def __init__(self, 
                 rknn_config: RKNNConfig = None,
                 yolo_model: str = "yolov8n.pt",
                 confidence: float = 0.5,
                 device: str = "cpu"):
        self.use_npu = False
        self.rknn_detector = None
        self.cpu_detector = None
        self.yolo_model = yolo_model
        self.confidence = confidence
        self.device = device
        
        # 尝试初始化 NPU
        if RKNN_AVAILABLE:
            self.rknn_detector = RK3588Detector(rknn_config)
            if self.rknn_detector.init():
                self.use_npu = True
                logger.info("Using RK3588 NPU for detection")
            else:
                logger.warning("RK3588 NPU init failed, falling back to CPU")
        
        # 如果 NPU 不可用，初始化 CPU 检测器
        if not self.use_npu:
            self._init_cpu_detector()
    
    def _init_cpu_detector(self):
        """初始化 CPU 检测器 (YOLOv8)"""
        try:
            from ultralytics import YOLO
            self.cpu_detector = YOLO(self.yolo_model)
            logger.info(f"CPU YOLOv8 detector loaded: {self.yolo_model}")
        except Exception as e:
            logger.error(f"Failed to load CPU detector: {e}")
            self.cpu_detector = None
    
    def detect(self, frame: np.ndarray, core_id: Optional[int] = None) -> List[dict]:
        """执行检测"""
        if self.use_npu and self.rknn_detector:
            return self.rknn_detector.detect(frame, core_id)
        
        if self.cpu_detector:
            results = self.cpu_detector(
                frame,
                conf=self.confidence,
                classes=[0],  # person
                device=self.device,
                verbose=False
            )
            
            boxes = []
            for result in results:
                for box in result.boxes:
                    boxes.append({
                        'xyxy': box.xyxy[0].tolist(),
                        'confidence': float(box.conf[0]),
                        'class_id': int(box.cls[0])
                    })
            return boxes
        
        return []
    
    def release(self):
        """释放资源"""
        if self.rknn_detector:
            self.rknn_detector.release()
        
    def get_status(self) -> dict:
        """获取检测器状态"""
        status = {
            "mode": "NPU" if self.use_npu else "CPU",
            "cpu_available": self.cpu_detector is not None,
        }
        
        if self.rknn_detector:
            status.update(self.rknn_detector.get_status())
        
        return status


# 模型转换工具函数
def convert_yolo_to_rknn(yolo_path: str, output_path: str, target: str = "rk3588"):
    """
    将 YOLO 模型转换为 RKNN 格式
    需要在 x86 Linux 环境使用 RKNN Toolkit2 进行转换
    
    使用示例:
        python -c "from rk3588_detector import convert_yolo_to_rknn; 
                   convert_yolo_to_rknn('yolov8n.pt', 'yolov8n.rknn')"
    """
    logger.info(f"Converting {yolo_path} to RKNN format...")
    logger.info(f"Target: {target}")
    logger.info(f"Output: {output_path}")
    
    # 这里应该调用 RKNN Toolkit2 的 API 进行转换
    # 由于需要在 x86 环境运行，这里只提供转换命令示例
    
    print("""
    # RKNN 模型转换步骤:
    
    1. 在 x86 Linux 环境安装 RKNN Toolkit2:
       pip install rknn-toolkit2
    
    2. 导出 ONNX 模型:
       yolo export model=yolov8n.pt format=onnx opset=12
    
    3. 使用 RKNN Toolkit 转换:
       python convert.py
    
    # convert.py 示例:
    from rknn.api import RKNN
    
    rknn = RKNN(verbose=True)
    
    # 配置
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], 
                target_platform='rk3588')
    
    # 加载 ONNX
    rknn.load_onnx(model='yolov8n.onnx')
    
    # 构建模型
    rknn.build(do_quantization=True, dataset='dataset.txt')
    
    # 导出 RKNN
    rknn.export_rknn('yolov8n.rknn')
    """)
