"""
多类型安全检测器
采用懒加载策略，CPU 模型全局单例，NPU 模型每核心独立实例
检测类型由 backend/detection_registry 配置驱动
"""

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple
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
    logger.warning("RKNNLite not available, NPU mode disabled")

# 防御性修复：rknnlite 导入会破坏 logging._nameToLevel，导致后续 torch/ultralytics 初始化时
# setLevel('WARNING') 抛出 ValueError: Unknown level: 'WARNING'。
# 在 torch 导入前强制 reload 标准 logging 模块，恢复原始状态。
import importlib
importlib.reload(logging)
logger = logging.getLogger(__name__)

# 尝试导入 ultralytics（CPU fallback）
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("ultralytics not available, CPU YOLO fallback disabled")

from backend.detection_registry import registry


# 项目根目录（用于解析相对路径）
PROJECT_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"


# 内部 person 模型路径（person 不在检测类型注册表中）
_PERSON_PATHS = {
    "cpu": [
        str(WEIGHTS_DIR / "yolov8n.pt"),
        str(PROJECT_ROOT / "yolov8n.pt"),
        "models/yolov8n.pt",
    ],
    "npu": [
        str(WEIGHTS_DIR / "yolov8n.rknn"),
        str(PROJECT_ROOT / "yolov8n.rknn"),
        "models/yolov8n.rknn",
    ],
}


def detect_npu_cores() -> int:
    """检测可用的 NPU 核心数（RK3588 为 3）。
    只要 rknnlite 能导入即认为 NPU 可用，不检查 /dev/rknpu 设备节点。
    """
    if not RKNN_AVAILABLE:
        return 0
    # RK3588 固定 3 核，实际初始化时由 RKNNLite 自行验证
    return 3


def detect_best_device() -> Tuple[str, int]:
    """检测最优可用设备，优先级: GPU > NPU > CPU

    Returns:
        (device, npu_cores)  device 为 "gpu" | "npu" | "cpu"
    """
    # 1. 检查 CUDA GPU（执行实际计算验证，捕获架构不兼容）
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.current_device()
            (torch.tensor([1.0]).cuda() * 2).cpu()
            logger.info("CUDA GPU detected and verified, using GPU mode")
            return "gpu", 0
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"CUDA detection failed: {e}")

    # 2. 检查 NPU
    npu_cores = detect_npu_cores()
    if npu_cores > 0:
        logger.info(f"NPU detected with {npu_cores} cores")
        return "npu", npu_cores

    # 3. 回退 CPU
    logger.info("No GPU/NPU detected, using CPU mode")
    return "cpu", 0


def _device_fallback_order(preferred: str) -> List[str]:
    """根据首选设备返回尝试顺序，优先级：gpu > npu > cpu"""
    order = []
    if preferred == "gpu":
        order = ["gpu", "npu", "cpu"]
    elif preferred == "npu":
        order = ["npu", "cpu"]
    else:
        order = ["cpu"]
    seen = set()
    result = []
    for d in order:
        if d not in seen:
            seen.add(d)
            result.append(d)
    return result


def _resolve_model_path(dtype: str, use_npu: bool) -> Optional[str]:
    """解析模型路径：优先环境变量，其次按注册表文件名在标准目录中查找

    model_path 存储当前环境实际的模型文件名（NPU 部署配 .rknn，CPU/GPU 部署配 .pt），
    程序不做后缀推导，读什么用什么。
    """
    env_key = f"{dtype.upper()}_RKNN_MODEL" if use_npu else f"{dtype.upper()}_MODEL"
    env_path = os.getenv(env_key)
    if env_path and os.path.exists(env_path):
        return env_path

    type_def = registry.get(dtype)
    if type_def is not None:
        filename = type_def.get("model_path")
        if filename is None:
            return None
        candidates = [
            str(WEIGHTS_DIR / filename),
            str(PROJECT_ROOT / filename),
            f"models/{filename}",
        ]
    elif dtype == "person":
        # 内部 person 模型不在注册表中，使用硬编码候选路径
        key = "npu" if use_npu else "cpu"
        candidates = _PERSON_PATHS.get(key, [])
    else:
        return None

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _process_yolo_box(raw_boxes: list, type_def: dict) -> dict:
    """yolo_box 后处理：按 classes 过滤框"""
    result = {"detected": False, "boxes": [], "scores": []}
    target_classes = type_def.get("classes")
    for box in raw_boxes:
        if target_classes is not None and box.get("class_id") not in target_classes:
            continue
        result["boxes"].append(box["xyxy"])
        result["scores"].append(box["confidence"])
    result["detected"] = len(result["boxes"]) > 0
    if result["scores"]:
        result["max_confidence"] = max(result["scores"])
    else:
        result["max_confidence"] = 0.0
    return result


def _process_yolo_pose(raw_output, type_def: dict, frame: np.ndarray) -> dict:
    """yolo_pose 后处理：姿态模型 + sleep_detect 分析"""
    result = {"detected": False, "boxes": [], "scores": [], "subjects": [], "count": 0}
    try:
        from safety_detection.sleep_detect import process_frame
        model = raw_output  # yolo_pose 时 raw_output 就是模型实例
        subjects = process_frame(model, frame, conf=type_def.get("model_confidence", 0.1))
        for s in subjects:
            result["boxes"].append(s["box"])
            result["scores"].append(s.get("sleep_confidence", 0))
            result["subjects"].append({
                "box": s["box"],
                "score": s.get("score", 0),
                "sleep_confidence": s.get("sleep_confidence", 0),
                "posture": s.get("posture_label", ""),
                "keypoints": s.get("keypoints"),
                "sleeping": s.get("sleeping", False),
            })
            if s.get("sleeping"):
                result["detected"] = True
                result["count"] += 1
        result["max_confidence"] = max(result["scores"]) if result["scores"] else 0.0
    except Exception as e:
        logger.error(f"Pose post-process error: {e}")
    return result


POST_PROCESSORS = {
    "yolo_box": _process_yolo_box,
    "yolo_pose": _process_yolo_pose,
}


class SafetyDetector:
    """
    安全检测器：支持多类型检测，懒加载模型
    CPU 模型：全局单例共享
    NPU 模型：每种模型 × NPU核心数 个独立实例
    """

    # NPU 核心掩码
    CORE_MASKS = [
        RKNNLite.NPU_CORE_0 if RKNN_AVAILABLE else 0,
        RKNNLite.NPU_CORE_1 if RKNN_AVAILABLE else 1,
        RKNNLite.NPU_CORE_2 if RKNN_AVAILABLE else 2,
    ]

    def __init__(self, npu_cores: int = 0, device: str = "cpu"):
        self._npu_cores = npu_cores
        self.device = device
        self._cpu_models: Dict[str, Any] = {}          # model_key -> model instance (also holds GPU models)
        self._npu_models: Dict[str, Dict[int, Any]] = {} # model_key -> {core_id: rknn}
        self._model_lock = threading.RLock()
        logger.info(f"SafetyDetector initialized (device={device}, npu_cores={npu_cores})")

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def ensure_models_loaded(self, detection_types: List[str], device: str = None) -> None:
        """懒加载指定类型所需的模型，共享 model_path 的类型只加载一次"""
        if device is None:
            device = self.device
        loaded_paths = set()
        with self._model_lock:
            for dtype in detection_types:
                type_def = registry.get(dtype)
                if type_def is None:
                    logger.warning(f"Unknown detection type: {dtype}")
                    continue
                model_key = type_def["model_path"]
                if model_key in loaded_paths:
                    continue
                loaded_paths.add(model_key)
                self._load_model(model_key, device)

    def _load_model(self, model_path: str, device: str) -> None:
        """通用模型加载：npu / gpu / cpu 三分支。model_path 为模型文件名（缓存 key）。"""
        # 按 model_path（文件名）查找使用此模型的第一个类型，用于解析实际路径
        def _first_dtype_by_path(mpath: str):
            return next(
                (dt for dt in registry.all_types() if (registry.get(dt) or {}).get("model_path") == mpath),
                None,
            )

        if device == "npu" and self._npu_cores > 0:
            if model_path not in self._npu_models:
                dtype = _first_dtype_by_path(model_path)
                if dtype is None:
                    return
                path = _resolve_model_path(dtype, use_npu=True)
                if path and RKNN_AVAILABLE:
                    self._npu_models[model_path] = {}
                    for core_id in range(self._npu_cores):
                        rknn = RKNNLite(verbose=False)
                        ret = rknn.load_rknn(path)
                        if ret != 0:
                            logger.error(f"Failed to load {model_path} RKNN for core {core_id}")
                            continue
                        ret = rknn.init_runtime(core_mask=self.CORE_MASKS[core_id])
                        if ret != 0:
                            logger.error(f"Failed to init {model_path} RKNN runtime for core {core_id}")
                            continue
                        self._npu_models[model_path][core_id] = rknn
                        logger.info(f"{model_path} RKNN loaded on core {core_id}")
                else:
                    logger.warning(f"{model_path} RKNN not found, falling back to CPU")
        elif device == "gpu":
            if model_path not in self._cpu_models:
                dtype = _first_dtype_by_path(model_path)
                if dtype is None:
                    return
                path = _resolve_model_path(dtype, use_npu=False)
                if path and ULTRALYTICS_AVAILABLE:
                    model = YOLO(path)
                    try:
                        model = model.to("cuda")
                        logger.info(f"{model_path} GPU model loaded from {path}")
                    except Exception as e:
                        logger.warning(f"Failed to move {model_path} to GPU: {e}")
                    self._cpu_models[model_path] = model
                else:
                    logger.warning(f"{model_path} GPU model not found")
        else:
            if model_path not in self._cpu_models:
                dtype = _first_dtype_by_path(model_path)
                if dtype is None:
                    return
                path = _resolve_model_path(dtype, use_npu=False)
                if path and ULTRALYTICS_AVAILABLE:
                    self._cpu_models[model_path] = YOLO(path)
                    logger.info(f"{model_path} CPU model loaded from {path}")
                else:
                    logger.warning(f"{model_path} model not found")

    def _load_person_model(self, device: str = None) -> None:
        """加载通用人员检测模型（yolov8n.pt / yolov8n.rknn），按 gpu>npu>cpu 自动适应"""
        if device is None:
            device = self.device
        dtype = "person"
        if dtype in self._cpu_models or dtype in self._npu_models:
            return

        order = _device_fallback_order(device)
        loaded = False

        for try_device in order:
            if loaded:
                break
            try:
                if try_device == "npu" and self._npu_cores > 0:
                    path = _resolve_model_path(dtype, use_npu=True)
                    if path and RKNN_AVAILABLE:
                        self._npu_models[dtype] = {}
                        for core_id in range(self._npu_cores):
                            rknn = RKNNLite(verbose=False)
                            ret = rknn.load_rknn(path)
                            if ret != 0:
                                logger.error(f"Failed to load person RKNN for core {core_id}")
                                continue
                            ret = rknn.init_runtime(core_mask=self.CORE_MASKS[core_id])
                            if ret != 0:
                                logger.error(f"Failed to init person RKNN runtime for core {core_id}")
                                continue
                            self._npu_models[dtype][core_id] = rknn
                            logger.info(f"Person RKNN loaded on core {core_id}")
                        if self._npu_models[dtype]:
                            loaded = True
                            logger.info(f"Person NPU model loaded from {path}")
                        else:
                            self._npu_models.pop(dtype, None)
                    else:
                        logger.warning("Person RKNN model not found")

                elif try_device == "gpu":
                    path = _resolve_model_path(dtype, use_npu=False)
                    if path and ULTRALYTICS_AVAILABLE:
                        model = YOLO(path)
                        model = model.to("cuda")
                        self._cpu_models[dtype] = model
                        loaded = True
                        logger.info(f"Person GPU model loaded from {path}")
                    else:
                        logger.warning("Person GPU model not found")

                else:  # cpu
                    path = _resolve_model_path(dtype, use_npu=False)
                    if path and ULTRALYTICS_AVAILABLE:
                        model = YOLO(path)
                        self._cpu_models[dtype] = model
                        loaded = True
                        logger.info(f"Person CPU model loaded from {path}")
                    else:
                        logger.warning("Person CPU model not found")
            except Exception as e:
                logger.warning(f"Person model load failed on {try_device}: {e}")

        if not loaded:
            logger.error("Failed to load person model on any device")

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def _run_model(self, model_key: str, frame: np.ndarray,
                   type_def: dict, core_id: int = 0):
        """执行模型推理，返回原始检测结果"""
        model = None
        use_npu = False
        with self._model_lock:
            if model_key in self._npu_models and core_id in self._npu_models[model_key]:
                model = self._npu_models[model_key][core_id]
                use_npu = True
            elif model_key in self._cpu_models:
                model = self._cpu_models[model_key]

        if model is None:
            logger.warning(f"Model {model_key} not loaded")
            return [] if type_def["post_process"] == "yolo_box" else model

        # yolo_pose 直接返回模型实例（由 _process_yolo_pose 自行调用 process_frame）
        if type_def["post_process"] == "yolo_pose":
            return model

        # yolo_box：统一推理流程
        try:
            if use_npu:
                input_frame = self._preprocess(frame)
                outputs = model.inference(inputs=[input_frame])
                conf = type_def.get("model_confidence", 0.5)
                return self._postprocess_rknn(outputs, frame.shape[:2], conf_threshold=conf)
            else:
                conf = type_def.get("model_confidence", 0.5)
                pred = model.predict(frame, conf=conf, verbose=False)
                boxes = []
                if pred and pred[0].boxes is not None:
                    for b in pred[0].boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        cls = int(b.cls[0])
                        score = float(b.conf[0])
                        boxes.append({"xyxy": [x1, y1, x2, y2], "class_id": cls, "confidence": score})
                return boxes
        except Exception as e:
            logger.error(f"Model {model_key} inference error: {e}")
            return []

    def detect(self, frame: np.ndarray, detection_types: List[str],
               core_id: int = 0) -> Dict[str, dict]:
        """
        对单帧执行多类型检测（注册表驱动）

        共享 model_key 的类型只推理一次，各自按 classes 过滤。
        """
        results: Dict[str, dict] = {}
        processed_models = set()

        for dtype in detection_types:
            type_def = registry.get(dtype)
            if type_def is None:
                logger.warning(f"Unknown detection type: {dtype}")
                continue

            # model_path 是模型文件名（缓存 key），model_key 是注册表 slug（用于 get_types_by_model）
            model_path = type_def.get("model_path")
            model_key = type_def.get("model_key")
            if model_path is None or model_path in processed_models:
                continue

            raw_output = self._run_model(model_path, frame, type_def, core_id)

            # 对共享此模型的所有请求类型执行后处理
            for related_dtype in registry.get_types_by_model(model_key):
                if related_dtype not in detection_types:
                    continue
                related_def = registry.get(related_dtype)
                strategy = related_def["post_process"]
                processor = POST_PROCESSORS.get(strategy)
                if processor is None:
                    logger.warning(f"Unknown post_process strategy: {strategy}")
                    continue
                if strategy == "yolo_pose":
                    results[related_dtype] = processor(raw_output, related_def, frame)
                else:
                    results[related_dtype] = processor(raw_output, related_def)

            processed_models.add(model_path)

        return results

    def _detect_persons(self, frame: np.ndarray, core_id: int = 0) -> List[dict]:
        """通用人员检测（yolov8n / yolov8n.rknn），仅返回 person 类别"""
        import time

        model = None
        use_npu = False
        with self._model_lock:
            if "person" in self._npu_models and core_id in self._npu_models["person"]:
                model = self._npu_models["person"][core_id]
                use_npu = True
            elif "person" in self._cpu_models:
                model = self._cpu_models["person"]

        if model is None:
            return []

        try:
            t0 = time.perf_counter()
            if use_npu:
                input_frame = self._preprocess(frame)
                outputs = model.inference(inputs=[input_frame])
                boxes = self._postprocess_rknn(outputs, frame.shape[:2])
                # 过滤 person 类别 (class_id == 0)
                boxes = [b for b in boxes if b.get("class_id") == 0]
            else:
                pred = model.predict(frame, classes=[0], verbose=False)
                boxes = []
                if pred and pred[0].boxes is not None:
                    for b in pred[0].boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        score = float(b.conf[0])
                        boxes.append({"xyxy": [x1, y1, x2, y2], "confidence": score})
            t1 = time.perf_counter()
            dev = "npu" if use_npu else ("gpu" if self.device == "gpu" else "cpu")
            logger.info(f"[PERSON_DETECT] device={dev} predict_time={(t1-t0)*1000:.1f}ms boxes={len(boxes)}")
            return boxes
        except Exception as e:
            logger.error(f"Person detection error: {e}")
            return []

    # ------------------------------------------------------------------
    # 预处理 / 后处理（NPU）
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(frame: np.ndarray, input_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """resize -> BGR->RGB，返回 uint8 NHWC（RKNN 内部自动归一化）"""
        img = cv2.resize(frame, input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.expand_dims(img, axis=0)  # (1, H, W, C)

    @staticmethod
    def _postprocess_rknn(outputs: List[np.ndarray], orig_shape: Tuple[int, int],
                          conf_threshold: float = 0.5, nms_threshold: float = 0.45) -> List[dict]:
        """
        RKNN YOLO 输出后处理（简化版，适配常见 YOLOv8 RKNN 输出格式）
        outputs[0] shape: (1, 84, 8400) 即 (batch, 4+80, num_anchors)
        """
        import numpy as np
        if not outputs or outputs[0] is None:
            logger.warning(f"RKNN inference returned empty/None outputs: {outputs}")
            return []
        preds = outputs[0]  # (1, 84, 8400)
        if preds.ndim == 3:
            preds = preds[0]  # (84, 8400)
        else:
            return []

        # transpose to (8400, 84)
        preds = np.transpose(preds, (1, 0))

        boxes = []
        for pred in preds:
            conf = pred[4:].max()
            if conf < conf_threshold:
                continue
            class_id = int(pred[4:].argmax())
            cx, cy, w, h = pred[:4]
            x1 = int((cx - w / 2) / 640 * orig_shape[1])
            y1 = int((cy - h / 2) / 640 * orig_shape[0])
            x2 = int((cx + w / 2) / 640 * orig_shape[1])
            y2 = int((cy + h / 2) / 640 * orig_shape[0])
            boxes.append({"xyxy": [x1, y1, x2, y2], "class_id": class_id, "confidence": float(conf)})

        # 简单 NMS（按类别分别做）
        if len(boxes) > 1:
            boxes = _nms_boxes(boxes, nms_threshold)
        return boxes

    # ------------------------------------------------------------------
    # 资源管理
    # ------------------------------------------------------------------

    def release(self) -> None:
        """释放所有模型资源"""
        with self._model_lock:
            for model_key, core_map in self._npu_models.items():
                for core_id, rknn in core_map.items():
                    try:
                        rknn.release()
                        logger.info(f"Released {model_key} RKNN on core {core_id}")
                    except Exception as e:
                        logger.error(f"Error releasing {model_key} RKNN core {core_id}: {e}")
            self._npu_models.clear()
            self._cpu_models.clear()

    @property
    def loaded_models(self) -> List[str]:
        """返回已加载的模型列表（model_key，不含内部 person 模型）"""
        models = []
        with self._model_lock:
            models.extend([k for k in self._cpu_models.keys() if k != "person"])
            models.extend(self._npu_models.keys())
        return list(dict.fromkeys(models))

    def get_model_status(self) -> List[Dict[str, Any]]:
        """返回模型状态详情（注册表驱动）"""
        status = []
        with self._model_lock:
            for dtype in registry.all_types():
                type_def = registry.get(dtype)
                model_key = type_def["model_path"]
                # 检查是否已加载
                is_loaded = model_key in self._cpu_models or model_key in self._npu_models
                if is_loaded:
                    if model_key in self._cpu_models:
                        model = self._cpu_models[model_key]
                        model_device = getattr(model, "device", None)
                        if model_device is not None:
                            device_type = str(model_device).split(":")[0]
                        else:
                            device_type = "cpu"
                        if device_type == "cuda":
                            entry = {"type": dtype, "backend": "gpu", "device": "cuda", "loaded": True}
                        else:
                            entry = {"type": dtype, "backend": "cpu", "device": "pytorch", "loaded": True}
                    else:
                        core_map = self._npu_models.get(model_key, {})
                        entry = {"type": dtype, "backend": "npu", "device": "rk3588",
                                 "cores": len(core_map), "loaded": True}
                else:
                    entry = {"type": dtype, "backend": self.device,
                             "device": self.device, "loaded": False}
                status.append(entry)
        return status


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------

def _nms_boxes(boxes: List[dict], threshold: float = 0.45) -> List[dict]:
    """简易 NMS"""
    if not boxes:
        return boxes
    # 按置信度降序
    boxes = sorted(boxes, key=lambda x: x["confidence"], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if _iou(best["xyxy"], b["xyxy"]) < threshold]
    return keep


def _iou(box_a: List[int], box_b: List[int]) -> float:
    """计算两个框的 IoU"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_b[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
