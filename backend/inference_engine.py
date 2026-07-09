"""
多类型安全检测器
支持 fire、smoke、uniform、mask、cigarette、sleep 六种检测类型
采用懒加载策略，CPU 模型全局单例，NPU 模型每核心独立实例
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


# 项目根目录（用于解析相对路径）
PROJECT_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# 模型默认路径配置
MODEL_PATHS = {
    "fire": {
        "cpu": [
            str(WEIGHTS_DIR / "fire_smoke.pt"),
            str(PROJECT_ROOT / "fire_smoke.pt"),
            "models/fire.pt",
        ],
        "npu": [
            str(WEIGHTS_DIR / "fire_smoke.rknn"),
            str(PROJECT_ROOT / "fire_smoke.rknn"),
            "models/fire.rknn",
        ],
    },
    "uniform": {
        "cpu": [
            str(WEIGHTS_DIR / "uniform.pt"),
            str(PROJECT_ROOT / "uniform.pt"),
            "models/uniform.pt",
        ],
        "npu": [
            str(WEIGHTS_DIR / "uniform.rknn"),
            str(PROJECT_ROOT / "uniform.rknn"),
            "models/uniform.rknn",
        ],
    },
    "person": {
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
    },
    "sleep": {
        "cpu": [
            str(WEIGHTS_DIR / "yolov8s-pose.pt"),
            str(WEIGHTS_DIR / "yolov8n-pose.pt"),
            str(PROJECT_ROOT / "yolov8s-pose.pt"),
            str(PROJECT_ROOT / "yolov8n-pose.pt"),
            "models/yolov8s-pose.pt",
            "models/yolov8n-pose.pt",
        ],
        "npu": [
            str(WEIGHTS_DIR / "yolov8s-pose.rknn"),
            str(WEIGHTS_DIR / "yolov8n-pose.rknn"),
            str(PROJECT_ROOT / "yolov8s-pose.rknn"),
            str(PROJECT_ROOT / "yolov8n-pose.rknn"),
            "models/yolov8s-pose.rknn",
            "models/yolov8n-pose.rknn",
        ],
    },
    "mask": {
        "cpu": [
            str(WEIGHTS_DIR / "mask.pt"),
            str(PROJECT_ROOT / "mask.pt"),
            "models/mask.pt",
        ],
        "npu": [
            str(WEIGHTS_DIR / "mask.rknn"),
            str(PROJECT_ROOT / "mask.rknn"),
            "models/mask.rknn",
        ],
    },
    "cigarette": {
        "cpu": [
            str(WEIGHTS_DIR / "cigarette.pt"),
            str(PROJECT_ROOT / "cigarette.pt"),
            "models/cigarette.pt",
        ],
        "npu": [
            str(WEIGHTS_DIR / "cigarette.rknn"),
            str(PROJECT_ROOT / "cigarette.rknn"),
            "models/cigarette.rknn",
        ],
    },
}

# YOLOv8 自定义模型目标类别（逗号分隔 class id，默认 0）
MASK_TARGET_CLASSES = [int(x) for x in os.getenv("MASK_TARGET_CLASSES", "0").split(",") if x.strip()] or [0]
CIGARETTE_TARGET_CLASSES = [int(x) for x in os.getenv("CIGARETTE_TARGET_CLASSES", "0").split(",") if x.strip()] or [0]
# 工服检测：默认 class 1 表示未穿工服/反光背心（class 0 表示已穿），可通过环境变量调整
UNIFORM_TARGET_CLASSES = [int(x) for x in os.getenv("UNIFORM_TARGET_CLASSES", "1").split(",") if x.strip()] or [1]

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
    """解析模型路径：优先环境变量，其次按候选列表查找"""
    env_key = f"{dtype.upper()}_MODEL" if not use_npu else f"{dtype.upper()}_RKNN_MODEL"
    env_path = os.getenv(env_key)
    if env_path and os.path.exists(env_path):
        return env_path

    key = "npu" if use_npu else "cpu"
    candidates = MODEL_PATHS.get(dtype, {}).get(key)
    if candidates is None:
        return None
    if isinstance(candidates, str):
        candidates = [candidates]

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


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
        self._cpu_models: Dict[str, Any] = {}          # dtype -> model instance (also holds GPU models)
        self._npu_models: Dict[str, Dict[int, Any]] = {} # dtype -> {core_id: rknn}
        self._model_lock = threading.RLock()
        logger.info(f"SafetyDetector initialized (device={device}, npu_cores={npu_cores})")

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def ensure_models_loaded(self, detection_types: List[str], device: str = None) -> None:
        """懒加载指定类型所需的模型（支持 gpu / npu / cpu）"""
        if device is None:
            device = self.device
        with self._model_lock:
            for dtype in detection_types:
                if dtype in ("fire", "smoke"):
                    self._load_fire_smoke_model(device)
                elif dtype == "uniform":
                    self._load_uniform_model(device)
                elif dtype == "mask":
                    self._load_mask_model(device)
                elif dtype == "cigarette":
                    self._load_cigarette_model(device)
                elif dtype == "sleep":
                    self._load_sleep_model(device)

    def _load_fire_smoke_model(self, device: str = None) -> None:
        """加载 fire/smoke 模型（共用同一个模型文件）"""
        if device is None:
            device = self.device
        dtype = "fire"
        if device == "npu" and self._npu_cores > 0:
            if dtype not in self._npu_models:
                path = _resolve_model_path(dtype, use_npu=True)
                if path and RKNN_AVAILABLE:
                    self._npu_models[dtype] = {}
                    for core_id in range(self._npu_cores):
                        rknn = RKNNLite(verbose=False)
                        ret = rknn.load_rknn(path)
                        if ret != 0:
                            logger.error(f"Failed to load fire RKNN for core {core_id}")
                            continue
                        ret = rknn.init_runtime(core_mask=self.CORE_MASKS[core_id])
                        if ret != 0:
                            logger.error(f"Failed to init fire RKNN runtime for core {core_id}")
                            continue
                        self._npu_models[dtype][core_id] = rknn
                        logger.info(f"Fire RKNN loaded on core {core_id}")
                else:
                    logger.warning("Fire RKNN model not found, falling back to CPU")
        elif device == "gpu":
            if dtype not in self._cpu_models:
                path = _resolve_model_path(dtype, use_npu=False)
                if path and ULTRALYTICS_AVAILABLE:
                    model = YOLO(path)
                    try:
                        model = model.to("cuda")
                        logger.info(f"Fire GPU model loaded from {path}")
                    except Exception as e:
                        logger.warning(f"Failed to move fire model to GPU, using CPU: {e}")
                    self._cpu_models[dtype] = model
                else:
                    logger.warning("Fire GPU model not found")
        else:
            if dtype not in self._cpu_models:
                path = _resolve_model_path(dtype, use_npu=False)
                if path and ULTRALYTICS_AVAILABLE:
                    self._cpu_models[dtype] = YOLO(path)
                    logger.info(f"Fire CPU model loaded from {path}")
                else:
                    logger.warning("Fire CPU model not found")

    def _load_uniform_model(self, device: str = None) -> None:
        """加载工服检测模型（YOLOv8 自训练模型）"""
        if device is None:
            device = self.device
        dtype = "uniform"
        if device == "npu" and self._npu_cores > 0:
            if dtype not in self._npu_models:
                path = _resolve_model_path(dtype, use_npu=True)
                if path and RKNN_AVAILABLE:
                    self._npu_models[dtype] = {}
                    for core_id in range(self._npu_cores):
                        rknn = RKNNLite(verbose=False)
                        ret = rknn.load_rknn(path)
                        if ret != 0:
                            logger.error(f"Failed to load uniform RKNN for core {core_id}")
                            continue
                        ret = rknn.init_runtime(core_mask=self.CORE_MASKS[core_id])
                        if ret != 0:
                            logger.error(f"Failed to init uniform RKNN runtime for core {core_id}")
                            continue
                        self._npu_models[dtype][core_id] = rknn
                        logger.info(f"Uniform RKNN loaded on core {core_id}")
                else:
                    logger.warning("Uniform RKNN model not found, falling back to CPU/GPU")
        elif device == "gpu":
            if dtype not in self._cpu_models:
                path = _resolve_model_path(dtype, use_npu=False)
                if path and ULTRALYTICS_AVAILABLE:
                    model = YOLO(path)
                    try:
                        model = model.to("cuda")
                        logger.info(f"Uniform GPU model loaded from {path}")
                    except Exception as e:
                        logger.warning(f"Failed to move uniform model to GPU, using CPU: {e}")
                    self._cpu_models[dtype] = model
                else:
                    logger.warning("Uniform GPU model not found")
        else:
            if dtype not in self._cpu_models:
                path = _resolve_model_path(dtype, use_npu=False)
                if path and ULTRALYTICS_AVAILABLE:
                    self._cpu_models[dtype] = YOLO(path)
                    logger.info(f"Uniform CPU model loaded from {path}")
                else:
                    logger.warning("Uniform CPU model not found")

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

    def _load_mask_model(self, device: str = None) -> None:
        """加载口罩检测模型（YOLOv8 自训练模型）"""
        if device is None:
            device = self.device
        dtype = "mask"
        if device == "npu" and self._npu_cores > 0:
            if dtype not in self._npu_models:
                path = _resolve_model_path(dtype, use_npu=True)
                if path and RKNN_AVAILABLE:
                    self._npu_models[dtype] = {}
                    for core_id in range(self._npu_cores):
                        rknn = RKNNLite(verbose=False)
                        ret = rknn.load_rknn(path)
                        if ret != 0:
                            logger.error(f"Failed to load mask RKNN for core {core_id}")
                            continue
                        ret = rknn.init_runtime(core_mask=self.CORE_MASKS[core_id])
                        if ret != 0:
                            logger.error(f"Failed to init mask RKNN runtime for core {core_id}")
                            continue
                        self._npu_models[dtype][core_id] = rknn
                        logger.info(f"Mask RKNN loaded on core {core_id}")
                else:
                    logger.warning("Mask RKNN model not found, falling back to CPU/GPU")
        elif device == "gpu":
            if dtype not in self._cpu_models:
                path = _resolve_model_path(dtype, use_npu=False)
                if path and ULTRALYTICS_AVAILABLE:
                    model = YOLO(path)
                    try:
                        model = model.to("cuda")
                        logger.info(f"Mask GPU model loaded from {path}")
                    except Exception as e:
                        logger.warning(f"Failed to move mask model to GPU, using CPU: {e}")
                    self._cpu_models[dtype] = model
                else:
                    logger.warning("Mask GPU model not found")
        else:
            if dtype not in self._cpu_models:
                path = _resolve_model_path(dtype, use_npu=False)
                if path and ULTRALYTICS_AVAILABLE:
                    self._cpu_models[dtype] = YOLO(path)
                    logger.info(f"Mask CPU model loaded from {path}")
                else:
                    logger.warning("Mask CPU model not found")

    def _load_cigarette_model(self, device: str = None) -> None:
        """加载吸烟检测模型（YOLOv8 自训练模型）"""
        if device is None:
            device = self.device
        dtype = "cigarette"
        if device == "npu" and self._npu_cores > 0:
            if dtype not in self._npu_models:
                path = _resolve_model_path(dtype, use_npu=True)
                if path and RKNN_AVAILABLE:
                    self._npu_models[dtype] = {}
                    for core_id in range(self._npu_cores):
                        rknn = RKNNLite(verbose=False)
                        ret = rknn.load_rknn(path)
                        if ret != 0:
                            logger.error(f"Failed to load cigarette RKNN for core {core_id}")
                            continue
                        ret = rknn.init_runtime(core_mask=self.CORE_MASKS[core_id])
                        if ret != 0:
                            logger.error(f"Failed to init cigarette RKNN runtime for core {core_id}")
                            continue
                        self._npu_models[dtype][core_id] = rknn
                        logger.info(f"Cigarette RKNN loaded on core {core_id}")
                else:
                    logger.warning("Cigarette RKNN model not found, falling back to CPU/GPU")
        elif device == "gpu":
            if dtype not in self._cpu_models:
                path = _resolve_model_path(dtype, use_npu=False)
                if path and ULTRALYTICS_AVAILABLE:
                    model = YOLO(path)
                    try:
                        model = model.to("cuda")
                        logger.info(f"Cigarette GPU model loaded from {path}")
                    except Exception as e:
                        logger.warning(f"Failed to move cigarette model to GPU, using CPU: {e}")
                    self._cpu_models[dtype] = model
                else:
                    logger.warning("Cigarette GPU model not found")
        else:
            if dtype not in self._cpu_models:
                path = _resolve_model_path(dtype, use_npu=False)
                if path and ULTRALYTICS_AVAILABLE:
                    self._cpu_models[dtype] = YOLO(path)
                    logger.info(f"Cigarette CPU model loaded from {path}")
                else:
                    logger.warning("Cigarette CPU model not found")

    def _load_sleep_model(self, device: str = None) -> None:
        """加载睡岗检测姿态模型（yolov8-pose，自动下载）"""
        if device is None:
            device = self.device
        dtype = "sleep"
        if device == "npu" and self._npu_cores > 0:
            logger.warning("Sleep pose model NPU not supported, using CPU/GPU")
        if dtype not in self._cpu_models:
            path = _resolve_model_path(dtype, use_npu=False)
            if path and ULTRALYTICS_AVAILABLE:
                model = YOLO(path)
                if device == "gpu":
                    try:
                        model = model.to("cuda")
                        logger.info(f"Sleep pose GPU model loaded from {path}")
                    except Exception as e:
                        logger.warning(f"Failed to move sleep model to GPU, using CPU: {e}")
                else:
                    logger.info(f"Sleep pose CPU model loaded from {path}")
                self._cpu_models[dtype] = model
            else:
                logger.warning("Sleep pose model not found, will try auto-download if available")

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray, detection_types: List[str], core_id: int = 0) -> Dict[str, dict]:
        """
        对单帧执行多类型检测

        Returns:
            {
                "fire": {"detected": bool, "boxes": List[List[int]], "scores": List[float]},
                "smoke": {"detected": bool, "boxes": List[List[int]], "scores": List[float]},
                ...
            }
        """
        results: Dict[str, dict] = {}
        for dtype in detection_types:
            if dtype in ("fire", "smoke"):
                fire_res, smoke_res = self._detect_fire_smoke(frame, core_id)
                results["fire"] = fire_res
                results["smoke"] = smoke_res
            elif dtype == "uniform":
                results[dtype] = self._detect_uniform(frame, core_id)
            elif dtype == "mask":
                results[dtype] = self._detect_mask(frame, core_id)
            elif dtype == "cigarette":
                results[dtype] = self._detect_cigarette(frame, core_id)
            elif dtype == "sleep":
                results[dtype] = self._detect_sleep(frame)
        return results

    def _detect_fire_smoke(self, frame: np.ndarray, core_id: int) -> Tuple[dict, dict]:
        """
        fire/smoke 检测：共用模型，按类别分离结果
        fire=cls0, smoke=cls1（约定）
        """
        fire_result = {"detected": False, "boxes": [], "scores": []}
        smoke_result = {"detected": False, "boxes": [], "scores": []}

        model = None
        use_npu = False
        with self._model_lock:
            if "fire" in self._npu_models and core_id in self._npu_models["fire"]:
                model = self._npu_models["fire"][core_id]
                use_npu = True
            elif "fire" in self._cpu_models:
                model = self._cpu_models["fire"]

        if model is None:
            logger.warning("Fire/smoke model not loaded")
            return fire_result, smoke_result

        try:
            if use_npu:
                # NPU 推理
                input_frame = self._preprocess(frame)
                outputs = model.inference(inputs=[input_frame])
                boxes = self._postprocess_rknn(outputs, frame.shape[:2])
            else:
                # CPU YOLO 推理
                pred = model.predict(frame, verbose=False)
                boxes = []
                if pred and pred[0].boxes is not None:
                    for b in pred[0].boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        cls = int(b.cls[0])
                        score = float(b.conf[0])
                        boxes.append({"xyxy": [x1, y1, x2, y2], "class_id": cls, "confidence": score})

            for box in boxes:
                if box.get("class_id") == 0:
                    fire_result["boxes"].append(box["xyxy"])
                    fire_result["scores"].append(box["confidence"])
                elif box.get("class_id") == 1:
                    smoke_result["boxes"].append(box["xyxy"])
                    smoke_result["scores"].append(box["confidence"])

            fire_result["detected"] = len(fire_result["boxes"]) > 0
            fire_result["max_confidence"] = max(fire_result["scores"]) if fire_result["scores"] else 0.0
            smoke_result["detected"] = len(smoke_result["boxes"]) > 0
            smoke_result["max_confidence"] = max(smoke_result["scores"]) if smoke_result["scores"] else 0.0

        except Exception as e:
            logger.error(f"Fire/smoke detection error: {e}")

        return fire_result, smoke_result

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

    def _detect_uniform(self, frame: np.ndarray, core_id: int = 0) -> dict:
        """工服检测：检测未穿工服/反光背心的人员"""
        result = {"detected": False, "boxes": [], "scores": [], "missing_vest": False}
        model = None
        use_npu = False
        with self._model_lock:
            if "uniform" in self._npu_models and core_id in self._npu_models["uniform"]:
                model = self._npu_models["uniform"][core_id]
                use_npu = True
            elif "uniform" in self._cpu_models:
                model = self._cpu_models["uniform"]

        if model is None:
            logger.warning("Uniform model not loaded")
            return result

        try:
            if use_npu:
                input_frame = self._preprocess(frame)
                outputs = model.inference(inputs=[input_frame])
                boxes = self._postprocess_rknn(outputs, frame.shape[:2])
                for box in boxes:
                    if box.get("class_id") not in UNIFORM_TARGET_CLASSES:
                        continue
                    result["boxes"].append(box["xyxy"])
                    result["scores"].append(box["confidence"])
            else:
                pred = model.predict(frame, classes=UNIFORM_TARGET_CLASSES, verbose=False)
                if pred and pred[0].boxes is not None:
                    for b in pred[0].boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        score = float(b.conf[0])
                        result["boxes"].append([x1, y1, x2, y2])
                        result["scores"].append(score)
            result["detected"] = len(result["boxes"]) > 0
            result["missing_vest"] = result["detected"]
            result["max_confidence"] = max(result["scores"]) if result["scores"] else 0.0
        except Exception as e:
            logger.error(f"Uniform detection error: {e}")
        return result

    def _detect_mask(self, frame: np.ndarray, core_id: int = 0) -> dict:
        """口罩检测（YOLOv8 自训练模型）"""
        result = {"detected": False, "boxes": [], "scores": []}
        model = None
        use_npu = False
        with self._model_lock:
            if "mask" in self._npu_models and core_id in self._npu_models["mask"]:
                model = self._npu_models["mask"][core_id]
                use_npu = True
            elif "mask" in self._cpu_models:
                model = self._cpu_models["mask"]
        if model is None:
            return result
        try:
            if use_npu:
                input_frame = self._preprocess(frame)
                outputs = model.inference(inputs=[input_frame])
                boxes = self._postprocess_rknn(outputs, frame.shape[:2])
                for box in boxes:
                    if box.get("class_id") not in MASK_TARGET_CLASSES:
                        continue
                    result["boxes"].append(box["xyxy"])
                    result["scores"].append(box["confidence"])
            else:
                pred = model.predict(frame, verbose=False)
                if pred and pred[0].boxes is not None:
                    for b in pred[0].boxes:
                        cls = int(b.cls[0])
                        if cls not in MASK_TARGET_CLASSES:
                            continue
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        score = float(b.conf[0])
                        result["boxes"].append([x1, y1, x2, y2])
                        result["scores"].append(score)
            result["detected"] = len(result["boxes"]) > 0
            result["max_confidence"] = max(result["scores"]) if result["scores"] else 0.0
        except Exception as e:
            logger.error(f"Mask detection error: {e}")
        return result

    def _detect_cigarette(self, frame: np.ndarray, core_id: int = 0) -> dict:
        """吸烟检测（YOLOv8 自训练模型）"""
        result = {"detected": False, "boxes": [], "scores": []}
        model = None
        use_npu = False
        with self._model_lock:
            if "cigarette" in self._npu_models and core_id in self._npu_models["cigarette"]:
                model = self._npu_models["cigarette"][core_id]
                use_npu = True
            elif "cigarette" in self._cpu_models:
                model = self._cpu_models["cigarette"]
        if model is None:
            return result
        try:
            if use_npu:
                input_frame = self._preprocess(frame)
                outputs = model.inference(inputs=[input_frame])
                boxes = self._postprocess_rknn(outputs, frame.shape[:2])
                for box in boxes:
                    if box.get("class_id") not in CIGARETTE_TARGET_CLASSES:
                        continue
                    result["boxes"].append(box["xyxy"])
                    result["scores"].append(box["confidence"])
            else:
                pred = model.predict(frame, verbose=False)
                if pred and pred[0].boxes is not None:
                    for b in pred[0].boxes:
                        cls = int(b.cls[0])
                        if cls not in CIGARETTE_TARGET_CLASSES:
                            continue
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        score = float(b.conf[0])
                        result["boxes"].append([x1, y1, x2, y2])
                        result["scores"].append(score)
            result["detected"] = len(result["boxes"]) > 0
            result["max_confidence"] = max(result["scores"]) if result["scores"] else 0.0
        except Exception as e:
            logger.error(f"Cigarette detection error: {e}")
        return result

    def _detect_sleep(self, frame: np.ndarray) -> dict:
        """睡岗检测（基于 YOLOv8-pose + sleep_detect 分析）"""
        result = {"detected": False, "boxes": [], "scores": [], "subjects": [], "count": 0}
        model = self._cpu_models.get("sleep")
        if model is None:
            return result
        try:
            from safety_detection.sleep_detect import process_frame
            subjects = process_frame(model, frame, conf=0.1)
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
            logger.error(f"Sleep detection error: {e}")
        return result

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
            for dtype, core_map in self._npu_models.items():
                for core_id, rknn in core_map.items():
                    try:
                        rknn.release()
                        logger.info(f"Released {dtype} RKNN on core {core_id}")
                    except Exception as e:
                        logger.error(f"Error releasing {dtype} RKNN core {core_id}: {e}")
            self._npu_models.clear()
            self._cpu_models.clear()

    @property
    def loaded_models(self) -> List[str]:
        """返回已加载的模型列表（不含内部 person 模型）"""
        models = []
        with self._model_lock:
            models.extend([k for k in self._cpu_models.keys() if k != "person"])
            models.extend(self._npu_models.keys())
        return list(dict.fromkeys(models))  # 去重

    def get_model_status(self) -> List[Dict[str, Any]]:
        """返回模型状态详情（用于 API /detector/models）"""
        status = []
        with self._model_lock:
            for dtype, model in self._cpu_models.items():
                if dtype == "person":
                    continue
                status.append({
                    "type": dtype,
                    "backend": "cpu",
                    "device": "pytorch",
                    "loaded": True,
                })
            for dtype, core_map in self._npu_models.items():
                status.append({
                    "type": dtype,
                    "backend": "npu",
                    "device": "rk3588",
                    "cores": len(core_map),
                    "loaded": True,
                })
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
