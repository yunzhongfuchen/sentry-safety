"""
GPU 动态多模型调度器。

支持：
- 多路摄像头，每路独立启用不同的检测类型和间隔
- 按检测类型分组 batch 推理
- 队列数可配置，默认每个模型一个队列（纯并行）
- 调度周期 0.5 秒
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """检测类型对应的模型配置"""
    model_path: str
    detection_type: str
    device: str = "cuda"
    confidence: float = 0.5
    classes: Optional[List[int]] = None


class ModelDetector:
    """轻量模型包装，支持自定义 classes"""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)
        self.device = cfg.device
        self.confidence = cfg.confidence
        self.classes = cfg.classes if cfg.classes is not None else [0]

    def predict(self, frames: List[np.ndarray], half: bool = False):
        """batch 推理，返回 ultralytics Results 列表"""
        return self.model(
            frames,
            conf=self.confidence,
            classes=self.classes,
            device=self.device,
            verbose=False,
            half=half,
        )


class QueueWorker(threading.Thread):
    """推理队列：内部串行执行多个模型，与其他队列并行"""

    def __init__(self, queue_id: int, detectors: List[ModelDetector], half: bool = False):
        super().__init__(daemon=True)
        self.queue_id = queue_id
        self.detectors = detectors
        self.half = half
        self.frames: Optional[List[np.ndarray]] = None
        self.cam_ids: Optional[List[str]] = None
        self.results: Optional[List] = None
        self.start_event = threading.Event()
        self.done_event = threading.Event()
        self.running = True
        self.model_times = [0.0 for _ in detectors]
        self.infer_count = 0
        self.total_infer_time = 0.0
        self._lock = threading.Lock()

    def run(self):
        while self.running:
            self.start_event.wait()
            self.start_event.clear()
            if not self.running:
                break

            queue_t0 = time.time()
            batch_results = []
            for idx, detector in enumerate(self.detectors):
                t0 = time.time()
                try:
                    res = detector.predict(self.frames, half=self.half)
                except Exception as e:
                    logger.error(f"[队列{self.queue_id}-模型{idx}] 推理出错: {e}")
                    res = None
                elapsed = time.time() - t0
                with self._lock:
                    self.model_times[idx] += elapsed
                batch_results.append(res)

            with self._lock:
                self.infer_count += 1
                self.total_infer_time += time.time() - queue_t0

            self.results = batch_results
            self.done_event.set()

    def set_frames(self, frames: List[np.ndarray], cam_ids: List[str]):
        self.frames = frames
        self.cam_ids = cam_ids
        self.start_event.set()

    def stop(self):
        self.running = False
        self.start_event.set()
        self.join(timeout=5.0)

    def reset_stats(self):
        with self._lock:
            self.infer_count = 0
            self.total_infer_time = 0.0
            self.model_times = [0.0 for _ in self.detectors]

    def get_model_avg_ms(self) -> List[float]:
        with self._lock:
            return [
                (t / self.infer_count * 1000) if self.infer_count else 0.0
                for t in self.model_times
            ]

    @property
    def avg_queue_ms(self) -> float:
        with self._lock:
            return (self.total_infer_time / self.infer_count * 1000) if self.infer_count else 0.0


class GPUDynamicScheduler(threading.Thread):
    """GPU 动态调度器主线程"""

    def __init__(
        self,
        camera_manager,
        model_configs: Dict[str, ModelConfig],
        num_queues: Optional[int] = None,
        interval: float = 0.5,
        on_result: Optional[Callable[[str, str, Any], None]] = None,
        half: bool = False,
        warmup: bool = True,
    ):
        super().__init__(daemon=True)
        self.camera_manager = camera_manager
        self.model_configs = model_configs
        self.interval = interval
        self.on_result = on_result
        self.half = half
        self.warmup = warmup
        self.running = True
        # (camera_id, detection_type) -> last_infer_timestamp
        self.last_infer: Dict[Tuple[str, str], float] = {}
        self._busy = False
        self.MAX_FRAME_AGE = 0.5  # 帧最大年龄 0.5 秒

        # 加载所有模型
        self.detectors: Dict[str, ModelDetector] = {}
        for dtype, cfg in model_configs.items():
            logger.info(f"加载模型 {dtype}: {cfg.model_path} ...")
            d = ModelDetector(cfg)
            self.detectors[dtype] = d
            logger.info(f"  -> 完成，设备: {cfg.device}")

        # 创建队列
        num_models = len(self.detectors)
        if num_queues is None or num_queues > num_models:
            num_queues = num_models
        self.num_queues = num_queues

        dtype_list = list(self.detectors.keys())
        queue_size = (num_models + num_queues - 1) // num_queues

        self.dtype_to_queue: Dict[str, int] = {}
        self.dtype_to_idx: Dict[str, int] = {}  # dtype 在队列内的 detector 索引
        self.queues: Dict[int, QueueWorker] = {}

        for q in range(num_queues):
            start = q * queue_size
            end = min(start + queue_size, num_models)
            queue_dtypes = dtype_list[start:end]
            queue_detectors = [self.detectors[dtype] for dtype in queue_dtypes]

            w = QueueWorker(q, queue_detectors, half=half)
            w.start()
            self.queues[q] = w
            for idx, dtype in enumerate(queue_dtypes):
                self.dtype_to_queue[dtype] = q
                self.dtype_to_idx[dtype] = idx
            logger.info(f"队列{q}: {queue_dtypes}")

        # dummy 预热
        if self.warmup:
            self._warmup()

    def _warmup(self):
        """用黑图预热所有队列，消除 CUDA 首次分配开销"""
        logger.info("正在 dummy 预热 ...")
        dummy = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(2)]
        for qid, w in self.queues.items():
            w.set_frames(dummy, ["dummy"] * len(dummy))
        for qid, w in self.queues.items():
            w.done_event.wait()
            w.done_event.clear()
            w.reset_stats()
        logger.info("预热完成")

    def _get_active_cameras(self) -> List[str]:
        """获取当前活跃摄像头 ID 列表"""
        # camera_manager._cameras 是内部 dict
        if hasattr(self.camera_manager, "_cameras"):
            return list(self.camera_manager._cameras.keys())
        return []

    def _get_camera_detection_types(self, cam_id: str) -> Dict[str, dict]:
        """获取摄像头的检测类型配置"""
        if not hasattr(self.camera_manager, "_cameras"):
            return {}
        state = self.camera_manager._cameras.get(cam_id)
        if not state or not state.config:
            return {}
        cfg = state.config
        if hasattr(cfg, "detection_types"):
            dt = cfg.detection_types
            if isinstance(dt, dict):
                return dt
        return {}

    def _is_camera_enabled(self, cam_id: str) -> bool:
        if not hasattr(self.camera_manager, "_cameras"):
            return False
        state = self.camera_manager._cameras.get(cam_id)
        if not state or not state.config:
            return False
        cfg = state.config
        return getattr(cfg, "enabled", True) and getattr(cfg, "detection_enabled", True)

    def _collect_due_frames(self, now: float) -> Dict[str, List[Tuple[str, np.ndarray]]]:
        """收集到期任务，过滤过旧帧"""
        tasks: Dict[str, List[Tuple[str, np.ndarray]]] = {}

        for cam_id in self._get_active_cameras():
            if not self._is_camera_enabled(cam_id):
                continue

            # 取帧时间作为帧年龄判断依据
            frame_capture_time = time.time()
            frame = self.camera_manager.request_frame(cam_id, timeout=1.0, store_history=True)
            if frame is None:
                continue

            frame_age = time.time() - frame_capture_time
            if frame_age > self.MAX_FRAME_AGE:
                logger.debug(f"Drop old frame from {cam_id}, age={frame_age:.2f}s")
                continue

            det_types = self._get_camera_detection_types(cam_id)
            for dtype, cfg in det_types.items():
                if dtype not in self.model_configs:
                    continue

                enabled = cfg.get("enabled", False) if isinstance(cfg, dict) else getattr(cfg, "enabled", False)
                if not enabled:
                    continue

                interval = (
                    cfg.get("interval", 1.0)
                    if isinstance(cfg, dict)
                    else getattr(cfg, "interval", 1.0)
                )
                key = (cam_id, dtype)
                last = self.last_infer.get(key, 0.0)
                if now - last >= interval:
                    tasks.setdefault(dtype, []).append((cam_id, frame.copy()))

        return tasks

    def run(self):
        logger.info(
            f"GPU 调度器启动: models={len(self.detectors)}, queues={self.num_queues}, "
            f"interval={self.interval}s"
        )
        while self.running:
            if self._busy:
                # 上一轮还没完成，丢弃本轮
                time.sleep(0.05)
                continue

            t0 = time.time()
            self._busy = True
            try:
                now = time.time()
                tasks = self._collect_due_frames(now)

                if tasks:
                    active_queues: set = set()
                    collected_keys = []
                    for dtype, cam_frames in tasks.items():
                        cam_ids, frames = zip(*cam_frames)
                        qid = self.dtype_to_queue[dtype]
                        self.queues[qid].set_frames(list(frames), list(cam_ids))
                        active_queues.add(qid)
                        for cid in cam_ids:
                            collected_keys.append((cid, dtype))

                    for qid in active_queues:
                        self.queues[qid].done_event.wait()
                        self.queues[qid].done_event.clear()

                    # 回调结果
                    if self.on_result:
                        for dtype, cam_frames in tasks.items():
                            cam_ids, _ = zip(*cam_frames)
                            qid = self.dtype_to_queue[dtype]
                            idx = self.dtype_to_idx[dtype]
                            results = self.queues[qid].results
                            if results is None or idx >= len(results):
                                continue
                            model_results = results[idx]
                            if model_results is None:
                                continue
                            for cam_id, result in zip(cam_ids, model_results):
                                try:
                                    self.on_result(cam_id, dtype, result)
                                except Exception as e:
                                    logger.error(f"回调出错 [{cam_id}/{dtype}]: {e}")

                    # 推理完成后更新 last_infer
                    completed_at = time.time()
                    for key in collected_keys:
                        self.last_infer[key] = completed_at
            finally:
                self._busy = False

            sleep_time = self.interval - (time.time() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        logger.info("GPU 调度器停止中...")
        self.running = False
        for w in self.queues.values():
            w.stop()
        self.join(timeout=5.0)
        logger.info("GPU 调度器已停止")
