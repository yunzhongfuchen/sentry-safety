"""
多类型安全检测调度器
支持策略模式调度（CorePinnedStrategy / SerialStrategy）
管理每摄像头每类型的独立调度、连续计数、冷却、VLM 提交
"""

import logging
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend import config
from backend.detection_registry import registry
from backend.frame_utils import encode_frame_to_jpg
from inference_engine import SafetyDetector, detect_npu_cores

logger = logging.getLogger(__name__)

MAX_VLM_REVIEW_FRAMES = 5

# CJK 字体候选路径（Windows 开发机 / Linux 部署环境）
_CJK_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]
_cjk_font_cache: Dict[int, Optional[ImageFont.FreeTypeFont]] = {}


def _get_cjk_font(size: int) -> Optional[ImageFont.FreeTypeFont]:
    """加载 CJK 字体（按尺寸缓存）。找不到时返回 None，调用方回退英文标签。"""
    if size in _cjk_font_cache:
        return _cjk_font_cache[size]
    font = None
    for path in _CJK_FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
    _cjk_font_cache[size] = font
    return font


def _draw_labels_pil(frame_bgr: np.ndarray, text_items: list, font) -> np.ndarray:
    """用 PIL 在帧上绘制中文标签（cv2.putText 不支持中文）。
    text_items: [(zh_text, x1, y1, color_bgr), ...]
    """
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for text, x1, y1, bgr in text_items:
        color_rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        pos = (x1 + 3, max(0, y1 - 20))
        tb = draw.textbbox(pos, text, font=font)
        draw.rectangle([tb[0] - 3, tb[1] - 2, tb[2] + 3, tb[3] + 2], fill=color_rgb)
        draw.text(pos, text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


@dataclass
class TypeSchedule:
    """单类型检测调度状态"""
    dtype: str
    enabled: bool
    interval: float
    threshold: float
    cooldown: float
    verification_frame_count: int = 1
    verification_frame_interval: float = 1.0
    consecutive_required: int = 1
    consecutive_count: int = 0
    last_run: float = 0.0
    use_vlm: bool = False
    # 当存在外部调度器（如 GPU scheduler）接管该类型推理时设为 True
    externally_managed: bool = False
    roi: list = None
    roi_invert: bool = False
    # 静态目标过滤：同一检测轮内的采样帧区域几乎无变化时判为误判
    static_filter: bool = False
    static_diff_threshold: float = 0.02
    # 当前检测轮的采样状态；成功完成一轮后才增加 consecutive_count
    sampling_active: bool = False
    sampled_frame_count: int = 0
    last_sample_time: float = 0.0
    last_sample_seq: int = -1
    task_regions: List[np.ndarray] = None

    def is_due(self, now: float) -> bool:
        if self.sampling_active:
            return now - self.last_sample_time >= self.verification_frame_interval
        return now - self.last_run >= self.interval


# ----------------------------------------------------------------------
# ROI 过滤
# ----------------------------------------------------------------------

def filter_by_roi(result: dict, roi: list, roi_invert: bool,
                  frame_width: int, frame_height: int) -> dict:
    """按 ROI 多边形过滤检测框，支持单个多边形或 polygon 列表，保持 subjects 与 boxes 索引一致"""
    if not roi:
        return result

    # 兼容旧格式：单个多边形 [[x, y], ...]；新格式为 polygon 列表
    if roi and len(roi) > 0 and len(roi[0]) > 0 and isinstance(roi[0][0], (int, float)):
        rois = [roi]
    else:
        rois = roi

    filtered_boxes, filtered_scores = [], []
    filtered_subjects = []
    subjects = result.get("subjects", [])

    for i, (box, score) in enumerate(zip(result.get("boxes", []), result.get("scores", []))):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        inside_any = False
        for polygon_pts in rois:
            polygon = np.array([
                [int(x * frame_width), int(y * frame_height)]
                for x, y in polygon_pts
            ], dtype=np.int32)
            if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                inside_any = True
                break
        keep = inside_any if not roi_invert else not inside_any
        if keep:
            filtered_boxes.append(box)
            filtered_scores.append(score)
            if i < len(subjects):
                filtered_subjects.append(subjects[i])

    out = {
        **result,
        "boxes": filtered_boxes,
        "scores": filtered_scores,
        "detected": len(filtered_boxes) > 0,
        "max_confidence": max(filtered_scores) if filtered_scores else 0.0,
    }
    if subjects:
        out["subjects"] = filtered_subjects
    return out


# ----------------------------------------------------------------------
# 静态目标过滤（关键帧图片比对）
# ----------------------------------------------------------------------

def _extract_box_region(frame: np.ndarray, box: list, margin_ratio: float = 0.1) -> Optional[np.ndarray]:
    """从帧中提取检测框区域（扩 margin_ratio 边界），缩放为 64x64 灰度图"""
    if frame is None or len(box) < 4:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box[:4])
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    mx, my = int(bw * margin_ratio), int(bh * margin_ratio)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return None
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 64))


def check_static_filter(regions: List[np.ndarray], diff_threshold: float) -> bool:
    """连续帧框区域是否全部几乎无变化（静态目标，视为误判）。

    相邻帧做绝对差分，统计差值 > 15 的像素占比；全部低于 diff_threshold 时返回 True。
    """
    if len(regions) < 2:
        return False
    for i in range(1, len(regions)):
        prev, curr = regions[i - 1], regions[i]
        if prev is None or curr is None:
            continue
        diff = cv2.absdiff(prev, curr)
        changed_ratio = float((diff > 15).sum()) / diff.size
        if changed_ratio >= diff_threshold:
            return False
    return True


# ----------------------------------------------------------------------
# 策略模式
# ----------------------------------------------------------------------

class DetectionStrategy(ABC):
    @abstractmethod
    def run(self, detector: "MultiDetector") -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...


class CorePinnedStrategy(DetectionStrategy):
    """NPU 核心绑定策略：每个 NPU 核心一个线程"""

    def __init__(self):
        self._running = False
        self._threads: List[threading.Thread] = []
        self._npu_cores = detect_npu_cores()

    def run(self, detector: "MultiDetector") -> None:
        self._running = True
        camera_ids = list(detector.camera_manager.get_camera_ids())
        if not camera_ids:
            logger.warning("No cameras registered, strategy idle")
            return
        groups = self._group_cameras(camera_ids, self._npu_cores)
        for core_id, group in enumerate(groups):
            if not group:
                continue
            t = threading.Thread(
                target=self._worker_loop,
                args=(detector, group, core_id),
                daemon=True,
                name=f"detector-core-{core_id}",
            )
            t.start()
            self._threads.append(t)
        logger.info(f"CorePinnedStrategy started: {len(camera_ids)} cameras across {self._npu_cores} cores")

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=2)
        self._threads.clear()
        logger.info("CorePinnedStrategy stopped")

    def _group_cameras(self, camera_ids: List[str], n_cores: int) -> List[List[str]]:
        if n_cores <= 0 or not camera_ids:
            return [camera_ids]
        groups: List[List[str]] = [[] for _ in range(n_cores)]
        for i, cam_id in enumerate(camera_ids):
            groups[i % n_cores].append(cam_id)
        return groups

    def _worker_loop(self, detector: "MultiDetector", camera_ids: List[str], core_id: int) -> None:
        logger.info(f"CorePinnedStrategy worker started for core {core_id} with cameras {camera_ids}")
        while self._running:
            has_work = False
            for camera_id in camera_ids:
                if not self._running:
                    break
                due_types = detector._get_due_types(camera_id, time.time())
                if due_types:
                    has_work = True
                    try:
                        detector._process_camera(camera_id, core_id)
                    except Exception as e:
                        logger.error(f"Worker error on {camera_id} core {core_id}: {e}")
                else:
                    # 该摄像头没有到期的检测，短暂休息避免空转
                    time.sleep(0.02)
            if not has_work:
                # 本轮没有任何工作，多休息一会儿
                time.sleep(0.1)
            else:
                # 有工作完成，至少休息 20ms 避免占满 CPU
                time.sleep(0.02)


class SerialStrategy(DetectionStrategy):
    """串行策略：单线程轮询所有摄像头（CPU-only 设备 fallback）"""

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def run(self, detector: "MultiDetector") -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            args=(detector,),
            daemon=True,
            name="detector-serial",
        )
        self._thread.start()
        logger.info("SerialStrategy started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("SerialStrategy stopped")

    def _worker_loop(self, detector: "MultiDetector") -> None:
        logger.info("SerialStrategy worker started")
        while self._running:
            cam_ids = detector.camera_manager.get_camera_ids()
            if not cam_ids:
                logger.warning("SerialStrategy: no cameras, waiting...")
                time.sleep(1.0)
                continue
            has_work = False
            for camera_id in cam_ids:
                if not self._running:
                    break
                due_types = detector._get_due_types(camera_id, time.time())
                if due_types:
                    has_work = True
                    try:
                        detector._process_camera(camera_id, core_id=0)
                    except Exception as e:
                        logger.error(f"Serial worker error on {camera_id}: {e}")
                else:
                    time.sleep(0.02)
            if not has_work:
                time.sleep(0.1)
            else:
                time.sleep(0.02)


def select_detection_strategy(strategy_setting: str, npu_cores: int) -> DetectionStrategy:
    """根据全局设置与核心数选择调度策略。

    auto: 核心数 >= 2 用 CorePinnedStrategy，否则 SerialStrategy；
    parallel / serial: 强制指定策略（1G 显存等受限设备可强制串行）。
    """
    setting = (strategy_setting or "auto").strip().lower()
    if setting == "serial":
        return SerialStrategy()
    if setting == "parallel":
        return CorePinnedStrategy()
    return CorePinnedStrategy() if npu_cores >= 2 else SerialStrategy()


# ----------------------------------------------------------------------
# MultiDetector
# ----------------------------------------------------------------------

class MultiDetector:
    """
    多类型检测调度器
    - 按摄像头注册检测类型配置
    - 策略模式执行调度
    - 管理连续计数、冷却、VLM 提交、睡岗状态机
    """

    def __init__(
        self,
        camera_manager,
        safety_detector: SafetyDetector,
        vlm_queue,
        strategy: DetectionStrategy,
        trigger_callback: Optional[Callable[[str, str, np.ndarray, dict], None]] = None,
        frame_callback: Optional[Callable[[str, np.ndarray], None]] = None,
        vlm_result_callback: Optional[Callable[[str, str, dict], None]] = None,
    ):
        self.camera_manager = camera_manager
        self.safety_detector = safety_detector
        self.vlm_queue = vlm_queue
        self.strategy = strategy
        self.trigger_callback = trigger_callback
        self.frame_callback = frame_callback
        self.vlm_result_callback = vlm_result_callback

        self._schedules: Dict[str, Dict[str, TypeSchedule]] = {}
        self._alert_states: Dict[str, Dict[str, dict]] = {}
        self._cooldowns: Dict[str, Dict[str, float]] = {}
        self._latest_results: Dict[str, Dict[str, dict]] = {}
        # 静态过滤：每摄像头每类型缓存连续帧的框区域图（64x64 灰度）
        self._static_regions: Dict[str, Dict[str, List[np.ndarray]]] = {}
        self._lock = threading.RLock()
        self._running = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.strategy.run(self)
        logger.info("MultiDetector started")

    def stop(self) -> None:
        self._running = False
        self.strategy.stop()
        logger.info("MultiDetector stopped")

    # ------------------------------------------------------------------
    # 摄像头注册
    # ------------------------------------------------------------------

    @staticmethod
    def _build_schedule(dtype: str, cfg: dict) -> TypeSchedule:
        """以注册表 defaults 为基底，叠加摄像头级配置（enabled/roi/roi_invert）构建调度"""
        merged = registry.merge_camera_config(dtype, cfg or {})
        return TypeSchedule(
            dtype=dtype,
            enabled=True,
            interval=merged.get("interval", 1.0),
            threshold=merged.get("threshold", 0.5),
            cooldown=merged.get("cooldown", 60.0),
            verification_frame_count=max(1, int(merged.get("verification_frame_count", 1))),
            verification_frame_interval=max(0.0, float(merged.get("verification_frame_interval", 1.0))),
            consecutive_required=max(1, int(merged.get("consecutive_required", 3))),
            use_vlm=merged.get("use_vlm", False),
            roi=merged.get("roi"),
            roi_invert=merged.get("roi_invert", False),
            static_filter=merged.get("static_filter", False),
            static_diff_threshold=merged.get("static_diff_threshold", 0.02),
        )

    def register_camera(self, camera_id: str, detection_types: Dict[str, dict]) -> None:
        """注册摄像头检测配置"""
        with self._lock:
            self._schedules[camera_id] = {}
            self._alert_states[camera_id] = {}
            self._cooldowns[camera_id] = {}
            for dtype, cfg in detection_types.items():
                if not cfg.get("enabled", False):
                    continue
                self._schedules[camera_id][dtype] = self._build_schedule(dtype, cfg)
            if self.camera_manager is not None:
                self.camera_manager.clear_all_detection_frames(camera_id)
            logger.info(f"Camera {camera_id} registered with {len(self._schedules[camera_id])} types")

    def mark_externally_managed(self, camera_id: str, dtypes: List[str]) -> None:
        """标记指定检测类型由外部调度器接管，本 Detector 不再对其运行推理"""
        with self._lock:
            schedules = self._schedules.get(camera_id, {})
            for dtype in dtypes:
                if dtype in schedules:
                    schedules[dtype].externally_managed = True
                    logger.info(f"Camera {camera_id} type {dtype} marked as externally managed")

    def unregister_camera(self, camera_id: str) -> None:
        """注销摄像头，清理内存"""
        with self._lock:
            if camera_id in self._schedules:
                del self._schedules[camera_id]
            self._alert_states.pop(camera_id, None)
            self._cooldowns.pop(camera_id, None)
            self._static_regions.pop(camera_id, None)
            logger.info(f"Camera {camera_id} unregistered")

    # ------------------------------------------------------------------
    # 单摄像头处理
    # ------------------------------------------------------------------

    @staticmethod
    def _annotate_frame(frame: np.ndarray, results: Dict[str, dict],
                        camera_id: str = "", due_types: list = None) -> np.ndarray:
        """在帧上绘制检测框和标签，返回标注后的帧副本"""
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        text_items = []  # (zh_text, en_text, x1, y1, color_bgr)

        for dtype, result in results.items():
            boxes = result.get("boxes", [])
            scores = result.get("scores", [])
            if not boxes:
                continue

            type_def = registry.get(dtype)
            base_color = registry.get_color_bgr(dtype) if type_def else (0, 255, 0)
            base_label = type_def.get("label", dtype) if type_def else dtype
            is_pose = type_def.get("post_process") == "yolo_pose" if type_def else False

            for i, box in enumerate(boxes):
                if len(box) < 4:
                    continue
                x1, y1, x2, y2 = map(int, box[:4])
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                if is_pose:
                    subjects = result.get("subjects", [])
                    is_sleeping = subjects[i].get("sleeping", False) if i < len(subjects) else False
                    color = base_color if is_sleeping else (255, 255, 0)
                    label = base_label if is_sleeping else "person"
                else:
                    color = base_color
                    label = base_label

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                conf = scores[i] if i < len(scores) else 0.0
                text_items.append((f"{label} {conf:.2f}", f"{dtype} {conf:.2f}", x1, y1, color))

            if is_pose:
                skeleton = [
                    (0, 1), (0, 2), (1, 3), (2, 4),
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                    (5, 11), (6, 12), (11, 12),
                    (11, 13), (13, 15), (12, 14), (14, 16),
                ]
                subjects = result.get("subjects", [])
                for subj in subjects:
                    kpts = subj.get("keypoints")
                    if kpts is None or len(kpts) < 17:
                        continue
                    is_sleeping = subj.get("sleeping", False)
                    sk_color = base_color if is_sleeping else (255, 255, 0)
                    for a, b in skeleton:
                        if a < len(kpts) and b < len(kpts):
                            xa, ya, ca = kpts[a]
                            xb, yb, cb = kpts[b]
                            if ca > 0.4 and cb > 0.4:
                                pt_a = (int(xa), int(ya))
                                pt_b = (int(xb), int(yb))
                                cv2.line(annotated, pt_a, pt_b, sk_color, 2)
                    for idx, (kx, ky, kc) in enumerate(kpts[:17]):
                        if kc > 0.4:
                            cv2.circle(annotated, (int(kx), int(ky)), 3, sk_color, -1)

        # 标签文本：优先 PIL 中文绘制；无 CJK 字体时回退英文 + cv2.putText
        font = _get_cjk_font(14)
        if font is not None and text_items:
            annotated = _draw_labels_pil(
                annotated,
                [(zh, x1, y1, color) for zh, _en, x1, y1, color in text_items],
                font,
            )
        else:
            for _zh, en, x1, y1, color in text_items:
                (tw, th), _ = cv2.getTextSize(en, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, en, (x1 + 2, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated

    def _process_camera(self, camera_id: str, core_id: int) -> None:
        """处理单个摄像头的检测循环（由策略线程调用）"""
        frame = self.camera_manager.get_latest_frame(camera_id)
        if frame is None:
            return
        frame_seq = self.camera_manager.get_frame_seq(camera_id)

        now = time.time()
        due_types = self._get_due_types(camera_id, now)
        with self._lock:
            schedules = self._schedules.get(camera_id, {})
            due_types = [
                dtype for dtype in due_types
                if schedules.get(dtype) is not None
                and schedules[dtype].last_sample_seq != frame_seq
            ]
        results = {}

        if due_types:
            detect_start = time.time()
            roi_map = {}
            with self._lock:
                for dt in due_types:
                    s = self._schedules.get(camera_id, {}).get(dt)
                    if s and s.roi:
                        roi_map[dt] = (s.roi, s.roi_invert)
            # 执行检测
            try:
                results = self.safety_detector.detect(
                    frame, due_types, core_id=core_id,
                    camera_id=camera_id, frame_seq=frame_seq, roi_map=roi_map,
                )
            except Exception as e:
                logger.error(f"Safety detection error for {camera_id}: {e}")
                results = {}
            detect_elapsed = time.time() - detect_start

            # 日志：汇总检测结果（带上最高置信度，方便排查阈值问题）
            if results:
                summary = []
                for dtype, res in results.items():
                    box_count = len(res.get("boxes", []))
                    detected = res.get("detected", False)
                    max_conf = res.get("max_confidence", 0.0)
                    summary.append(f"{dtype}({'Y' if detected else 'N'}:{box_count},conf={max_conf:.2f})")
                logger.info(f"Detection {camera_id}: {' | '.join(summary)} ({detect_elapsed:.2f}s)")
            else:
                logger.info(f"Detection {camera_id}: no results ({detect_elapsed:.2f}s)")

            # 缓存最新检测结果用于画框（避免框闪烁）
            # 按类型合并更新，避免多类型 interval 不同步时互相覆盖
            if results:
                cached = self._latest_results.setdefault(camera_id, {})
                cached.update(results)

            # 处理每类型结果
            with self._lock:
                schedules = self._schedules.get(camera_id, {})
                for dtype in due_types:
                    if dtype not in schedules:
                        continue
                    schedule = schedules[dtype]
                    schedule.last_sample_seq = frame_seq
                    result = results.get(dtype, {"detected": False})

                    self._handle_standard_detection(camera_id, dtype, frame, result, schedule, now=now)

        # 注：视频流渲染已拆分到独立 overlay 线程，此处不再送流

    def _is_in_cooldown_unlocked(self, camera_id: str, dtype: str, now: float) -> bool:
        last = self._cooldowns.get(camera_id, {}).get(dtype, 0)
        schedule = self._schedules.get(camera_id, {}).get(dtype)
        cooldown = schedule.cooldown if schedule else 3.0
        return now - last < cooldown

    def _get_due_types(self, camera_id: str, now: float) -> List[str]:
        """获取当前到期的检测类型（跳过冷却中和由外部调度器管理的类型）"""
        with self._lock:
            schedules = self._schedules.get(camera_id, {})
            due = []
            for dtype, s in schedules.items():
                if s.externally_managed:
                    continue
                if not s.is_due(now):
                    continue
                if self._is_in_cooldown_unlocked(camera_id, dtype, now):
                    continue
                due.append(dtype)
            return due

    def is_type_due(self, camera_id: str, dtype: str, now: float) -> bool:
        """外部调度器查询某类型当前是否需要采样。"""
        with self._lock:
            schedule = self._schedules.get(camera_id, {}).get(dtype)
            return bool(schedule and schedule.is_due(now))

    def is_in_cooldown(self, camera_id: str, dtype: str, now: float) -> bool:
        with self._lock:
            return self._is_in_cooldown_unlocked(camera_id, dtype, now)

    # ------------------------------------------------------------------
    # 标准检测处理（fire / smoke / mask / cigarette）
    # ------------------------------------------------------------------

    def _reset_detection_progress(
        self, camera_id: str, dtype: str, schedule: TypeSchedule
    ) -> None:
        """清空当前采样轮和此前连续命中轮的证据。"""
        schedule.consecutive_count = 0
        schedule.sampling_active = False
        schedule.sampled_frame_count = 0
        schedule.last_sample_time = 0.0
        schedule.task_regions = []
        if self.camera_manager is not None:
            self.camera_manager.clear_detection_frames(camera_id, dtype)

    def _handle_standard_detection(
        self, camera_id: str, dtype: str, frame: np.ndarray,
        result: dict, schedule: TypeSchedule, now: float = None
    ) -> None:
        now = time.time() if now is None else now
        # ROI 过滤（relation 结果已在 detect 内按 roi_map 过滤，跳过二次过滤）
        if schedule.roi and not result.get("roi_applied"):
            h, w = frame.shape[:2]
            result = filter_by_roi(result, schedule.roi, schedule.roi_invert, w, h)

        detected = result.get("detected", False)
        max_conf = result.get("max_confidence", 0.0)

        if not detected or max_conf < schedule.threshold:
            if not detected and result.get("boxes"):
                logger.warning(f"{camera_id} {dtype} has boxes but detected=False, resetting progress")
            elif detected and max_conf < schedule.threshold:
                logger.info(f"{camera_id} {dtype} blocked by threshold: conf={max_conf:.2f} < threshold={schedule.threshold}")
            self._reset_detection_progress(camera_id, dtype, schedule)
            schedule.last_run = now
            return

        if not schedule.sampling_active:
            schedule.sampling_active = True
            schedule.sampled_frame_count = 0
            schedule.task_regions = []

        schedule.sampled_frame_count += 1
        schedule.last_sample_time = now
        logger.info(
            f"{camera_id} {dtype} sample={schedule.sampled_frame_count}/"
            f"{schedule.verification_frame_count} conf={max_conf:.2f}"
        )

        # 静态过滤只比较同一检测轮内的采样帧（取最高置信度框区域）
        if schedule.static_filter and result.get("boxes"):
            scores = result.get("scores", [])
            best_idx = scores.index(max(scores)) if scores else 0
            region = _extract_box_region(frame, result["boxes"][best_idx])
            if region is not None:
                schedule.task_regions.append(region)

        # 命中采样帧先进入证据缓存；本轮或后续轮失败时统一清空
        if self.camera_manager is not None:
            settings = config.load_global_settings()
            jpeg_bytes = encode_frame_to_jpg(
                frame,
                quality=settings.get("frame_quality", 60),
                draw_ts=settings.get("save_image_timestamp", True),
                timestamp=now,
            )
            max_frames = schedule.verification_frame_count * schedule.consecutive_required
            self.camera_manager.add_detection_frame(
                camera_id, dtype, now, jpeg_bytes, maxlen=max_frames
            )

        if schedule.sampled_frame_count < schedule.verification_frame_count:
            return

        # 同一轮的全部采样帧都命中后执行静态过滤
        if schedule.static_filter and check_static_filter(
            schedule.task_regions, schedule.static_diff_threshold
        ):
            logger.info(
                f"{camera_id} {dtype} static filter: box region unchanged across "
                f"{len(schedule.task_regions)} samples, resetting progress"
            )
            self._reset_detection_progress(camera_id, dtype, schedule)
            schedule.last_run = now
            return

        schedule.sampling_active = False
        schedule.sampled_frame_count = 0
        schedule.last_sample_time = 0.0
        schedule.task_regions = []
        schedule.last_run = now
        schedule.consecutive_count += 1
        logger.info(
            f"{camera_id} {dtype} round={schedule.consecutive_count}/"
            f"{schedule.consecutive_required} conf={max_conf:.2f}"
        )

        if schedule.consecutive_count < schedule.consecutive_required:
            return

        # 达到连续命中轮数，触发告警流程
        logger.info(f"{camera_id} {dtype} TRIGGERING alarm (conf={max_conf:.2f})")
        self._cooldowns[camera_id][dtype] = now
        schedule.consecutive_count = 0

        # 把 level 和 reason 写入 result，供 trigger_callback 创建记录时使用
        result["level"] = "small_model_alarm"
        type_def = registry.get(dtype)
        label = type_def.get("label", dtype) if type_def else dtype
        alarm_description = type_def.get("alarm_description", "").strip() if type_def else ""
        if alarm_description:
            result["reason"] = alarm_description
        else:
            result["reason"] = f"检测到{label}异常"

        # 达到阈值，统一触发告警流程：先创建记录，再按需提交 VLM 复核
        # 告警记录会在 trigger_callback 中立即创建；VLM 复核结果通过 vlm_result_callback 更新同一条记录。
        self._alert_states[camera_id][dtype] = {"active": True, "time": now, "level": "small_model_alarm"}
        result["detection_frames"] = (
            self.camera_manager.get_detection_frames(camera_id, dtype)
            if self.camera_manager is not None
            else []
        )
        if schedule.use_vlm:
            result["pending_vlm_review"] = True
            vlm_frames = result["detection_frames"][-MAX_VLM_REVIEW_FRAMES:]
            self._submit_vlm_review(camera_id, dtype, vlm_frames, schedule, result)
        if self.trigger_callback:
            try:
                self.trigger_callback(camera_id, dtype, frame, result)
            except Exception as e:
                logger.error(f"Trigger callback error: {e}")

        # 触发后清空缓存
        if self.camera_manager is not None:
            self.camera_manager.clear_detection_frames(camera_id, dtype)

    # ------------------------------------------------------------------
    # VLM 提交辅助
    # ------------------------------------------------------------------

    def _submit_vlm_review(
        self, camera_id: str, dtype: str,
        frames: List[Tuple[float, bytes]],
        schedule: TypeSchedule, result: dict
    ) -> None:
        """异步复核（fire/smoke）"""
        if self.vlm_queue is None:
            return
        numpy_frames = [
            cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            for _, jpg in frames
        ]
        task = {
            "task_id": str(uuid.uuid4()),
            "camera_id": camera_id,
            "dtype": dtype,
            "level": "small_model_alarm",
            "frames": numpy_frames,
            "prompt_type": f"{dtype}_review",
            "callback": lambda res: self._on_vlm_review(camera_id, dtype, res),
        }
        self.vlm_queue.submit(task=task)

    # ------------------------------------------------------------------
    # VLM 回调
    # ------------------------------------------------------------------

    def _on_vlm_review(self, camera_id: str, dtype: str, vlm_result: dict) -> None:
        """VLM 复核回调：透传给上层，由上层更新记录。

        注意：小模型告警记录已在 trigger_callback 中创建（当检测命中时）。
        此回调仅将 VLM 复核结果通过 vlm_result_callback 转发给上层，
        以便上层更新现有记录的 level。它不会再次调用 trigger_callback。
        """
        logger.info(f"VLM review result for {camera_id} {dtype}: {vlm_result}")
        if self.vlm_result_callback:
            try:
                self.vlm_result_callback(camera_id, dtype, vlm_result)
            except Exception as e:
                logger.error(f"VLM result callback error: {e}")

    # ------------------------------------------------------------------
    # 巡检注入
    # ------------------------------------------------------------------

    def inject_detection(self, camera_id: str, dtype: str, simulated_result: dict) -> None:
        """VLM 巡检发现漏检时，注入模拟检测结果"""
        with self._lock:
            schedules = self._schedules.get(camera_id, {})
            if dtype not in schedules:
                return
            schedule = schedules[dtype]
            now = time.time()

            # 三重去重
            if self.has_active_alert(camera_id, dtype):
                return
            if self.is_in_cooldown(camera_id, dtype, now):
                return

            # 高置信度巡检结果可跳过二次复核
            if simulated_result.get("confidence", 0) > 0.85:
                simulated_result["vlm_pre_confirmed"] = True

            # 作为普通检测处理：巡检注入时通常没有原始帧，直接标记告警
            self._cooldowns[camera_id][dtype] = now
            self._alert_states[camera_id][dtype] = {
                "active": True, "time": now, "level": "small_model_alarm", "source": "vlm_inspection"
            }
            if not simulated_result.get("reason"):
                simulated_result["reason"] = f"VLM 巡检发现 {dtype}"
            logger.info(f"Injected {dtype} detection for {camera_id} from VLM inspection")

    # ------------------------------------------------------------------
    # 去重查询
    # ------------------------------------------------------------------

    def has_active_alert(self, camera_id: str, dtype: str) -> bool:
        with self._lock:
            return self._alert_states.get(camera_id, {}).get(dtype, {}).get("active", False)

    def is_in_cooldown(self, camera_id: str, dtype: str, now: float) -> bool:
        with self._lock:
            last = self._cooldowns.get(camera_id, {}).get(dtype, 0)
            schedule = self._schedules.get(camera_id, {}).get(dtype)
            cooldown = schedule.cooldown if schedule else 3.0
            return now - last < cooldown

    # ------------------------------------------------------------------
    # 状态查询（兼容旧接口）
    # ------------------------------------------------------------------

    def get_all_states(self) -> Dict[str, dict]:
        """获取所有摄像头状态（用于 API /status）"""
        with self._lock:
            return {
                cam_id: {
                    "camera_id": cam_id,
                    "types": {
                        dtype: {
                            "enabled": s.enabled,
                            "consecutive_count": s.consecutive_count,
                            "last_run": s.last_run,
                        }
                        for dtype, s in schedules.items()
                    },
                    "alerts": self._alert_states.get(cam_id, {}),
                }
                for cam_id, schedules in self._schedules.items()
            }

    def get_camera_schedules(self, camera_id: str) -> Optional[Dict[str, TypeSchedule]]:
        with self._lock:
            return self._schedules.get(camera_id)

    def update_camera_config(self, camera_id: str, detection_types: Dict[str, dict]) -> None:
        """动态更新摄像头检测配置（热更新）"""
        self.unregister_camera(camera_id)
        self.register_camera(camera_id, detection_types)
        # 懒加载新类型模型（复用当前设备配置，不再硬编码 use_npu=False）
        enabled = [dtype for dtype, cfg in detection_types.items() if cfg.get("enabled")]
        self.safety_detector.ensure_models_loaded(enabled)

    def refresh_type_schedule(self, dtype: str) -> int:
        """算法 defaults 变更后热同步所有启用该算法的摄像头（保留摄像头级 roi/roi_invert），返回同步的摄像头数"""
        synced = 0
        with self._lock:
            for schedules in self._schedules.values():
                old = schedules.get(dtype)
                if old is None:
                    continue
                schedules[dtype] = self._build_schedule(
                    dtype, {"enabled": True, "roi": old.roi, "roi_invert": old.roi_invert}
                )
                synced += 1
        if synced:
            logger.info(f"Type {dtype} schedule refreshed on {synced} cameras")
        return synced
