"""
多类型安全检测调度器
支持策略模式调度（CorePinnedStrategy / SerialStrategy）
管理每摄像头每类型的独立调度、连续计数、冷却、VLM 提交
"""

import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend import config
from backend.frame_utils import encode_frame_to_jpg
from inference_engine import SafetyDetector, detect_npu_cores

logger = logging.getLogger(__name__)

MAX_VLM_REVIEW_FRAMES = 5


@dataclass
class TypeSchedule:
    """单类型检测调度状态"""
    dtype: str
    enabled: bool
    interval: float
    threshold: float
    cooldown: float
    consecutive_required: int = 1
    consecutive_count: int = 0
    last_run: float = 0.0
    use_vlm: bool = False
    # 当存在外部调度器（如 GPU scheduler）接管该类型推理时设为 True
    externally_managed: bool = False

    def is_due(self, now: float) -> bool:
        return now - self.last_run >= self.interval


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

    def register_camera(self, camera_id: str, detection_types: Dict[str, dict]) -> None:
        """注册摄像头检测配置"""
        with self._lock:
            self._schedules[camera_id] = {}
            self._alert_states[camera_id] = {}
            self._cooldowns[camera_id] = {}
            for dtype, cfg in detection_types.items():
                if not cfg.get("enabled", False):
                    continue
                schedule = TypeSchedule(
                    dtype=dtype,
                    enabled=True,
                    interval=cfg.get("interval", 1.0),
                    threshold=cfg.get("threshold", 0.5),
                    cooldown=cfg.get("cooldown", 60.0),
                    consecutive_required=cfg.get("consecutive_required", 3),
                    use_vlm=cfg.get("use_vlm", False),
                )
                self._schedules[camera_id][dtype] = schedule
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

        type_colors = {
            "fire": (0, 0, 255),      # 红
            "smoke": (128, 128, 128), # 灰
            "uniform": (255, 0, 0),   # 蓝
            "mask": (0, 255, 255),    # 黄
            "cigarette": (0, 255, 0), # 绿
            "sleep": (255, 0, 255),   # 紫
        }
        type_labels = {
            "fire": "fire",
            "smoke": "smoke",
            "uniform": "uniform",
            "mask": "mask",
            "cigarette": "cigarette",
            "sleep": "sleep",
        }

        total_boxes = 0
        for dtype, result in results.items():
            boxes = result.get("boxes", [])
            scores = result.get("scores", [])
            detected = result.get("detected", False)
            if not boxes:
                continue

            total_boxes += len(boxes)
            base_color = type_colors.get(dtype, (0, 255, 0))
            base_label = type_labels.get(dtype, dtype)

            for i, box in enumerate(boxes):
                if len(box) < 4:
                    continue
                x1, y1, x2, y2 = map(int, box[:4])
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                # sleep 类型按 sleeping 状态区分颜色
                if dtype == "sleep":
                    subjects = result.get("subjects", [])
                    is_sleeping = subjects[i].get("sleeping", False) if i < len(subjects) else False
                    color = base_color if is_sleeping else (255, 255, 0)  # 青色
                    label = "sleep" if is_sleeping else "person"
                else:
                    color = base_color
                    label = base_label

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                conf = scores[i] if i < len(scores) else 0.0
                text = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, text, (x1 + 2, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 睡岗检测：绘制骨架
            if dtype == "sleep":
                skeleton = [
                    (0, 1), (0, 2), (1, 3), (2, 4),       # head
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
                    (5, 11), (6, 12), (11, 12),            # torso
                    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
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
                    # 关键点圆点
                    for idx, (kx, ky, kc) in enumerate(kpts[:17]):
                        if kc > 0.4:
                            cv2.circle(annotated, (int(kx), int(ky)), 3, sk_color, -1)

            # 如果该类型有检测到目标，在右上角显示类型状态
            if detected:
                status_text = f"[ALERT] {base_label}"
                (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (w - tw - 10, 5), (w - 5, 5 + th + 10), base_color, -1)
                cv2.putText(annotated, status_text, (w - tw - 5, 5 + th + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 左上角调试信息：始终显示，帮助确认画框逻辑在运行
        debug_lines = [f"cam:{camera_id}"]
        if due_types:
            debug_lines.append(f"detect:{','.join(due_types)}")
        debug_lines.append(f"boxes:{total_boxes}")
        y_offset = 20
        for line in debug_lines:
            cv2.putText(annotated, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 18

        return annotated

    def _process_camera(self, camera_id: str, core_id: int) -> None:
        """处理单个摄像头的检测循环（由策略线程调用）"""
        frame = self.camera_manager.get_latest_frame(camera_id)
        if frame is None:
            return

        now = time.time()
        due_types = self._get_due_types(camera_id, now)
        results = {}

        if due_types:
            detect_start = time.time()
            # 执行检测
            try:
                results = self.safety_detector.detect(frame, due_types, core_id=core_id)
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
                    scores = res.get("scores", [])
                    max_conf = max(scores) if scores else 0.0
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
                    # 如果检测耗时超过该类型的 interval，用完成时间作为 last_run，
                    # 避免检测还没做完下一轮又到期，导致 CPU/GPU 100% 占满卡死
                    if detect_elapsed >= schedule.interval:
                        schedule.last_run = time.time()
                    else:
                        schedule.last_run = now
                    result = results.get(dtype, {"detected": False})

                    if dtype == "uniform":
                        if result.get("detected") and not result.get("reason"):
                            result["reason"] = "检测到未穿工服/反光背心的人员"
                        self._handle_standard_detection(camera_id, dtype, frame, result, schedule)
                    elif dtype == "sleep":
                        self._handle_sleep_detection(camera_id, frame, result, schedule)
                    else:
                        self._handle_standard_detection(camera_id, dtype, frame, result, schedule)

        # 注：视频流渲染已拆分到独立 overlay 线程，此处不再送流

    def _get_due_types(self, camera_id: str, now: float) -> List[str]:
        """获取当前到期的检测类型（跳过由外部调度器管理的类型）"""
        with self._lock:
            schedules = self._schedules.get(camera_id, {})
            return [dtype for dtype, s in schedules.items() if not s.externally_managed and s.is_due(now)]

    # ------------------------------------------------------------------
    # 标准检测处理（fire / smoke / mask / cigarette）
    # ------------------------------------------------------------------

    def _handle_standard_detection(
        self, camera_id: str, dtype: str, frame: np.ndarray,
        result: dict, schedule: TypeSchedule
    ) -> None:
        detected = result.get("detected", False)
        max_conf = max(result.get("scores", [0]) or [0])

        if not detected or max_conf < schedule.threshold:
            if not detected and result.get("boxes"):
                logger.warning(f"{camera_id} {dtype} has boxes but detected=False, resetting count")
            elif detected and max_conf < schedule.threshold:
                logger.info(f"{camera_id} {dtype} blocked by threshold: conf={max_conf:.2f} < threshold={schedule.threshold}")
            if self.camera_manager is not None:
                self.camera_manager.clear_detection_frames(camera_id, dtype)
            schedule.consecutive_count = 0
            return

        schedule.consecutive_count += 1
        logger.info(f"{camera_id} {dtype} consecutive={schedule.consecutive_count}/{schedule.consecutive_required} conf={max_conf:.2f}")

        # 编码并写入检测帧缓存
        if self.camera_manager is not None:
            ts = time.time()
            settings = config.load_global_settings()
            jpeg_bytes = encode_frame_to_jpg(
                frame,
                quality=settings.get("frame_quality", 60),
                draw_ts=settings.get("save_image_timestamp", True),
                timestamp=ts,
            )
            self.camera_manager.add_detection_frame(
                camera_id, dtype, ts, jpeg_bytes, maxlen=schedule.consecutive_required
            )

        if schedule.consecutive_count < schedule.consecutive_required:
            return

        now = time.time()
        if self.is_in_cooldown(camera_id, dtype, now):
            logger.info(f"{camera_id} {dtype} blocked by cooldown")
            return

        # 达到阈值，触发告警流程
        logger.info(f"{camera_id} {dtype} TRIGGERING alarm (conf={max_conf:.2f})")
        self._cooldowns[camera_id][dtype] = now

        # 把 level 和 reason 写入 result，供 trigger_callback 创建记录时使用
        result["level"] = "small_model_alarm"
        if not result.get("reason"):
            if dtype == "sleep":
                result["reason"] = f"睡岗检测连续 {schedule.consecutive_required} 次命中"
            else:
                result["reason"] = f"检测到 {dtype}，置信度 {max_conf:.2f}"

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
    # 睡岗检测状态机
    # ------------------------------------------------------------------

    def _handle_sleep_detection(
        self, camera_id: str, frame: np.ndarray, result: dict, schedule: TypeSchedule
    ) -> None:
        """睡岗检测统一为标准检测逻辑：命中写缓存，未命中清空，触发后清空。"""
        self._handle_standard_detection(camera_id, "sleep", frame, result, schedule)

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

            # 作为普通检测处理
            if dtype == "sleep":
                # 巡检不缩短 60s 间隔，仅作为一次独立命中
                simulated_result["skip_interval_check"] = True
                # TODO: 需要 frame 参数
            else:
                # 标准类型需要 frame，但巡检注入时可能没有
                # 简化：直接标记告警
                self._cooldowns[camera_id][dtype] = now
                self._alert_states[camera_id][dtype] = {
                    "active": True, "time": now, "level": "small_model_alarm", "source": "vlm_inspection"
                }
                if dtype == "uniform" and not simulated_result.get("reason"):
                    simulated_result["reason"] = "VLM 巡检发现未穿工服/反光背心的人员"
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
