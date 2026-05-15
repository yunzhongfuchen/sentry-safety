"""
有限空间人员计数与进出监控核心逻辑

监控模式：
- entrance: YOLO检测入口有人 + VLM窗口统计进入/离开人数
"""

import base64
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ZoneConfig:
    """有限空间区域配置（扁平化：一个摄像头即一个区域）"""
    camera_id: str
    name: str
    roi: List[int]              # [x1, y1, x2, y2]
    enable_vlm_review: bool = True


@dataclass
class ZoneState:
    """有限空间区域运行时状态"""
    config: ZoneConfig

    frame_buffer: deque = field(default_factory=lambda: deque(maxlen=10))  # 1fps 采样，保留约10秒
    is_active: bool = False         # 入口是否有人
    active_empty_frames: int = 0    # 连续无人帧计数
    last_sample_time: float = 0.0   # 控制 1fps 采样频率
    window_center: float = 0.0      # 当前待提交 VLM 窗口的中心时刻
    window_submit_at: float = 0.0   # 当前窗口计划提交时间
    window_submitted: bool = True   # 当前窗口是否已提交（防止重复提交）

    # 通用
    last_event_time: float = field(default_factory=time.time)
    event_history: List[dict] = field(default_factory=list)
    vlm_review_pending: bool = False
    last_review_result: Optional[dict] = None
    # 最近一次检测的人体框，供前端可视化使用 [{xyxy, score}, ...]
    last_detections: List[dict] = field(default_factory=list)


class ZoneCounter:
    """
    有限空间人员计数器（入口模式）

    处理流程：
    - YOLO检测入口 -> 状态跟踪 -> 窗口提交VLM统计 -> 更新
    """

    def __init__(
        self,
        camera_manager,
        inference_engine,
        vlm_queue,
        storage=None,
        performance_storage=None,
        settings=None,
    ):
        self.camera_manager = camera_manager
        self.inference_engine = inference_engine
        self.vlm_queue = vlm_queue
        self.storage = storage or performance_storage
        self.settings = settings or {}

        self._zones: Dict[str, ZoneState] = {}
        self._lock = threading.RLock()

        # 确保 person 模型已加载
        try:
            self.inference_engine._load_person_model("cpu")
        except Exception as e:
            logger.warning(f"Failed to preload person model: {e}")

    # -- 区域管理 --

    def register_camera(self, config: ZoneConfig) -> None:
        """注册（或更新）一个有限空间摄像头（一摄像头一区域）"""
        cid = config.camera_id
        with self._lock:
            if cid in self._zones:
                old = self._zones[cid]
                self._zones[cid] = ZoneState(
                    config=config,
                    event_history=old.event_history,
                )
                logger.info(f"Camera updated: {cid}")
            else:
                self._zones[cid] = ZoneState(config=config)
                logger.info(f"Camera registered: {cid}")

    def unregister_camera(self, camera_id: str) -> None:
        """注销摄像头"""
        with self._lock:
            if camera_id in self._zones:
                del self._zones[camera_id]
                logger.info(f"Camera unregistered: {camera_id}")

    def get_camera_state(self, camera_id: str) -> Optional[dict]:
        """获取摄像头当前状态（供 API 查询）"""
        with self._lock:
            state = self._zones.get(camera_id)
            if not state:
                return None
            return {
                "camera_id": camera_id,
                "name": state.config.name,
                "roi": list(state.config.roi or []),
                "enable_vlm_review": state.config.enable_vlm_review,
                "vlm_review_pending": state.vlm_review_pending,
                "last_review_result": state.last_review_result,
                "event_history": state.event_history[-20:],
            }

    def get_camera_visualization(self, camera_id: str) -> dict:
        """获取摄像头的 ROI 与最近检测框,供前端/标注帧使用"""
        with self._lock:
            state = self._zones.get(camera_id)
            if not state:
                return {"camera_id": camera_id, "zones": []}
            roi = state.config.roi
            detections_in_roi = [
                d for d in (state.last_detections or [])
                if self._in_roi(d.get("xyxy", []), roi)
            ]
            zone_data = {
                "zone_id": camera_id,
                "name": state.config.name,
                "roi": list(roi or []),
                "detected_count": len(detections_in_roi),
                "detections": detections_in_roi,
            }
            return {"camera_id": camera_id, "zones": [zone_data]}

    def list_cameras(self) -> List[dict]:
        """列出所有摄像头状态（一摄像头一区域）"""
        with self._lock:
            return [
                {
                    "camera_id": z.config.camera_id,
                    "name": z.config.name,
                    "roi": list(z.config.roi or []),
                    "enable_vlm_review": z.config.enable_vlm_review,
                }
                for z in self._zones.values()
            ]

    def reset_buffer(self, camera_id: str) -> None:
        """清空指定摄像头的帧缓存（视频从头播放时调用，避免旧帧混入新窗口）"""
        with self._lock:
            state = self._zones.get(camera_id)
            if not state:
                return
            state.frame_buffer.clear()
            state.last_sample_time = 0.0
            logger.info(f"Camera {camera_id}: frame buffer cleared")

    # -- 核心处理 --

    def process_frame(self, camera_id: str, frame: np.ndarray) -> Optional[List[dict]]:
        """
        处理单帧，返回事件列表（如果发生了人数变化）
        """
        with self._lock:
            state = self._zones.get(camera_id)
            if not state:
                logger.warning(f"Camera not found: {camera_id}")
                return None

        cfg = state.config

        # 人员检测
        boxes = self.inference_engine._detect_persons(frame)
        boxes_in_roi = [b for b in boxes if self._in_roi(b.get("xyxy", []), cfg.roi)]
        detected_count = len(boxes_in_roi)

        # 缓存最近检测,供前端可视化
        with self._lock:
            state.last_detections = list(boxes)

        logger.info(f"[DETECT] camera={camera_id} persons={len(boxes)} in_roi={detected_count}")

        events = self._process_entrance(camera_id, state, cfg, detected_count, frame)

        if events:
            for ev in events:
                self._save_event(ev, frame)
        return events

    # -- entrance 模式处理 --

    def _process_entrance(self, camera_id, state, cfg, detected_count, frame) -> List[dict]:
        events = []
        now = time.time()
        vlm_task = None  # (window_frames, trigger) or None

        sample_interval = self.settings.get("sample_interval", 1.0)
        window_delay = self.settings.get("window_delay", 3.0)
        consecutive_required = self.settings.get("consecutive_required", 3)

        with self._lock:
            # 动态调整 buffer 容量，确保覆盖完整窗口 + 额外缓冲历史
            required_len = max(10, int(window_delay * 3 / sample_interval) + 1)
            if state.frame_buffer.maxlen != required_len:
                new_buffer = deque(maxlen=required_len)
                new_buffer.extend(state.frame_buffer)
                state.frame_buffer = new_buffer

            # 按 sample_interval 采样写入滚动缓存
            if now - state.last_sample_time >= sample_interval:
                state.frame_buffer.append((now, frame.copy()))
                state.last_sample_time = now

            has_person = detected_count > 0

            if not state.is_active:
                # IDLE -> ACTIVE：检测到人则启动对称窗口
                if has_person:
                    state.is_active = True
                    state.active_empty_frames = 0
                    state.window_center = now
                    state.window_submit_at = now + window_delay
                    state.window_submitted = False
                    logger.info(f"Camera {camera_id}: IDLE -> ACTIVE, center={now:.2f}, delay={window_delay}")
            else:
                # ACTIVE 状态：不管中间是否漏检，坚持等到 window_submit_at 再提交完整窗口
                if now >= state.window_submit_at and not state.window_submitted:
                    start_t = state.window_center - window_delay
                    end_t = state.window_center + window_delay
                    window_frames = [(t, f) for t, f in state.frame_buffer if start_t <= t <= end_t]
                    if window_frames:
                        vlm_task = (window_frames, "interval")
                    state.window_submitted = True

                # 窗口已提交后，才决定是否退出或启动下一个窗口
                if state.window_submitted:
                    if has_person:
                        # 人还在，启动下一个重叠窗口
                        state.window_center = now
                        state.window_submit_at = now + window_delay
                        state.window_submitted = False
                        state.active_empty_frames = 0
                        logger.debug(f"Camera {camera_id}: next window center={now:.2f}")
                    else:
                        state.active_empty_frames += 1
                        if state.active_empty_frames >= consecutive_required:
                            state.is_active = False
                            state.active_empty_frames = 0
                            logger.info(f"Camera {camera_id}: ACTIVE -> IDLE")

        # 在锁外调用 VLM 提交，避免长时间持有锁阻塞检测循环、渲染线程和 API
        if vlm_task:
            window_frames, trigger = vlm_task
            vlm_events = self._submit_vlm_window_review(camera_id, window_frames, cfg, trigger)
            events.extend(vlm_events)

        return events

    def force_submit_pending_window(self, camera_id: str) -> None:
        """视频播放到末尾时，强制提交尚未提交的 VLM 窗口（避免短视频窗口永远等不到 window_delay）"""
        with self._lock:
            state = self._zones.get(camera_id)
            if not state or not state.is_active or state.window_submitted:
                return

            window_delay = self.settings.get("window_delay", 3.0)
            start_t = state.window_center - window_delay
            window_frames = [(t, f) for t, f in state.frame_buffer if t >= start_t]
            if not window_frames:
                state.is_active = False
                state.active_empty_frames = 0
                return

            state.window_submitted = True
            state.is_active = False
            state.active_empty_frames = 0
            cfg = state.config

        self._submit_vlm_window_review(camera_id, window_frames, cfg, "forced")

    # -- VLM 复核 --

    @staticmethod
    def _draw_roi_on_frame(frame: np.ndarray, roi: List[int]) -> np.ndarray:
        """在帧副本上画出 ROI 框，标注入口区域，供 VLM 参考"""
        annotated = frame.copy()
        if roi and len(roi) == 4:
            x1, y1, x2, y2 = map(int, roi)
            h, w = annotated.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = "ENTRANCE"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 8, y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        return annotated

    def _submit_vlm_window_review(self, camera_id: str, buffer: List[Tuple[float, np.ndarray]], cfg: ZoneConfig, trigger: str) -> List[dict]:
        """entrance 模式：提交窗口帧给 VLM 统计，返回事件列表"""
        if not buffer or self.vlm_queue is None:
            return []

        # 根据窗口配置动态计算最大帧数：窗口总时长 / 采样间隔
        sample_interval = self.settings.get("sample_interval", 1.0)
        window_delay = self.settings.get("window_delay", 3.0)
        max_frames = int(window_delay * 2 / sample_interval)

        frames = [f for _, f in buffer]
        if len(frames) > max_frames:
            step = len(frames) / max_frames
            frames = [frames[min(int(i * step), len(frames) - 1)] for i in range(max_frames)]

        # 在帧副本上画出 ROI 框，标注入口区域，不影响原始帧和推流
        frames = [self._draw_roi_on_frame(f, cfg.roi) for f in frames]

        with self._lock:
            state = self._zones.get(camera_id)
            if not state:
                return []
            state.vlm_review_pending = True

        event_id = f"cs-{uuid.uuid4().hex[:8]}"
        event_placeholder = {
            "event_id": event_id,
            "zone_id": camera_id,
            "zone_name": cfg.name,
            "camera_id": cfg.camera_id,
            "timestamp": datetime.now().isoformat(),
            "trigger": trigger,
        }

        # 保存窗口帧序列（均匀采样覆盖整个窗口）
        raw_frames = [f for _, f in buffer]
        if len(raw_frames) > max_frames:
            step = len(raw_frames) / max_frames
            raw_frames = [raw_frames[min(int(i * step), len(raw_frames) - 1)] for i in range(max_frames)]
        if raw_frames and self.storage:
            for i, f in enumerate(raw_frames):
                _, buf = cv2.imencode('.jpg', f)
                frame_b64 = base64.b64encode(buf).decode('utf-8')
                self.storage.save_frame(event_id, frame_b64, i)

        # 创建占位记录（VLM 结果回来前显示为"检测中"）
        placeholder_event = {
            "event_id": event_id,
            "zone_id": camera_id,
            "zone_name": cfg.name,
            "camera_id": cfg.camera_id,
            "timestamp": event_placeholder["timestamp"],
            "event_type": "pending",
            "monitor_mode": "entrance",
            "diff": 0,
            "description": f"{cfg.name} VLM 检测中...",
            "vlm_reviewed": False,
            "vlm_result": None,
            "alert": False,
            "trigger": trigger,
            "frame_count": len(raw_frames),
        }
        with self._lock:
            state.event_history.append(placeholder_event)
        if self.storage:
            self.storage.add_record(placeholder_event)

        task = {
            "task_id": f"cs-window-{event_id}",
            "camera_id": cfg.camera_id,
            "dtype": "confined_window",
            "level": "P1",
            "frames": frames,
            "prompt_type": "confined_window_review",
            "extra_context": {
                "zone_name": cfg.name,
                "N": len(frames),
            },
            "callback": lambda result: self._on_vlm_window_review(camera_id, event_id, result),
        }
        self.vlm_queue.submit(task)

        # entrance 模式下事件已在 placeholder 中生成
        return []

    def _on_vlm_window_review(self, camera_id: str, placeholder_id: str, result: dict) -> None:
        """entrance 模式 VLM 窗口统计回调（不保留人数累积，更新占位记录）"""
        with self._lock:
            state = self._zones.get(camera_id)
            if not state:
                return
            state.vlm_review_pending = False
            state.last_review_result = result
            cfg = state.config

        vlm_entered = result.get("entered", False)
        vlm_left = result.get("left", False)
        vlm_other = result.get("other", False)
        entered_count = result.get("entered_count", 0)
        left_count = result.get("left_count", 0)
        confidence = result.get("confidence", 0)

        logger.info(
            f"VLM window review for camera {camera_id}: "
            f"entered={vlm_entered}({entered_count}), left={vlm_left}({left_count}), "
            f"other={vlm_other}, confidence={confidence}"
        )

        # 确定事件类型：同时有进有出 -> both；否则按 enter/leave/other 优先级
        if vlm_entered and entered_count > 0 and vlm_left and left_count > 0:
            primary_type = "both"
            description = f"{cfg.name} 检测到 {entered_count} 人进入、{left_count} 人离开"
            updates = {
                "event_type": "both",
                "description": description,
                "diff": entered_count - left_count,
                "vlm_reviewed": True,
                "vlm_result": result,
            }
        elif vlm_entered and entered_count > 0:
            updates = {
                "event_type": "enter",
                "description": self._build_event_description(cfg, "enter", entered_count),
                "diff": entered_count,
                "vlm_reviewed": True,
                "vlm_result": result,
            }
        elif vlm_left and left_count > 0:
            updates = {
                "event_type": "leave",
                "description": self._build_event_description(cfg, "leave", left_count),
                "diff": -left_count,
                "vlm_reviewed": True,
                "vlm_result": result,
            }
        elif vlm_other:
            updates = {
                "event_type": "other",
                "description": self._build_event_description(cfg, "other", 0),
                "diff": 0,
                "vlm_reviewed": True,
                "vlm_result": result,
            }
        else:
            # VLM 未检测到任何事件
            updates = {
                "event_type": "other",
                "description": f"{cfg.name} 未检测到人员进出",
                "diff": 0,
                "vlm_reviewed": True,
                "vlm_result": result,
            }

        # 更新内存中的占位记录
        with self._lock:
            for i, ev in enumerate(state.event_history):
                if ev.get("event_id") == placeholder_id:
                    state.event_history[i] = {**ev, **updates}
                    break

        # 更新持久化记录
        if self.storage:
            self.storage.update_record(placeholder_id, updates)

    @staticmethod
    def _build_event_description(cfg, event_type, count) -> str:
        """生成事件描述文本"""
        if event_type == "enter":
            return f"{cfg.name} 检测到 {count} 人进入"
        elif event_type == "leave":
            return f"{cfg.name} 检测到 {count} 人离开"
        else:
            return f"{cfg.name} 检测到人员在入口附近活动但未进出"

    # -- 事件保存 --

    def _save_event(self, event: dict, frame: Optional[np.ndarray]) -> None:
        """保存事件到存储"""
        if self.storage is None:
            return
        try:
            self.storage.add_record(event)
            if frame is not None:
                # TODO: 编码帧为base64并保存快照
                pass
        except Exception as e:
            logger.error(f"Failed to save confined space event: {e}")

    # -- 工具 --

    @staticmethod
    def _in_roi(box: List[int], roi: List[int]) -> bool:
        """判断检测框中心点是否在 ROI 内。ROI 全零视为未配置，全图生效。"""
        if not box or len(box) < 4 or not roi or len(roi) < 4:
            return True
        rx1, ry1, rx2, ry2 = roi
        # ROI 全零视为未配置，不过滤
        if rx1 == 0 and ry1 == 0 and rx2 == 0 and ry2 == 0:
            return True
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2
