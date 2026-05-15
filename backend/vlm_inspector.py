"""
VLM 周期性巡检器
每 30 秒对所有摄像头进行一次全类型综合检查
发现漏检时注入模拟检测结果
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


class VLMInspector:
    """
    VLM 巡检器
    - interval: 巡检间隔（秒）
    - max_cameras_per_inspection: 每次巡检最多检查几个摄像头
    """

    def __init__(
        self,
        camera_manager,
        multi_detector,
        vlm_queue,
        understander,
        interval: float = 30.0,
        max_cameras_per_inspection: int = 3,
        timeout: float = 10.0,
    ):
        self.camera_manager = camera_manager
        self.multi_detector = multi_detector
        self.vlm_queue = vlm_queue
        self.understander = understander
        self.interval = interval
        self.max_cameras = max_cameras_per_inspection
        self.timeout = timeout
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stats = {"inspections": 0, "injections": 0, "misses": 0}
        self._stats_lock = threading.Lock()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._inspection_loop,
            daemon=True,
            name="vlm-inspector",
        )
        self._thread.start()
        logger.info(f"VLMInspector started (interval={self.interval}s)")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("VLMInspector stopped")

    def _inspection_loop(self) -> None:
        while self._running:
            time.sleep(self.interval)
            if not self._running:
                break
            try:
                self._run_inspection()
            except Exception as e:
                logger.error(f"VLM inspection error: {e}")

    def _run_inspection(self) -> None:
        """执行一次巡检"""
        # 1. 收集启用的摄像头
        camera_ids = self.camera_manager.get_camera_ids()
        if not camera_ids:
            return

        # 最多取 max_cameras 个
        selected_ids = list(camera_ids)[: self.max_cameras]

        # 2. 收集帧和启用类型
        frames: Dict[str, np.ndarray] = {}
        enabled_types_per_camera: Dict[str, List[str]] = {}
        all_enabled_types: Set[str] = set()

        for cam_id in selected_ids:
            frame = self.camera_manager.get_frame(cam_id)
            if frame is None:
                continue
            frames[cam_id] = frame.copy()

            schedules = self.multi_detector.get_camera_schedules(cam_id)
            if schedules:
                types = [dtype for dtype, s in schedules.items() if s.enabled]
                enabled_types_per_camera[cam_id] = types
                all_enabled_types.update(types)

        if not frames:
            return

        # 3. 对每个摄像头分别调用 VLM
        for cam_id, frame in frames.items():
            types = enabled_types_per_camera.get(cam_id, [])
            if not types:
                continue

            prompt = self._build_inspection_prompt(types)
            try:
                result = self.understander.analyze_multi(
                    frames=[frame],
                    prompt_type="inspection",
                    extra_context={
                        "prompt": prompt,
                        "camera_id": cam_id,
                        "enabled_types": types,
                    },
                )
            except Exception as e:
                logger.error(f"Inspection VLM call failed for {cam_id}: {e}")
                continue

            # 4. 解析结果并注入
            detections = self._parse_inspection_result(result, types)
            for dtype, detection in detections.items():
                if not detection.get("detected", False):
                    continue

                # 四重去重
                now = time.time()
                if self.multi_detector.has_active_alert(cam_id, dtype):
                    continue
                if self.multi_detector.is_pending_vlm(cam_id, dtype):
                    continue
                if self.multi_detector.is_in_cooldown(cam_id, dtype, now):
                    continue
                if dtype == "sleep" and self.multi_detector.sleep_has_pending_vlm(cam_id):
                    continue

                # 注入
                self.multi_detector.inject_detection(cam_id, dtype, detection)
                with self._stats_lock:
                    self._stats["injections"] += 1
                logger.info(f"VLM inspection injected {dtype} for {cam_id}")

        with self._stats_lock:
            self._stats["inspections"] += 1

    def _build_inspection_prompt(self, enabled_types: List[str]) -> str:
        """构建巡检 prompt"""
        type_desc = {
            "fire": "明火",
            "smoke": "烟雾",
            "uniform": "未穿工服",
            "mask": "未戴口罩",
            "cigarette": "吸烟",
            "sleep": "睡岗/打盹",
        }
        checks = [f"- {type_desc.get(t, t)}" for t in enabled_types]
        checks_str = "\n".join(checks)

        return f"""你正在执行工业安全监控巡检。请仔细检查这张监控画面，判断是否存在以下安全隐患：
{checks_str}

请以 JSON 格式返回，不要其他内容：
{{
"detections": {{
{chr(10).join([f'  "{t}": {{"detected": true/false, "confidence": 0.0-1.0, "reason": "判断理由"}}' for t in enabled_types])}
}}
}}

注意：
- 只检查上述列出的类型，不要自行扩展
- confidence 范围 0.0-1.0
- 如果没有发现任何异常，所有 detected 都返回 false"""

    def _parse_inspection_result(
        self, result: dict, expected_types: List[str]
    ) -> Dict[str, dict]:
        """解析巡检结果，返回 dtype -> detection 的映射"""
        detections: Dict[str, dict] = {}
        if not result or "error" in result:
            return detections

        # 支持两种格式：
        # 1. result["detections"][dtype] = {"detected": bool, "confidence": float}
        # 2. result[dtype] = {"detected": bool, "confidence": float}
        raw = result.get("detections", result)

        for dtype in expected_types:
            det = raw.get(dtype, {})
            if isinstance(det, dict) and det.get("detected"):
                detections[dtype] = {
                    "detected": True,
                    "confidence": det.get("confidence", 0.5),
                    "reason": det.get("reason", ""),
                    "source": "vlm_inspection",
                }
        return detections

    def get_stats(self) -> Dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)
