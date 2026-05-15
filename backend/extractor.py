import logging
import time
from typing import List, Tuple, Optional
import cv2
import numpy as np
import threading
import config

logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    连续抽帧器 - 从 RTSPReader 获取帧，不再自己开连接。
    使用 snapshot() 获取帧副本，避免与检测循环竞争。
    """

    def __init__(self, frame_count: int = 30, frame_interval: float = 1.0):
        self.frame_count = frame_count
        self.frame_interval = frame_interval

    def extract_from_reader(self, reader) -> Tuple[List[np.ndarray], List[float]]:
        """
        从已有的 RTSPReader 定时抽帧。

        核心逻辑：每隔 frame_interval 秒取一帧快照。
        只跳过与上一帧完全相同的像素（说明 reader 还没更新），
        不做"相似度"过滤——监控场景连续帧本来就很像，但时间不同，VLM 需要这些帧来判断运动。

        Args:
            reader: RTSPReader 实例（需支持 snapshot() 方法）

        Returns:
            (帧列表, 时间戳列表)
        """
        frames = []
        timestamps = []
        last_frame = None

        logger.info(
            f"Starting frame extraction: {self.frame_count} frames "
            f"at {self.frame_interval}s interval"
        )

        for i in range(self.frame_count):
            # 尝试获取一帧新的（不同于上一帧的）画面
            frame = self._wait_for_new_frame(reader, last_frame, timeout=2.0)

            if frame is None:
                logger.warning(f"Frame {i}: failed to get new frame, skipping")
                # 即使拿不到新帧，也继续等下一个间隔
                if i < self.frame_count - 1:
                    time.sleep(self.frame_interval)
                continue

            frames.append(frame)
            timestamps.append(time.time())
            last_frame = frame

            # 最后一帧不需要等
            if i < self.frame_count - 1:
                time.sleep(self.frame_interval)

        logger.info(f"Extracted {len(frames)}/{self.frame_count} frames")
        return frames, timestamps

    def _wait_for_new_frame(
        self, reader, last_frame: Optional[np.ndarray], timeout: float = 2.0
    ) -> Optional[np.ndarray]:
        """
        等待 reader 产出一帧与 last_frame 不完全相同的画面。
        最多等 timeout 秒，超时则返回当前帧（即使和上一帧相同）。
        """
        deadline = time.time() + timeout

        while True:
            if hasattr(reader, 'snapshot'):
                frame = reader.snapshot()
            else:
                ret, frame = reader.read()
                if not ret:
                    frame = None

            if frame is None:
                if time.time() >= deadline:
                    return None
                time.sleep(0.03)
                continue

            # 如果没有上一帧，直接返回
            if last_frame is None:
                return frame.copy()

            # 如果像素不完全相同，说明 reader 已经更新了
            if not _frames_identical(last_frame, frame):
                return frame.copy()

            # 像素完全相同 → reader 还没更新，短暂等待后重试
            if time.time() >= deadline:
                # 超时了，返回当前帧（总比没有好）
                return frame.copy()

            time.sleep(0.03)


def _frames_identical(a: np.ndarray, b: np.ndarray) -> bool:
    """判断两帧是否像素完全相同（说明 reader 还没刷新）。"""
    if a.shape != b.shape:
        return False
    return np.array_equal(a, b)
