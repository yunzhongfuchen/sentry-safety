"""
高性能视频流处理模块
使用多线程和缓冲队列优化视频流性能
"""

import threading
import time
import logging
from typing import Dict, Optional
from queue import Queue, Empty
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoStreamBuffer:
    """
    视频流缓冲区
    使用生产者-消费者模式，避免阻塞主线程
    """
    
    def __init__(self, maxsize: int = 5):
        self._queue: Queue = Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_jpeg: Optional[bytes] = None
        self._frame_seq = 0
        self._jpeg_seq = 0
        self._fps = 0
        self._frame_count = 0
        self._last_time = time.time()

    def put(self, frame: np.ndarray, quality: int = 85) -> bool:
        """添加帧到缓冲区"""
        try:
            # 非阻塞放入，如果满了就丢弃最旧的
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
            self._queue.put_nowait(frame)

            with self._lock:
                self._latest_frame = frame
                self._frame_seq += 1
                self._frame_count += 1
                self._encode_jpeg(frame, quality)

                # 计算 FPS
                now = time.time()
                if now - self._last_time >= 1.0:
                    self._fps = self._frame_count / (now - self._last_time)
                    self._frame_count = 0
                    self._last_time = now

            return True
        except:
            return False

    def _encode_jpeg(self, frame: np.ndarray, quality: int) -> None:
        """缓存当前帧的 JPEG 字节，避免每个 HTTP 客户端独立编码。"""
        try:
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if _:
                self._latest_jpeg = jpeg.tobytes()
                self._jpeg_seq = self._frame_seq
        except Exception:
            self._latest_jpeg = None

    def get(self) -> Optional[np.ndarray]:
        """获取一帧"""
        try:
            return self._queue.get_nowait()
        except Empty:
            with self._lock:
                return self._latest_frame
    
    def get_latest(self) -> Optional[np.ndarray]:
        """获取最新帧"""
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_latest_with_seq(self) -> tuple:
        """获取最新帧及其序列号"""
        with self._lock:
            return self._frame_seq, self._latest_frame.copy() if self._latest_frame is not None else None

    def get_latest_jpeg_with_seq(self) -> tuple:
        """获取最新帧的 JPEG 字节及其序列号"""
        with self._lock:
            return self._jpeg_seq, self._latest_jpeg

    def get_fps(self) -> float:
        """获取当前 FPS"""
        with self._lock:
            return self._fps


class MJPEGStreamServer:
    """
    MJPEG 流服务器
    为多摄像头提供高性能的视频流服务
    支持双缓冲：原始帧 + 标注帧（画框）
    """

    def __init__(self):
        self._buffers: Dict[str, VideoStreamBuffer] = {}
        self._raw_buffers: Dict[str, VideoStreamBuffer] = {}
        self._lock = threading.Lock()
        self._quality = 85
        self._max_fps = 30

    def register_camera(self, camera_id: str, buffer_size: int = 3):
        """注册摄像头缓冲区（同时注册原始和标注两个缓冲区）"""
        with self._lock:
            if camera_id not in self._buffers:
                self._buffers[camera_id] = VideoStreamBuffer(maxsize=buffer_size)
                self._raw_buffers[camera_id] = VideoStreamBuffer(maxsize=buffer_size)
                logger.info(f"Registered stream buffers for {camera_id}")

    def unregister_camera(self, camera_id: str):
        """注销摄像头缓冲区"""
        with self._lock:
            if camera_id in self._buffers:
                del self._buffers[camera_id]
            if camera_id in self._raw_buffers:
                del self._raw_buffers[camera_id]
            logger.info(f"Unregistered stream buffers for {camera_id}")

    def update_frame(self, camera_id: str, frame: np.ndarray, raw: bool = False):
        """更新摄像头帧
        Args:
            raw: True 更新原始帧缓冲区，False 更新标注帧缓冲区
        """
        with self._lock:
            if raw:
                if camera_id in self._raw_buffers:
                    self._raw_buffers[camera_id].put(frame, quality=self._quality)
            else:
                if camera_id in self._buffers:
                    self._buffers[camera_id].put(frame, quality=self._quality)

    def _placeholder_frame(self) -> bytes:
        """生成黑屏占位帧，避免无帧时 HTTP 连接挂起导致浏览器并发槽耗尽"""
        if not hasattr(self, '_placeholder_jpeg'):
            black = np.zeros((360, 640, 3), dtype=np.uint8)
            _, jpeg = cv2.imencode('.jpg', black, [cv2.IMWRITE_JPEG_QUALITY, 60])
            self._placeholder_jpeg = jpeg.tobytes() if _ else b''
        return (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                self._placeholder_jpeg + b'\r\n')

    def generate_frames(self, camera_id: str, raw: bool = False):
        """
        生成 MJPEG 帧流
        只在帧内容变化时才推送，避免重复帧和固定帧率不匹配导致的卡顿
        无帧时推送黑屏占位帧，防止浏览器连接挂起阻塞其他 API
        Args:
            raw: True 输出原始帧流，False 输出标注帧流
        """
        last_seq = -1
        placeholder_sent = False

        while True:
            with self._lock:
                buffer = self._raw_buffers.get(camera_id) if raw else self._buffers.get(camera_id)

            if buffer is None:
                # 缓冲区不存在（摄像头未注册），发送一次占位帧后慢轮询
                if not placeholder_sent:
                    yield self._placeholder_frame()
                    placeholder_sent = True
                time.sleep(0.1)
                continue

            seq, jpeg = buffer.get_latest_jpeg_with_seq()
            if jpeg is not None and seq != last_seq:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       jpeg + b'\r\n')
                last_seq = seq
                placeholder_sent = False
            elif jpeg is None and not placeholder_sent:
                # 缓冲区存在但尚无帧（摄像头连接中或读帧失败），发送一次占位帧
                yield self._placeholder_frame()
                placeholder_sent = True

            # 短轮询，降低 CPU 占用同时保证低延迟
            time.sleep(0.02)
    
    def generate_multi_view(self, camera_ids: list, layout: str = 'grid'):
        """
        生成多画面合流
        
        Args:
            camera_ids: 摄像头ID列表
            layout: 布局方式 'grid' | 'horizontal' | 'vertical'
        """
        frame_interval = 1.0 / 15  # 多画面降低帧率
        
        while True:
            start_time = time.time()
            
            frames = []
            for cid in camera_ids:
                with self._lock:
                    buffer = self._buffers.get(cid)
                if buffer:
                    frame = buffer.get_latest()
                    if frame is not None:
                        frames.append((cid, frame))
            
            if frames:
                # 创建合流画面
                combined = self._create_combined_frame(frames, layout)
                
                if combined is not None:
                    _, jpeg = cv2.imencode('.jpg', combined,
                        [cv2.IMWRITE_JPEG_QUALITY, 75])
                    
                    if _:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               jpeg.tobytes() + b'\r\n')
            
            # 控制帧率
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _create_combined_frame(self, frames: list, layout: str) -> Optional[np.ndarray]:
        """创建合流画面"""
        if not frames:
            return None
        
        n = len(frames)
        
        # 统一大小
        target_size = (320, 240)  # 每个画面大小
        resized = []
        for cid, frame in frames:
            if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
                r = cv2.resize(frame, target_size)
            else:
                r = frame
            resized.append(r)
        
        if layout == 'grid':
            # 网格布局
            if n == 1:
                return resized[0]
            elif n == 2:
                return np.hstack(resized)
            elif n <= 4:
                rows = 2
                cols = 2
            elif n <= 6:
                rows = 2
                cols = 3
            else:
                rows = 3
                cols = 3
            
            # 填充空白
            while len(resized) < rows * cols:
                resized.append(np.zeros((*target_size[::-1], 3), dtype=np.uint8))
            
            # 拼接
            grid_rows = []
            for i in range(rows):
                row = np.hstack(resized[i*cols:(i+1)*cols])
                grid_rows.append(row)
            
            return np.vstack(grid_rows)
        
        elif layout == 'horizontal':
            return np.hstack(resized)
        
        elif layout == 'vertical':
            return np.vstack(resized)
        
        return resized[0]


# 全局流服务器实例
_stream_server = None
_lock = threading.Lock()


def get_stream_server() -> MJPEGStreamServer:
    """获取全局流服务器实例（单例）"""
    global _stream_server
    with _lock:
        if _stream_server is None:
            _stream_server = MJPEGStreamServer()
        return _stream_server
