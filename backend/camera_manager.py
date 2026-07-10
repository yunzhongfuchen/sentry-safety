"""
多摄像头管理模块 - 支持 RK3588 边缘端多路视频流管理
"""

import logging
import os
import queue
import threading
import time
from typing import Dict, List, Optional, Callable, Tuple
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

# RTSP 默认使用 TCP 传输，避免 WSL2 / 某些网络环境下 UDP 丢包导致超时
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import numpy as np

import gpu_decoder
from decode_scheduler import DecodeScheduler

logger = logging.getLogger(__name__)


class CameraStatus(Enum):
    """摄像头状态"""
    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass
class CameraConfig:
    """摄像头配置"""
    camera_id: str
    source: str  # 摄像头索引、RTSP地址或视频流URL
    name: str = ""  # 摄像头显示名称
    enabled: bool = True
    # 视频流配置
    width: int = 640
    height: int = 480
    fps: int = 15
    buffer_size: int = 1
    # 视频源类型
    source_type: str = "auto"  # "camera" | "rtsp" | "auto"
    # 重连配置
    reconnect_interval: int = 5
    max_reconnect_attempts: int = 0  # 0 = 无限重连（生产环境摄像头检修/网络抖动常见）
    # 检测配置
    detection_enabled: bool = True
    detection_types: Optional[dict] = None  # 各检测类型配置
    detection_roi: Optional[List[int]] = None  # [x, y, w, h] 检测区域
    # NPU配置 (RK3588)
    use_npu: bool = False
    npu_core: int = 0  # 0-2, RK3588有3个NPU核心


@dataclass
class CameraState:
    """摄像头运行时状态"""
    config: CameraConfig
    status: CameraStatus = CameraStatus.IDLE
    cap: Optional[cv2.VideoCapture] = None
    current_frame: Optional[np.ndarray] = None
    last_frame_time: float = 0
    frame_count: int = 0
    error_count: int = 0
    reconnect_attempts: int = 0
    thread: Optional[threading.Thread] = None
    running: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    # 按需解码调度状态
    last_decode_time: float = 0.0
    decode_queued: bool = False
    # 统计信息
    fps_stats: List[float] = field(default_factory=list)
    # 时间窗口帧历史 (用于触发时保存前5秒帧)
    frame_history: deque = field(default_factory=lambda: deque(maxlen=10))
    # 录制状态
    recording: bool = False
    record_writer: Optional[cv2.VideoWriter] = None
    record_path: Optional[str] = None
    record_lock: threading.Lock = field(default_factory=threading.Lock)
    record_queue: Optional[queue.Queue] = None
    record_thread: Optional[threading.Thread] = None
    # 预览录制（低分辨率 VP8，供浏览器播放）
    record_preview_writer: Optional[cv2.VideoWriter] = None
    record_preview_path: Optional[str] = None
    record_preview_queue: Optional[queue.Queue] = None
    record_preview_thread: Optional[threading.Thread] = None

    def get_avg_fps(self) -> float:
        """计算平均FPS"""
        if not self.fps_stats:
            return 0.0
        return sum(self.fps_stats) / len(self.fps_stats)


class CameraManager:
    """
    多摄像头管理器
    支持多路视频流的统一管理、自动重连、状态监控
    """
    
    def __init__(self):
        self._cameras: Dict[str, CameraState] = {}
        self._lock = threading.RLock()
        self._global_frame_callback: Optional[Callable[[str, np.ndarray], None]] = None
        self._main_camera_id: Optional[str] = None
        self.decode_scheduler = DecodeScheduler(self, num_workers=4)
    def register_camera(self, config: CameraConfig) -> bool:
        """注册摄像头"""
        with self._lock:
            if config.camera_id in self._cameras:
                logger.warning(f"Camera {config.camera_id} already registered")
                return False
            
            state = CameraState(config=config)
            self._cameras[config.camera_id] = state
            logger.info(f"Camera {config.camera_id} registered: {config.source}")
            return True
    
    def unregister_camera(self, camera_id: str) -> bool:
        """注销摄像头"""
        with self._lock:
            if camera_id not in self._cameras:
                return False

        # 在锁外停止线程，避免 join 阻塞其他需要锁的操作（如 get_frame）
        self.stop_camera(camera_id)

        with self._lock:
            self._cameras.pop(camera_id, None)
            if self._main_camera_id == camera_id:
                self._main_camera_id = None
                self.decode_scheduler.set_main_camera(None)
        logger.info(f"Camera {camera_id} unregistered")
        return True
    
    def start_camera(self, camera_id: str) -> bool:
        """启动指定摄像头"""
        with self._lock:
            if camera_id not in self._cameras:
                logger.error(f"Camera {camera_id} not found")
                return False

            state = self._cameras[camera_id]
            if state.running:
                logger.warning(f"Camera {camera_id} already running")
                return True

            state.running = True
            state.error_count = 0
            state.reconnect_attempts = 0
            state.last_decode_time = 0.0
            state.frame_count = 0
            state.last_frame_count = 0
            state.last_fps_time = 0.0
            state.fps_stats.clear()

            # 启动 DecodeScheduler（首次启动时）
            if not self.decode_scheduler._running.is_set():
                self.decode_scheduler.start()

        # 在锁外打开视频源
        def _open_and_retry():
            try:
                self._open_capture(camera_id)
            except Exception as e:
                logger.error(f"Camera {camera_id} open failed: {e}")
                # 重连逻辑交给 DecodeScheduler 的 decode 失败计数

        threading.Thread(target=_open_and_retry, daemon=True, name=f"open-{camera_id}").start()

        logger.info(f"Camera {camera_id} started")
        return True
    
    def stop_camera(self, camera_id: str) -> bool:
        """停止指定摄像头"""
        with self._lock:
            if camera_id not in self._cameras:
                return False

            state = self._cameras[camera_id]
            state.running = False

        # 给 DecodeScheduler worker 留出时间完成当前 cap.read()
        time.sleep(0.15)

        with self._lock:
            if camera_id not in self._cameras:
                return True

            state = self._cameras[camera_id]
            with state.lock:
                if state.cap:
                    state.cap.release()
                    state.cap = None

            state.status = CameraStatus.IDLE
            logger.info(f"Camera {camera_id} stopped")
            return True
    
    def start_all(self):
        """启动所有已启用的摄像头"""
        with self._lock:
            for camera_id, state in self._cameras.items():
                if state.config.enabled:
                    self.start_camera(camera_id)
    
    def stop_all(self):
        """停止所有摄像头"""
        self.decode_scheduler.stop()
        with self._lock:
            camera_ids = list(self._cameras.keys())
        for camera_id in camera_ids:
            self.stop_camera(camera_id)
    
    def get_frame(self, camera_id: str, allow_paused: bool = True) -> Optional[np.ndarray]:
        """获取指定摄像头的当前帧。"""
        with self._lock:
            if camera_id not in self._cameras:
                return None

            state = self._cameras[camera_id]
            with state.lock:
                return state.current_frame.copy() if state.current_frame is not None else None

    def get_latest_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """获取指定摄像头的最新帧"""
        return self.get_frame(camera_id, allow_paused=True)

    def set_main_camera(self, camera_id: Optional[str]) -> bool:
        """设置主画面摄像头"""
        with self._lock:
            if camera_id is not None and camera_id not in self._cameras:
                logger.warning(f"Cannot set main camera {camera_id}: not found")
                self._main_camera_id = None
                self.decode_scheduler.set_main_camera(None)
                return False

            self._main_camera_id = camera_id
            self.decode_scheduler.set_main_camera(camera_id)
            logger.info(f"Main camera set to {camera_id}")
            return True

    def get_main_camera(self) -> Optional[str]:
        """获取当前主画面摄像头 ID"""
        with self._lock:
            return self._main_camera_id

    def get_window_frames(self, camera_id: str, start_time: float, end_time: float) -> List[Tuple[float, np.ndarray]]:
        """
        获取指定摄像头在某个时间窗口内的历史帧
        返回 [(timestamp, frame), ...] 按时间升序排列
        """
        with self._lock:
            if camera_id not in self._cameras:
                return []

            state = self._cameras[camera_id]
            with state.lock:
                return [
                    (ts, fr.copy()) for ts, fr in state.frame_history
                    if start_time <= ts <= end_time
                ]
    
    def get_all_frames(self) -> Dict[str, np.ndarray]:
        """获取所有摄像头的当前帧"""
        frames = {}
        with self._lock:
            for camera_id, state in self._cameras.items():
                if state.status == CameraStatus.CONNECTED:
                    with state.lock:
                        if state.current_frame is not None:
                            frames[camera_id] = state.current_frame.copy()
        return frames
    
    def get_camera_status(self, camera_id: str) -> Optional[dict]:
        """获取摄像头状态信息"""
        with self._lock:
            if camera_id not in self._cameras:
                return None

            state = self._cameras[camera_id]
            # 判断解码后端
            decoder_backend = "cpu"
            if state.cap is not None:
                if hasattr(state.cap, "backend"):
                    decoder_backend = state.cap.backend
                elif hasattr(state.cap, "isOpened"):
                    decoder_backend = "cpu"

            # 状态修正：cap 已打开但还没读到第一帧时，实际仍是连接中
            actual_status = state.status.value
            if state.status == CameraStatus.CONNECTED and state.current_frame is None:
                actual_status = CameraStatus.CONNECTING.value

            return {
                "camera_id": camera_id,
                "name": state.config.name,
                "source": state.config.source,
                "status": actual_status,
                "enabled": state.config.enabled,
                "fps": state.get_avg_fps(),
                "frame_count": state.frame_count,
                "error_count": state.error_count,
                "reconnect_attempts": state.reconnect_attempts,
                "decoder_backend": decoder_backend,
            }
    
    def get_all_status(self) -> List[dict]:
        """获取所有摄像头状态"""
        with self._lock:
            return [
                self.get_camera_status(camera_id)
                for camera_id in self._cameras.keys()
            ]
    
    def set_frame_callback(self, callback: Callable[[str, np.ndarray], None]):
        """设置全局帧回调函数"""
        self._global_frame_callback = callback

    def set_camera_source(self, camera_id: str, source: str, source_type: str = "auto") -> bool:
        """动态切换视频源"""
        with self._lock:
            if camera_id not in self._cameras:
                return False
            state = self._cameras[camera_id]
            was_running = state.running

        # 在锁外停止摄像头，避免 stop_camera 的 join 与旧线程抢锁
        self.stop_camera(camera_id)

        with self._lock:
            if camera_id not in self._cameras:
                return False
            state = self._cameras[camera_id]
            state.config.source = source
            state.config.source_type = source_type
            logger.info(f"Camera {camera_id} source switched to {source} (type={source_type})")

        if was_running:
            self.start_camera(camera_id)
        return True

    def get_camera_ids(self) -> List[str]:
        """获取所有摄像头ID列表"""
        with self._lock:
            return list(self._cameras.keys())

    def start_recording(self, camera_id: str, output_dir: Optional[str] = None) -> Optional[str]:
        """开始录制原始视频帧，返回高清录制文件路径。
        同时生成一份 640px 限制的 VP8 预览视频，供浏览器直接播放。"""
        with self._lock:
            if camera_id not in self._cameras:
                return None
            state = self._cameras[camera_id]

        if state.recording:
            return state.record_path

        # 默认输出目录
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "data", "recordings")
        os.makedirs(output_dir, exist_ok=True)
        preview_dir = os.path.join(output_dir, "previews")
        os.makedirs(preview_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 获取当前 cap 的分辨率与帧率
        cap = state.cap
        if cap is None or not cap.isOpened():
            logger.warning(f"Camera {camera_id} not connected, cannot start recording")
            return None

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        # ---- 高清轨道：mp4v + 原始分辨率（存档/下载） ----
        record_path = os.path.join(output_dir, f"{camera_id}_{timestamp}.mp4")
        hd_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        hd_writer = cv2.VideoWriter(record_path, hd_fourcc, fps, (w, h))
        if not hd_writer.isOpened():
            logger.error(f"Failed to open HD VideoWriter for {record_path}")
            return None

        # ---- 预览轨道：VP8 + 最长边 640（浏览器播放） ----
        max_edge = max(w, h)
        if max_edge > 640:
            scale = 640 / max_edge
            pw = int(w * scale) // 2 * 2
            ph = int(h * scale) // 2 * 2
        else:
            pw, ph = w, h
        preview_path = os.path.join(preview_dir, f"{camera_id}_{timestamp}.webm")
        sd_fourcc = cv2.VideoWriter_fourcc(*"VP80")
        sd_writer = cv2.VideoWriter(preview_path, sd_fourcc, fps, (pw, ph))
        if not sd_writer.isOpened():
            logger.error(f"Failed to open preview VideoWriter for {preview_path}, falling back to HD only")
            sd_writer = None

        # 创建帧队列和后台写入线程
        hd_queue: queue.Queue = queue.Queue(maxsize=120)
        sd_queue: queue.Queue = queue.Queue(maxsize=120)

        def _hd_writer_loop():
            dropped = 0
            written = 0
            try:
                while True:
                    try:
                        item = hd_queue.get(timeout=0.5)
                    except queue.Empty:
                        if not state.recording:
                            break
                        continue
                    if item is None:
                        break
                    try:
                        hd_writer.write(item)
                        written += 1
                    except Exception as e:
                        logger.error(f"Camera {camera_id} HD write error: {e}")
                        dropped += 1
            finally:
                try:
                    hd_writer.release()
                except Exception:
                    pass
                logger.info(f"Camera {camera_id} HD writer done: written={written}")

        def _sd_writer_loop():
            dropped = 0
            written = 0
            try:
                while True:
                    try:
                        item = sd_queue.get(timeout=0.5)
                    except queue.Empty:
                        if not state.recording:
                            break
                        continue
                    if item is None:
                        break
                    try:
                        if sd_writer is not None:
                            # 动态缩放到预览分辨率
                            if item.shape[1] != pw or item.shape[0] != ph:
                                item = cv2.resize(item, (pw, ph))
                            sd_writer.write(item)
                            written += 1
                    except Exception as e:
                        logger.error(f"Camera {camera_id} preview write error: {e}")
                        dropped += 1
            finally:
                if sd_writer is not None:
                    try:
                        sd_writer.release()
                    except Exception:
                        pass
                logger.info(f"Camera {camera_id} preview writer done: written={written}")

        hd_thread = threading.Thread(target=_hd_writer_loop, name=f"rec-hd-{camera_id}", daemon=True)
        sd_thread = threading.Thread(target=_sd_writer_loop, name=f"rec-sd-{camera_id}", daemon=True)

        with state.record_lock:
            state.recording = True
            state.record_writer = hd_writer
            state.record_path = record_path
            state.record_queue = hd_queue
            state.record_thread = hd_thread
            state.record_preview_writer = sd_writer
            state.record_preview_path = preview_path
            state.record_preview_queue = sd_queue
            state.record_preview_thread = sd_thread

        hd_thread.start()
        sd_thread.start()
        logger.info(f"Camera {camera_id} recording started: HD={record_path} ({w}x{h}), Preview={preview_path} ({pw}x{ph})")
        return record_path

    def stop_recording(self, camera_id: str) -> Optional[str]:
        """停止录制，返回录制文件路径。不阻塞等待后台线程，避免卡住事件循环。"""
        with self._lock:
            if camera_id not in self._cameras:
                return None
            state = self._cameras[camera_id]

        if not state.recording:
            return None

        with state.record_lock:
            path = state.record_path
            frame_queue = state.record_queue
            writer_thread = state.record_thread
            preview_queue = state.record_preview_queue
            preview_thread = state.record_preview_thread
            state.recording = False
            state.record_writer = None
            state.record_path = None
            state.record_queue = None
            state.record_thread = None
            state.record_preview_writer = None
            state.record_preview_path = None
            state.record_preview_queue = None
            state.record_preview_thread = None

        # 发送停止信号，后台线程自行完成写入和释放
        if frame_queue is not None:
            try:
                frame_queue.put_nowait(None)
            except queue.Full:
                pass
        if preview_queue is not None:
            try:
                preview_queue.put_nowait(None)
            except queue.Full:
                pass
        # 极短超时：给已空队列的快速收尾一个同步窗口，不阻塞主循环
        if writer_thread is not None and writer_thread.is_alive():
            writer_thread.join(timeout=0.2)
        if preview_thread is not None and preview_thread.is_alive():
            preview_thread.join(timeout=0.2)

        logger.info(f"Camera {camera_id} recording stopped: {path}")
        return path

    def get_recording_status(self, camera_id: str) -> Optional[dict]:
        """获取录制状态"""
        with self._lock:
            if camera_id not in self._cameras:
                return None
            state = self._cameras[camera_id]
        return {
            "recording": state.recording,
            "record_path": state.record_path,
        }

    def _open_capture(self, camera_id: str):
        """打开视频源并初始化 cap"""
        with self._lock:
            if camera_id not in self._cameras:
                return
            state = self._cameras[camera_id]
            state.status = CameraStatus.CONNECTING

        source = state.config.source
        if source.isdigit():
            source = int(source)

        source_str = str(source)
        is_network = source_str.startswith(("http://", "https://", "rtsp://"))
        is_rtsp = source_str.startswith("rtsp://")
        cap = None

        # RTSP 流优先尝试 GPU 硬件解码
        if is_rtsp and gpu_decoder.gpu_available():
            gpu_reader = gpu_decoder.GPUVideoReader(str(source))
            if gpu_reader.start():
                cap = gpu_reader
                logger.info(f"Camera {camera_id} using GPU decoder")

        # 回退到 OpenCV CPU 解码：网络流强制用 FFMPEG，本地摄像头可回退
        # 网络流的 VideoCapture() 构造会在 FFMPEG 层阻塞 15-30s，环境变量 stimeout 在此版本无效。
        # 用独立线程 + Event 给构造过程设置 3s 上限，超时直接放弃本次连接。
        if cap is None:
            if is_network:
                result_holder = [None]
                done = threading.Event()

                def _open_in_thread():
                    try:
                        result_holder[0] = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                    except Exception:
                        pass
                    finally:
                        done.set()

                t = threading.Thread(target=_open_in_thread, daemon=True)
                t.start()
                if not done.wait(timeout=5.0):
                    raise RuntimeError(f"Failed to open video source: {source} (connect timeout)")
                cap = result_holder[0]
            else:
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, state.config.buffer_size)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, state.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, state.config.height)
        cap.set(cv2.CAP_PROP_FPS, state.config.fps)

        with self._lock:
            state.cap = cap
            state.status = CameraStatus.CONNECTED
            state.reconnect_attempts = 0

        logger.info(f"Camera {camera_id} connected")

    def _reopen_capture(self, camera_id: str):
        """重新打开视频源"""
        with self._lock:
            if camera_id not in self._cameras:
                return
            state = self._cameras[camera_id]
            state.status = CameraStatus.RECONNECTING
            state.reconnect_attempts += 1

            max_attempts = state.config.max_reconnect_attempts
            if max_attempts > 0 and state.reconnect_attempts > max_attempts:
                logger.error(f"Camera {camera_id} max reconnect attempts reached")
                state.status = CameraStatus.ERROR
                state.running = False
                if state.cap:
                    try:
                        state.cap.release()
                    except Exception:
                        pass
                    state.cap = None
                return

            if state.cap:
                try:
                    state.cap.release()
                except Exception:
                    pass
                state.cap = None

        try:
            self._open_capture(camera_id)
        except Exception as e:
            logger.error(f"Camera {camera_id} reopen failed: {e}")


class CameraConfigLoader:
    """摄像头配置加载器"""
    
    @staticmethod
    def from_env() -> List[CameraConfig]:
        """从环境变量加载摄像头配置"""
        import os
        configs = []
        
        # 解析 CAMERAS 环境变量 (格式: id:source:id2:source2...)
        cameras_env = os.getenv("CAMERAS", "")
        if cameras_env:
            parts = cameras_env.split(":")
            for i in range(0, len(parts), 2):
                if i + 1 < len(parts):
                    camera_id = parts[i]
                    source = parts[i + 1]
                    configs.append(CameraConfig(
                        camera_id=camera_id,
                        source=source,
                        name=f"Camera {camera_id}",
                        enabled=True
                    ))
        
        # 如果没有配置，使用默认摄像头
        if not configs:
            video_source = os.getenv("VIDEO_SOURCE", "0")
            configs.append(CameraConfig(
                camera_id="cam_0",
                source=video_source,
                name="Default Camera",
                enabled=True
            ))
        
        return configs
    
    @staticmethod
    def from_dict(data: List[dict]) -> List[CameraConfig]:
        """从字典列表加载配置"""
        configs = []
        for item in data:
            roi = item.get("detection_roi")
            if roi:
                roi = [int(x) for x in roi]

            configs.append(CameraConfig(
                camera_id=item["camera_id"],
                source=item["source"],
                name=item.get("name", ""),
                enabled=item.get("enabled", True),
                width=item.get("width", 640),
                height=item.get("height", 480),
                fps=item.get("fps", 15),
                source_type=item.get("source_type", "auto"),
                detection_enabled=item.get("detection_enabled", True),
                detection_types=item.get("detection_types"),
                detection_roi=roi,
                use_npu=item.get("use_npu", False),
                npu_core=item.get("npu_core", 0),
            ))
        return configs
