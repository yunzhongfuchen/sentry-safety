"""
GPU 硬件视频解码器 (NVIDIA NVDEC via PyAV + CUVID)
在 NVIDIA 显卡可用且视频编码支持时自动启用，失败时回退到 CPU。
"""

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_av = None


def _import_av():
    """懒加载 PyAV，避免无 GPU 环境强制依赖。"""
    global _av
    if _av is None:
        try:
            import av as _av_module

            _av = _av_module
        except ImportError:
            _av = False
    return _av


# NVIDIA CUVID 解码器映射（FFmpeg 内置）
_CUVID_CODECS = {
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "mpeg2video": "mpeg2_cuvid",
    "mpeg4": "mpeg4_cuvid",
    "vp8": "vp8_cuvid",
    "vp9": "vp9_cuvid",
}


def gpu_available() -> bool:
    """检测 NVIDIA GPU 是否可用（优先 torch.cuda，其次 pynvml）。"""
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except ImportError:
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetCount() > 0
    except Exception:
        pass
    return False


def _try_cuvid_context(stream) -> Optional:
    """尝试为视频流创建 CUVID 硬件解码上下文。"""
    av = _import_av()
    if not av:
        return None

    orig_codec = stream.codec_context.codec.name
    cuvid_name = _CUVID_CODECS.get(orig_codec)
    if not cuvid_name:
        logger.debug(f"codec {orig_codec} has no CUVID mapping")
        return None

    try:
        codec = av.Codec(cuvid_name, "r")
        ctx = av.codec.CodecContext.create(codec)

        orig = stream.codec_context
        if orig.extradata:
            ctx.extradata = orig.extradata
        ctx.width = orig.width
        ctx.height = orig.height
        ctx.pix_fmt = orig.pix_fmt
        ctx.open()

        logger.info(f"CUVID decoder created: {cuvid_name} ({orig.width}x{orig.height})")
        return ctx
    except Exception as e:
        logger.warning(f"CUVID init failed: {e}")
        return None


class GPUVideoReader:
    """
    NVIDIA GPU 硬件解码视频读取器。
    接口与 cv2.VideoCapture / RTSPReader 兼容，支持 duck-typing 替换。

    支持的输入：RTSP / HTTP / 本地视频文件（H.264/H.265 等）
    不支持的输入：本地摄像头索引（如 0、1）
    """

    def __init__(self, source: str):
        self.source = str(source)
        self.container = None
        self.stream = None
        self.ctx = None
        self._use_gpu = False

        self._frame: Optional[np.ndarray] = None
        self._ret = False
        self._lock = threading.Lock()
        self.running = False
        self._thread: Optional[threading.Thread] = None

        # 缓存属性，兼容 cv2.VideoCapture.get()
        self._width = 0
        self._height = 0
        self._fps = 0.0

    def start(self) -> bool:
        """打开视频源并启动后台解码线程。"""
        av = _import_av()
        if not av:
            logger.debug("PyAV not installed, skip GPU reader")
            return False

        # PyAV 不支持本地摄像头索引
        if self.source.isdigit():
            return False

        try:
            options = {"rtsp_transport": "tcp"} if self.source.startswith("rtsp") else {}
            self.container = av.open(self.source, options=options)
            self.stream = self.container.streams.video[0]

            # 缓存流属性
            self._width = self.stream.codec_context.width or 0
            self._height = self.stream.codec_context.height or 0
            fps = self.stream.codec_context.framerate
            self._fps = float(fps) if fps else 0.0

            # 尝试 GPU 解码
            if gpu_available():
                cuvid_ctx = _try_cuvid_context(self.stream)
                if cuvid_ctx:
                    self.ctx = cuvid_ctx
                    self._use_gpu = True

            # 回退到 PyAV 默认解码器（CPU，仍通过 FFmpeg）
            if self.ctx is None:
                self.ctx = self.stream.codec_context
                logger.info("Using CPU decoder via PyAV/FFmpeg")

            self.running = True
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()
            return True

        except Exception as e:
            logger.error(f"Failed to open {self.source}: {e}")
            self._cleanup()
            return False

    def _reader_loop(self):
        """后台线程：持续 demux -> decode -> 转 BGR -> 缓存最新帧。"""
        try:
            for packet in self.container.demux(self.stream):
                if not self.running:
                    break

                frames = self.ctx.decode(packet)
                for frame in frames:
                    # 硬件/软件解码输出多为 nv12/yuv420p，统一 reformat 到 rgb24
                    rgb = frame.reformat(format="rgb24")
                    arr = rgb.to_ndarray()
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

                    with self._lock:
                        self._ret = True
                        self._frame = bgr
        except Exception as e:
            logger.error(f"GPU reader loop error: {e}")
            with self._lock:
                self._ret = False

    def read(self):
        """返回 (ret, frame) 元组，与 cv2.VideoCapture.read() 兼容。"""
        with self._lock:
            return self._ret, self._frame.copy() if self._frame is not None else None

    def snapshot(self):
        """返回当前最新帧副本，不影响内部状态。"""
        with self._lock:
            if self._ret and self._frame is not None:
                return self._frame.copy()
            return None

    def isOpened(self):
        """与 cv2.VideoCapture.isOpened() 兼容。"""
        return self.container is not None

    @property
    def backend(self) -> str:
        """返回当前使用的解码后端：gpu | cpu"""
        return "gpu" if self._use_gpu else "cpu"

    def is_opened(self):
        """与 RTSPReader.is_opened() 兼容。"""
        return self.isOpened()

    def release(self):
        """与 cv2.VideoCapture.release() 兼容。"""
        self.stop()

    def stop(self):
        """停止解码线程并释放资源。"""
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._cleanup()

    def _cleanup(self):
        if self.container:
            try:
                self.container.close()
            except Exception:
                pass
            self.container = None
        self.stream = None
        self.ctx = None
        with self._lock:
            self._frame = None
            self._ret = False

    def get(self, prop):
        """
        兼容 cv2.VideoCapture.get() 的常用属性。
        未实现属性返回 0.0。
        """
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop == cv2.CAP_PROP_FPS:
            return float(self._fps)
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            try:
                return float(self.stream.frames) if self.stream and self.stream.frames else 0.0
            except Exception:
                return 0.0
        return 0.0

    def set(self, prop, value):
        """
        兼容 cv2.VideoCapture.set()，当前为空实现。
        返回 False 表示未生效（调用方可据此选择是否继续）。
        """
        return False
