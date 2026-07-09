"""
显示模块专用检测工作进程

为选中摄像头的显示模块提供独立进程推理，避免 CPU/NPU 推理占用 GIL 阻塞前端推流。
"""

import logging
import multiprocessing as mp
import queue
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _worker_loop(input_queue: mp.Queue, output_queue: mp.Queue, npu_cores: int, device: str):
    """子进程主循环：加载模型并等待检测任务"""
    # 子进程里重新初始化日志，避免继承父进程的 handler 状态
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    worker_logger = logging.getLogger(__name__)

    # 在子进程里导入模型，避免序列化已加载的模型
    from inference_engine import SafetyDetector

    detector = SafetyDetector(npu_cores=npu_cores, device=device)
    worker_logger.info(f"Display detection worker started (device={device}, npu_cores={npu_cores})")

    while True:
        try:
            item = input_queue.get()
        except Exception as e:
            worker_logger.error(f"Worker input queue error: {e}")
            continue

        if item is None:
            break

        frame, types_to_detect = item
        try:
            detector.ensure_models_loaded(types_to_detect)
            results = detector.detect(frame, types_to_detect)
            output_queue.put(results)
        except Exception as e:
            worker_logger.error(f"Display detection worker error: {e}")
            output_queue.put({})

    worker_logger.info("Display detection worker stopped")


class DisplayDetectionWorker:
    """独立进程检测器封装"""

    def __init__(self, npu_cores: int, device: str):
        self._input_queue: mp.Queue = mp.Queue(maxsize=2)
        self._output_queue: mp.Queue = mp.Queue(maxsize=2)
        self._process = mp.Process(
            target=_worker_loop,
            args=(self._input_queue, self._output_queue, npu_cores, device),
            daemon=True,
        )
        self._process.start()

    def detect(
        self,
        frame: np.ndarray,
        types_to_detect: List[str],
        timeout: float = 10.0,
    ) -> Optional[Dict]:
        """向子进程发送一帧，返回检测结果"""
        try:
            self._input_queue.put((frame, types_to_detect), timeout=1.0)
            return self._output_queue.get(timeout=timeout)
        except Exception as e:
            logger.error(f"DisplayDetectionWorker detect failed: {e}")
            return None

    def submit(self, frame: np.ndarray, types_to_detect: List[str]) -> bool:
        """非阻塞提交一帧检测任务。队列满时返回 False，调用方应跳过本次。"""
        try:
            self._input_queue.put_nowait((frame, types_to_detect))
            return True
        except queue.Full:
            return False
        except Exception as e:
            logger.error(f"DisplayDetectionWorker submit failed: {e}")
            return False

    def get_result(self, timeout: float = 0.1) -> Optional[Dict]:
        """非阻塞读取一次检测结果，无结果时返回 None。"""
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(f"DisplayDetectionWorker get_result failed: {e}")
            return None

    def stop(self):
        """停止子进程"""
        try:
            self._input_queue.put(None, timeout=1.0)
        except Exception:
            pass
        self._process.join(timeout=5.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
