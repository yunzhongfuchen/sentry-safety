"""
VLM 异步队列
双优先级队列（P0 > P1），信号量控制最大并发
"""

import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VLMQueue:
    """
    VLM 任务队列
    - P0 队列（fire/smoke 复核） maxlen=50
    - P1 队列（mask/cigarette/uniform/sleep 确认） maxlen=100
    - 单消费者线程 + Semaphore(max_concurrent=3)
    """

    def __init__(self, understander, max_concurrent: int = 3):
        self.understander = understander
        self.semaphore = threading.Semaphore(max_concurrent)
        self.p0_queue: deque = deque(maxlen=50)
        self.p1_queue: deque = deque(maxlen=100)
        self._running = False
        self._consumer_thread: Optional[threading.Thread] = None
        self._stats = {"p0_submitted": 0, "p1_submitted": 0, "completed": 0, "failed": 0}
        self._stats_lock = threading.Lock()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._consumer_thread = threading.Thread(
            target=self._consume,
            daemon=True,
            name="vlm-consumer",
        )
        self._consumer_thread.start()
        logger.info("VLMQueue started")

    def stop(self) -> None:
        self._running = False
        if self._consumer_thread:
            self._consumer_thread.join(timeout=5)
        logger.info("VLMQueue stopped")

    def submit(self, task: Dict[str, Any]) -> None:
        """
        提交 VLM 任务
        task 格式：
        {
            "task_id": str,
            "camera_id": str,
            "dtype": str,
            "level": "P0" | "P1",
            "frames": List[np.ndarray],
            "prompt_type": str,
            "extra_context": Optional[dict],
            "callback": Callable[[dict], None],
        }
        """
        level = task.get("level", "P1")
        with self._stats_lock:
            if level == "P0":
                self.p0_queue.append(task)
                self._stats["p0_submitted"] += 1
                logger.info(f"P0 task submitted: {task['camera_id']} {task['dtype']}")
            else:
                self.p1_queue.append(task)
                self._stats["p1_submitted"] += 1
                logger.info(f"P1 task submitted: {task['camera_id']} {task['dtype']}")

    def _consume(self) -> None:
        while self._running:
            task = None
            # 优先取 P0
            if self.p0_queue:
                task = self.p0_queue.popleft()
            elif self.p1_queue:
                task = self.p1_queue.popleft()
            else:
                time.sleep(0.1)
                continue

            # 获取信号量，在新线程中执行
            self.semaphore.acquire()
            threading.Thread(
                target=self._run_vlm,
                args=(task,),
                daemon=True,
                name=f"vlm-task-{task.get('task_id', 'unknown')}",
            ).start()

    def _run_vlm(self, task: Dict[str, Any]) -> None:
        try:
            prompt_type = task.get("prompt_type", "review")
            frames = task.get("frames", [])
            extra = task.get("extra_context")

            if not frames:
                result = {"error": "No frames provided"}
            else:
                # 调用 understander 的多图分析接口
                result = self.understander.analyze_multi(
                    frames=frames,
                    prompt_type=prompt_type,
                    extra_context=extra,
                )

            callback = task.get("callback")
            if callback:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"VLM callback error: {e}")

            with self._stats_lock:
                self._stats["completed"] += 1

        except Exception as e:
            logger.error(f"VLM task failed: {e}")
            callback = task.get("callback")
            if callback:
                try:
                    callback({"error": str(e)})
                except Exception:
                    pass
            with self._stats_lock:
                self._stats["failed"] += 1
        finally:
            self.semaphore.release()

    def get_stats(self) -> Dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    @property
    def queue_depth(self) -> Dict[str, int]:
        return {
            "p0": len(self.p0_queue),
            "p1": len(self.p1_queue),
        }
