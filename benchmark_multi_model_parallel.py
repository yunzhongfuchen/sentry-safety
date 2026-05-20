"""
多模型队列式压测：所有流独立解码，每秒收集最新帧。
模型分成 N 个队列，队列之间并行，队列内部串行。

用法:
    # 5 个模型分 2 个队列（队列内串行，队列间并行）
    python benchmark_multi_model_parallel.py --source test.mp4 --streams 15 --duration 30 \
        --models yolov8s.pt --num-models 5 --num-queues 2

    # 5 个模型 5 个队列 = 纯并行
    python benchmark_multi_model_parallel.py --source test.mp4 --streams 15 --duration 30 \
        --models yolov8s.pt --num-models 5 --num-queues 5

    # 5 个模型 1 个队列 = 纯串行
    python benchmark_multi_model_parallel.py --source test.mp4 --streams 15 --duration 30 \
        --models yolov8s.pt --num-models 5 --num-queues 1
"""

import argparse
import signal
import numpy as np
import sys
import threading
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from detector import PersonDetector
from gpu_decoder import GPUVideoReader, gpu_available


class StreamBench:
    def __init__(self, source: str, stream_id: int, fps_limit: float = 30.0):
        self.source = source
        self.stream_id = stream_id
        self.fps_limit = fps_limit
        self.reader: GPUVideoReader | None = None
        self.frames = 0
        self.running = False
        self.thread: threading.Thread | None = None
        self.start_time = 0.0

    def start(self) -> bool:
        self.reader = GPUVideoReader(self.source)
        if not self.reader.start():
            print(f"[stream-{self.stream_id}] 启动失败")
            return False
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"[stream-{self.stream_id}] 已启动，后端={self.reader.backend}")
        return True

    def _loop(self):
        target_interval = 1.0 / self.fps_limit if self.fps_limit > 0 else 0.0
        while self.running:
            loop_start = time.time()
            ret, frame = self.reader.read()
            if ret and frame is not None:
                self.frames += 1
            elapsed = time.time() - loop_start
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def snapshot(self):
        if self.reader:
            return self.reader.snapshot()
        return None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.reader:
            self.reader.stop()

    @property
    def fps(self) -> float:
        elapsed = time.time() - self.start_time
        return self.frames / elapsed if elapsed > 0 else 0.0

    @property
    def backend(self) -> str:
        return self.reader.backend if self.reader else "none"


class QueueWorker(threading.Thread):
    """推理队列：内部串行执行多个模型，与其他队列并行"""

    def __init__(self, queue_id: int, detectors: List[PersonDetector], half: bool):
        super().__init__(daemon=True)
        self.queue_id = queue_id
        self.detectors = detectors
        self.half = half
        self.frames = None
        self.start_event = threading.Event()
        self.done_event = threading.Event()
        self.running = True
        self.model_times = [0.0 for _ in detectors]
        self.infer_count = 0
        self.total_infer_time = 0.0
        self._lock = threading.Lock()

    def run(self):
        while self.running:
            self.start_event.wait()
            self.start_event.clear()
            if not self.running:
                break

            queue_t0 = time.time()
            for idx, detector in enumerate(self.detectors):
                t0 = time.time()
                try:
                    detector.model(
                        self.frames,
                        conf=detector.confidence,
                        classes=[0],
                        device=detector.device,
                        verbose=False,
                        half=self.half,
                    )
                except Exception as e:
                    print(f"[队列{self.queue_id}-模型{idx}] 推理出错: {e}")
                elapsed = time.time() - t0
                with self._lock:
                    self.model_times[idx] += elapsed

            with self._lock:
                self.infer_count += 1
                self.total_infer_time += time.time() - queue_t0

            self.done_event.set()

    def set_frames(self, frames):
        self.frames = frames
        self.start_event.set()

    def stop(self):
        self.running = False
        self.start_event.set()
        self.join(timeout=5.0)

    def reset_stats(self):
        """清空统计，用于 dummy 预热后归零"""
        with self._lock:
            self.infer_count = 0
            self.total_infer_time = 0.0
            self.model_times = [0.0 for _ in self.detectors]

    def get_model_avg_ms(self) -> List[float]:
        with self._lock:
            return [
                (t / self.infer_count * 1000) if self.infer_count else 0.0
                for t in self.model_times
            ]

    @property
    def avg_queue_ms(self) -> float:
        with self._lock:
            return (self.total_infer_time / self.infer_count * 1000) if self.infer_count else 0.0


def main():
    parser = argparse.ArgumentParser(description="多模型队列式批处理压测")
    parser.add_argument("--source", required=True, help="视频文件路径或 RTSP 地址")
    parser.add_argument("--streams", type=int, default=4, help="并发路数（默认 4）")
    parser.add_argument("--duration", type=int, default=30, help="压测时长秒数（默认 30）")
    parser.add_argument("--interval", type=int, default=5, help="状态打印间隔秒数（默认 5）")
    parser.add_argument(
        "--infer-interval", type=float, default=1.0, help="Batch 推理间隔秒数（默认 1.0）"
    )
    parser.add_argument(
        "--fps-limit", type=float, default=10.0, help="每路解码帧率上限（默认 30.0）"
    )
    parser.add_argument(
        "--models",
        default="yolov8s.pt",
        help="YOLO 模型路径，逗号分隔；若数量不足，自动用最后一个补齐（默认 yolov8s.pt）",
    )
    parser.add_argument(
        "--num-models", type=int, default=5, help="模型数量（默认 5）"
    )
    parser.add_argument(
        "--num-queues", type=int, default=2, help="并行队列数量（默认 2）"
    )
    parser.add_argument(
        "--device", default="cuda", help="推理设备：cuda / cpu / 0（默认 cuda）"
    )
    parser.add_argument(
        "--half", action="store_true", help="启用 FP16 半精度推理（降低显存占用）"
    )
    parser.add_argument(
        "--warmup", action="store_true", help="启用 dummy 预热（消除 CUDA 首次分配开销）"
    )
    parser.add_argument(
        "--pad-batch", action="store_true", help="启用固定 batch size（不足时用黑图填充）"
    )
    args = parser.parse_args()

    if args.num_queues < 1:
        print("错误：队列数至少为 1")
        sys.exit(1)
    if args.num_queues > args.num_models:
        print(f"警告：队列数({args.num_queues}) > 模型数({args.num_models})，自动调整为 {args.num_models}")
        args.num_queues = args.num_models

    model_paths = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_paths:
        print("错误：至少指定一个模型")
        sys.exit(1)
    while len(model_paths) < args.num_models:
        model_paths.append(model_paths[-1])

    if not gpu_available():
        print("警告：未检测到 NVIDIA GPU，硬件解码不会生效")

    detectors: List[PersonDetector] = []
    for idx, path in enumerate(model_paths):
        print(f"加载模型 {idx}: {path} ...")
        d = PersonDetector()
        d.model_path = path
        d.device = args.device
        d.load_model()
        detectors.append(d)
        print(f"  -> 完成，设备: {d.device}")
    print()

    # 将模型均匀分配到各队列
    queue_size = (args.num_models + args.num_queues - 1) // args.num_queues
    workers: List[QueueWorker] = []
    for q in range(args.num_queues):
        start = q * queue_size
        end = min(start + queue_size, args.num_models)
        queue_detectors = detectors[start:end]
        w = QueueWorker(q, queue_detectors, args.half)
        w.start()
        workers.append(w)
        print(f"队列{q}: 模型{start}-{end-1} ({len(queue_detectors)}个模型)")
    print()

    benches: List[StreamBench] = []
    stop_event = threading.Event()

    def on_signal(*_):
        print("\n收到中断信号，正在停止...")
        stop_event.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    print(
        f"开始压测：source={args.source}, streams={args.streams}, "
        f"models={args.num_models}, queues={args.num_queues}, duration={args.duration}s, "
        f"batch_interval={args.infer_interval}s, fps_limit={args.fps_limit}, half={args.half}"
    )
    print("提示：另开终端执行  watch -n 0.5 nvidia-smi  观察显存占用\n")

    for i in range(args.streams):
        b = StreamBench(args.source, i + 1, fps_limit=args.fps_limit)
        if b.start():
            benches.append(b)
        else:
            print(f"[stream-{i + 1}] 启动失败，跳过后续路数")
            break
        time.sleep(0.2)

    if not benches:
        print("没有一路启动成功，退出")
        sys.exit(1)

    # dummy 预热：固定 batch size 跑一轮，把显存池和 CUDA 缓存热透
    if args.warmup:
        target_batch = len(benches)
        print(f"\n正在 dummy 预热，batch_size={target_batch} ...")
        template = None
        for _ in range(100):
            template = benches[0].snapshot() if benches else None
            if template is not None:
                break
            time.sleep(0.05)
        if template is not None:
            dummy = [template.copy() for _ in range(target_batch)]
        else:
            dummy = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(target_batch)]

        for w in workers:
            w.set_frames(dummy)
        for w in workers:
            w.done_event.wait()
            w.done_event.clear()
            w.reset_stats()
        print("预热完成，开始正式压测\n")

    infer_count = 0
    total_frames_inferred = 0
    start = time.time()
    last_print = start

    while not stop_event.is_set():
        time.sleep(0.5)
        now = time.time()
        if now - last_print >= args.interval:
            elapsed = now - start
            total_frames = sum(b.frames for b in benches)
            print(f"\n--- [{elapsed:5.1f}s] 汇总 ---")
            print(
                f"总计: 解码帧={total_frames}, 解码FPS={total_frames / elapsed:.2f}, "
                f"Batch次数={infer_count}, 总推理帧={total_frames_inferred}"
            )
            for w in workers:
                model_avgs = w.get_model_avg_ms()
                queue_ms = w.avg_queue_ms
                print(f"  队列{w.queue_id}: {queue_ms:.1f} ms/轮")
                for idx, avg_ms in enumerate(model_avgs):
                    global_idx = w.queue_id * queue_size + idx
                    print(f"    模型{global_idx}: {avg_ms:.1f} ms/次")
            queue_avgs = [w.avg_queue_ms for w in workers]
            print(f"  瓶颈队列: {max(queue_avgs):.1f} ms (总延迟 ≈ 最慢队列)")
            last_print = now

        if now - start >= args.duration:
            print(f"\n达到设定时长 {args.duration}s，自动停止...")
            break

        # 推理调度
        t0 = time.time()
        frames = []
        for b in benches:
            f = b.snapshot()
            if f is not None:
                frames.append(f)

        if frames:
            # 固定 batch size，不足用黑图填充
            if args.pad_batch:
                target_batch = len(benches)
                while len(frames) < target_batch:
                    frames.append(np.zeros_like(frames[0]))

            for w in workers:
                w.set_frames(frames)
            for w in workers:
                w.done_event.wait()
                w.done_event.clear()

            infer_count += 1
            total_frames_inferred += len(frames)

        sleep_time = args.infer_interval - (time.time() - t0)
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("\n========== 最终报告 ==========")
    for w in workers:
        w.stop()
    for b in benches:
        b.stop()
    elapsed = time.time() - start
    total_frames = sum(b.frames for b in benches)

    print(f"压测时长:      {elapsed:.1f}s")
    print(f"并发路数:      {len(benches)}")
    print(f"模型数量:      {args.num_models}")
    print(f"队列数量:      {args.num_queues}")
    print(f"半精度推理:    {args.half}")
    print(f"总解码帧数:    {total_frames}")
    print(f"总解码 FPS:    {total_frames / elapsed:.2f}")
    print(f"平均每路解码:  {total_frames / elapsed / len(benches):.2f} FPS")
    print(f"Batch推理次数: {infer_count}")
    print(f"总推理帧数:    {total_frames_inferred}")
    print()
    queue_avgs = []
    for w in workers:
        model_avgs = w.get_model_avg_ms()
        queue_ms = w.avg_queue_ms
        queue_avgs.append(queue_ms)
        print(f"队列{w.queue_id}: {queue_ms:.1f} ms/轮")
        for idx, avg_ms in enumerate(model_avgs):
            global_idx = w.queue_id * queue_size + idx
            print(f"  模型{global_idx}: {avg_ms:.1f} ms/次")
    print(f"\n瓶颈队列:      {max(queue_avgs):.1f} ms/轮")
    print(f"串行参考:      {sum(queue_avgs):.1f} ms/轮")
    print("==============================")


if __name__ == "__main__":
    main()
