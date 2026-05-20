"""
多模型批处理压测：所有流独立解码，每秒收集最新帧，依次串行跑多个 YOLO 模型。

用法:
    python benchmark_multi_model.py --source test.mp4 --streams 15 --duration 30 \
        --models yolov8s.pt,yolov8s.pt,yolov8s.pt,yolov8s.pt,yolov8s.pt
"""

import argparse
import signal
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


class MultiModelInferer(threading.Thread):
    def __init__(self, benches: List[StreamBench], detectors: List[PersonDetector], interval: float):
        super().__init__(daemon=True)
        self.benches = benches
        self.detectors = detectors
        self.interval = interval
        self.running = True
        self.infer_count = 0
        self.total_frames_inferred = 0
        # 每个模型的耗时统计
        self.model_times = [0.0 for _ in detectors]
        self._lock = threading.Lock()

    def run(self):
        while self.running:
            t0 = time.time()
            frames = []
            for b in self.benches:
                f = b.snapshot()
                if f is not None:
                    frames.append(f)

            if frames:
                batch_size = len(frames)
                for idx, detector in enumerate(self.detectors):
                    infer_t0 = time.time()
                    try:
                        detector.model(
                            frames,
                            conf=detector.confidence,
                            classes=[0],
                            device=detector.device,
                            verbose=False,
                        )
                    except Exception as e:
                        print(f"[model-{idx}] 推理出错: {e}")
                    elapsed = time.time() - infer_t0
                    with self._lock:
                        self.model_times[idx] += elapsed

                with self._lock:
                    self.infer_count += 1
                    self.total_frames_inferred += batch_size

            sleep_time = self.interval - (time.time() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        self.running = False
        self.join(timeout=2.0)

    def get_model_avg_ms(self) -> List[float]:
        with self._lock:
            return [
                (t / self.infer_count * 1000) if self.infer_count else 0.0
                for t in self.model_times
            ]

    @property
    def avg_batch_size(self) -> float:
        with self._lock:
            return (
                self.total_frames_inferred / self.infer_count if self.infer_count else 0.0
            )


def main():
    parser = argparse.ArgumentParser(description="多模型批处理压测")
    parser.add_argument("--source", required=True, help="视频文件路径或 RTSP 地址")
    parser.add_argument("--streams", type=int, default=4, help="并发路数（默认 4）")
    parser.add_argument("--duration", type=int, default=30, help="压测时长秒数（默认 30）")
    parser.add_argument("--interval", type=int, default=5, help="状态打印间隔秒数（默认 5）")
    parser.add_argument(
        "--infer-interval", type=float, default=1.0, help="Batch 推理间隔秒数（默认 1.0）"
    )
    parser.add_argument(
        "--fps-limit", type=float, default=30.0, help="每路解码帧率上限（默认 30.0）"
    )
    parser.add_argument(
        "--models",
        default="yolov8n.pt",
        help="YOLO 模型路径，逗号分隔；若数量不足，自动用最后一个补齐（默认 yolov8n.pt）",
    )
    parser.add_argument(
        "--num-models", type=int, default=5, help="模型数量（默认 5）"
    )
    parser.add_argument(
        "--device", default="cuda", help="推理设备：cuda / cpu / 0（默认 cuda）"
    )
    args = parser.parse_args()

    model_paths = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_paths:
        print("错误：至少指定一个模型")
        sys.exit(1)
    # 如果模型路径数量不足，自动用最后一个补齐
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

    benches: List[StreamBench] = []
    stop_event = threading.Event()

    def on_signal(*_):
        print("\n收到中断信号，正在停止...")
        stop_event.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    print(
        f"开始压测：source={args.source}, streams={args.streams}, "
        f"models={len(detectors)}, duration={args.duration}s, "
        f"batch_interval={args.infer_interval}s, fps_limit={args.fps_limit}"
    )
    print("提示：另开终端执行  nvidia-smi dmon -s pucm  观察 GPU 占用\n")

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

    inferer = MultiModelInferer(benches, detectors, args.infer_interval)
    inferer.start()

    start = time.time()
    last_print = start

    while not stop_event.is_set():
        time.sleep(0.5)
        now = time.time()
        if now - last_print >= args.interval:
            elapsed = now - start
            total_frames = sum(b.frames for b in benches)
            model_avgs = inferer.get_model_avg_ms()
            print(f"\n--- [{elapsed:5.1f}s] 汇总 ---")
            print(
                f"总计: 解码帧={total_frames}, 解码FPS={total_frames / elapsed:.2f}, "
                f"Batch次数={inferer.infer_count}, 总推理帧={inferer.total_frames_inferred}, "
                f"平均Batch={inferer.avg_batch_size:.1f}"
            )
            for idx, avg_ms in enumerate(model_avgs):
                print(f"  模型{idx}: {avg_ms:.1f} ms/次")
            total_infer_ms = sum(model_avgs)
            print(f"  五模型合计: {total_infer_ms:.1f} ms/轮")
            last_print = now

        if now - start >= args.duration:
            print(f"\n达到设定时长 {args.duration}s，自动停止...")
            break

    print("\n========== 最终报告 ==========")
    inferer.stop()
    for b in benches:
        b.stop()
    elapsed = time.time() - start
    total_frames = sum(b.frames for b in benches)
    model_avgs = inferer.get_model_avg_ms()

    print(f"压测时长:      {elapsed:.1f}s")
    print(f"并发路数:      {len(benches)}")
    print(f"模型数量:      {len(detectors)}")
    print(f"总解码帧数:    {total_frames}")
    print(f"总解码 FPS:    {total_frames / elapsed:.2f}")
    print(f"平均每路解码:  {total_frames / elapsed / len(benches):.2f} FPS")
    print(f"Batch推理次数: {inferer.infer_count}")
    print(f"总推理帧数:    {inferer.total_frames_inferred}")
    print(f"平均Batch大小: {inferer.avg_batch_size:.1f}")
    print()
    for idx, avg_ms in enumerate(model_avgs):
        print(f"模型{idx}平均延迟: {avg_ms:.1f} ms")
    print(f"五模型串行合计: {sum(model_avgs):.1f} ms/轮")
    print("==============================")


if __name__ == "__main__":
    main()
