"""
完整链路压测：解码 + YOLO 推理。
同时启动 N 路视频流，每路独立解码，共享 YOLO 模型做推理。

用法:
    python benchmark_pipeline.py --source test.mp4 --streams 4 --duration 30 --stride 2

另开终端监控 GPU:
    nvidia-smi dmon -s pucm
    watch -n 2 nvidia-smi
"""

import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List

import numpy as np

# 兼容从项目根目录运行：backend/ 目录加入模块路径
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from detector import PersonDetector
from gpu_decoder import GPUVideoReader, gpu_available


class StreamBench:
    def __init__(self, source: str, stream_id: int, detector: PersonDetector, infer_lock: threading.Lock, infer_interval: float):
        self.source = source
        self.stream_id = stream_id
        self.detector = detector
        self.infer_lock = infer_lock
        self.infer_interval = infer_interval

        self.reader: GPUVideoReader | None = None
        self.frames = 0
        self.inferences = 0
        self.infer_time_total = 0.0
        self.running = False
        self.thread: threading.Thread | None = None
        self.start_time = 0.0
        self._last_infer_time = 0.0

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
        while self.running:
            ret, frame = self.reader.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            self.frames += 1

            # 按时间间隔做推理（默认每秒一次）
            now = time.time()
            if now - self._last_infer_time >= self.infer_interval:
                self._last_infer_time = now
                t0 = time.time()
                with self.infer_lock:
                    self.detector.detect(frame)
                self.infer_time_total += time.time() - t0
                self.inferences += 1

            time.sleep(0.001)

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
    def infer_fps(self) -> float:
        elapsed = time.time() - self.start_time
        return self.inferences / elapsed if elapsed > 0 else 0.0

    @property
    def avg_infer_ms(self) -> float:
        return (self.infer_time_total / self.inferences * 1000) if self.inferences > 0 else 0.0

    @property
    def backend(self) -> str:
        return self.reader.backend if self.reader else "none"


def main():
    parser = argparse.ArgumentParser(description="完整链路压测：解码 + YOLO 推理")
    parser.add_argument("--source", required=True, help="视频文件路径或 RTSP 地址")
    parser.add_argument("--streams", type=int, default=4, help="并发路数（默认 4）")
    parser.add_argument("--duration", type=int, default=30, help="压测时长秒数（默认 30）")
    parser.add_argument("--interval", type=int, default=5, help="状态打印间隔秒数（默认 5）")
    parser.add_argument("--infer-interval", type=float, default=1.0, help="推理间隔秒数（默认 1.0，即每秒推理一次）")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLO 模型路径（默认 yolov8s.pt）")
    parser.add_argument("--device", default="cuda", help="推理设备：cuda / cpu / 0（默认 cuda）")
    args = parser.parse_args()

    if not gpu_available():
        print("警告：未检测到 NVIDIA GPU，硬件解码不会生效")

    print(f"加载模型: {args.model} ...")
    detector = PersonDetector()
    detector.model_path = args.model
    detector.device = args.device
    detector.load_model()
    print(f"模型加载完成，推理设备: {detector.device}\n")

    infer_lock = threading.Lock()
    benches: List[StreamBench] = []
    stop_event = threading.Event()

    def on_signal(*_):
        print("\n收到中断信号，正在停止...")
        stop_event.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    print(f"开始压测：source={args.source}, streams={args.streams}, duration={args.duration}s, infer_interval={args.infer_interval}s")
    print("提示：另开终端执行  nvidia-smi dmon -s pucm  观察 dec/sm/mem 占用\n")

    for i in range(args.streams):
        b = StreamBench(args.source, i + 1, detector, infer_lock, args.infer_interval)
        if b.start():
            benches.append(b)
        else:
            print(f"[stream-{i + 1}] 启动失败，跳过后续路数")
            break
        time.sleep(0.2)

    if not benches:
        print("没有一路启动成功，退出")
        sys.exit(1)

    start = time.time()
    last_print = start

    while not stop_event.is_set():
        time.sleep(0.5)
        now = time.time()
        if now - last_print >= args.interval:
            elapsed = now - start
            total_frames = sum(b.frames for b in benches)
            total_infers = sum(b.inferences for b in benches)
            print(f"\n--- [{elapsed:5.1f}s] 汇总 ---")
            print(f"{'路数':>6} | {'后端':>6} | {'解码帧':>8} | {'解码FPS':>8} | {'推理次':>8} | {'推理FPS':>8} | {'推理ms':>8}")
            for b in benches:
                print(
                    f"{b.stream_id:>6} | {b.backend:>6} | {b.frames:>8} | {b.fps:>8.2f} | "
                    f"{b.inferences:>8} | {b.infer_fps:>8.2f} | {b.avg_infer_ms:>8.1f}"
                )
            print(
                f"总计: 解码帧={total_frames}, 解码FPS={total_frames / elapsed:.2f}, "
                f"推理次={total_infers}, 推理FPS={total_infers / elapsed:.2f}"
            )
            last_print = now

        if now - start >= args.duration:
            print(f"\n达到设定时长 {args.duration}s，自动停止...")
            break

    print("\n========== 最终报告 ==========")
    for b in benches:
        b.stop()
    elapsed = time.time() - start
    total_frames = sum(b.frames for b in benches)
    total_infers = sum(b.inferences for b in benches)
    total_infer_ms = sum(b.infer_time_total for b in benches)

    print(f"压测时长:    {elapsed:.1f}s")
    print(f"并发路数:    {len(benches)}")
    print(f"推理间隔:    {args.infer_interval}s")
    print(f"总解码帧数:  {total_frames}")
    print(f"总解码 FPS:  {total_frames / elapsed:.2f}")
    print(f"总推理次数:  {total_infers}")
    print(f"总推理 FPS:  {total_infers / elapsed:.2f}")
    print(f"平均推理延迟: {total_infer_ms / total_infers * 1000:.1f} ms/次" if total_infers else "N/A")
    print(f"平均每路解码: {total_frames / elapsed / len(benches):.2f} FPS")
    print("==============================")


if __name__ == "__main__":
    main()
