"""
GPU 解码压测脚本。
同时启动 N 路 GPUVideoReader，统计每路实际帧率，观察 NVDEC 瓶颈。

用法:
    python benchmark_decoder.py --source test.mp4 --streams 4 --duration 30
"""

import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List

# 兼容从项目根目录运行：backend/ 目录加入模块路径
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from gpu_decoder import GPUVideoReader, gpu_available


class StreamBench:
    def __init__(self, source: str, stream_id: int):
        self.source = source
        self.stream_id = stream_id
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
        while self.running:
            ret, frame = self.reader.read()
            if ret and frame is not None:
                self.frames += 1
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
    def backend(self) -> str:
        return self.reader.backend if self.reader else "none"


def main():
    parser = argparse.ArgumentParser(description="GPU 解码压测")
    parser.add_argument("--source", required=True, help="视频文件路径或 RTSP 地址")
    parser.add_argument("--streams", type=int, default=4, help="并发路数（默认 4）")
    parser.add_argument("--duration", type=int, default=30, help="压测时长秒数（默认 30）")
    parser.add_argument("--interval", type=int, default=5, help="状态打印间隔秒数（默认 5）")
    args = parser.parse_args()

    if not gpu_available():
        print("警告：未检测到 NVIDIA GPU，硬件解码不会生效")

    benches: List[StreamBench] = []
    stop_event = threading.Event()

    def on_signal(*_):
        print("\n收到中断信号，正在停止...")
        stop_event.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    print(f"开始压测：source={args.source}, streams={args.streams}, duration={args.duration}s")
    print("提示：可另开终端执行  nvidia-smi dmon -s pucm  观察 dec 利用率\n")

    for i in range(args.streams):
        b = StreamBench(args.source, i + 1)
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
            print(f"\n--- [{elapsed:5.1f}s] 汇总 ---")
            print(f"{'路数':>6} | {'后端':>6} | {'帧数':>8} | {'FPS':>6}")
            for b in benches:
                print(f"{b.stream_id:>6} | {b.backend:>6} | {b.frames:>8} | {b.fps:>6.2f}")
            print(f"总计帧数: {total_frames}, 平均总 FPS: {total_frames / elapsed:.2f}")
            last_print = now

        if now - start >= args.duration:
            print(f"\n达到设定时长 {args.duration}s，自动停止...")
            break

    print("\n========== 最终报告 ==========")
    for b in benches:
        b.stop()
    elapsed = time.time() - start
    total_frames = sum(b.frames for b in benches)

    print(f"压测时长: {elapsed:.1f}s")
    print(f"并发路数: {len(benches)}")
    print(f"总帧数:   {total_frames}")
    print(f"总 FPS:   {total_frames / elapsed:.2f}")
    print(f"平均每路 FPS: {total_frames / elapsed / len(benches):.2f}")
    print("==============================")


if __name__ == "__main__":
    main()
