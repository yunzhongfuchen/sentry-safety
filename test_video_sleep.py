"""对视频文件进行睡岗检测测试，输出带标注的视频和报告

Usage:
    conda run -n py312 python test_video_sleep.py ./video/sleep1.mp4
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent / "backend"))
from safety_detection.sleep_detect import process_frame

MODEL_PATH = Path(__file__).parent / "weights" / "yolov8s-pose.pt"
OUTPUT_DIR = Path(__file__).parent / "test_results" / "sleep_video"

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)


def draw_result(frame, results):
    h, w = frame.shape[:2]
    for i, r in enumerate(results):
        x1, y1, x2, y2 = map(int, r["box"])
        info = r["_info"]
        is_sleep = info["is_sleeping"]
        color = COLOR_RED if is_sleep else COLOR_GREEN
        label = f"#{i+1} {'SLEEP' if is_sleep else 'normal'} AR={info['aspect_ratio']:.2f} drop={info['head_drop']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        kp = r.get("keypoints")
        if kp is not None:
            for j, (kx, ky, kc) in enumerate(kp):
                if kc > 0.4:
                    cv2.circle(frame, (int(kx), int(ky)), 2, (255, 255, 0), -1)
    return frame


def process_video(video_path: str):
    video_path = Path(video_path)
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = out_dir / f"{video_path.stem}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    model = YOLO(str(MODEL_PATH))
    stats = defaultdict(int)
    frame_idx = 0
    sample_frames = []

    print(f"\nProcessing: {video_path.name} ({w}x{h}, {fps:.1f}fps, {total} frames)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = process_frame(model, frame, conf=0.1)
        annotated = frame.copy()
        draw_result(annotated, results)
        writer.write(annotated)

        sleeping_count = sum(1 for r in results if r["_info"]["is_sleeping"])
        stats["total_frames"] += 1
        stats["total_people"] += len(results)
        if sleeping_count:
            stats["sleep_frames"] += 1
            stats["total_sleep_detections"] += sleeping_count

        # 保存有 sleeping 的帧作为缩略图样本
        if sleeping_count and len(sample_frames) < 5:
            sample_path = out_dir / f"{video_path.stem}_sample_{len(sample_frames)}.jpg"
            cv2.imwrite(str(sample_path), annotated)
            sample_frames.append(frame_idx)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  {frame_idx}/{total} frames processed...")

    cap.release()
    writer.release()

    print(f"\n{'='*50}")
    print(f"Video: {video_path.name}")
    print(f"Output: {out_path}")
    print(f"Total frames: {stats['total_frames']}")
    print(f"Total people detected: {stats['total_people']}")
    print(f"Frames with sleeping: {stats['sleep_frames']}")
    print(f"Total sleeping detections: {stats['total_sleep_detections']}")
    print(f"{'='*50}")

    return stats


if __name__ == "__main__":
    video_dir = Path(__file__).parent / "video"
    video_files = sorted(video_dir.glob("*.mp4"))

    if not video_files:
        print("No .mp4 files found in ./video/")
        sys.exit(1)

    all_stats = []
    for vf in video_files:
        stats = process_video(str(vf))
        stats["video"] = vf.name
        all_stats.append(stats)

    # 汇总报告
    report_path = OUTPUT_DIR / "video_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("睡岗检测视频测试报告\n")
        f.write("=" * 50 + "\n\n")
        for s in all_stats:
            f.write(f"视频: {s['video']}\n")
            f.write(f"  总帧数: {s['total_frames']}\n")
            f.write(f"  检出人数: {s['total_people']}\n")
            f.write(f"  睡岗帧数: {s['sleep_frames']}\n")
            f.write(f"  睡岗人次: {s['total_sleep_detections']}\n")
            f.write("\n")

    print(f"\n汇总报告已保存: {report_path}")
