"""大规模睡岗检测批量测试

对 data/frames 中各类标签图片抽样检测，统计召回率和误报率。

Usage:
    conda run -n py312 python batch_test_sleep_large.py
"""
import sys
from pathlib import Path
from collections import defaultdict
import random

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent / "backend"))
from safety_detection.sleep_detect import process_frame

MODEL_PATH = Path(__file__).parent / "weights" / "yolov8s-pose.pt"
FRAMES_DIR = Path(__file__).parent / "data" / "frames"
OUTPUT_DIR = Path(__file__).parent / "test_results" / "sleep_large"

# 各类别抽样数量
SAMPLE_LIMITS = {
    "_sleep_": 200,
    "_mask_": 200,
    "_cigarette_": 100,
    "_fire_": 100,
    "camera216": 200,  # 摄像头216的真实监控帧
}


def sample_files(pattern, limit):
    files = sorted(FRAMES_DIR.glob(f"*{pattern}*.jpg"))
    if len(files) > limit:
        random.seed(42)
        files = random.sample(files, limit)
    return files


def run_batch():
    model = YOLO(str(MODEL_PATH))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "sleep_tagged": sample_files("_sleep_", SAMPLE_LIMITS["_sleep_"]),
        "mask_tagged": sample_files("_mask_", SAMPLE_LIMITS["_mask_"]),
        "cigarette_tagged": sample_files("_cigarette_", SAMPLE_LIMITS["_cigarette_"]),
        "fire_tagged": sample_files("_fire_", SAMPLE_LIMITS["_fire_"]),
        "camera216_real": sample_files("摄像头 216_", SAMPLE_LIMITS["camera216"]),
    }

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("大规模睡岗检测批量测试报告")
    report_lines.append("=" * 70)
    report_lines.append("")

    overall = {"total_images": 0, "total_people": 0, "sleep_detections": 0}

    for set_name, files in datasets.items():
        stats = defaultdict(int)
        sleep_samples = []
        no_sleep_samples = []

        print(f"\nProcessing {set_name}: {len(files)} images...")

        for i, p in enumerate(files):
            img = cv2.imread(str(p))
            if img is None:
                continue
            results = process_frame(model, img, conf=0.1)

            stats["total_images"] += 1
            stats["total_people"] += len(results)

            sleeping_count = sum(1 for r in results if r["_info"]["is_sleeping"])
            if sleeping_count:
                stats["sleep_detections"] += sleeping_count
                if len(sleep_samples) < 5:
                    sleep_samples.append((p.name, sleeping_count, len(results)))
            else:
                if len(no_sleep_samples) < 3 and results:
                    no_sleep_samples.append((p.name, len(results)))

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(files)} done...")

        report_lines.append(f"{'=' * 70}")
        report_lines.append(f"数据集: {set_name} ({stats['total_images']} 张)")
        report_lines.append(f"{'=' * 70}")
        report_lines.append(f"  检出人数: {stats['total_people']}")
        report_lines.append(f"  sleeping 触发人次: {stats['sleep_detections']}")
        report_lines.append(f"  触发率: {stats['sleep_detections'] / max(stats['total_people'], 1) * 100:.1f}%")
        if sleep_samples:
            report_lines.append(f"  触发示例:")
            for name, s, t in sleep_samples:
                report_lines.append(f"    {name}: {s}/{t} sleeping")
        if no_sleep_samples:
            report_lines.append(f"  未触发示例:")
            for name, t in no_sleep_samples:
                report_lines.append(f"    {name}: {t} people, no sleeping")
        report_lines.append("")

        overall["total_images"] += stats["total_images"]
        overall["total_people"] += stats["total_people"]
        overall["sleep_detections"] += stats["sleep_detections"]

    report_lines.append(f"{'=' * 70}")
    report_lines.append("总体统计")
    report_lines.append(f"{'=' * 70}")
    report_lines.append(f"总图片数: {overall['total_images']}")
    report_lines.append(f"总检出人数: {overall['total_people']}")
    report_lines.append(f"总 sleeping 触发: {overall['sleep_detections']}")
    report_lines.append(f"整体触发率: {overall['sleep_detections'] / max(overall['total_people'], 1) * 100:.1f}%")
    report_lines.append("")

    report_path = OUTPUT_DIR / "batch_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    run_batch()
