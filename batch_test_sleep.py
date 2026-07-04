"""批量测试睡岗检测算法

Usage:
    conda run -n py312 python batch_test_sleep.py data/frames/*_sleep_*.jpg
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent / "backend"))
from safety_detection.sleep_detect import process_frame

MODEL_PATH = Path(__file__).parent / "weights" / "yolov8s-pose.pt"


def batch_test(paths):
    model = YOLO(str(MODEL_PATH))
    stats = defaultdict(int)
    details = []

    for p in sorted(paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        results = process_frame(model, img, conf=0.1)

        sleeping_count = sum(1 for r in results if r["_info"]["is_sleeping"])
        total_people = len(results)

        reasons = [r["_info"]["sleep_reason"] for r in results if r["_info"]["is_sleeping"]]
        ars = [round(r["_info"]["aspect_ratio"], 3) for r in results]
        drops = [round(r["_info"]["head_drop"], 3) for r in results]

        stats["total_images"] += 1
        stats["total_people"] += total_people
        if sleeping_count:
            stats["images_with_sleep"] += 1
            stats["total_sleep_detections"] += sleeping_count

        details.append({
            "file": Path(p).name,
            "people": total_people,
            "sleeping": sleeping_count,
            "reasons": reasons,
            "aspect_ratios": ars,
            "head_drops": drops,
        })

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"Total images: {stats['total_images']}")
    print(f"Total people detected: {stats['total_people']}")
    print(f"Images with sleeping detection: {stats['images_with_sleep']} ({stats['images_with_sleep'] / max(stats['total_images'], 1) * 100:.1f}%)")
    print(f"Total sleeping detections: {stats['total_sleep_detections']}")
    print(f"\n{'=' * 60}")
    print("Details (images with sleeping only)")
    print(f"{'=' * 60}")
    for d in details:
        if d["sleeping"]:
            print(f"  {d['file']}: {d['sleeping']}/{d['people']} sleeping | AR={d['aspect_ratios']} | drop={d['head_drops']} | reasons={d['reasons']}")

    print(f"\n{'=' * 60}")
    print("Details (images with people but NO sleeping)")
    print(f"{'=' * 60}")
    for d in details:
        if d["people"] and not d["sleeping"]:
            print(f"  {d['file']}: {d['people']} people | AR={d['aspect_ratios']} | drop={d['head_drops']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_test_sleep.py <glob or file list>")
        sys.exit(1)
    batch_test(sys.argv[1:])
