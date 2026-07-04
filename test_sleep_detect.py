"""睡岗检测单图测试脚本

Usage:
    conda run -n py312 python test_sleep_detect.py /path/to/image.jpg
"""
import sys
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))
from safety_detection.sleep_detect import process_frame

MODEL_PATH = Path(__file__).parent / "weights" / "yolov8s-pose.pt"


def test_image(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    print(f"\n{'=' * 60}")
    print(f"Image: {image_path}")
    print(f"Size: {img.shape[1]}x{img.shape[0]}")
    print(f"{'=' * 60}")

    model = YOLO(str(MODEL_PATH))
    results = process_frame(model, img, conf=0.1)

    if not results:
        print("No person detected.")
        return

    for i, r in enumerate(results):
        info = r["_info"]
        print(f"\nPerson #{i + 1}:")
        print(f"  Box: {[round(x, 1) for x in r['box']]}")
        print(f"  Score: {r['score']:.3f}")
        print(f"  Is Sleeping: {info['is_sleeping']}  (confidence: {info['sleep_confidence']:.3f})")
        print(f"  Reason: {info['sleep_reason']}")
        print(f"  Posture: {info['posture_label']}  (confidence: {info['posture_confidence']:.3f})")
        print(f"  Aspect Ratio: {info['aspect_ratio']:.3f}")
        print(f"  Head Below Shoulder: {info['head_below_shoulder']}")
        print(f"  Head Drop: {info['head_drop']:.3f}")
        print(f"  Head Hidden: {info['head_hidden']}")
        print(f"  Shoulder Sym: {info['shoulder_sym']:.3f}")
        print(f"  Hip Sym: {info['hip_sym']:.3f}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_sleep_detect.py <image1> [<image2> ...]")
        sys.exit(1)

    for p in sys.argv[1:]:
        test_image(p)
