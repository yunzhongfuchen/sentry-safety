"""生成睡岗检测测试报告（带可视化标注）

Usage:
    conda run -n py312 python generate_sleep_test_report.py
"""
import sys
import cv2
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent / "backend"))
from safety_detection.sleep_detect import process_frame

MODEL_PATH = Path(__file__).parent / "weights" / "yolov8s-pose.pt"
FRAMES_DIR = Path(__file__).parent / "data" / "frames"
OUTPUT_DIR = Path(__file__).parent / "test_results" / "sleep_detect"

# 颜色
COLOR_GREEN = (0, 255, 0)      # 正确检测
COLOR_RED = (0, 0, 255)        # 误报
COLOR_GRAY = (128, 128, 128)   # 未触发

TEST_SETS = {
    "sleep_tagged": sorted(Path(FRAMES_DIR).glob("*_sleep_*.jpg"))[:20],
    "mask_tagged": sorted(Path(FRAMES_DIR).glob("*_mask_*.jpg"))[:20],
    "camera216_random": sorted(Path(FRAMES_DIR).glob("摄像头 216_*.jpg"))[:50],
}


def draw_result(img, results):
    h, w = img.shape[:2]
    for i, r in enumerate(results):
        x1, y1, x2, y2 = map(int, r["box"])
        info = r["_info"]
        is_sleep = info["is_sleeping"]
        color = COLOR_RED if is_sleep else COLOR_GRAY
        label = f"#{i+1} {'SLEEP' if is_sleep else 'normal'} AR={info['aspect_ratio']:.2f} drop={info['head_drop']:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # 画关键点
        kp = r.get("keypoints")
        if kp is not None:
            for j, (kx, ky, kc) in enumerate(kp):
                if kc > 0.4:
                    cv2.circle(img, (int(kx), int(ky)), 2, (255, 255, 0), -1)
    return img


def run_tests():
    model = YOLO(str(MODEL_PATH))
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("睡岗检测算法测试报告")
    report_lines.append("=" * 70)
    report_lines.append("")

    overall = {"total_images": 0, "total_people": 0, "sleep_detections": 0, "false_positives": 0}

    for set_name, paths in TEST_SETS.items():
        report_lines.append(f"\n{'=' * 70}")
        report_lines.append(f"测试集: {set_name} ({len(paths)} 张图片)")
        report_lines.append(f"{'=' * 70}")

        stats = defaultdict(int)
        set_out = OUTPUT_DIR / set_name
        set_out.mkdir(parents=True, exist_ok=True)

        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            results = process_frame(model, img, conf=0.1)

            stats["total_images"] += 1
            stats["total_people"] += len(results)

            vis_img = img.copy()
            draw_result(vis_img, results)
            out_path = set_out / p.name
            cv2.imwrite(str(out_path), vis_img)

            for r in results:
                info = r["_info"]
                if info["is_sleeping"]:
                    stats["sleep_detections"] += 1
                    # 在 mask/sleep 标记集里，_sleep_ 标签不一定真睡，_mask_ 标签应该不睡
                    # 这里简单标记：camera216_random 中 sleeping 视为误报，sleep_tagged 中视为可能正确
                    if set_name == "camera216_random":
                        stats["false_positives"] += 1
                        tag = "[误报]"
                    else:
                        tag = ""
                    report_lines.append(f"  {p.name}: {tag} AR={info['aspect_ratio']:.3f} drop={info['head_drop']:.3f} reason={info['sleep_reason']}")

        report_lines.append(f"\n  统计: {stats['total_images']} 张图, {stats['total_people']} 人检出, {stats['sleep_detections']} 次 sleeping 触发")
        if set_name == "camera216_random":
            report_lines.append(f"  误报数: {stats['false_positives']} ({stats['false_positives']/max(stats['total_people'],1)*100:.1f}%)")

        overall["total_images"] += stats["total_images"]
        overall["total_people"] += stats["total_people"]
        overall["sleep_detections"] += stats["sleep_detections"]
        overall["false_positives"] += stats.get("false_positives", 0)

    report_lines.append(f"\n{'=' * 70}")
    report_lines.append("总体统计")
    report_lines.append(f"{'=' * 70}")
    report_lines.append(f"总图片数: {overall['total_images']}")
    report_lines.append(f"总检出人数: {overall['total_people']}")
    report_lines.append(f"总 sleeping 触发: {overall['sleep_detections']}")
    report_lines.append(f"camera216 误报: {overall['false_positives']} ({overall['false_positives']/max(overall['total_people'],1)*100:.1f}%)")
    report_lines.append("")

    report_path = OUTPUT_DIR / "report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"报告已保存: {report_path}")
    print(f"可视化图片: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_tests()
