"""睡岗检测单帧测试脚本"""
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from inference_engine import SafetyDetector

def main():
    video_path = "/home/yangrunfu/project/detection/sentry-rk3588-v1.0.0/uploads/videos/20260428_141630_sleep.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    detector = SafetyDetector(npu_cores=0)
    detector.ensure_models_loaded(["sleep"], use_npu=False)

    print(f"已加载模型: {detector.loaded_models}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector._detect_sleep(frame)
        print(f"帧 {frame_idx}: detected={result['detected']}, boxes={len(result['boxes'])}, count={result['count']}")
        for i, subj in enumerate(result.get("subjects", [])):
            print(f"  目标{i}: sleeping={subj.get('sleeping')}, posture={subj.get('posture')}, conf={subj.get('sleep_confidence', 0):.2f}")

        # 保存第10帧的标注图用于查看
        if frame_idx == 10 and result["boxes"]:
            from safety_detection.detector_core import MultiDetector
            annotated = MultiDetector._annotate_frame(frame, {"sleep": result}, "test", ["sleep"])
            out_path = "/tmp/sleep_test_frame.jpg"
            cv2.imwrite(out_path, annotated)
            print(f"已保存标注图: {out_path}")
            break

        frame_idx += 1
        if frame_idx > 50:
            break

    cap.release()
    detector.release()
    print("测试完成")

if __name__ == "__main__":
    main()
