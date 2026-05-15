"""测试摄像头1实际使用的视频文件"""
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from inference_engine import SafetyDetector
from safety_detection.detector_core import MultiDetector

def main():
    video_path = "/home/yangrunfu/project/detection/sentry-rk3588-v1.0.0/uploads/videos/20260428_154318_sleep.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    detector = SafetyDetector(npu_cores=0)
    detector.ensure_models_loaded(["sleep"], use_npu=False)
    print(f"已加载模型: {detector.loaded_models}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total_frames}")

    frame_idx = 0
    found_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector._detect_sleep(frame)
        boxes = len(result.get("boxes", []))
        subjects = result.get("subjects", [])
        sleeping = sum(1 for s in subjects if s.get("sleeping"))

        if frame_idx % 10 == 0 or boxes > 0:
            print(f"帧 {frame_idx}: boxes={boxes}, sleeping={sleeping}, detected={result['detected']}")
            for i, subj in enumerate(subjects):
                print(f"  目标{i}: sleeping={subj.get('sleeping')}, posture={subj.get('posture')}, conf={subj.get('sleep_confidence', 0):.2f}")

        if boxes > 0 and found_frame is None:
            found_frame = frame.copy()
            found_result = result

        frame_idx += 1
        if frame_idx > 200:
            break

    cap.release()
    detector.release()

    if found_frame is not None:
        print(f"\n找到带检测目标的帧，保存标注图...")
        annotated = MultiDetector._annotate_frame(found_frame, {"sleep": found_result}, "1", ["sleep"])
        out_path = "/tmp/sleep_cam1_test.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"已保存: {out_path}")
    else:
        print("\n前200帧均未检测到人体")

if __name__ == "__main__":
    main()
