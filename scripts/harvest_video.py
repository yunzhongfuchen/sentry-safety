"""离线扫描视频文件：1Hz 抽帧跑姿态模型，dump 关键点 JSONL + 定期标注帧

用法: python scripts/harvest_video.py data/simulator_uploads/睡岗6.mp4
输出: data/harvest_<视频名>.jsonl, data/hv_<视频名>_<秒>.jpg
"""

import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "weights" / "yolov8s-pose_1.pt"


def main():
    src = Path(sys.argv[1])
    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    step = max(int(fps), 1)  # 1Hz
    out_path = ROOT / "data" / f"harvest_{src.stem}.jsonl"
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if n % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                ts = n / fps
                r = model.predict(frame, conf=0.1, iou=0.5, verbose=False)[0]
                persons = []
                vis = frame.copy()
                for i in range(len(r.boxes)):
                    bbox = r.boxes.xyxy[i].cpu().numpy()
                    kp = r.keypoints.data[i].cpu().numpy()
                    persons.append({
                        "box": [round(float(v), 1) for v in bbox],
                        "conf": round(float(r.boxes.conf[i]), 3),
                        "kp": [[round(float(x), 1), round(float(y), 1), round(float(c), 3)] for x, y, c in kp],
                    })
                    x1, y1, x2, y2 = bbox.astype(int)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    for x, y, c in kp[:11]:
                        if c > 0.4:
                            cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)
                f.write(json.dumps({"ts": round(ts, 2), "persons": persons}) + "\n")
                if int(ts) % 30 == 0:
                    cv2.imwrite(str(ROOT / "data" / f"hv_{src.stem}_{int(ts):04d}.jpg"), vis)
            n += 1
    cap.release()
    print(f"done: {n} frames ({n/fps:.0f}s) -> {out_path}")


if __name__ == "__main__":
    main()
