"""利用视频内嵌的红/绿标签框自动标注醒睡样本

红框+「睡岗」文字 = 睡觉，绿框+「工作」文字 = 清醒。
逐帧提取红绿矩形框，与姿态检出的人框 IoU 匹配，输出带标签的 JSONL。

用法: python scripts/harvest_labeled.py data/simulator_uploads/睡岗1.mp4
输出: data/labeled_<视频名>.jsonl  (每条: ts, label, box, conf, kp)
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "weights" / "yolov8s-pose_1.pt"


def imread_cn(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def find_label_boxes(frame):
    """提取红/绿标签文字条（「睡岗」红、「工作状态/工作中」绿），返回 [(label, (cx, cy)), ...]

    文字条特征：宽 60-200px、高 15-60px 的高饱和色块。标签锚点用于最近邻匹配人框。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 60, 80), (12, 255, 255)) | cv2.inRange(hsv, (168, 60, 80), (180, 255, 255))
    green = cv2.inRange(hsv, (45, 60, 80), (90, 255, 255))
    out = []
    for mask, label in ((red, "sleep"), (green, "awake")):
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if 60 <= w <= 200 and 15 <= h <= 60:
                out.append((label, (x + w / 2, y + h / 2)))
    return out


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-6)


def main():
    src = Path(sys.argv[1])
    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    step = max(int(fps // 2), 1)  # 2Hz，标签可能闪烁，多采一些
    out_path = ROOT / "data" / f"labeled_{src.stem}.jsonl"
    n = kept = 0
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
                anchors = find_label_boxes(frame)
                r = model.predict(frame, conf=0.1, iou=0.5, verbose=False)[0]
                for i in range(len(r.boxes)):
                    bbox = [float(v) for v in r.boxes.xyxy[i].cpu().numpy()]
                    # 标签条一般在人框上方约 30px，但趴睡者的红条可能压在框内：
                    # 接受「锚点在人框上方 110px 到框底之间、水平对齐」的空间关系，
                    # 角落的时间戳/目标数角标下方没有人框，自然被排除
                    bx1, by1, bx2, by2 = bbox
                    bw = bx2 - bx1
                    best_label, best_d = None, 1e9
                    for label, (ax, ay) in anchors:
                        if not (bx1 - 0.3 * bw <= ax <= bx2 + 0.3 * bw):
                            continue
                        if not (by1 - 110 <= ay <= by2):
                            continue
                        d = abs(ay - (by1 - 33)) + abs(ax - (bx1 + bx2) / 2) * 0.2
                        if d < best_d:
                            best_label, best_d = label, d
                    if best_label is None:
                        continue
                    kp = r.keypoints.data[i].cpu().numpy()
                    f.write(json.dumps({
                        "ts": round(ts, 2), "label": best_label,
                        "box": [round(v, 1) for v in bbox],
                        "conf": round(float(r.boxes.conf[i]), 3),
                        "kp": [[round(float(x), 1), round(float(y), 1), round(float(c), 3)] for x, y, c in kp],
                    }) + "\n")
                    kept += 1
            n += 1
    cap.release()
    print(f"done: {kept} labeled samples -> {out_path}")


if __name__ == "__main__":
    main()
