"""睡岗漏检诊断：抓取摄像头实时帧，打印每个人的姿态分析细节

用法:
  python scripts/diagnose_sleep.py [stream_url]          # 单帧详细输出
  python scripts/diagnose_sleep.py --loop [秒数]          # 循环紧凑输出，等睡觉场景出现
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from safety_detection.sleep_detect import analyze_sleep  # noqa: E402

STREAM_URL = "http://localhost:18765/video1/stream"
MODEL_PATH = ROOT / "weights" / "yolov8s-pose_1.pt"


def analyze_frame(model, frame, verbose=True):
    r = model.predict(frame, conf=0.1, iou=0.5, verbose=False)[0]
    lines = []
    for i in range(len(r.boxes)):
        bbox = r.boxes.xyxy[i].cpu().numpy()
        kp = r.keypoints.data[i].cpu().numpy()
        score = float(r.boxes.conf[i])
        info = analyze_sleep(kp, bbox)

        x1, y1, x2, y2 = bbox
        bh = y2 - y1
        head_confs = kp[0:5, 2]

        # 候选信号：可见手腕到可见头部关键点的最近距离（相对框高）
        head_pts = kp[0:5][kp[0:5, 2] > 0.4][:, :2]
        wrist_dists = []
        for w in kp[[9, 10]]:
            if w[2] > 0.4 and len(head_pts) > 0:
                d = np.min(np.linalg.norm(head_pts - w[:2], axis=1)) / max(bh, 1)
                wrist_dists.append(float(d))

        # 候选信号：头-肩二维位移（相对框高），不限方向，适应任意相机角度
        sh_pts = kp[[5, 6]][kp[[5, 6], 2] > 0.4][:, :2]
        if len(head_pts) > 0 and len(sh_pts) > 0:
            head_center = head_pts.mean(axis=0)
            sh_center = sh_pts.mean(axis=0)
            disp = float(np.linalg.norm(head_center - sh_center) / max(bh, 1))
            dy = float((head_center[1] - sh_center[1]) / max(bh, 1))
            disp_s = f"{disp:.2f} (dy={dy:+.2f})"
        else:
            disp_s = "N/A"

        if verbose:
            lines.append(f"--- 人 {i} (检测分 {score:.2f}) ---")
            lines.append(f"  框: [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] AR={info['aspect_ratio']:.2f}")
            lines.append(f"  头评分: {[f'{c:.2f}' for c in head_confs]}  肩: {kp[5,2]:.2f}/{kp[6,2]:.2f}  腕: {[f'{c:.2f}' for c in kp[[9,10],2]]}")
            lines.append(f"  head_drop={info['head_drop']:.3f} head_hidden={info['head_hidden']} head_conf={info['head_conf']:.2f}")
            lines.append(f"  腕-头距离: {[f'{d:.2f}' for d in wrist_dists] or 'N/A'}  头-肩位移: {disp_s}")
            lines.append(f"  判定: sleeping={info['is_sleeping']} reason={info['sleep_reason']} conf={info['sleep_confidence']:.2f}")
        else:
            wd = "/".join(f"{d:.2f}" for d in wrist_dists) or "-"
            hc = "/".join(f"{c:.2f}" for c in head_confs)
            lines.append(
                f"  p{i} AR={info['aspect_ratio']:.2f} head[{hc}] drop={info['head_drop']:.2f} "
                f"hid={int(info['head_hidden'])} wd={wd} disp={disp_s} -> {info['sleep_reason']} conf={info['sleep_confidence']:.2f}"
            )
    return lines, len(r.boxes)


def grab_frame(cap):
    ok, frame = cap.read()
    return frame if ok else None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    loop = "--loop" in sys.argv
    harvest = "--harvest" in sys.argv
    url = args[0] if args and "://" in args[0] else STREAM_URL
    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(url)

    if harvest:
        # 连续采集原始关键点：1Hz 写 JSONL，每 20s 存标注帧用于人工标注醒/睡
        duration = next((int(a) for a in args if a.isdigit()), 600)
        out_path = ROOT / "data" / "pose_harvest.jsonl"
        start = time.time()
        last_snap = 0.0
        n = 0
        with open(out_path, "a", encoding="utf-8") as f:
            while time.time() - start < duration:
                frame = grab_frame(cap)
                if frame is None:
                    time.sleep(1)
                    continue
                ts = time.time() - start
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
                f.flush()
                n += 1
                if ts - last_snap >= 20:
                    cv2.imwrite(str(ROOT / "data" / f"harvest_{int(ts):04d}.jpg"), vis)
                    last_snap = ts
                time.sleep(1)
        cap.release()
        print(f"harvest done: {n} samples -> {out_path}")
        return

    if not loop:
        frame = grab_frame(cap)
        cap.release()
        if frame is None:
            print("ERROR: 无法从流中读取帧")
            return
        lines, n = analyze_frame(model, frame, verbose=True)
        print(f"帧尺寸: {frame.shape[1]}x{frame.shape[0]}, 检出 {n} 人\n")
        print("\n".join(lines))
        out = ROOT / "data" / "diagnose_frame.jpg"
        cv2.imwrite(str(out), frame)
        print(f"\n帧已保存: {out}")
        return

    duration = next((int(a) for a in args if a.isdigit()), 300)
    start = time.time()
    round_no = 0
    while time.time() - start < duration:
        frame = grab_frame(cap)
        if frame is None:
            time.sleep(1)
            continue
        round_no += 1
        lines, n = analyze_frame(model, frame, verbose=False)
        t = time.strftime("%H:%M:%S")
        print(f"[{t}] #{round_no} 检出 {n} 人", flush=True)
        print("\n".join(lines), flush=True)
        # 有人判睡或头部评分偏低（疑似趴睡）时保存帧
        suspect = any("conf=0.5" in l or "sleeping=True" in l for l in lines)
        if suspect:
            out = ROOT / "data" / f"diagnose_suspect_{round_no}.jpg"
            cv2.imwrite(str(out), frame)
            print(f"  >>> 疑似帧已保存: {out}", flush=True)
        time.sleep(3)
    cap.release()


if __name__ == "__main__":
    main()
