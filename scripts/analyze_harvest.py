"""离线分析 harvest 数据：跟踪每个人（按座位位置聚类），输出姿态特征时间序列与分布

用法: python scripts/analyze_harvest.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "pose_harvest.jsonl"

KPT = 0.4  # 关键点可见阈值


def feats(p):
    """从一条 person 记录提取候选特征"""
    kp = np.array(p["kp"])  # (17,3)
    x1, y1, x2, y2 = p["box"]
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)

    def vis(idx):
        return kp[idx, 2] > KPT

    def pt(idx):
        return kp[idx, :2]

    f = {"ar": bw / bh, "conf": p["conf"]}

    head_idx = [i for i in range(5) if vis(i)]
    f["head_vis"] = len(head_idx)
    head_pts = np.array([pt(i) for i in head_idx]) if head_idx else None
    head_c = head_pts.mean(axis=0) if head_pts is not None else None

    sh_idx = [i for i in (5, 6) if vis(i)]
    sh_pts = np.array([pt(i) for i in sh_idx]) if sh_idx else None
    sh_c = sh_pts.mean(axis=0) if sh_pts is not None else None
    f["sh_vis"] = len(sh_idx)

    if head_c is not None and sh_c is not None:
        d = head_c - sh_c
        f["disp"] = float(np.linalg.norm(d) / bh)
        f["dy"] = float(d[1] / bh)
        # 头相对肩的方向角（-pi..pi，图像坐标 y 向下）
        f["ang"] = float(np.arctan2(d[1], d[0]))
    if head_c is not None:
        # 头中心在框内的相对位置（0.5,0.5=框中心；y<0.5=偏上）
        f["head_bx"] = float((head_c[0] - x1) / bw)
        f["head_by"] = float((head_c[1] - y1) / bh)
    if sh_c is not None:
        f["sh_by"] = float((sh_c[1] - y1) / bh)

    # 肘宽/肩宽（趴睡时小臂垫头，肘外张）
    if all(vis(i) for i in (5, 6, 7, 8)):
        sh_w = float(np.linalg.norm(pt(5) - pt(6)))
        elb_w = float(np.linalg.norm(pt(7) - pt(8)))
        f["elb_sh"] = elb_w / max(sh_w, 1)
    # 腕-头最近距离（垫头睡时腕贴头）
    if head_pts is not None:
        ds = []
        for w in (9, 10):
            if vis(w):
                ds.append(float(np.min(np.linalg.norm(head_pts - pt(w), axis=1)) / bh))
        if ds:
            f["wr_hd"] = min(ds)
    # 肘-头最近距离
    if head_pts is not None:
        ds = []
        for e in (7, 8):
            if vis(e):
                ds.append(float(np.min(np.linalg.norm(head_pts - pt(e), axis=1)) / bh))
        if ds:
            f["elb_hd"] = min(ds)
    # 头肩"前后"分离度：头中心是否比肩中心更靠近框边缘（头探出躯干）
    if head_c is not None and sh_c is not None:
        bc = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        f["head_out"] = float((np.linalg.norm(head_c - bc) - np.linalg.norm(sh_c - bc)) / bh)
    return f


def main():
    samples = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    print(f"{len(samples)} 个采样")

    # 按框中心聚类到人（座位固定，用首帧位置做锚点，最近邻关联）
    anchors = []  # [cx, cy]
    series = []   # per anchor: list of (ts, feats)
    for s in samples:
        for p in s["persons"]:
            x1, y1, x2, y2 = p["box"]
            c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            best, bd = None, 120.0
            for i, a in enumerate(anchors):
                d = float(np.linalg.norm(c - a))
                if d < bd:
                    best, bd = i, d
            if best is None:
                anchors.append(c)
                series.append([])
                best = len(anchors) - 1
            else:
                anchors[best] = 0.9 * anchors[best] + 0.1 * c
            series[best].append((s["ts"], feats(p)))

    print(f"跟踪到 {len(series)} 个座位\n")
    KEYS = ["ar", "head_vis", "disp", "dy", "head_by", "sh_by", "elb_sh", "wr_hd", "elb_hd", "head_out"]
    for i, tr in enumerate(series):
        if len(tr) < 100:
            continue
        print(f"== 座位{i} 锚点=({anchors[i][0]:.0f},{anchors[i][1]:.0f}) 样本={len(tr)} ==")
        # 全时段分布
        for k in KEYS:
            vs = [f[k] for _, f in tr if k in f]
            if not vs:
                continue
            q = np.percentile(vs, [10, 50, 90])
            print(f"  {k:9s} p10={q[0]:+.2f} p50={q[1]:+.2f} p90={q[2]:+.2f} (n={len(vs)})")
        # 前 120s vs 后段对比（视频前段偏抬头、后段全趴）
        early = [f for t, f in tr if t < 120]
        late = [f for t, f in tr if t > 300]
        print("  -- 前120s vs >300s 中位数 --")
        for k in KEYS:
            e = [f[k] for f in early if k in f]
            l = [f[k] for f in late if k in f]
            if e and l:
                print(f"  {k:9s} early={np.median(e):+.2f} (n={len(e)})  late={np.median(l):+.2f} (n={len(l)})")
        print()


if __name__ == "__main__":
    sys.exit(main())
