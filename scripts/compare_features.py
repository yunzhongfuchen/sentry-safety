"""对比醒/睡样本的几何特征分布，找可分信号

用法: python scripts/compare_features.py
数据源:
  醒: data/labeled_睡岗1.jsonl + labeled_睡岗4.jsonl (label=awake)
  睡-俯视: data/harvest_睡岗6_1.jsonl + harvest_睡岗5_1.jsonl (全员趴睡)
  睡-斜视: labeled 中 label=sleep 的少量样本
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_harvest import feats  # noqa: E402

KEYS = ["ar", "head_vis", "disp", "dy", "head_by", "sh_by", "elb_sh", "wr_hd", "elb_hd", "head_out"]


def load_labeled(name, label):
    rows = [json.loads(l) for l in open(ROOT / "data" / f"labeled_{name}.jsonl", encoding="utf-8")]
    return [feats(r) for r in rows if r["label"] == label]


def load_harvest(name):
    rows = [json.loads(l) for l in open(ROOT / "data" / f"harvest_{name}.jsonl", encoding="utf-8")]
    return [feats(p) for r in rows for p in r["persons"]]


def show(name, samples):
    print(f"== {name} (n={len(samples)}) ==")
    for k in KEYS:
        vs = [f[k] for f in samples if k in f]
        if not vs:
            print(f"  {k:9s} n=0")
            continue
        q = np.percentile(vs, [5, 25, 50, 75, 95])
        print(f"  {k:9s} p05={q[0]:+.2f} p25={q[1]:+.2f} p50={q[2]:+.2f} p75={q[3]:+.2f} p95={q[4]:+.2f} (n={len(vs)})")
    print()


def main():
    awake = load_labeled("睡岗1", "awake") + load_labeled("睡岗4", "awake")
    sleep_labeled = load_labeled("睡岗1", "sleep") + load_labeled("睡岗4", "sleep")
    sleep_top = load_harvest("睡岗6_1") + load_harvest("睡岗5_1")
    show("醒（斜视，标注）", awake)
    show("睡（斜视，标注）", sleep_labeled)
    show("睡（俯视，全员）", sleep_top)


if __name__ == "__main__":
    main()
