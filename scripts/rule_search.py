"""在标注数据集上搜索组合规则：俯视睡者命中率高、醒者误触发率低

用法: python scripts/rule_search.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))
from analyze_harvest import feats  # noqa: E402
from safety_detection.sleep_detect import analyze_sleep  # noqa: E402


def load_labeled(name, label):
    rows = [json.loads(l) for l in open(ROOT / "data" / f"labeled_{name}.jsonl", encoding="utf-8")]
    return [r for r in rows if r["label"] == label]


def load_harvest(name):
    rows = [json.loads(l) for l in open(ROOT / "data" / f"harvest_{name}.jsonl", encoding="utf-8")]
    return [p for r in rows for p in r["persons"]]


def existing_verdict(p):
    kp = np.array(p["kp"])
    return analyze_sleep(kp, p["box"])["is_sleeping"]


def main():
    awake = load_labeled("睡岗1", "awake") + load_labeled("睡岗4", "awake")
    sleep_top = load_harvest("睡岗6_1") + load_harvest("睡岗5_1")

    # 现有规则的命中情况
    aw_hit = sum(existing_verdict(p) for p in awake)
    st_hit = sum(existing_verdict(p) for p in sleep_top)
    print(f"现有规则: 醒误报 {aw_hit}/{len(awake)}  俯视睡命中 {st_hit}/{len(sleep_top)}")

    # 候选组合规则（只评估现有规则漏掉的样本）
    def rule(p, disp_t, dy_t, hv_t, wr_t):
        f = feats(p)
        if "disp" not in f or "dy" not in f:
            return False
        if f["disp"] < disp_t or f["dy"] < dy_t or f["head_vis"] > hv_t:
            return False
        if wr_t is not None and f.get("wr_hd", 9) > wr_t:
            return False
        return True

    missed_sleep = [p for p in sleep_top if not existing_verdict(p)]
    safe_awake = [p for p in awake if not existing_verdict(p)]
    print(f"现有规则漏: 睡 {len(missed_sleep)}, 醒 {len(safe_awake)}\n")

    print(f"{'规则':38s} 睡命中率  醒误触发率")
    for disp_t in (0.13, 0.15, 0.17):
        for dy_t in (-0.18, -0.15, -0.12):
            for hv_t in (2, 3):
                for wr_t in (None, 0.25):
                    s = sum(rule(p, disp_t, dy_t, hv_t, wr_t) for p in missed_sleep)
                    a = sum(rule(p, disp_t, dy_t, hv_t, wr_t) for p in safe_awake)
                    name = f"disp>{disp_t} dy>{dy_t} hv<={hv_t} wr<{wr_t}"
                    print(f"{name:38s} {s/len(missed_sleep):7.1%}  {a/len(safe_awake):8.1%} ({a}/{len(safe_awake)})")


if __name__ == "__main__":
    main()
