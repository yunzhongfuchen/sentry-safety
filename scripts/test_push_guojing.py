# -*- coding: utf-8 -*-
"""手动测试：取一条历史报警记录 + 本地快照/检测帧图片，推送到国经 webhook 接收地址。"""

import base64
import json
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "backend"))

from backend.integrations.guojing.channel import GuojingWebhookChannel
from backend.integrations.dingtalk.channel import DingTalkChannel

TARGET_URL = os.getenv("GUOJING_PUSH_URL", "http://192.168.4.16:5004/v1/alarmRecord/sync")
# 钉钉凭证从环境变量读取，避免把群机器人密钥提交进版本库
DINGTALK_WEBHOOK = os.getenv("DINGTALK_WEBHOOK", "")
DINGTALK_SECRET = os.getenv("DINGTALK_SECRET", "")
RECORDS_FILE = "data/records.json"
FRAMES_DIR = "data/frames"


def load_record_with_images():
    with open(RECORDS_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    for record in reversed(records):  # 从最近的找
        rid = record.get("id")
        snap_path = f"{FRAMES_DIR}/{rid}_snapshot.jpg"
        try:
            with open(snap_path, "rb") as f:
                snapshot_b64 = base64.b64encode(f.read()).decode("utf-8")
        except FileNotFoundError:
            continue

        frames_b64 = []
        idx = 0
        while True:
            try:
                with open(f"{FRAMES_DIR}/{rid}_frame_{idx:03d}.jpg", "rb") as f:
                    frames_b64.append(base64.b64encode(f.read()).decode("utf-8"))
                idx += 1
            except FileNotFoundError:
                break

        return record, snapshot_b64, frames_b64

    return None, None, []


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "created"

    if mode.startswith("dingtalk"):
        return main_dingtalk(reviewed="reviewed" in mode)

    record, snapshot_b64, frames_b64 = load_record_with_images()
    if record is None:
        print("未找到带本地图片的历史报警记录")
        sys.exit(1)

    logs = []
    channel = GuojingWebhookChannel(
        TARGET_URL, timeout=10.0,
        log=lambda msg, level="info": logs.append((level, msg)),
    )

    if mode == "reviewed":
        # 伪造一条 VLM 复核结果：alarm.reviewed 不带图片，vlm_review 有值
        record = dict(record)
        record["level"] = "vlm_alarm"
        record["reason"] = "[VLM 确认] 画面中人员确实未佩戴口罩"
        record["vlm_review"] = {
            "confirmed": True,
            "confidence": 0.93,
            "reason": "画面中人员确实未佩戴口罩",
        }
        payload = channel.build_event("alarm.reviewed", record, None, [])
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("-" * 50)
        ok = channel.send_reviewed(record)
    else:
        print(f"记录ID      : {record['id']}")
        print(f"检测类型    : {record['detection_type']}")
        print(f"时间        : {record['time']}")
        print(f"快照大小    : {len(snapshot_b64)} base64 chars")
        print(f"检测帧数量  : {len(frames_b64)}")
        print(f"推送目标    : {TARGET_URL}")
        print("-" * 50)
        payload = channel.build_event("alarm.created", record, snapshot_b64, frames_b64)
        preview = json.loads(json.dumps(payload))
        preview["data"]["snapshot"] = f"<base64 {len(snapshot_b64)} chars>"
        preview["data"]["frames"] = [f"<base64 {len(x)} chars>" for x in frames_b64]
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        print("-" * 50)
        ok = channel.send_created(record, snapshot_b64, frames_b64)

    for level, msg in logs:
        print(f"[{level}] {msg}")
    print(f"\n推送结果: {'成功' if ok else '失败'}")
    sys.exit(0 if ok else 1)


def main_dingtalk(reviewed: bool):
    if not DINGTALK_WEBHOOK:
        print("请先设置环境变量 DINGTALK_WEBHOOK（需要加签时再设 DINGTALK_SECRET）")
        sys.exit(1)

    record, snapshot_b64, frames_b64 = load_record_with_images()
    if record is None:
        print("未找到带本地图片的历史报警记录")
        sys.exit(1)

    logs = []
    channel = DingTalkChannel(
        DINGTALK_WEBHOOK, secret=DINGTALK_SECRET,
        log=lambda msg, level="info": logs.append((level, msg)),
    )

    if reviewed:
        record = dict(record)
        record["level"] = "vlm_alarm"
        record["reason"] = "[VLM 确认] 画面中人员确实未佩戴口罩"
        record["vlm_review"] = {
            "confirmed": True,
            "confidence": 0.93,
            "reason": "画面中人员确实未佩戴口罩",
        }
        payload = channel.build_message("alarm.reviewed", record)
        ok = channel.send_reviewed(record)
    else:
        payload = channel.build_message("alarm.created", record)
        ok = channel.send_created(record, snapshot_b64, frames_b64)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("-" * 50)
    for level, msg in logs:
        print(f"[{level}] {msg}")
    print(f"\n推送结果: {'成功' if ok else '失败'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
