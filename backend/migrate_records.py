#!/usr/bin/env python3
"""
迁移脚本：将旧的 records.json（内嵌 base64 图片）拆分为
  - 轻量 records.json（仅元数据）
  - data/frames/ 目录下的 jpg 文件

用法：python migrate_records.py
"""
import json
import base64
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RECORDS_FILE = DATA_DIR / "records.json"
FRAMES_DIR = DATA_DIR / "frames"
BACKUP_FILE = DATA_DIR / "records_backup.json"


def migrate():
    if not RECORDS_FILE.exists():
        print("records.json not found, nothing to migrate.")
        return

    print(f"Loading {RECORDS_FILE} ...")
    with open(RECORDS_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 检查是否已经迁移过（没有 snapshot 字段说明已迁移）
    if records and "snapshot" not in records[0] and "frames" not in records[0]:
        print("Already migrated, skipping.")
        return

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Migrating {len(records)} records ...")
    new_records = []
    for i, r in enumerate(records):
        rid = r.get("id", str(i))

        # 保存 snapshot
        snapshot_b64 = r.pop("snapshot", None)
        if snapshot_b64:
            path = FRAMES_DIR / f"{rid}_snapshot.jpg"
            path.write_bytes(base64.b64decode(snapshot_b64))

        # 保存 frames
        frames_b64 = r.pop("frames", [])
        for idx, fb64 in enumerate(frames_b64):
            path = FRAMES_DIR / f"{rid}_frame_{idx:03d}.jpg"
            path.write_bytes(base64.b64decode(fb64))

        r["frame_count"] = len(frames_b64)
        new_records.append(r)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(records)} done")

    # 备份旧文件
    print(f"Backing up original to {BACKUP_FILE} ...")
    RECORDS_FILE.rename(BACKUP_FILE)

    # 写入新的轻量 records.json
    with open(RECORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(new_records, f, ensure_ascii=False, indent=2)

    new_size = RECORDS_FILE.stat().st_size
    print(f"Migration complete!")
    print(f"  New records.json: {new_size / 1024:.1f} KB")
    print(f"  Frames saved to: {FRAMES_DIR}")
    print(f"  Backup at: {BACKUP_FILE}")
    print(f"\nYou can delete the backup after verifying: rm {BACKUP_FILE}")


if __name__ == "__main__":
    migrate()
