"""
打包脚本：排除 weights/、data/、logs/、uploads/ 等运行时产物，
仅保留源码、配置、前端、文档与部署脚本。
"""

import zipfile
import os
import sys


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # sentry-safety/
output_dir = os.path.dirname(project_root)                                   # D:/project/
output_name = "sentry-safety-20260901.zip"
output_path = os.path.join(output_dir, output_name)

# 顶级目录完全排除
EXCLUDE_TOP_DIRS = {
    "weights", "data", "logs", "uploads",
    "交大交付", "sophon-stream-master",
    ".claude", ".git", ".codegraph", ".playwright-mcp",
    ".pytest_cache", "__pycache__", ".superpowers", ".worktrees",
    ".VSCodeCounter",
}

# 任意层级排除的目录名
EXCLUDE_ANY_DIRS = {"__pycache__", ".mypy_cache"}

# 排除文件扩展名（权重、运行日志）
EXCLUDE_EXTS = {".pt", ".onnx", ".rknn", ".log", ".pid"}

# 根目录下排除的具体文件
EXCLUDE_ROOT_FILES = {
    "backend.log", "backend.pid", "yolov8s.pt",
    "camera-dialog-2.yml", "camera-dialog-3.yml",
    "camera-dialog-accordion.yml", "snapshot-initial.yml",
    "视频综合平台&视频智能应用OpenAPI对接文档2.3.9.pdf",
}


def should_include(rel_path: str) -> bool:
    parts = rel_path.replace("\\", "/").split("/")

    # 顶级排除目录
    if parts[0] in EXCLUDE_TOP_DIRS:
        return False

    # 根目录排除文件
    if len(parts) == 1 and parts[0] in EXCLUDE_ROOT_FILES:
        return False

    # 任意层级排除目录
    for p in parts[:-1]:
        if p in EXCLUDE_ANY_DIRS:
            return False

    # 扩展名排除
    ext = os.path.splitext(parts[-1])[1].lower()
    if ext in EXCLUDE_EXTS:
        return False

    # .pyc
    if parts[-1].endswith(".pyc"):
        return False

    return True


def main():
    print(f"Project root : {project_root}")
    print(f"Output       : {output_path}")

    count = 0
    raw_bytes = 0

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(project_root):
            # 剪枝：原地修改 dirs 以避免 os.walk 进入排除目录
            dirs[:] = [
                d for d in dirs
                if d not in EXCLUDE_TOP_DIRS and d not in EXCLUDE_ANY_DIRS
            ]

            for fname in files:
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, project_root)

                if not should_include(rel_path):
                    continue

                try:
                    fsize = os.path.getsize(full_path)
                    zf.write(full_path, rel_path)
                    count += 1
                    raw_bytes += fsize
                except Exception as exc:
                    print(f"  SKIP {rel_path}: {exc}", file=sys.stderr)

    zip_bytes = os.path.getsize(output_path)
    print(
        f"\nDone! {count} files | "
        f"raw {raw_bytes / 1024 / 1024:.1f} MB -> "
        f"zip {zip_bytes / 1024 / 1024:.1f} MB"
    )


if __name__ == "__main__":
    main()
