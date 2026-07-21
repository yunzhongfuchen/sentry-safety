#!/usr/bin/env python3
"""
视频流模拟器（带 Web 管理界面）

把本地视频文件转换为 HTTP MJPEG 实时流，供 Sentry 等系统当作普通摄像头/RTSP 流使用。
支持通过浏览器上传视频、控制播放/暂停/循环、复制流地址。

用法示例:
    python video_simulator.py
    python video_simulator.py --port 18765 --quality 80
    python video_simulator.py video1.mp4 video2.mp4 --port 18765

然后在浏览器打开 http://localhost:18765 进行管理。
在 Sentry 摄像头配置中填写:
    source: http://localhost:18765/video1/stream
    source_type: rtsp
"""

import argparse
import json
import logging
import mimetypes
import os
import re
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_UPLOAD_DIR = Path(__file__).parent / "data" / "simulator_uploads"


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


class ImageFeed:
    """静态图片按固定帧率循环推流。"""

    def __init__(self, image_path: str, quality: int = 85, max_width: int = 1280, fps: float = 25.0):
        self.video_path = image_path
        self.quality = quality
        self.max_width = max_width
        self._fps = fps
        self._current_jpeg: bytes | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.playing = True
        self.loop = True

    def start(self) -> None:
        buf = np.frombuffer(Path(self.video_path).read_bytes(), dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"无法读取图片: {self.video_path}")
        h, w = frame.shape[:2]
        if self.max_width > 0 and w > self.max_width:
            scale = self.max_width / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise RuntimeError(f"图片编码失败: {self.video_path}")
        with self._lock:
            self._current_jpeg = jpeg.tobytes()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"img-{Path(self.video_path).stem}")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        interval = 1.0 / self._fps
        while not self._stop.is_set():
            self._stop.wait(interval)

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._current_jpeg

    def get_status(self) -> dict:
        return {"fps": self._fps, "playing": self.playing, "loop": self.loop}


class VideoFeed:
    """单个视频源的循环播放线程。"""

    def __init__(self, video_path: str, quality: int = 85, max_width: int = 1280):
        self.video_path = video_path
        self.quality = quality
        self.max_width = max_width

        self._cap: cv2.VideoCapture | None = None
        self._fps = 25.0
        self._frame_interval = 1.0 / self._fps
        self._current_jpeg: bytes | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.playing = True
        self.loop = True

    def start(self) -> None:
        self._open()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"feed-{Path(self.video_path).stem}")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
            self._cap = None

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.video_path}")

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(fps) if isinstance(fps, (int, float)) and fps > 0 else 25.0
        self._frame_interval = 1.0 / self._fps
        logger.info(f"{self.video_path}: fps={self._fps:.2f}, interval={self._frame_interval:.4f}s")

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._cap is None:
                time.sleep(0.1)
                continue

            if not self.playing:
                time.sleep(0.05)
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                if self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.playing = False
                    continue

            # 等比例缩放过大的帧，降低带宽
            h, w = frame.shape[:2]
            if self.max_width > 0 and w > self.max_width:
                scale = self.max_width / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if ok:
                with self._lock:
                    self._current_jpeg = jpeg.tobytes()

            time.sleep(self._frame_interval)

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._current_jpeg

    def get_status(self) -> dict:
        return {
            "fps": self._fps,
            "playing": self.playing,
            "loop": self.loop,
        }


def _parse_multipart(body: bytes, content_type: str) -> dict:
    """简单 multipart/form-data 解析，不依赖 cgi 模块。"""
    if not content_type.startswith("multipart/form-data"):
        return {}
    match = re.search(r'boundary=([^\s;]+)', content_type)
    if not match:
        return {}
    boundary = match.group(1).strip('"\'')
    boundary_bytes = ("--" + boundary).encode()
    parts = body.split(boundary_bytes)
    files = {}
    for part in parts:
        part = part.lstrip(b"\r\n")
        if not part or part == b"--\r\n":
            continue
        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        headers = part[:header_end].decode("utf-8", errors="ignore")
        data = part[header_end + 4:]
        if data.endswith(b"\r\n"):
            data = data[:-2]

        filename = None
        name = None
        for line in headers.split("\r\n"):
            if line.lower().startswith("content-disposition"):
                for item in line.split(";"):
                    item = item.strip()
                    if item.startswith("name="):
                        name = item[5:].strip('"\'')
                    elif item.startswith("filename="):
                        filename = item[9:].strip('"\'')
        if filename and name:
            files[name] = {"filename": filename, "data": data}
    return files


class MJPEGHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器。"""

    feeds: dict[str, VideoFeed] = {}
    upload_dir: Path = DEFAULT_UPLOAD_DIR
    quality: int = 85
    max_width: int = 1280

    def log_message(self, format, *args):
        # 抑制默认访问日志，避免刷屏
        pass

    def _json_response(self, status: int, data: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    def _next_path(self) -> str:
        idx = 1
        while f"/video{idx}" in self.feeds:
            idx += 1
        return f"/video{idx}"

    def _base_url(self) -> str:
        host = self.headers.get("Host", f"localhost:{self.server.server_port}")
        return f"http://{host}"

    def do_GET(self):
        path = self.path.split("?", 1)[0]

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_HTML_PAGE.encode("utf-8"))
            return

        if path == "/api/feeds":
            feeds = []
            for fpath, feed in sorted(self.feeds.items()):
                status = feed.get_status()
                status["path"] = fpath
                status["source"] = Path(feed.video_path).name
                status["url"] = f"{self._base_url()}{fpath}/stream"
                feeds.append(status)
            self._json_response(200, {"feeds": feeds})
            return

        if path.endswith("/snapshot"):
            base_path = path[: -len("/snapshot")] or "/"
            feed = self.feeds.get(base_path)
            if feed is None:
                self.send_error(404, "Unknown stream")
                return
            frame = feed.get_frame()
            if frame is None:
                self.send_error(503, "No frame yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(frame)
            return

        if path.endswith("/stream"):
            base_path = path[: -len("/stream")] or "/"
            feed = self.feeds.get(base_path)
            if feed is None:
                self.send_error(404, "Unknown stream")
                return

            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            interval = 1.0 / max(feed._fps, 1.0)
            try:
                while True:
                    frame = feed.get_frame()
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    time.sleep(interval)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        self.send_error(404)

    def do_POST(self):
        if self.path == "/upload":
            content_type = self.headers.get("Content-Type", "")
            body = self._read_body()
            files = _parse_multipart(body, content_type)
            video = files.get("video")
            if video is None:
                self._json_response(400, {"error": "未找到上传文件"})
                return

            self.upload_dir.mkdir(parents=True, exist_ok=True)
            filename = Path(video["filename"]).name
            # 避免重名
            save_path = self.upload_dir / filename
            counter = 1
            stem = save_path.stem
            suffix = save_path.suffix
            while save_path.exists():
                save_path = self.upload_dir / f"{stem}_{counter}{suffix}"
                counter += 1

            save_path.write_bytes(video["data"])
            path = self._next_path()
            try:
                ext = Path(filename).suffix.lower()
                if ext in IMAGE_EXTS:
                    feed = ImageFeed(str(save_path), quality=self.quality, max_width=self.max_width)
                else:
                    feed = VideoFeed(str(save_path), quality=self.quality, max_width=self.max_width)
                feed.start()
                self.feeds[path] = feed
                logger.info(f"上传并启动 {path}/stream -> {save_path}")
                self._json_response(200, {
                    "path": path,
                    "url": f"{self._base_url()}{path}/stream"
                })
            except Exception as e:
                logger.error(f"启动视频失败: {e}")
                self._json_response(500, {"error": str(e)})
            return

        if self.path == "/api/control":
            try:
                body = self._read_body()
                data = json.loads(body.decode("utf-8"))
                path = data.get("path")
                action = data.get("action")
                feed = self.feeds.get(path)
                if feed is None:
                    self._json_response(404, {"error": "视频流不存在"})
                    return

                if action == "toggle":
                    feed.playing = not feed.playing
                elif action == "play":
                    feed.playing = True
                elif action == "pause":
                    feed.playing = False
                elif action == "loop":
                    feed.loop = bool(data.get("loop", True))
                else:
                    self._json_response(400, {"error": "未知操作"})
                    return

                self._json_response(200, {"success": True})
            except Exception as e:
                self._json_response(500, {"error": str(e)})
            return

        if self.path == "/api/remove":
            try:
                body = self._read_body()
                data = json.loads(body.decode("utf-8"))
                path = data.get("path")
                feed = self.feeds.pop(path, None)
                if feed is None:
                    self._json_response(404, {"error": "视频流不存在"})
                    return
                feed.stop()
                logger.info(f"移除 {path}")
                self._json_response(200, {"success": True})
            except Exception as e:
                self._json_response(500, {"error": str(e)})
            return

        self.send_error(404)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>视频流模拟器</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 24px;
            background: #f5f7fa;
            color: #333;
        }
        h1 { margin-top: 0; font-size: 24px; }
        .upload-area {
            border: 2px dashed #c0c4cc;
            border-radius: 8px;
            padding: 40px 24px;
            text-align: center;
            background: #fff;
            margin-bottom: 24px;
            transition: all 0.2s;
        }
        .upload-area.dragover {
            border-color: #409eff;
            background: #ecf5ff;
        }
        .upload-area p { margin: 0 0 12px; color: #606266; }
        .btn {
            padding: 8px 16px;
            border: 1px solid #dcdfe6;
            border-radius: 4px;
            background: #fff;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        .btn:hover { background: #f5f7fa; }
        .btn-primary {
            background: #409eff;
            color: #fff;
            border-color: #409eff;
        }
        .btn-primary:hover { background: #66b1ff; }
        .btn-danger {
            background: #f56c6c;
            color: #fff;
            border-color: #f56c6c;
        }
        .btn-danger:hover { background: #f78989; }
        .feed-card {
            background: #fff;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 2px 12px 0 rgba(0,0,0,0.05);
        }
        .feed-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .feed-title { font-weight: 600; font-size: 16px; }
        .feed-meta { color: #909399; font-size: 13px; }
        .feed-url {
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            background: #f5f7fa;
            padding: 10px 12px;
            border-radius: 4px;
            word-break: break-all;
            font-size: 13px;
            color: #606266;
            margin-bottom: 12px;
        }
        .controls { display: flex; gap: 8px; flex-wrap: wrap; }
        .preview {
            width: 100%;
            max-width: 320px;
            height: 180px;
            object-fit: contain;
            background: #000;
            border-radius: 4px;
            margin-bottom: 12px;
        }
        .empty {
            text-align: center;
            color: #909399;
            padding: 40px;
        }
        .toast {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #67c23a;
            color: #fff;
            padding: 10px 16px;
            border-radius: 4px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.15);
            display: none;
        }
    </style>
</head>
<body>
    <h1>视频流模拟器</h1>

    <div class="upload-area" id="uploadArea">
        <p>拖拽视频文件到此处，或点击选择文件</p>
        <input type="file" id="fileInput" accept="video/*,image/*" multiple style="display:none">
        <button class="btn btn-primary" onclick="document.getElementById('fileInput').click()">选择文件</button>
    </div>

    <div id="feedList">
        <div class="empty">暂无视频流，请先上传视频</div>
    </div>

    <div class="toast" id="toast"></div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const feedList = document.getElementById('feedList');
        const toast = document.getElementById('toast');

        function showToast(msg, isError) {
            toast.textContent = msg;
            toast.style.background = isError ? '#f56c6c' : '#67c23a';
            toast.style.display = 'block';
            setTimeout(() => toast.style.display = 'none', 3000);
        }

        async function loadFeeds() {
            try {
                const res = await fetch('/api/feeds');
                const data = await res.json();
                renderFeeds(data.feeds);
            } catch (e) {
                console.error('加载失败', e);
            }
        }

        function renderFeeds(feeds) {
            if (feeds.length === 0) {
                feedList.innerHTML = '<div class="empty">暂无视频流，请先上传视频</div>';
                return;
            }
            feedList.innerHTML = feeds.map(feed => `
                <div class="feed-card">
                    <div class="feed-header">
                        <span class="feed-title">${escapeHtml(feed.source)}</span>
                        <span class="feed-meta">${feed.fps.toFixed(2)} FPS · ${feed.playing ? '播放中' : '已暂停'}</span>
                    </div>
                    <img class="preview" src="${feed.path}/snapshot?t=${Date.now()}" alt="预览">
                    <div class="feed-url">${feed.url}</div>
                    <div class="controls">
                        <button class="btn ${feed.playing ? 'btn-danger' : 'btn-primary'}" onclick="control('${feed.path}', 'toggle')">
                            ${feed.playing ? '⏸ 暂停' : '▶ 播放'}
                        </button>
                        <button class="btn" onclick="control('${feed.path}', 'loop', ${!feed.loop})">
                            ${feed.loop ? '🔁 循环：开' : '➡️ 循环：关'}
                        </button>
                        <button class="btn" onclick="copyUrl('${feed.url}')">📋 复制地址</button>
                        <button class="btn btn-danger" onclick="removeFeed('${feed.path}')">🗑 删除</button>
                    </div>
                </div>
            `).join('');
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        async function control(path, action, loop) {
            const body = { path, action };
            if (loop !== undefined) body.loop = loop;
            try {
                await fetch('/api/control', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });
                loadFeeds();
            } catch (e) {
                console.error('控制失败', e);
            }
        }

        async function removeFeed(path) {
            if (!confirm('确定删除这个视频流吗？')) return;
            try {
                await fetch('/api/remove', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path })
                });
                loadFeeds();
            } catch (e) {
                console.error('删除失败', e);
            }
        }

        async function copyUrl(url) {
            try {
                await navigator.clipboard.writeText(url);
                showToast('已复制到剪贴板');
            } catch (e) {
                // fallback
                const input = document.createElement('input');
                input.value = url;
                document.body.appendChild(input);
                input.select();
                document.execCommand('copy');
                document.body.removeChild(input);
                showToast('已复制到剪贴板');
            }
        }

        async function uploadFiles(files) {
            for (const file of files) {
                showToast(`上传中: ${file.name}`);
                const formData = new FormData();
                formData.append('video', file);
                try {
                    const res = await fetch('/upload', { method: 'POST', body: formData });
                    const data = await res.json();
                    if (!res.ok) {
                        showToast(`上传失败: ${data.error || res.status}`, true);
                    } else {
                        showToast(`已添加: ${file.name}`);
                    }
                } catch (e) {
                    showToast(`上传异常: ${e.message}`, true);
                }
            }
            fileInput.value = '';
            loadFeeds();
        }

        fileInput.addEventListener('change', e => uploadFiles(e.target.files));

        uploadArea.addEventListener('dragover', e => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
        uploadArea.addEventListener('drop', e => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            uploadFiles(e.dataTransfer.files);
        });

        loadFeeds();
        setInterval(loadFeeds, 2000);
    </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="把本地视频转换为 HTTP MJPEG 实时流")
    parser.add_argument("videos", nargs="*", help="启动时预加载的本地视频文件路径")
    parser.add_argument("--port", type=int, default=18765, help="HTTP 端口 (默认 18765)")
    parser.add_argument("--quality", type=int, default=85, help="JPEG 质量 0-100 (默认 85)")
    parser.add_argument("--max-width", type=int, default=1280, help="最大宽度，超过则等比例缩放 (默认 1280)")
    parser.add_argument("--upload-dir", type=str, default=str(DEFAULT_UPLOAD_DIR),
                        help=f"上传文件保存目录 (默认 {DEFAULT_UPLOAD_DIR})")
    args = parser.parse_args()

    MJPEGHandler.upload_dir = Path(args.upload_dir)
    MJPEGHandler.quality = args.quality
    MJPEGHandler.max_width = args.max_width

    feeds: dict[str, VideoFeed] = {}
    for idx, video in enumerate(args.videos or [], start=1):
        path = f"/video{idx}"
        try:
            feed = VideoFeed(video, quality=args.quality, max_width=args.max_width)
            feed.start()
            feeds[path] = feed
            logger.info(f"预加载 {path}/stream -> {video}")
        except Exception as e:
            logger.error(f"初始化 {video} 失败: {e}")

    MJPEGHandler.feeds = feeds
    server = ThreadedHTTPServer(("", args.port), MJPEGHandler)
    logger.info(f"模拟器已启动: http://localhost:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("正在停止...")
    finally:
        for feed in MJPEGHandler.feeds.values():
            feed.stop()
        server.server_close()


if __name__ == "__main__":
    main()
