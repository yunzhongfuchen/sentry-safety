import os
import logging
import threading
import time
import asyncio
import cv2
import numpy as np
import base64
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Optional, List
from collections import deque
import config
import storage
from detector import PersonDetector
from extractor import FrameExtractor
from understander import VideoUnderstander
import gpu_decoder

# 配置日志：全局默认 WARNING（抑制第三方库 info 噪音），但保留项目自身重要 info
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI(title="Sentry API")

# 挂载前端静态文件
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 全局状态 ──
detector = None
extractor = None
understander = None

# 用轻量级锁保护状态读写，不做任何重操作
_status_lock = threading.Lock()
current_status = {
    "detecting": False,
    "person_detected": False,
    "analyzing": False,
    "result": None,
    "logs": deque(maxlen=50),
}

# 记录列表 + 独立锁，避免和状态锁互相阻塞
_records_lock = threading.Lock()
detection_records: List[dict] = []

stream_active = False

# 触发冷却
_last_trigger_time = 0
_trigger_lock = threading.Lock()
MAX_CONCURRENT_ANALYSIS = 3

# 异步保存：用一个标记位，由专门线程定期写盘
_records_dirty = threading.Event()


class RTSPReader:
    """独立线程持续读取 RTSP/视频帧，外部只取最新帧。
    优先尝试 NVIDIA GPU 硬件解码 (NVDEC)，失败时回退到 OpenCV CPU 解码。"""

    def __init__(self, source: str):
        self.source = source
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.running = False
        self.cap = None
        self._gpu: Optional[gpu_decoder.GPUVideoReader] = None

    @property
    def backend(self) -> str:
        """返回当前使用的解码后端：gpu | cpu"""
        if self._gpu and self._gpu.is_opened():
            return "gpu"
        return "cpu"

    def start(self) -> bool:
        # 1. 尝试 GPU 硬件解码（仅限 NVIDIA + 非摄像头索引）
        if gpu_decoder.gpu_available():
            gpu = gpu_decoder.GPUVideoReader(self.source)
            if gpu.start():
                self._gpu = gpu
                self.running = True
                logger.info(f"GPU decoder active for {self.source}")
                return True
            logger.debug(f"GPU decoder failed for {self.source}, fallback to CPU")

        # 2. 回退到 OpenCV CPU 解码
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = True
        threading.Thread(target=self._reader_loop, daemon=True).start()
        return True

    def _reader_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
            if not ret:
                time.sleep(0.01)

    def read(self):
        if self._gpu:
            return self._gpu.read()
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def snapshot(self):
        """返回当前最新帧的副本，不影响内部状态。供抽帧器使用。"""
        if self._gpu:
            return self._gpu.snapshot()
        with self.lock:
            if self.ret and self.frame is not None:
                return self.frame.copy()
            return None

    def is_opened(self):
        if self._gpu:
            return self._gpu.is_opened()
        return self.cap is not None and self.cap.isOpened()

    def stop(self):
        self.running = False
        if self._gpu:
            self._gpu.stop()
            self._gpu = None
        if self.cap:
            self.cap.release()
            self.cap = None


# 全局 reader
video_reader: Optional[RTSPReader] = None


# ── 工具函数 ──

def log_message(msg: str):
    timestamp = time.strftime("%H:%M:%S")
    with _status_lock:
        current_status["logs"].append(f"[{timestamp}] {msg}")


def can_trigger() -> bool:
    global _last_trigger_time
    with _trigger_lock:
        return time.time() - _last_trigger_time >= config.TRIGGER_COOLDOWN


def record_trigger():
    global _last_trigger_time
    with _trigger_lock:
        _last_trigger_time = time.time()


def get_running_analysis_count() -> int:
    return sum(1 for t in threading.enumerate()
               if t.name and 'analysis-' in t.name)


def mark_records_dirty():
    """标记记录需要保存，由后台线程异步写盘"""
    _records_dirty.set()


def _records_saver_loop():
    """后台线程：检测到 dirty 标记后批量写盘，避免频繁 IO"""
    while True:
        _records_dirty.wait()  # 阻塞直到有人标记 dirty
        _records_dirty.clear()
        time.sleep(1)  # 合并 1 秒内的多次写入
        try:
            with _records_lock:
                data = list(detection_records)
            storage.save_records(data)
        except Exception as e:
            logger.error(f"Failed to save records: {e}")


def init_modules():
    global detector, extractor, understander, detection_records
    detector = PersonDetector()
    extractor = FrameExtractor(
        frame_count=config.FRAME_COUNT,
        frame_interval=config.FRAME_INTERVAL
    )
    understander = VideoUnderstander()

    # 加载历史记录
    detection_records = storage.load_records()

    # 启动异步保存线程
    threading.Thread(target=_records_saver_loop, daemon=True).start()

    logger.info("Modules initialized")


def encode_frame_to_base64(frame: np.ndarray, quality: int = 70) -> str:
    """将帧编码为base64，可调节 JPEG 质量减小体积"""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer.tobytes()).decode('utf-8')


# ── 视频流 + 检测循环（拆分为两个独立线程） ──

# 最新的标注帧，供视频流输出使用
_display_frame: Optional[np.ndarray] = None
_display_lock = threading.Lock()


def _detection_loop():
    """
    独立线程：持续从 RTSPReader 取帧 → YOLO 检测 → 触发分析。
    不再阻塞视频流输出。
    """
    global video_reader, stream_active, _display_frame

    source = config.VIDEO_SOURCE

    while True:
        video_reader = RTSPReader(source)
        if not video_reader.start():
            log_message("Failed to open video source, retrying...")
            time.sleep(2)
            continue

        log_message(f"Video source opened: {source}")
        stream_active = True
        consecutive_failures = 0

        while True:
            ret, frame = video_reader.read()

            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= 30:
                    log_message("Too many failures, reconnecting...")
                    break
                time.sleep(0.03)
                continue

            consecutive_failures = 0

            # YOLO 检测
            has_person = detector.detect(frame)
            with _status_lock:
                current_status["person_detected"] = has_person

            # 触发分析
            if has_person and can_trigger():
                if get_running_analysis_count() < MAX_CONCURRENT_ANALYSIS:
                    record_trigger()
                    threading.Thread(
                        target=run_analysis,
                        args=(frame.copy(),),
                        daemon=True,
                        name=f"analysis-{int(time.time()*1000)}"
                    ).start()

            # 绘制标注
            display = frame
            if has_person:
                cv2.rectangle(display, (10, 10), (250, 50), (0, 0, 255), 2)
                cv2.putText(display, "Person Detected!", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            with _status_lock:
                status_text = "Analyzing..." if current_status["analyzing"] else "Monitoring"
            cv2.putText(display, status_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 更新显示帧
            with _display_lock:
                _display_frame = display

            # 检测不需要太快，~15fps 足够
            time.sleep(0.06)

        video_reader.stop()
        video_reader = None
        stream_active = False
        time.sleep(1)


def generate_video_frames():
    """生成 MJPEG 视频流 —— 只做编码输出，不做检测"""
    while True:
        with _display_lock:
            frame = _display_frame

        if frame is None:
            time.sleep(0.05)
            continue

        ok, jpeg = cv2.imencode('.jpg', frame)
        if ok:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        time.sleep(0.03)


# ── 分析逻辑 ──

def run_analysis(snapshot: np.ndarray):
    """后台线程：抽帧 → VLM 分析 → 存储结果"""
    global detection_records

    # 记录开始时间
    start_time = time.time()
    trigger_time = time.strftime("%Y-%m-%d %H:%M:%S")
    record_id = f"{int(time.time()*1000)}"

    # 快照存文件系统
    snapshot_b64 = encode_frame_to_base64(snapshot)
    storage.save_image(record_id, "snapshot", snapshot_b64)

    # 元数据记录（不含图片数据）
    record = {
        "id": record_id,
        "camera_id": "default",
        "time": trigger_time,
        "frame_count": 0,
        "result": None,
        "action": None,
        "confidence": None,
        "reason": "分析中...",
        "timing": None,
    }

    with _records_lock:
        detection_records.insert(0, record)
        if len(detection_records) > 100:
            # 删除溢出记录的图片文件
            for old in detection_records[100:]:
                storage.delete_record_images(old["id"])
            detection_records = detection_records[:100]
    mark_records_dirty()

    with _status_lock:
        current_status["detecting"] = True
        current_status["analyzing"] = True
    log_message("Starting video analysis...")

    try:
        if video_reader is None:
            log_message("Video reader not available, skip analysis")
            record["reason"] = "视频源不可用"
            mark_records_dirty()
            return

        frames, timestamps = extractor.extract_from_reader(video_reader)

        if len(frames) == 0:
            log_message("Failed to extract frames")
            record["reason"] = "抽帧失败"
            return

        log_message(f"Extracted {len(frames)} frames")

        # 帧存文件系统
        for i, f in enumerate(frames):
            b64 = encode_frame_to_base64(f, quality=60)
            storage.save_image(record_id, "frame", b64, i)
        record["frame_count"] = len(frames)
        mark_records_dirty()

        # VLM 分析
        result = understander.analyze(frames)

        if result:
            with _status_lock:
                current_status["result"] = result
            record["result"] = result
            record["action"] = result.get("action", "none")
            record["confidence"] = result.get("confidence", 0)
            record["reason"] = result.get("reason", "")

            action = result.get("action", "none")
            conf = result.get("confidence", 0)
            if action == "enter":
                log_message(f"ALERT: Person entered! Confidence: {conf:.2f}")
            elif action == "leave":
                log_message(f"ALERT: Person left! Confidence: {conf:.2f}")
            else:
                log_message("Analysis result: No entry/leave detected")
        else:
            record["reason"] = "分析失败"
            log_message("Analysis failed")

        elapsed = time.time() - start_time
        record["timing"] = {
            "total_seconds": round(elapsed, 2),
            "total_display": f"{elapsed:.1f}秒",
        }
        log_message(f"Analysis completed in {elapsed:.2f}s")
        mark_records_dirty()

    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        log_message(f"Analysis error: {e}")
        record["reason"] = f"错误: {str(e)[:30]}"
        record["timing"] = {
            "total_seconds": round(time.time() - start_time, 2),
            "total_display": f"{time.time() - start_time:.1f}秒",
        }
        mark_records_dirty()
    finally:
        with _status_lock:
            current_status["detecting"] = False
            current_status["analyzing"] = False


# ── API 端点 ──

@app.get("/video")
async def video_feed():
    async def async_video_frames():
        loop = asyncio.get_event_loop()
        gen = generate_video_frames()
        while True:
            frame_data = await loop.run_in_executor(None, next, gen)
            yield frame_data

    return StreamingResponse(
        async_video_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/status")
async def get_status():
    with _status_lock:
        status_snapshot = {
            "detecting": current_status["detecting"],
            "person_detected": current_status["person_detected"],
            "analyzing": current_status["analyzing"],
            "result": current_status["result"],
            "logs": list(current_status["logs"]),
            "stream_active": stream_active,
            "decoder_backend": video_reader.backend if video_reader else "none",
        }

    # 只返回最近 10 条的元数据，不含图片
    with _records_lock:
        simple_records = [
            {
                "id": r.get("id"),
                "time": r.get("time"),
                "action": r.get("action"),
                "confidence": r.get("confidence"),
                "reason": r.get("reason"),
            }
            for r in detection_records[:10]
        ]

    status_snapshot["records"] = simple_records
    return status_snapshot


@app.get("/record/{record_id}")
async def get_record(record_id: str, include_frames: bool = True):
    import urllib.parse
    decoded_id = urllib.parse.unquote(record_id)

    with _records_lock:
        meta = None
        for r in detection_records:
            if r.get("id") == decoded_id:
                meta = dict(r)
                break
    if meta is None:
        return {"error": "Record not found"}

    # 检查快照是否存在
    meta["has_snapshot"] = storage.load_image_b64(decoded_id, "snapshot") is not None
    meta["camera_id"] = meta.get("camera_id", "default")

    # 按需加载帧
    if include_frames:
        frame_count = meta.get("frame_count", 0)
        frames = []
        for i in range(min(frame_count, 30)):
            b64 = storage.load_image_b64(decoded_id, "frame", i)
            if b64:
                frames.append(b64)
        meta["frames"] = frames

    return meta


@app.get("/record/{record_id}/snapshot")
async def get_record_snapshot(record_id: str):
    """单独获取记录快照图片"""
    import urllib.parse
    decoded_id = urllib.parse.unquote(record_id)
    snapshot = storage.load_image_b64(decoded_id, "snapshot")
    if snapshot:
        return {"snapshot": snapshot}
    return {"error": "Snapshot not found"}


@app.get("/record/{record_id}/frames")
async def get_record_frames(record_id: str, start: int = 0, count: int = 30):
    """分页获取记录帧图片"""
    import urllib.parse
    decoded_id = urllib.parse.unquote(record_id)
    count = min(count, 30)

    frames = []
    for i in range(start, start + count):
        b64 = storage.load_image_b64(decoded_id, "frame", i)
        if b64:
            frames.append(b64)

    return {
        "record_id": decoded_id,
        "start": start,
        "count": len(frames),
        "frames": frames
    }


@app.get("/records")
async def get_all_records(
    page: int = 1,
    page_size: int = 20,
    camera_id: str = None,
    action: str = None,
):
    """获取记录列表 - 支持分页和过滤"""
    with _records_lock:
        filtered = list(detection_records)

    # 过滤
    if camera_id:
        filtered = [r for r in filtered if r.get("camera_id") == camera_id]
    if action:
        filtered = [r for r in filtered if r.get("action") == action]

    total = len(filtered)
    total_pages = max(1, (total + page_size - 1) // page_size)
    start = (page - 1) * page_size
    page_records = filtered[start:start + page_size]

    records = []
    for r in page_records:
        records.append({
            "id": r.get("id"),
            "camera_id": r.get("camera_id", "default"),
            "time": r.get("time"),
            "action": r.get("action"),
            "confidence": r.get("confidence"),
            "reason": r.get("reason"),
            "timing": r.get("timing"),
            "frame_count": r.get("frame_count", 0),
        })

    return {
        "records": records,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


@app.get("/records/summary")
async def get_records_summary():
    """获取记录统计摘要"""
    with _records_lock:
        records = list(detection_records)

    total = len(records)
    enter_count = sum(1 for r in records if r.get("action") == "enter")
    leave_count = sum(1 for r in records if r.get("action") == "leave")

    return {
        "total": total,
        "enter": enter_count,
        "leave": leave_count,
    }


@app.get("/cameras")
async def list_cameras():
    """返回摄像头列表（单摄像头模式返回默认摄像头）"""
    return {
        "cameras": [
            {
                "camera_id": "default",
                "name": f"摄像头 ({config.VIDEO_SOURCE})",
                "status": "connected" if stream_active else "disconnected",
            }
        ]
    }


@app.get("/")
async def root():
    fp = Path(__file__).parent.parent / "frontend" / "safety_detection" / "monitor.html"
    if fp.exists():
        return HTMLResponse(fp.read_text(encoding="utf-8"))
    return {"message": "Sentry API", "status": "running"}


@app.get("/records.html")
async def records_page():
    fp = Path(__file__).parent.parent / "frontend" / "records.html"
    if fp.exists():
        return HTMLResponse(fp.read_text(encoding="utf-8"))
    return {"error": "Records page not found"}


# ── 提示词管理 API ──

@app.get("/prompt")
async def get_prompt():
    """获取当前提示词"""
    prompt, question = config.load_prompt()
    return {"prompt": prompt, "question": question}


@app.post("/prompt")
async def update_prompt(data: dict):
    """保存提示词"""
    prompt = data.get("prompt", "")
    question = data.get("question", "")
    success = config.save_prompt(prompt, question)
    if success:
        return {"success": True, "message": "提示词已保存"}
    return {"success": False, "message": "保存失败"}


@app.on_event("startup")
async def startup():
    init_modules()
    # 启动检测循环（独立线程，不阻塞视频流）
    threading.Thread(target=_detection_loop, daemon=True).start()
    logger.info(f"Sentry backend started on {config.API_HOST}:{config.API_PORT}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, log_level="warning")
