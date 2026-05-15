"""
Sentry 有限空间监控独立服务入口
端口 8001，可独立启动
"""

import os
import sys
import json
import logging
import threading
import time
import asyncio
from typing import Dict, List, Optional
from collections import deque
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# 配置日志：全局默认 WARNING（抑制第三方库 info 噪音），但保留项目自身重要 info
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# 终极修复：RK3588 conda 等边缘环境中，某些库（如 coloredlogs、uvicorn 特定版本）
# 会重新加载或污染 logging 模块，导致 _nameToLevel 被清空。
# 在 torch 导入前强制 reload 标准 logging 模块，恢复其原始内部状态。
import importlib
importlib.reload(logging)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# 导入模块
try:
    from camera_manager import CameraManager, CameraConfig
    from inference_engine import SafetyDetector, detect_best_device
    from vlm_queue import VLMQueue
    from understander import VideoUnderstander
    from video_stream import get_stream_server
    import config as app_config
    import performance_storage as storage
    from confined_space.zone_counter import ZoneCounter, ZoneConfig
    from confined_space.api import router as confined_router
    from confined_space import storage as confined_storage
except ImportError as e:
    logger.error(f"Import error: {e}")
    raise

# 创建 FastAPI 应用
app = FastAPI(title="Sentry Confined Space Monitoring API")

# 挂载前端静态文件
frontend_path = Path(__file__).parent.parent / "frontend" / "confined_space"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载业务路由
app.include_router(confined_router)

# -- 全局组件 --
camera_manager: Optional[CameraManager] = None
safety_detector: Optional[SafetyDetector] = None
vlm_queue: Optional[VLMQueue] = None
zone_counter: Optional[ZoneCounter] = None
stream_server = get_stream_server()
_global_settings: dict = {}

# 状态管理
_status_lock = threading.Lock()
_system_status = {
    "started_at": None,
    "camera_count": 0,
    "active_cameras": 0,
    "total_events": 0,
    "logs": deque(maxlen=100),
}


def log_message(msg: str, level: str = "info") -> None:
    """记录系统日志（控制台仅输出 warning/error，其余降为 debug）"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with _status_lock:
        _system_status["logs"].append(f"[{timestamp}] {msg}")
    if level == "error":
        logger.error(msg)
    elif level == "warning":
        logger.warning(msg)
    else:
        logger.debug(msg)


# -- 初始化 --

def init_components():
    """初始化所有组件"""
    global camera_manager, safety_detector, vlm_queue, zone_counter, _global_settings

    log_message("Initializing Confined Space Monitoring System...")

    # 1. 加载全局配置
    _global_settings = app_config.load_confined_settings()
    app.state.global_settings = _global_settings
    log_message("Confined settings loaded")

    # 2. 初始化摄像头管理器
    camera_manager = CameraManager()
    app.state.camera_manager = camera_manager

    camera_configs = app_config.load_confined_cameras()
    for cam_data in camera_configs:
        cfg = CameraConfig(
            camera_id=cam_data["camera_id"],
            source=cam_data["source"],
            name=cam_data.get("name", ""),
            enabled=cam_data.get("enabled", True),
            width=cam_data.get("width", 640),
            height=cam_data.get("height", 480),
            fps=cam_data.get("fps", 15),
            source_type=cam_data.get("source_type", "auto"),
        )
        camera_manager.register_camera(cfg)
        stream_server.register_camera(cfg.camera_id)

    log_message(f"Registered {len(camera_configs)} cameras")

    # 3. 检测设备
    device, npu_cores = detect_best_device()
    log_message(f"Detection device: {device}, npu_cores={npu_cores}")

    # 4. 初始化推理引擎
    safety_detector = SafetyDetector(npu_cores=npu_cores, device=device)
    app.state.safety_detector = safety_detector
    safety_detector._load_person_model(device)
    log_message("Person model loaded")

    # 5. 初始化 VLMQueue
    understander = VideoUnderstander()
    vlm_queue = VLMQueue(
        understander=understander,
        max_concurrent=_global_settings.get("vlm_max_concurrent", 3),
    )
    app.state.vlm_queue = vlm_queue

    # 6. 初始化存储
    confined_storage.init()
    confined_storage.start_saver()
    app.state.storage = confined_storage

    # 7. 初始化 ZoneCounter
    zone_counter = ZoneCounter(
        camera_manager=camera_manager,
        inference_engine=safety_detector,
        vlm_queue=vlm_queue,
        storage=confined_storage,
        settings=_global_settings,
    )
    app.state.zone_counter = zone_counter

    # 从配置加载摄像头（一摄像头一区域，扁平格式）
    for cam_data in camera_configs:
        config = ZoneConfig(
            camera_id=cam_data["camera_id"],
            name=cam_data.get("name", cam_data["camera_id"]),
            roi=cam_data.get("roi", []),
            enable_vlm_review=cam_data.get("enable_vlm_review", True),
        )
        zone_counter.register_camera(config)

    log_message("ZoneCounter initialized")

    # 更新系统状态
    with _status_lock:
        _system_status["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _system_status["camera_count"] = len(camera_configs)


# -- 页面路由 --

@app.get("/")
@app.get("/monitor")
async def monitor_page():
    """监控主页"""
    fp = Path(__file__).parent.parent / "frontend" / "confined_space" / "monitor.html"
    if fp.exists():
        return HTMLResponse(fp.read_text(encoding="utf-8"))
    return {"error": "Monitor page not found"}


@app.get("/records")
async def records_page():
    """记录页面"""
    fp = Path(__file__).parent.parent / "frontend" / "confined_space" / "records.html"
    if fp.exists():
        return HTMLResponse(fp.read_text(encoding="utf-8"))
    return {"error": "Records page not found"}


@app.get("/settings")
async def settings_page():
    """设置页面"""
    fp = Path(__file__).parent.parent / "frontend" / "confined_space" / "settings.html"
    if fp.exists():
        return HTMLResponse(fp.read_text(encoding="utf-8"))
    return {"error": "Settings page not found"}


@app.get("/recordings")
async def recordings_page():
    """录屏管理页面"""
    fp = Path(__file__).parent.parent / "frontend" / "confined_space" / "recordings.html"
    if fp.exists():
        return HTMLResponse(fp.read_text(encoding="utf-8"))
    return {"error": "Recordings page not found"}


# -- 系统 API --

@app.get("/status")
async def system_status():
    """系统状态"""
    with _status_lock:
        status = dict(_system_status)
    return status


@app.get("/system/mode")
async def get_system_mode():
    """获取当前运行模式和检测设备"""
    mode = os.environ.get("SENTRY_MODE", "multi")
    device, npu_cores = detect_best_device()
    return {"mode": mode, "device": device, "npu_cores": npu_cores}


@app.post("/system/restart")
async def restart_system():
    """重启服务"""
    try:
        log_message("System restart requested")
        stop_confined_threads()
        if zone_counter:
            pass  # ZoneCounter 无需单独停止
        if camera_manager:
            camera_manager.stop_all()
        if safety_detector:
            safety_detector.release()

        init_components()
        if camera_manager:
            camera_manager.start_all()
        if vlm_queue:
            vlm_queue.start()
        start_confined_threads()

        log_message("System restart completed")
        return {"success": True}
    except Exception as e:
        log_message(f"Restart failed: {e}", "error")
        return JSONResponse({"error": str(e)}, status_code=500)


# -- 视频流 --

@app.get("/cameras/{camera_id}/stream")
async def camera_stream(camera_id: str, raw: bool = False):
    """摄像头视频流 (raw=true 输出原始帧, 默认输出标注帧)"""
    if camera_manager is None:
        return JSONResponse({"error": "Not initialized"}, status_code=500)

    return StreamingResponse(
        stream_server.generate_frames(camera_id, raw=raw),
        media_type='multipart/x-mixed-replace; boundary=frame',
    )


@app.get("/cameras/{camera_id}/snapshot")
async def camera_snapshot(camera_id: str):
    """获取摄像头当前一帧 JPEG (用于 ROI 编辑器)"""
    if camera_manager is None:
        return JSONResponse({"error": "Not initialized"}, status_code=500)
    frame = camera_manager.get_frame(camera_id)
    if frame is None:
        return JSONResponse({"error": "no frame available"}, status_code=404)
    h, w = frame.shape[:2]
    quality = _global_settings.get("snapshot_quality", 85)
    ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return JSONResponse({"error": "encode failed"}, status_code=500)
    from fastapi import Response
    return Response(
        content=buffer.tobytes(),
        media_type='image/jpeg',
        headers={"X-Frame-Width": str(w), "X-Frame-Height": str(h)},
    )


@app.post("/cameras/{camera_id}/playback")
async def camera_playback_control(camera_id: str, data: dict):
    """视频播放控制：play / pause / loop / speed / seek"""
    if camera_manager is None:
        return JSONResponse({"error": "Not initialized"}, status_code=500)
    action = data.get("action")
    if not action:
        return JSONResponse({"error": "action required"}, status_code=400)
    result = camera_manager.control_playback(camera_id, action, **{k: v for k, v in data.items() if k != "action"})
    if result is None:
        return JSONResponse({"error": "Camera not found"}, status_code=404)
    return {"success": True, "playback": result}


@app.get("/cameras/{camera_id}/playback")
async def camera_playback_status(camera_id: str):
    """获取视频播放状态"""
    if camera_manager is None:
        return JSONResponse({"error": "Not initialized"}, status_code=500)
    status = camera_manager.get_playback_status(camera_id)
    if status is None:
        return JSONResponse({"error": "Camera not found"}, status_code=404)
    return {"playback": status}


@app.post("/cameras/{camera_id}/record/start")
async def camera_record_start(camera_id: str):
    """开始录制原始视频"""
    if camera_manager is None:
        return JSONResponse({"error": "Not initialized"}, status_code=500)
    record_path = await asyncio.to_thread(camera_manager.start_recording, camera_id)
    if record_path is None:
        return JSONResponse({"error": "Camera not found or not connected"}, status_code=404)
    return {"success": True, "camera_id": camera_id, "record_path": record_path, "recording": True}


@app.post("/cameras/{camera_id}/record/stop")
async def camera_record_stop(camera_id: str):
    """停止录制"""
    if camera_manager is None:
        return JSONResponse({"error": "Not initialized"}, status_code=500)
    record_path = await asyncio.to_thread(camera_manager.stop_recording, camera_id)
    if record_path is None:
        return JSONResponse({"error": "Camera not found or not recording"}, status_code=404)
    return {"success": True, "camera_id": camera_id, "record_path": record_path, "recording": False}


@app.get("/cameras/{camera_id}/record/status")
async def camera_record_status(camera_id: str):
    """获取录制状态"""
    if camera_manager is None:
        return JSONResponse({"error": "Not initialized"}, status_code=500)
    status = camera_manager.get_recording_status(camera_id)
    if status is None:
        return JSONResponse({"error": "Camera not found"}, status_code=404)
    return {"recording": status["recording"], "record_path": status["record_path"]}


# -- 录屏文件管理 --

RECORDINGS_DIR = Path(__file__).parent.parent / "data" / "recordings"


def _parse_recording_meta(path: Path) -> dict:
    """解析录制文件元数据"""
    meta = {
        "filename": path.name,
        "size": path.stat().st_size,
        "camera_id": "",
        "recorded_at": "",
        "duration_seconds": 0,
        "width": 0,
        "height": 0,
        "fps": 0,
    }
    # 文件名格式: {camera_id}_{timestamp}.mp4
    stem = path.stem
    parts = stem.split("_", 1)
    if len(parts) >= 1:
        meta["camera_id"] = parts[0]
    if len(parts) >= 2:
        ts = parts[1]
        # 尝试解析时间戳 YYYYMMDD_HHMMSS
        try:
            meta["recorded_at"] = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}:{ts[13:15]}"
        except Exception:
            meta["recorded_at"] = ts

    # 用 OpenCV 读取视频元数据
    try:
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            meta["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            meta["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            meta["fps"] = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if meta["fps"] > 0 and total_frames > 0:
                meta["duration_seconds"] = total_frames / meta["fps"]
        cap.release()
    except Exception:
        pass

    return meta


@app.get("/api/recordings")
async def list_recordings(camera_id: Optional[str] = None):
    """列出所有录制文件。只扫描根目录 .mp4，自动关联 previews/ 下的 .webm 预览。"""
    recordings = []
    total_size = 0
    today = time.strftime("%Y%m%d")
    today_count = 0

    if RECORDINGS_DIR.exists():
        mp4_paths = [p for p in RECORDINGS_DIR.glob("*.mp4")]
        for path in sorted(mp4_paths, key=lambda p: p.stat().st_mtime, reverse=True):
            meta = _parse_recording_meta(path)
            total_size += meta["size"]
            if today in path.stem:
                today_count += 1
            if camera_id and meta["camera_id"] != camera_id:
                continue
            # 查找对应的预览文件
            preview_path = RECORDINGS_DIR / "previews" / f"{path.stem}.webm"
            if preview_path.exists():
                meta["preview_filename"] = preview_path.name
            recordings.append(meta)

    return {
        "recordings": recordings,
        "total": len(recordings),
        "today": today_count,
        "total_size": total_size,
    }


@app.get("/recordings/{filename}")
async def get_recording_file(filename: str):
    """获取录制视频文件"""
    file_path = RECORDINGS_DIR / filename
    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    from fastapi.responses import FileResponse
    media_type = "video/webm" if file_path.suffix.lower() == ".webm" else "video/mp4"
    return FileResponse(
        str(file_path),
        media_type=media_type,
    )


@app.head("/recordings/{filename}")
async def head_recording_file(filename: str):
    """HEAD 请求用于浏览器视频预检"""
    file_path = RECORDINGS_DIR / filename
    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    from fastapi import Response
    media_type = "video/webm" if file_path.suffix.lower() == ".webm" else "video/mp4"
    return Response(
        headers={
            "content-type": media_type,
            "content-length": str(file_path.stat().st_size),
            "accept-ranges": "bytes",
        }
    )


@app.get("/recordings/previews/{filename}")
async def get_preview_file(filename: str):
    """获取预览视频文件（低分辨率 VP8/webm）"""
    file_path = RECORDINGS_DIR / "previews" / filename
    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    from fastapi.responses import FileResponse
    return FileResponse(str(file_path), media_type="video/webm")


@app.head("/recordings/previews/{filename}")
async def head_preview_file(filename: str):
    """HEAD 请求用于预览视频预检"""
    file_path = RECORDINGS_DIR / "previews" / filename
    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    from fastapi import Response
    return Response(
        headers={
            "content-type": "video/webm",
            "content-length": str(file_path.stat().st_size),
            "accept-ranges": "bytes",
        }
    )


@app.delete("/api/recordings/{filename}")
async def delete_recording_file(filename: str):
    """删除录制文件"""
    file_path = RECORDINGS_DIR / filename
    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    try:
        file_path.unlink()
        return {"success": True, "filename": filename}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# -- 后台处理线程 --

_confined_running = False
_confined_thread: Optional[threading.Thread] = None
_overlay_running = False
_overlay_thread: Optional[threading.Thread] = None


def _annotate_frame(frame: np.ndarray, viz: dict) -> np.ndarray:
    """根据 ROI 与检测框,在帧上绘制可视化标注（纯英文，兼容 cv2.putText）"""
    if frame is None:
        return frame
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for zone in viz.get("zones", []):
        roi = zone.get("roi") or []
        roi_color = (0, 200, 255)  # BGR

        # 画 ROI 框
        if len(roi) == 4:
            x1, y1, x2, y2 = [int(v) for v in roi]
            x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), roi_color, 2)
            detected = zone.get("detected_count", 0)
            label = f"D:{detected}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), roi_color, -1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # 画人体检测框
        for det in zone.get("detections", []) or []:
            xyxy = det.get("xyxy") or det.get("bbox") or []
            if len(xyxy) != 4:
                continue
            bx1, by1, bx2, by2 = [int(v) for v in xyxy]
            score = det.get("score", det.get("confidence", 0))
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (60, 220, 60), 2)
            tag = f"person {score:.2f}" if score else "person"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (bx1, max(0, by1 - th - 6)), (bx1 + tw + 6, by1), (60, 220, 60), -1)
            cv2.putText(annotated, tag, (bx1 + 3, by1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    return annotated


def _overlay_loop():
    """独立渲染线程：定期从 camera_manager 取帧 + 画区域框/人数 + 送 stream_server
    同时推送原始帧和标注帧到双缓冲区，与安全检测 main_multi.py 的 _overlay_loop 保持一致。
    """
    global _overlay_running
    log_message("Confined overlay render thread started")
    while _overlay_running:
        loop_t0 = time.perf_counter()
        try:
            if camera_manager is None or stream_server is None:
                time.sleep(0.1)
                continue

            cam_ids = camera_manager.get_camera_ids()
            for cam_id in cam_ids:
                if not _overlay_running:
                    break
                frame = camera_manager.get_frame(cam_id)
                if frame is None:
                    continue

                # 推送原始帧到 raw 缓冲区
                stream_server.update_frame(cam_id, frame, raw=True)

                # 获取区域可视化并画标注帧
                if zone_counter is not None:
                    viz = zone_counter.get_camera_visualization(cam_id)
                    annotated = _annotate_frame(frame, viz) if viz else frame
                    # 诊断：统计画了多少个检测框
                    if viz:
                        det_count = sum(len(z.get("detections", []) or []) for z in (viz.get("zones") or []))
                        if det_count > 0:
                            logger.debug(f"[OVERLAY] cam={cam_id} drawn_boxes={det_count}")
                else:
                    annotated = frame

                # 推送标注帧到 annotated 缓冲区
                stream_server.update_frame(cam_id, annotated, raw=False)
        except Exception as e:
            logger.error(f"Confined overlay loop error: {e}")
        loop_dt = time.perf_counter() - loop_t0
        if loop_dt > 0.1:
            logger.warning(f"[OVERLAY_SLOW] loop took {loop_dt*1000:.1f}ms")
        # 25fps 渲染间隔（与安全检测保持一致）
        time.sleep(max(0.0, 0.04 - loop_dt))
    log_message("Confined overlay render thread stopped")


def _confined_space_loop():
    """有限空间推理线程 —— 只负责推理，不碰推流"""
    global _confined_running
    log_message("Confined inference thread started")
    last_inference_ts: Dict[str, float] = {}
    _prev_video_frame_idx: Dict[str, int] = {}  # 跟踪视频摄像头的帧索引，检测从头播放
    _diag_loop_count = 0
    _diag_process_count = 0
    _diag_none_count = 0
    _diag_skip_count = 0
    _diag_process_time_acc = 0.0
    _diag_last_log = time.perf_counter()
    while _confined_running:
        try:
            if camera_manager is None or zone_counter is None:
                time.sleep(0.5)
                continue

            now = time.perf_counter()
            inference_interval = _global_settings.get("inference_interval", 0.2)
            cameras = zone_counter.list_cameras()
            _diag_loop_count += 1

            for cam_info in cameras:
                if not _confined_running:
                    break
                camera_id = cam_info.get("camera_id")
                if not camera_id:
                    continue

                # 本地视频停止播放时自动归零人数，避免残留计数持续触发 VLM
                pb = camera_manager.get_playback_status(camera_id)
                if pb and pb.get("is_video_file"):
                    current_idx = pb.get("current_frame_idx", 0)
                    prev_idx = _prev_video_frame_idx.get(camera_id, 0)
                    # 检测到视频发生回退（循环播放或手动 seek 到开头），清空 frame_buffer 避免旧帧混入
                    if current_idx < prev_idx:
                        zone_counter.reset_buffer(camera_id)
                    _prev_video_frame_idx[camera_id] = current_idx

                    if not pb.get("playing", True):
                        # 视频播放到末尾（不循环），强制提交尚未完成的窗口，避免短视频永远等不到 window_delay
                        total_frames = pb.get("total_frames", 0)
                        if total_frames > 0 and current_idx >= total_frames - 1:
                            zone_counter.force_submit_pending_window(camera_id)
                        continue

                elapsed = now - last_inference_ts.get(camera_id, 0)
                if elapsed < inference_interval:
                    _diag_skip_count += 1
                    continue

                frame = camera_manager.get_frame(camera_id)
                if frame is None:
                    _diag_none_count += 1
                    continue

                t0 = time.perf_counter()
                events = zone_counter.process_frame(camera_id, frame)
                t1 = time.perf_counter()
                proc_dt = t1 - t0
                last_inference_ts[camera_id] = now
                _diag_process_count += 1
                _diag_process_time_acc += proc_dt
                logger.debug(f"[PROCESS] camera={camera_id} proc_time={proc_dt*1000:.1f}ms elapsed_since_last={elapsed*1000:.1f}ms")
                if events:
                    with _status_lock:
                        _system_status["total_events"] += len(events)

            # 诊断日志：每 5 秒汇总一次推理情况
            if now - _diag_last_log >= 5.0:
                avg_proc = (_diag_process_time_acc / _diag_process_count * 1000) if _diag_process_count > 0 else 0
                logger.debug(
                    f"[DIAG] loops={_diag_loop_count} process={_diag_process_count} "
                    f"none={_diag_none_count} skip={_diag_skip_count} "
                    f"cameras={len(cameras)} avg_proc_ms={avg_proc:.1f}"
                )
                _diag_loop_count = _diag_process_count = _diag_none_count = _diag_skip_count = 0
                _diag_process_time_acc = 0.0
                _diag_last_log = now

        except Exception as e:
            logger.error(f"Confined inference loop error: {e}")
        time.sleep(0.02)
    log_message("Confined inference thread stopped")


def start_confined_threads():
    """启动处理线程（推理 + 推流分离）"""
    global _confined_running, _confined_thread, _overlay_running, _overlay_thread
    if _confined_thread and _confined_thread.is_alive():
        return
    _confined_running = True
    _overlay_running = True
    _confined_thread = threading.Thread(
        target=_confined_space_loop, daemon=True, name="confined-inference"
    )
    _overlay_thread = threading.Thread(
        target=_overlay_loop, daemon=True, name="confined-overlay"
    )
    _confined_thread.start()
    _overlay_thread.start()


def stop_confined_threads():
    """停止处理线程"""
    global _confined_running, _overlay_running
    _confined_running = False
    _overlay_running = False
    if _confined_thread:
        _confined_thread.join(timeout=2)
    if _overlay_thread:
        _overlay_thread.join(timeout=2)


@app.on_event("startup")
async def startup():
    """服务启动"""
    init_components()
    if camera_manager:
        camera_manager.start_all()
    if vlm_queue:
        vlm_queue.start()
    start_confined_threads()
    port = _global_settings.get("api_port", 8001)
    log_message(f"Confined Space Monitoring started on port {port}")


@app.on_event("shutdown")
async def shutdown():
    """服务关闭"""
    log_message("Shutting down...")
    stop_confined_threads()
    if vlm_queue:
        vlm_queue.stop()
    if camera_manager:
        camera_manager.stop_all()
    if safety_detector:
        safety_detector.release()


if __name__ == "__main__":
    settings = app_config.load_confined_settings()
    port = settings.get("api_port", 8001)
    host = settings.get("api_host", "0.0.0.0")
    uvicorn.run(app, host=host, port=port, log_level="warning")
