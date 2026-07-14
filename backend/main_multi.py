"""
Sentry 多摄像头版本主程序
支持 RK3588 边缘端部署，多路视频流并行处理
"""

import asyncio
import concurrent.futures
import os
import sys
import json
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple
from collections import deque
from pathlib import Path

# 把项目根目录和 backend 目录都加入 sys.path，确保：
# 1. 直接运行 python backend/main_multi.py 时，子模块中的 `from backend import xxx` 能解析。
# 2. 以模块方式 python -m backend.main_multi 运行时，`from camera_manager` 等能解析。
_project_root = Path(__file__).resolve().parent.parent
_backend_dir = _project_root / "backend"
for _path in (_project_root, _backend_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

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
logger.setLevel(logging.INFO)

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
logger.setLevel(logging.INFO)

# 导入模块
try:
    from camera_manager import CameraManager, CameraConfig
    from inference_engine import SafetyDetector, detect_best_device
    from safety_detection.detector_core import MultiDetector, CorePinnedStrategy, SerialStrategy
    from vlm_queue import VLMQueue
    from vlm_inspector import VLMInspector
    from understander import VideoUnderstander
    import performance_storage as storage
    import config as app_config
    from backend.detection_registry import registry
    from video_stream import get_stream_server
    from safety_detection.api import router as safety_router
    from alarm_state import create_record, apply_vlm_review, confirm_alarm, confirm_false_positive
    from backend.frame_utils import draw_timestamp_on_frame
except ImportError as e:
    logger.error(f"Import error: {e}")
    raise

# 创建 FastAPI 应用
app = FastAPI(title="Sentry Multi-Camera Safety Detection API")

# 挂载前端静态文件
frontend_path = Path(__file__).parent.parent / "frontend" / "safety_detection"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载安全检测业务路由
app.include_router(safety_router)

# ── 全局组件 ──
camera_manager: Optional[CameraManager] = None
safety_detector: Optional[SafetyDetector] = None
multi_detector: Optional[MultiDetector] = None
vlm_queue: Optional[VLMQueue] = None
vlm_inspector: Optional[VLMInspector] = None
stream_server = get_stream_server()
_global_settings: dict = {}
gpu_scheduler = None
_save_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

# 状态管理
_status_lock = threading.Lock()
_system_status = {
    "started_at": None,
    "camera_count": 0,
    "active_cameras": 0,
    "total_detections": 0,
    "logs": deque(maxlen=100),
}

# 显示帧缓存 (用于视频流)
_display_frames: Dict[str, np.ndarray] = {}
_display_lock = threading.Lock()

# 记录管理
_records_lock = threading.Lock()
detection_records: List[dict] = []
_records_dirty = threading.Event()
# 单次 API 返回的最大记录帧数
_MAX_RECORD_FRAMES_PER_REQUEST = 20
# 小模型告警 VLM 复核待更新记录映射 (camera_id, dtype) -> deque(record_id, ...)
_pending_reviews: Dict[Tuple[str, str], "deque[str]"] = {}
_pending_reviews_lock = threading.Lock()

def log_message(msg: str, level: str = "info"):
    """记录日志"""
    timestamp = time.strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    
    with _status_lock:
        _system_status["logs"].append({
            "time": timestamp,
            "level": level,
            "message": msg
        })
    
    if level == "error":
        logger.error(msg)
    elif level == "warning":
        logger.warning(msg)
    else:
        logger.info(msg)


def encode_frame_to_bytes(frame: np.ndarray, quality: int = 70) -> bytes:
    """将帧编码为 JPEG 字节"""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes()


def _save_detection_frames_async(
    record_id: str,
    detection_frames: List[Tuple[float, bytes]],
):
    """后台保存检测帧序列。"""
    try:
        for i, (ts, jpg_bytes) in enumerate(detection_frames):
            storage.save_image(record_id, "frame", jpg_bytes, i)
        log_message(f"Saved {len(detection_frames)} detection frames for {record_id}")
    except Exception as e:
        logger.error(f"Failed to save detection frames for {record_id}: {e}")


def on_trigger(camera_id: str, dtype: str, frame: Optional[np.ndarray], result: dict):
    """检测触发回调（创建告警记录）"""
    global detection_records

    with _status_lock:
        _system_status["total_detections"] += 1

    log_message(f"Camera {camera_id}: {dtype} detected, level={result.get('level', 'small_model_alarm')}")

    trigger_ts = time.time()
    record_id = f"{camera_id}_{dtype}_{int(trigger_ts * 1000)}"

    # 保存快照（只画触发类型的框）
    if frame is not None:
        trigger_results = {dtype: result}
        annotated = MultiDetector._annotate_frame(frame, trigger_results, camera_id, [])
        if _global_settings.get("save_image_timestamp", True):
            annotated = draw_timestamp_on_frame(annotated.copy(), trigger_ts)
        snapshot_bytes = encode_frame_to_bytes(annotated, quality=_global_settings.get("snapshot_quality", 70))
        storage.save_image(record_id, "snapshot", snapshot_bytes)

    record = create_record(camera_id, dtype, result, record_id=record_id)
    record["time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(trigger_ts))

    # 如果该类型启用了 VLM 复核，记录待更新映射，供 VLM 回调更新 level
    if result.get("pending_vlm_review"):
        with _pending_reviews_lock:
            _pending_reviews.setdefault((camera_id, dtype), deque()).append(record_id)

    with _records_lock:
        detection_records.insert(0, record)
        max_records = _global_settings.get("max_records", 100)
        if len(detection_records) > max_records:
            ratio = _global_settings.get("emergency_cleanup_ratio", 0.2)
            remove_count = max(1, int(len(detection_records) * ratio))
            to_remove = detection_records[-remove_count:]
            for old in to_remove:
                storage.delete_record_images(old["id"])
            detection_records = detection_records[:-remove_count]
    mark_records_dirty()

    # 保存检测帧序列
    try:
        detection_frames = result.get("detection_frames", [])
        record["frame_count"] = len(detection_frames)

        if _save_executor is not None:
            _save_executor.submit(
                _save_detection_frames_async,
                record_id,
                detection_frames,
            )
        else:
            _save_detection_frames_async(
                record_id,
                detection_frames,
            )
    except Exception as e:
        logger.error(f"Failed to schedule detection frames save for {record_id}: {e}")


def _convert_ultralytics_result(dtype: str, result) -> Optional[dict]:
    """将 ultralytics Results 转换为 SafetyDetector 风格的 dict"""
    type_def = registry.get(dtype)
    is_pose = type_def.get("post_process") == "yolo_pose" if type_def else False

    if result is None or result.boxes is None or len(result.boxes) == 0:
        if is_pose:
            return {"detected": False, "boxes": [], "scores": [], "subjects": [], "count": 0}
        return {"detected": False, "boxes": [], "scores": [], "max_confidence": 0.0}

    boxes = []
    scores = []
    for b in result.boxes:
        boxes.append(list(map(int, b.xyxy[0])))
        scores.append(float(b.conf[0]))

    if is_pose:
        subjects = []
        detected = False
        count = 0
        if result.keypoints is not None and result.keypoints.data is not None:
            for i in range(len(result.boxes)):
                bbox = result.boxes.xyxy[i].cpu().numpy()
                kp = result.keypoints.data[i].cpu().numpy()
                if len(kp) >= 17:
                    from safety_detection.sleep_detect import analyze_sleep
                    info = analyze_sleep(kp, bbox)
                    subjects.append({
                        "box": bbox.tolist(),
                        "score": float(result.boxes.conf[i]),
                        "sleeping": info["is_sleeping"],
                        "posture_label": info["posture_label"],
                        "sleep_confidence": info["sleep_confidence"],
                        "keypoints": kp,
                    })
                    if info["is_sleeping"]:
                        detected = True
                        count += 1
        return {
            "detected": detected,
            "boxes": boxes,
            "scores": scores,
            "subjects": subjects,
            "count": count,
            "max_confidence": max(scores) if scores else 0.0,
        }

    max_conf = max(scores) if scores else 0.0
    return {
        "detected": len(boxes) > 0,
        "boxes": boxes,
        "scores": scores,
        "max_confidence": max_conf,
    }


# ── 初始化 ──

def init_components():
    """初始化所有组件（新架构）"""
    global camera_manager, safety_detector, multi_detector
    global vlm_queue, vlm_inspector, stream_server, _global_settings, gpu_scheduler

    log_message("Initializing Sentry Safety Detection System...")

    # 1. 加载全局配置
    _global_settings = app_config.load_global_settings()
    log_message(f"Global settings loaded")

    # 2. 初始化摄像头管理器
    camera_manager = CameraManager()
    app.state.camera_manager = camera_manager

    # 从配置加载摄像头（应用全局默认值）
    camera_globals = app_config.load_camera_globals()
    camera_configs_data = app_config.load_camera_configs()
    for cam_data in camera_configs_data:
        # 应用全局默认值（不覆盖已有配置）
        cam_data = app_config.apply_camera_globals(cam_data, camera_globals)
        cfg = CameraConfig(
            camera_id=cam_data["camera_id"],
            source=cam_data["source"],
            name=cam_data.get("name", ""),
            enabled=cam_data.get("enabled", True),
            width=cam_data.get("width", 640),
            height=cam_data.get("height", 480),
            fps=cam_data.get("fps", 15),
            source_type=cam_data.get("source_type", "auto"),
            detection_types=cam_data.get("detection_types"),
        )
        camera_manager.register_camera(cfg)
        # 不在此处注册所有流缓冲，只由 set_main_camera 注册主画面

    log_message(f"Registered {len(camera_configs_data)} cameras")

    # 3. 检测设备优先级: GPU > NPU > CPU
    device, npu_cores = detect_best_device()
    use_npu = device == "npu"
    log_message(f"Detection device: {device}, npu_cores={npu_cores}")

    # 4. 初始化 SafetyDetector
    safety_detector = SafetyDetector(npu_cores=npu_cores, device=device)
    app.state.safety_detector = safety_detector
    # 懒加载所有摄像头启用的模型类型
    all_enabled_types: set = set()
    for cam_data in camera_configs_data:
        for dtype, cfg in cam_data.get("detection_types", {}).items():
            if cfg.get("enabled", False):
                all_enabled_types.add(dtype)
    use_gpu_scheduler = _global_settings.get("use_gpu_scheduler", app_config.USE_GPU_SCHEDULER)
    if all_enabled_types:
        if use_gpu_scheduler and device == "gpu":
            log_message(f"GPU scheduler mode: deferring model load to scheduler")
        else:
            safety_detector.ensure_models_loaded(list(all_enabled_types))
            log_message(f"Models loaded for types: {list(all_enabled_types)} on {device}")

    # 5. 初始化 VLMQueue
    understander = VideoUnderstander()
    vlm_queue = VLMQueue(
        understander=understander,
        max_concurrent=_global_settings.get("vlm_max_concurrent", 3),
    )

    # 6. 选择调度策略
    if npu_cores >= 2:
        strategy = CorePinnedStrategy()
        log_message("Using CorePinnedStrategy")
    else:
        strategy = SerialStrategy()
        log_message("Using SerialStrategy")

    # 7. 初始化 MultiDetector

    def on_vlm_result(camera_id: str, dtype: str, vlm_result: dict):
        """VLM 复核/确认结果回调：更新已有记录 level，不改 status"""
        global detection_records
        record_id = None
        with _pending_reviews_lock:
            review_queue = _pending_reviews.get((camera_id, dtype))
            if review_queue:
                record_id = review_queue.popleft()
                if not review_queue:
                    del _pending_reviews[(camera_id, dtype)]
        if not record_id:
            return

        updated_record = None
        with _records_lock:
            for r in detection_records:
                if r.get("id") == record_id:
                    apply_vlm_review(r, vlm_result)
                    updated_record = r
                    break
        if updated_record:
            mark_records_dirty()
            log_message(f"Record {record_id} updated by VLM: level={updated_record['level']}, status={updated_record['status']}")

    multi_detector = MultiDetector(
        camera_manager=camera_manager,
        safety_detector=safety_detector,
        vlm_queue=vlm_queue,
        strategy=strategy,
        trigger_callback=on_trigger,
        vlm_result_callback=on_vlm_result,
    )
    app.state.multi_detector = multi_detector

    # 8. 初始化选中摄像头独立显示模块（与检测解耦）
    global selected_camera_display
    display_types = _global_settings.get("display_detection_types", app_config.DEFAULT_GLOBAL_SETTINGS["display_detection_types"])
    display_interval = _global_settings.get("display_detection_interval", app_config.DEFAULT_GLOBAL_SETTINGS["display_detection_interval"])
    selected_camera_display = SelectedCameraDisplay(
        camera_manager=camera_manager,
        stream_server=stream_server,
        npu_cores=npu_cores,
        device=device,
        display_types=display_types,
        display_interval=display_interval,
    )

    # 注册摄像头到检测器（detection_types 已由 apply_camera_globals 填充全局默认值）
    for cam_data in camera_configs_data:
        detection_types = cam_data.get("detection_types", {})
        if detection_types:
            multi_detector.register_camera(cam_data["camera_id"], detection_types)

    # 7.5 初始化 GPU 动态调度器（可选，仅 GPU 模式）
    gpu_scheduler = None
    app.state.gpu_scheduler = None
    if use_gpu_scheduler and device == "gpu":
        try:
            from gpu_scheduler import ModelConfig, GPUDynamicScheduler
            from inference_engine import _resolve_model_path

            def _gpu_on_result(cam_id: str, dtype: str, result):
                if multi_detector is None:
                    return
                res_dict = _convert_ultralytics_result(dtype, result)
                if res_dict is None:
                    return
                frame = getattr(result, "orig_img", None)
                multi_detector._latest_results.setdefault(cam_id, {})[dtype] = res_dict
                with multi_detector._lock:
                    schedule = multi_detector._schedules.get(cam_id, {}).get(dtype)
                    if schedule is None:
                        return
                    now = time.time()
                    if multi_detector.is_in_cooldown(cam_id, dtype, now):
                        return
                    schedule.last_run = now
                    try:
                        multi_detector._handle_standard_detection(cam_id, dtype, frame, res_dict, schedule)
                    except Exception as e:
                        logger.error(f"GPU scheduler result handling error [{cam_id}/{dtype}]: {e}")

            model_configs = {}
            # 为每个检测类型单独创建 ModelConfig，即使它们共享同一个模型文件。
            # 这样 GPUDynamicScheduler 才能为 fire/smoke 等共享模型的类型分别调度
            # 并按各自的 classes 过滤结果；否则后加载的类型会被跳过，导致漏检。
            for dtype in registry.all_types():
                type_def = registry.get(dtype)
                model_path = _resolve_model_path(dtype, use_npu=False)
                if not model_path:
                    continue
                classes = type_def.get("classes")
                confidence = type_def.get("model_confidence", 0.5)
                model_configs[dtype] = ModelConfig(
                    model_path, dtype, device="cuda",
                    confidence=confidence,
                    classes=classes,
                )

            num_queues = _global_settings.get("gpu_scheduler_num_queues", app_config.GPU_SCHEDULER_NUM_QUEUES) or None
            gpu_scheduler = GPUDynamicScheduler(
                camera_manager=camera_manager,
                model_configs=model_configs,
                num_queues=num_queues,
                interval=_global_settings.get("gpu_scheduler_interval", app_config.GPU_SCHEDULER_INTERVAL),
                on_result=_gpu_on_result,
                half=_global_settings.get("gpu_scheduler_half", app_config.GPU_SCHEDULER_HALF),
                cooldown_checker=multi_detector.is_in_cooldown,
            )
            log_message(f"GPU scheduler initialized: {len(model_configs)} models, {gpu_scheduler.num_queues} queues")
            app.state.gpu_scheduler = gpu_scheduler

            # 让 MultiDetector 跳过已由 GPU scheduler 推理的类型，避免重复检测
            scheduler_types = list(model_configs.keys())
            for cam_data in camera_configs_data:
                camera_id = cam_data["camera_id"]
                multi_detector.mark_externally_managed(camera_id, scheduler_types)
        except Exception as e:
            log_message(f"GPU scheduler init failed: {e}", "error")
            gpu_scheduler = None

    # 8. 初始化告警图片后台保存线程池（避免阻塞检测流水线）
    global _save_executor
    if _save_executor is not None:
        _save_executor.shutdown(wait=False)
    _save_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=4, thread_name_prefix="save-image"
    )

    # 8. 初始化 VLMInspector
    vlm_inspector = VLMInspector(
        camera_manager=camera_manager,
        multi_detector=multi_detector,
        vlm_queue=vlm_queue,
        understander=understander,
        interval=_global_settings.get("vlm_inspection_interval", 30.0),
        max_cameras_per_inspection=3,
    )

    # 9. 加载历史记录
    global detection_records
    detection_records = storage.load_records()
    log_message(f"Loaded {len(detection_records)} historical records")

    # 10. 启动后台保存线程
    threading.Thread(target=_records_saver_loop, daemon=True).start()

    # 11. 启动存储清理线程
    def _cleanup_loop():
        storage.storage_cleanup_loop(
            max_records=_global_settings.get("max_records", 100),
            max_storage_mb=_global_settings.get("max_storage_mb", 500),
            memory_threshold_percent=_global_settings.get("memory_threshold_percent", 80),
            emergency_cleanup_ratio=_global_settings.get("emergency_cleanup_ratio", 0.2),
            interval_seconds=3600,
        )
    threading.Thread(target=_cleanup_loop, daemon=True).start()

    # 更新系统状态
    with _status_lock:
        _system_status["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _system_status["camera_count"] = len(camera_configs_data)

    log_message("All components initialized successfully")


def _pick_default_main_camera() -> Optional[str]:
    """选择第一个启用的摄像头作为默认主画面"""
    if camera_manager is None:
        return None
    cam_ids = camera_manager.get_camera_ids()
    if not cam_ids:
        return None
    for cid in cam_ids:
        state = camera_manager._cameras.get(cid)
        if state and state.config.enabled:
            return cid
    return cam_ids[0]


def set_main_camera(camera_id: Optional[str]):
    """切换选中的摄像头，同步更新显示模块。保留旧流缓冲，避免前端切换时连接抖动。"""
    global stream_server

    if camera_manager:
        camera_manager.set_main_camera(camera_id)

    if camera_id:
        stream_server.register_camera(camera_id)
        log_message(f"Registered stream for selected camera {camera_id}")

    if selected_camera_display is not None:
        selected_camera_display.set_selected_camera(camera_id)


def _records_saver_loop():
    """后台记录保存线程"""
    while True:
        _records_dirty.wait()
        _records_dirty.clear()
        time.sleep(1)
        
        try:
            with _records_lock:
                data = list(detection_records)
            storage.save_records(data)
        except Exception as e:
            logger.error(f"Failed to save records: {e}")


def mark_records_dirty():
    """标记记录需要保存"""
    _records_dirty.set()


# ── 视频流生成 ──

def generate_camera_frames(camera_id: str, raw: bool = False):
    """生成单摄像头的 MJPEG 视频流 - 使用高性能流服务器
    Args:
        raw: True 输出原始帧流，False 输出标注帧流
    """
    yield from stream_server.generate_frames(camera_id, raw=raw)


# ── API 端点 ──

@app.get("/")
async def root():
    """根路径 - 返回监控中心"""
    fp = Path(__file__).parent.parent / "frontend" / "safety_detection" / "monitor.html"
    if fp.exists():
        return HTMLResponse(fp.read_text(encoding="utf-8"))
    return {
        "message": "Sentry Multi-Camera API",
        "version": "2.0",
        "cameras": len(camera_manager._cameras) if camera_manager else 0,
        "endpoints": [
            "/monitor - 监控中心",
            "/records.html - 检测记录",
            "/settings.html - 系统设置",
            "/cameras - 摄像头列表",
            "/status - 系统状态",
        ]
    }


@app.get("/monitor")
async def monitor_view():
    """Glass-clay 风格监控中心"""
    fp = Path(__file__).parent.parent / "frontend" / "safety_detection" / "monitor.html"
    if fp.exists():
        return HTMLResponse(fp.read_text(encoding="utf-8"))
    return {"error": "Monitor page not found"}


@app.get("/cameras")
async def list_cameras():
    """列出所有摄像头（包含 detection_types）"""
    if camera_manager is None:
        return {"cameras": []}

    status_list = camera_manager.get_all_status()

    # 合并检测状态和配置
    cameras_config = {c["camera_id"]: c for c in app_config.load_camera_configs()}
    if multi_detector:
        det_states = multi_detector.get_all_states()
        for s in status_list:
            cam_id = s["camera_id"]
            if cam_id in det_states:
                s["detection"] = det_states[cam_id]
            cfg = cameras_config.get(cam_id, {})
            s["source_type"] = cfg.get("source_type", "auto")
            s["detection_types"] = cfg.get("detection_types", {})

    return {"cameras": status_list}


@app.post("/cameras/{camera_id}/select")
async def select_main_camera(camera_id: str):
    """切换主画面摄像头"""
    if camera_manager is None or camera_id not in camera_manager._cameras:
        return JSONResponse({"error": "Camera not found"}, status_code=404)
    set_main_camera(camera_id)
    return {"success": True, "main_camera": camera_id}


@app.get("/cameras/{camera_id}/stream")
async def camera_stream(camera_id: str, raw: bool = False):
    """单摄像头视频流
    Args:
        raw: True 返回原始帧（无画框），False 返回标注帧（带画框）
    """
    if camera_manager is None or camera_id not in camera_manager._cameras:
        return JSONResponse({"error": "Camera not found"}, status_code=404)
    if camera_id != camera_manager.get_main_camera():
        return JSONResponse({"error": "Camera is not the main stream"}, status_code=404)
    return StreamingResponse(
        generate_camera_frames(camera_id, raw=raw),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/status")
async def get_status():
    """获取系统状态"""
    with _status_lock:
        status = dict(_system_status)
        status["logs"] = list(_system_status["logs"])
    
    # 添加检测器状态
    if gpu_scheduler:
        status["detector"] = {
            "mode": "gpu_scheduler",
            "loaded_models": list(gpu_scheduler.detectors.keys()),
            "queues": gpu_scheduler.num_queues,
        }
    elif safety_detector:
        status["detector"] = {
            "loaded_models": safety_detector.loaded_models,
            "model_status": safety_detector.get_model_status(),
        }
    
    # 添加摄像头状态
    if camera_manager:
        cameras = camera_manager.get_all_status()
        status["active_cameras"] = sum(1 for c in cameras if c["status"] == "connected")
        # 汇总所有摄像头的解码后端
        status["decoder_backends"] = {
            c["camera_id"]: c.get("decoder_backend", "cpu")
            for c in cameras
        }

    # 修正 total_detections 为实际记录数（包含从磁盘加载的历史记录）
    with _records_lock:
        status["total_detections"] = len(detection_records)
        status["recent_records"] = [
            {
                "id": r.get("id"),
                "camera_id": r.get("camera_id", "unknown"),
                "time": r.get("time"),
                "detection_type": r.get("detection_type"),
                "level": r.get("level"),
                "confidence": r.get("confidence"),
            }
            for r in detection_records[:10]
        ]
    
    return status


@app.get("/records")
async def get_records():
    """获取所有记录"""
    with _records_lock:
        records = []
        for r in detection_records:
            item = {
                "id": r.get("id"),
                "camera_id": r.get("camera_id", "unknown"),
                "time": r.get("time"),
                "action": r.get("action"),
                "confidence": r.get("confidence"),
                "reason": r.get("reason"),
                "timing": r.get("timing"),
            }
            # 加载快照缩略图
            snap = storage.load_image_b64(r.get("id", ""), "snapshot")
            if snap:
                item["snapshot"] = snap
            records.append(item)
        return {"records": records}


@app.get("/records/summary")
async def get_records_summary():
    """获取记录统计摘要"""
    return storage.get_record_summary()


@app.get("/record/{record_id}")
async def get_record(record_id: str, include_frames: bool = True):
    """获取单条记录详情"""
    import urllib.parse
    # URL 解码
    decoded_id = urllib.parse.unquote(record_id)
    
    logger.info(f"Looking for record: {decoded_id}")
    
    with _records_lock:
        meta = None
        for r in detection_records:
            if r.get("id") == decoded_id:
                meta = dict(r)
                break
    
    if meta is None:
        logger.warning(f"Record not found: {decoded_id}, available: {[r.get('id') for r in detection_records[:5]]}")
        return JSONResponse({"error": "Record not found"}, status_code=404)
    
    # 检查是否有快照
    snapshot_path = storage.FRAMES_DIR / f"{record_id}_snapshot.jpg"
    meta["has_snapshot"] = snapshot_path.exists()
    
    # 按需加载图片
    if meta["has_snapshot"]:
        snapshot = storage.load_image_b64(record_id, "snapshot")
        if snapshot:
            meta["snapshot"] = snapshot
    
    # 按需加载帧
    if include_frames:
        frame_count = meta.get("frame_count", 0)
        frames = []
        for i in range(min(frame_count, _MAX_RECORD_FRAMES_PER_REQUEST)):
            b64 = storage.load_image_b64(record_id, "frame", i)
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
    return JSONResponse({"error": "Snapshot not found"}, status_code=404)


@app.get("/record/{record_id}/frames")
async def get_record_frames(record_id: str, start: int = 0, count: int = 10):
    """分页获取记录帧图片"""
    import urllib.parse
    decoded_id = urllib.parse.unquote(record_id)
    count = min(count, _MAX_RECORD_FRAMES_PER_REQUEST)  # 限制最大数量
    
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


@app.post("/cameras/{camera_id}/enable")
async def enable_camera(camera_id: str):
    """启用摄像头"""
    if camera_manager is None:
        return JSONResponse({"error": "Camera manager not initialized"}, status_code=500)
    
    # 更新配置中的 enabled 状态
    with camera_manager._lock:
        if camera_id in camera_manager._cameras:
            camera_manager._cameras[camera_id].config.enabled = True
    
    camera_manager.start_camera(camera_id)
    
    # 保存配置到文件
    save_camera_configs()
    
    return {"success": True, "camera_id": camera_id}


@app.post("/cameras/{camera_id}/disable")
async def disable_camera(camera_id: str):
    """禁用摄像头"""
    if camera_manager is None:
        return JSONResponse({"error": "Camera manager not initialized"}, status_code=500)
    
    # 更新配置中的 enabled 状态
    with camera_manager._lock:
        if camera_id in camera_manager._cameras:
            camera_manager._cameras[camera_id].config.enabled = False
    
    camera_manager.stop_camera(camera_id)
    
    # 保存配置到文件
    save_camera_configs()
    
    return {"success": True, "camera_id": camera_id}


# ── 全局设置 API ──

@app.get("/settings")
async def get_settings():
    """获取全局配置（包含摄像头全局默认参数）"""
    settings = app_config.load_global_settings()
    camera_globals = app_config.load_camera_globals()
    settings["default_detection_types"] = camera_globals.get("detection_types", {})
    settings["default_camera_width"] = camera_globals.get("width", 640)
    settings["default_camera_height"] = camera_globals.get("height", 480)
    settings["default_camera_fps"] = camera_globals.get("fps", 15)
    return settings


@app.post("/settings")
async def update_settings(data: dict):
    """修改全局配置，实时生效并持久化"""
    try:
        # 分离摄像头全局默认参数
        camera_globals_update = {}
        if "default_detection_types" in data:
            camera_globals_update["detection_types"] = data.pop("default_detection_types")
        for key in ("default_camera_width", "default_camera_height", "default_camera_fps"):
            if key in data:
                camera_globals_update[key.replace("default_camera_", "")] = data.pop(key)

        # 保存摄像头全局默认参数
        if camera_globals_update:
            camera_globals = app_config.load_camera_globals()
            camera_globals.update(camera_globals_update)
            app_config.save_camera_globals(camera_globals)
            log_message(f"Camera globals updated: {list(camera_globals_update.keys())}")

        # 保存全局设置（排除摄像头默认参数）
        settings = app_config.load_global_settings()
        settings.update(data)
        app_config.save_global_settings(settings)
        global _global_settings
        _global_settings = settings

        # 动态更新运行中组件的参数
        if multi_detector:
            # 冷却与 VLM 开关已下沉到 per-type 配置，动态更新通过摄像头配置接口处理
            pass
        if vlm_inspector:
            vlm_inspector.interval = settings.get("vlm_inspection_interval", 30.0)
            new_interval = settings.get("vlm_inspection_interval", 30.0)
            if new_interval > 0 and not vlm_inspector._running:
                vlm_inspector.start()
            elif new_interval <= 0 and vlm_inspector._running:
                vlm_inspector.stop()
        if selected_camera_display and ("display_detection_types" in data or "display_detection_interval" in data):
            selected_camera_display.set_display_config(
                data.get("display_detection_types") if "display_detection_types" in data else None,
                data.get("display_detection_interval") if "display_detection_interval" in data else None,
            )
        log_message("Global settings updated")
        return {"success": True, "settings": settings}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/display-types")
async def get_display_types():
    """获取显示类型开关与刷新频率"""
    settings = app_config.load_global_settings()
    return {
        "display_detection_types": settings.get(
            "display_detection_types", app_config.DEFAULT_GLOBAL_SETTINGS["display_detection_types"]
        ),
        "display_detection_interval": settings.get(
            "display_detection_interval", app_config.DEFAULT_GLOBAL_SETTINGS["display_detection_interval"]
        ),
    }


@app.post("/display-types")
async def update_display_types(data: dict):
    """更新显示类型开关与刷新频率"""
    try:
        display_types = data.get("display_detection_types", {})
        display_interval = data.get("display_detection_interval")
        if not isinstance(display_types, dict):
            return JSONResponse({"error": "display_detection_types must be a dict"}, status_code=400)
        if display_interval is not None:
            try:
                display_interval = float(display_interval)
            except (TypeError, ValueError):
                return JSONResponse({"error": "display_detection_interval must be a number"}, status_code=400)

        settings = app_config.load_global_settings()
        settings["display_detection_types"] = display_types
        if display_interval is not None:
            settings["display_detection_interval"] = SelectedCameraDisplay._clamp_display_interval(display_interval)
        app_config.save_global_settings(settings)
        global _global_settings
        _global_settings = settings

        if selected_camera_display:
            selected_camera_display.set_display_config(
                display_types,
                settings.get("display_detection_interval"),
            )

        log_message(f"Display config updated: types={display_types}, interval={settings.get('display_detection_interval')}")
        return {
            "success": True,
            "display_detection_types": display_types,
            "display_detection_interval": settings.get("display_detection_interval"),
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ── 告警记录 API ──

@app.get("/alerts")
async def get_alerts(
    camera_id: Optional[str] = None,
    level: Optional[str] = None,
    type: Optional[str] = None,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
):
    """分页获取告警记录，支持过滤"""
    records, total = storage.get_records_paginated(
        page=page, page_size=page_size,
        camera_id=camera_id, level=level, dtype=type, status=status,
        date_from=date_from, date_to=date_to,
    )
    return {"records": records, "total": total, "page": page, "page_size": page_size}


@app.get("/alerts/stats")
async def get_alerts_stats():
    """获取告警统计"""
    summary = storage.get_record_summary()
    return {
        "total": summary.get("total", 0),
        "pending": summary.get("by_status", {}).get("pending", 0),
        "confirmed": summary.get("by_status", {}).get("confirmed", 0),
        "false_positive": summary.get("by_status", {}).get("false_positive", 0),
    }


@app.post("/alerts/{record_id}/confirm")
async def confirm_alert(record_id: str):
    """标记告警为已确认"""
    global detection_records
    with _records_lock:
        found = False
        for r in detection_records:
            if r.get("id") == record_id:
                confirm_alarm(r)
                found = True
                break
        if not found:
            return JSONResponse({"error": "Record not found"}, status_code=404)
        data = list(detection_records)
    storage.save_records(data)
    return {"success": True}


@app.post("/alerts/{record_id}/ignore")
async def ignore_alert(record_id: str):
    """标记告警为误报"""
    global detection_records
    with _records_lock:
        found = False
        for r in detection_records:
            if r.get("id") == record_id:
                confirm_false_positive(r)
                found = True
                break
        if not found:
            return JSONResponse({"error": "Record not found"}, status_code=404)
        data = list(detection_records)
    storage.save_records(data)
    return {"success": True}


# ── 摄像头配置 API ──

@app.post("/cameras/{camera_id}/config")
async def update_camera_config(camera_id: str, data: dict):
    """动态修改单路摄像头配置（检测类型 + 名称/源/分辨率等）"""
    if camera_manager is None or multi_detector is None:
        return JSONResponse({"error": "Not initialized"}, status_code=500)

    detection_types = data.get("detection_types", {})
    if detection_types:
        multi_detector.update_camera_config(camera_id, detection_types)

    # 更新运行时配置
    state = camera_manager._cameras.get(camera_id)
    if state:
        cfg = state.config
        if "name" in data:
            cfg.name = data["name"]
        if "source" in data and "source_type" in data:
            camera_manager.set_camera_source(camera_id, data["source"], data["source_type"])
        if "width" in data:
            cfg.width = int(data["width"])
        if "height" in data:
            cfg.height = int(data["height"])
        if "enabled" in data:
            cfg.enabled = bool(data["enabled"])
            if cfg.enabled:
                camera_manager.start_camera(camera_id)
            else:
                camera_manager.stop_camera(camera_id)
        if detection_types:
            cfg.detection_types = detection_types

    # 持久化
    cameras = app_config.load_camera_configs()
    for cam in cameras:
        if cam["camera_id"] == camera_id:
            if "name" in data:
                cam["name"] = data["name"]
            if "source" in data:
                cam["source"] = data["source"]
            if "source_type" in data:
                cam["source_type"] = data["source_type"]
            if "width" in data:
                cam["width"] = int(data["width"])
            if "height" in data:
                cam["height"] = int(data["height"])
            if "enabled" in data:
                cam["enabled"] = bool(data["enabled"])
            if detection_types:
                cam["detection_types"] = detection_types
            break
    app_config.save_camera_configs(cameras)

    log_message(f"Camera {camera_id} config updated")
    return {"success": True}


@app.post("/cameras/batch-config")
async def batch_camera_config(data: dict):
    """批量修改多路摄像头配置"""
    if camera_manager is None or multi_detector is None:
        return JSONResponse({"error": "Not initialized"}, status_code=500)

    camera_ids = data.get("camera_ids", [])
    detection_types = data.get("detection_types", {})

    cameras = app_config.load_camera_configs()
    for cam in cameras:
        if cam["camera_id"] in camera_ids:
            cam["detection_types"] = detection_types
            multi_detector.update_camera_config(cam["camera_id"], detection_types)

    app_config.save_camera_configs(cameras)
    log_message(f"Batch config updated for {len(camera_ids)} cameras")
    return {"success": True, "updated": camera_ids}


@app.post("/cameras/{camera_id}/reset-config")
async def reset_camera_config(camera_id: str):
    """将单个摄像头配置恢复为全局默认值（保留 source/name/camera_id）"""
    if camera_manager is None or multi_detector is None:
        return JSONResponse({"error": "Not initialized"}, status_code=500)

    cameras = app_config.load_camera_configs()
    target_cam = None
    for cam in cameras:
        if cam["camera_id"] == camera_id:
            target_cam = cam
            break

    if target_cam is None:
        return JSONResponse({"error": "Camera not found"}, status_code=404)

    # 加载全局默认值
    camera_globals = app_config.load_camera_globals()

    # 构建恢复后的配置（保留摄像头身份信息和视频源）
    restored = {
        "camera_id": target_cam["camera_id"],
        "source": target_cam.get("source", ""),
        "name": target_cam.get("name", ""),
        "enabled": target_cam.get("enabled", True),
        "source_type": target_cam.get("source_type", "auto"),
    }
    # 其他参数全部用全局默认值覆盖
    restored["width"] = camera_globals.get("width", 640)
    restored["height"] = camera_globals.get("height", 480)
    restored["fps"] = camera_globals.get("fps", 15)
    restored["detection_types"] = {
        k: dict(v) for k, v in camera_globals.get("detection_types", {dtype: registry.get_defaults(dtype) for dtype in registry.all_types()}).items()
    }

    # 更新内存中的摄像头配置
    state = camera_manager._cameras.get(camera_id)
    if state:
        state.config.width = restored["width"]
        state.config.height = restored["height"]
        state.config.fps = restored["fps"]
        state.config.detection_types = restored["detection_types"]

    # 更新检测器
    multi_detector.update_camera_config(camera_id, restored["detection_types"])

    # 持久化
    for cam in cameras:
        if cam["camera_id"] == camera_id:
            cam.update(restored)
            break
    app_config.save_camera_configs(cameras)

    log_message(f"Camera {camera_id} config reset to global defaults")
    return {"success": True, "camera_id": camera_id}


# ── 视频源切换 API ──

@app.post("/cameras/{camera_id}/source")
async def switch_camera_source(camera_id: str, data: dict):
    """切换视频源（摄像头索引 / RTSP / 本地视频文件路径）"""
    if camera_manager is None:
        return JSONResponse({"error": "Camera manager not initialized"}, status_code=500)

    source = data.get("source")
    source_type = data.get("source_type", "auto")
    if not source:
        return JSONResponse({"error": "source is required"}, status_code=400)

    success = camera_manager.set_camera_source(camera_id, source, source_type)
    if success:
        # 持久化
        cameras = app_config.load_camera_configs()
        for cam in cameras:
            if cam["camera_id"] == camera_id:
                cam["source"] = source
                cam["source_type"] = source_type
                break
        app_config.save_camera_configs(cameras)
        log_message(f"Camera {camera_id} source switched to {source}")
        return {"success": True}
    return JSONResponse({"error": "Failed to switch source"}, status_code=400)


def save_camera_configs():
    """保存摄像头配置到文件（新格式）"""
    try:
        cameras_data = []
        if camera_manager:
            for cam_id, state in camera_manager._cameras.items():
                cfg = state.config
                cam_data = {
                    "camera_id": cfg.camera_id,
                    "source": cfg.source,
                    "name": cfg.name,
                    "enabled": cfg.enabled,
                    "width": cfg.width,
                    "height": cfg.height,
                    "fps": cfg.fps,
                    "source_type": cfg.source_type,
                    "detection_types": cfg.detection_types or {},
                }
                if cfg.detection_roi:
                    cam_data["detection_roi"] = cfg.detection_roi
                cameras_data.append(cam_data)

        app_config.save_camera_configs(cameras_data)
        log_message("Camera configs saved")
        return True
    except Exception as e:
        log_message(f"Failed to save camera configs: {e}", "error")
        return False


@app.post("/cameras/add")
async def add_camera(data: dict):
    """动态添加摄像头"""
    if camera_manager is None:
        return JSONResponse({"error": "Camera manager not initialized"}, status_code=500)

    try:
        from camera_manager import CameraConfig

        # 新摄像头：自动应用全局默认值（传入的配置可覆盖）
        camera_globals = app_config.load_camera_globals()
        cam_data = dict(data)
        cam_data = app_config.apply_camera_globals(cam_data, camera_globals)

        cfg = CameraConfig(
            camera_id=cam_data["camera_id"],
            source=cam_data["source"],
            name=cam_data.get("name", ""),
            enabled=cam_data.get("enabled", True),
            width=cam_data.get("width", 640),
            height=cam_data.get("height", 480),
            fps=cam_data.get("fps", 15),
            source_type=cam_data.get("source_type", "auto"),
            detection_types=cam_data.get("detection_types"),
        )

        success = camera_manager.register_camera(cfg)
        if not success:
            return JSONResponse({"error": "Camera ID already exists"}, status_code=400)

        # 不在这里注册流缓冲；只有主画面才注册
        # 如果当前没有主画面，可以设为新主画面
        if camera_manager.get_main_camera() is None:
            set_main_camera(cfg.camera_id)

        if multi_detector and cfg.detection_types:
            multi_detector.register_camera(cfg.camera_id, cfg.detection_types)

        if cfg.enabled:
            camera_manager.start_camera(cfg.camera_id)

        save_camera_configs()

        log_message(f"Camera {cfg.camera_id} added dynamically")
        return {"success": True, "camera_id": cfg.camera_id}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.delete("/cameras/{camera_id}")
async def delete_camera(camera_id: str):
    """删除摄像头"""
    if camera_manager is None:
        return JSONResponse({"error": "Camera manager not initialized"}, status_code=500)
    
    # 从检测器注销
    if multi_detector:
        multi_detector.unregister_camera(camera_id)
    
    # 从流服务器注销
    stream_server.unregister_camera(camera_id)
    
    # 从管理器注销
    success = camera_manager.unregister_camera(camera_id)

    # 如果删除的是主画面，自动选择新的默认主画面
    if success and camera_manager.get_main_camera() is None:
        new_main = _pick_default_main_camera()
        if new_main:
            set_main_camera(new_main)

    if success:
        # 保存配置到文件
        save_camera_configs()

        log_message(f"Camera {camera_id} deleted")
        return {"success": True}
    else:
        return JSONResponse({"error": "Camera not found"}, status_code=404)


@app.get("/records.html")
async def records_page():
    """记录页面"""
    fp = Path(__file__).parent.parent / "frontend" / "safety_detection" / "records.html"
    if fp.exists():
        return HTMLResponse(fp.read_text(encoding="utf-8"))
    return {"error": "Records page not found"}


@app.get("/settings.html")
async def settings_page():
    """设置页面"""
    fp = Path(__file__).parent.parent / "frontend" / "safety_detection" / "settings.html"
    if fp.exists():
        return HTMLResponse(fp.read_text(encoding="utf-8"))
    return {"error": "Settings page not found"}


@app.get("/types.html")
async def types_page():
    """类型管理页面"""
    fp = Path(__file__).parent.parent / "frontend" / "safety_detection" / "types.html"
    if fp.exists():
        return HTMLResponse(fp.read_text(encoding="utf-8"))
    return {"error": "Types page not found"}


@app.get("/system/mode")
async def get_system_mode():
    """获取当前运行模式和检测设备"""
    mode = os.environ.get("SENTRY_MODE", "multi")
    device, npu_cores = detect_best_device()
    return {"mode": mode, "device": device, "npu_cores": npu_cores}


@app.post("/system/restart")
async def restart_system():
    """重启检测服务（重新初始化所有组件）"""
    try:
        log_message("System restart requested via API")
        # 停止现有组件
        stop_selected_camera_display()
        if vlm_inspector:
            vlm_inspector.stop()
        if vlm_queue:
            vlm_queue.stop()
        if gpu_scheduler:
            gpu_scheduler.stop()
        elif multi_detector:
            multi_detector.stop()
        if camera_manager:
            camera_manager.stop_all()
        if safety_detector:
            safety_detector.release()

        # 关闭旧保存线程池
        global _save_executor
        if _save_executor is not None:
            _save_executor.shutdown(wait=False)
            _save_executor = None

        # 重新初始化
        init_components()
        if camera_manager:
            camera_manager.start_all()
        if gpu_scheduler:
            gpu_scheduler.start()
        elif multi_detector:
            multi_detector.start()
        if vlm_queue:
            vlm_queue.start()
        if vlm_inspector and _global_settings.get("vlm_inspection_interval", 30.0) > 0:
            vlm_inspector.start()
        start_selected_camera_display()

        log_message("System restart completed")
        return {"success": True, "message": "Detection service restarted"}
    except Exception as e:
        log_message(f"Restart failed: {e}", "error")
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------------------------------------------
# 选中摄像头独立显示模块（与检测完全解耦）
# ------------------------------------------------------------------
selected_camera_display: Optional["SelectedCameraDisplay"] = None


class SelectedCameraDisplay:
    """
    独立显示模块：只负责当前选中摄像头的前端展示。
    - 自己单独解码，25 FPS 流畅播放。
    - 每秒独立检测一次（只测显示开关开启的类型），只画框，不告警。
    - 检测与推流分两个线程，避免检测耗时阻塞前端帧率。
    """

    def __init__(
        self,
        camera_manager,
        stream_server,
        npu_cores: int,
        device: str,
        display_types: Optional[Dict[str, bool]] = None,
        display_interval: float = 1.0,
    ):
        self.camera_manager = camera_manager
        self.stream_server = stream_server
        self._npu_cores = npu_cores
        self._device = device
        self._display_types: Dict[str, bool] = dict(display_types) if display_types else {}
        self._display_interval: float = self._clamp_display_interval(display_interval)
        self._lock = threading.RLock()
        self._running = False
        self._reader_thread: Optional[threading.Thread] = None
        self._display_thread: Optional[threading.Thread] = None
        self._detect_thread: Optional[threading.Thread] = None
        self._selected_camera_id: Optional[str] = None
        self._active_session_id: int = 0
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_session_id: int = 0
        self._frame_timestamp: float = 0.0
        self._last_detection_results: Dict[str, dict] = {}
        self._overlay_expires_at: float = 0.0

    @staticmethod
    def _clamp_display_interval(value: float) -> float:
        """把刷新频率限制在 0.1 ~ 10.0 秒之间。"""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 1.0
        if value < 0.1:
            return 0.1
        if value > 10.0:
            return 10.0
        return value

    def set_selected_camera(self, camera_id: Optional[str]):
        """切换当前选中的摄像头：启动新的独立 reader session，旧 session 自行退出。"""
        with self._lock:
            if self._selected_camera_id == camera_id:
                return
            self._active_session_id += 1
            session_id = self._active_session_id
            self._selected_camera_id = camera_id
            # 不清空 _latest_frame：继续显示旧画面直到新 session 有帧，避免切换黑屏
            self._last_detection_results = {}
            self._overlay_expires_at = 0.0

        if camera_id is not None and self._running:
            threading.Thread(
                target=self._reader_session,
                args=(session_id, camera_id),
                daemon=True,
                name=f"selected-camera-reader-{camera_id}",
            ).start()

    def set_display_types(self, display_types: Dict[str, bool]):
        """更新显示类型开关（向后兼容）。"""
        self.set_display_config(display_types)

    def set_display_config(
        self,
        display_types: Optional[Dict[str, bool]] = None,
        display_interval: Optional[float] = None,
    ):
        """更新显示类型开关和/或刷新频率。"""
        with self._lock:
            if display_types is not None:
                self._display_types = dict(display_types)
                if not any(self._display_types.values()):
                    self._last_detection_results = {}
                    self._overlay_expires_at = 0.0
            if display_interval is not None:
                self._display_interval = self._clamp_display_interval(display_interval)

    def start(self):
        """启动显示线程和检测线程"""
        if self._running:
            return
        self._running = True
        with self._lock:
            session_id = self._active_session_id
            camera_id = self._selected_camera_id
        if camera_id is not None:
            self._reader_thread = threading.Thread(
                target=self._reader_session,
                args=(session_id, camera_id),
                daemon=True,
                name=f"selected-camera-reader-{camera_id}",
            )
            self._reader_thread.start()
        self._display_thread = threading.Thread(
            target=self._display_loop, daemon=True, name="selected-camera-display"
        )
        self._display_thread.start()
        self._detect_thread = threading.Thread(
            target=self._detect_loop, daemon=True, name="selected-camera-detect"
        )
        self._detect_thread.start()
        log_message("SelectedCameraDisplay started")

    def stop(self):
        """停止所有线程"""
        self._running = False
        if self._reader_thread:
            self._reader_thread.join(timeout=2)
            self._reader_thread = None
        if self._display_thread:
            self._display_thread.join(timeout=2)
            self._display_thread = None
        if self._detect_thread:
            self._detect_thread.join(timeout=2)
            self._detect_thread = None
        log_message("SelectedCameraDisplay stopped")

    def _reader_session(self, session_id: int, camera_id: str):
        """主画面读取会话：从 camera_manager 共享帧轮询，切换时旧 session 自行退出。
        不再自建 VideoCapture，消除切换时重建 RTSP 连接的延迟。
        """
        log_message(f"SelectedCameraDisplay reader session started: {camera_id}#{session_id}")
        try:
            while self._running:
                with self._lock:
                    if session_id != self._active_session_id:
                        break

                frame = self.camera_manager.get_latest_frame(camera_id)
                if frame is None:
                    time.sleep(0.04)
                    continue

                if not self._update_session_frame(session_id, frame):
                    break
        finally:
            log_message(f"SelectedCameraDisplay reader session stopped: {camera_id}#{session_id}")

    def _update_session_frame(self, session_id: int, frame: np.ndarray) -> bool:
        """只有当前 session 才能更新最新帧，防止旧 session 迟到覆盖画面。"""
        with self._lock:
            if session_id != self._active_session_id:
                return False
            self._latest_frame = frame
            self._frame_session_id = session_id
            self._frame_timestamp = time.time()
            return True

    def _scale_for_display(self, frame: np.ndarray, camera_id: str) -> np.ndarray:
        """按摄像头配置的最大分辨率等比例缩放，降低推流编码压力（不改变检测分辨率）。"""
        state = self.camera_manager._cameras.get(camera_id)
        if state is None:
            return frame
        max_w = getattr(state.config, "width", 640)
        max_h = getattr(state.config, "height", 480)
        src_h, src_w = frame.shape[:2]
        if src_w <= max_w and src_h <= max_h:
            return frame
        scale = min(max_w / src_w, max_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        return cv2.resize(frame, (new_w, new_h))

    def _draw_timestamp(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        """在帧右上角绘制采集时间戳（白字黑边）。"""
        return draw_timestamp_on_frame(frame, timestamp)

    def _display_loop(self):
        """显示主循环：从缓存取最新帧，25 FPS 推流（只推标注帧，前端不消费原始帧）"""
        log_message("SelectedCameraDisplay display loop started")
        frame_count = 0
        last_log_time = time.time()
        while self._running:
            try:
                with self._lock:
                    camera_id = self._selected_camera_id
                    frame = self._latest_frame
                    results = dict(self._last_detection_results)
                    display_types = dict(self._display_types)
                    frame_session_id = self._frame_session_id
                    active_session_id = self._active_session_id
                    frame_timestamp = self._frame_timestamp

                if camera_id is None or frame is None:
                    time.sleep(0.05)
                    continue

                # 跳过旧 session 残留帧，防止切换后旧帧闪入新摄像头流（A-B-A-B 问题）
                if frame_session_id != active_session_id:
                    time.sleep(0.05)
                    continue

                now = time.time()
                frame_to_push = frame
                enabled_types = [dtype for dtype, enabled in display_types.items() if enabled]
                if results and enabled_types:
                    frame_to_push = MultiDetector._annotate_frame(
                        frame.copy(), results, camera_id, enabled_types
                    )

                display_frame = self._scale_for_display(frame_to_push, camera_id)
                display_frame = self._draw_timestamp(display_frame, frame_timestamp)
                self.stream_server.update_frame(camera_id, display_frame, raw=False)

                frame_count += 1
                if now - last_log_time >= 5.0:
                    fps = frame_count / (now - last_log_time)
                    logger.info(f"SelectedCameraDisplay {camera_id} display FPS: {fps:.1f}")
                    frame_count = 0
                    last_log_time = now

                time.sleep(0.04)
            except Exception as e:
                logger.error(f"SelectedCameraDisplay display loop error: {e}")
                time.sleep(0.1)
        log_message("SelectedCameraDisplay display loop stopped")

    def _detect_loop(self):
        """检测循环：按配置间隔直接对最新帧做推理并更新显示缓存。"""
        log_message("SelectedCameraDisplay detect loop started")
        next_time = time.time()
        while self._running:
            try:
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                next_time = time.time() + self._display_interval

                with self._lock:
                    camera_id = self._selected_camera_id
                    frame = self._latest_frame
                    display_types = dict(self._display_types)

                if camera_id is None or frame is None:
                    continue
                if not any(display_types.values()):
                    continue
                if safety_detector is None:
                    continue

                types_to_detect = [dtype for dtype, enabled in display_types.items() if enabled]
                try:
                    safety_detector.ensure_models_loaded(types_to_detect)
                    results = safety_detector.detect(frame.copy(), types_to_detect)
                except Exception as e:
                    logger.error(f"SelectedCameraDisplay {camera_id} detection error: {e}")
                    continue

                with self._lock:
                    visible_results = {
                        dtype: results.get(dtype)
                        for dtype in types_to_detect
                        if results.get(dtype)
                    }
                    self._overlay_expires_at = time.time() + max(self._display_interval, 0.1)
                    self._last_detection_results = visible_results
            except Exception as e:
                logger.error(f"SelectedCameraDisplay detect loop error: {e}")
        log_message("SelectedCameraDisplay detect loop stopped")


def start_selected_camera_display():
    """启动选中摄像头显示模块"""
    global selected_camera_display
    if selected_camera_display is not None:
        selected_camera_display.start()


def stop_selected_camera_display():
    """停止选中摄像头显示模块"""
    global selected_camera_display
    if selected_camera_display is not None:
        selected_camera_display.stop()


@app.on_event("startup")
async def startup():
    """服务启动：耗时初始化放到后台线程，避免阻塞事件循环，让前端尽快可访问"""
    def _do_startup():
        init_components()

        # 启动摄像头
        if camera_manager:
            camera_manager.start_all()

        # 设置默认主画面
        main_id = _pick_default_main_camera()
        if main_id:
            set_main_camera(main_id)

        # 启动检测器
        if gpu_scheduler:
            gpu_scheduler.start()
            log_message("GPU scheduler started")
        elif multi_detector:
            multi_detector.start()

        # 启动 VLM 队列
        if vlm_queue:
            vlm_queue.start()

        # 启动 VLM 巡检（由 vlm_inspection_interval 控制）
        if vlm_inspector and _global_settings.get("vlm_inspection_interval", 30.0) > 0:
            vlm_inspector.start()

        # 启动选中摄像头独立显示模块（画框 + 送流，与检测解耦）
        start_selected_camera_display()

        log_message(f"Sentry Safety Detection started on {app_config.API_HOST}:{app_config.API_PORT}")
        log_message(f"Access the monitoring center at http://{app_config.API_HOST}:{app_config.API_PORT}/monitor")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _do_startup)


@app.on_event("shutdown")
async def shutdown():
    """服务关闭"""
    log_message("Shutting down...")
    stop_selected_camera_display()

    if vlm_inspector:
        vlm_inspector.stop()

    if vlm_queue:
        vlm_queue.stop()

    if gpu_scheduler:
        gpu_scheduler.stop()
    elif multi_detector:
        multi_detector.stop()

    if camera_manager:
        camera_manager.stop_all()

    if safety_detector:
        safety_detector.release()

    log_message("Shutdown complete")


# ── 主入口 ──

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=app_config.API_HOST,
        port=app_config.API_PORT,
        log_level="warning"
    )
