"""
有限空间监控 REST API
提供区域管理、状态查询、事件记录、系统控制等端点
"""

import asyncio
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Optional

from .zone_counter import ZoneConfig

router = APIRouter(prefix="/api", tags=["confined-space"])


def _get_zone_counter(request: Request):
    return getattr(request.app.state, "zone_counter", None)


def _get_storage(request: Request):
    return getattr(request.app.state, "storage", None)


def _get_camera_manager(request: Request):
    return getattr(request.app.state, "camera_manager", None)


def _persist_camera(data: dict) -> None:
    """把摄像头+区域配置写入 confined_cameras.json（扁平格式）"""
    import config as app_config
    cameras = app_config.load_confined_cameras()
    camera_id = data.get("camera_id")
    cameras = [c for c in cameras if c.get("camera_id") != camera_id]
    cameras.append(data)
    app_config.save_confined_cameras(cameras)


def _remove_camera_from_config(camera_id: str) -> None:
    """从 confined_cameras.json 移除指定摄像头"""
    import config as app_config
    cameras = app_config.load_confined_cameras()
    cameras = [c for c in cameras if c.get("camera_id") != camera_id]
    app_config.save_confined_cameras(cameras)


def _sync_create_camera(data: dict, camera_manager, zone_counter) -> str:
    """同步执行摄像头注册与持久化（供 asyncio.to_thread 使用）"""
    from camera_manager import CameraConfig
    camera_id = str(data.get("camera_id", "")).strip()
    source = str(data.get("source", "")).strip()

    cfg = CameraConfig(
        camera_id=camera_id,
        source=source,
        name=data.get("name", camera_id),
        enabled=bool(data.get("enabled", True)),
        width=int(data.get("width", 640)),
        height=int(data.get("height", 480)),
        fps=int(data.get("fps", 15)),
        source_type=data.get("source_type", "auto"),
    )
    if camera_id in camera_manager._cameras:
        camera_manager.unregister_camera(camera_id)
    camera_manager.register_camera(cfg)
    camera_manager.start_camera(camera_id)

    try:
        from video_stream import get_stream_server
        get_stream_server().register_camera(camera_id)
    except Exception:
        pass

    roi = []
    if data.get("roi"):
        if isinstance(data["roi"], str):
            parts = [int(x.strip()) for x in data["roi"].split(",") if x.strip().isdigit()]
            if len(parts) == 4:
                roi = parts
        elif isinstance(data["roi"], list) and len(data["roi"]) == 4:
            roi = [int(x) for x in data["roi"]]

    zcfg = ZoneConfig(
        camera_id=camera_id,
        name=data.get("name", camera_id),
        roi=roi,
        enable_vlm_review=data.get("enable_vlm_review", True),
    )
    zone_counter.register_camera(zcfg)

    cam_dict = {
        "camera_id": camera_id,
        "source": source,
        "name": data.get("name", camera_id),
        "enabled": bool(data.get("enabled", True)),
        "width": int(data.get("width", 640)),
        "height": int(data.get("height", 480)),
        "fps": int(data.get("fps", 15)),
        "source_type": data.get("source_type", "auto"),
        "roi": roi,
        "enable_vlm_review": zcfg.enable_vlm_review,
    }
    _persist_camera(cam_dict)
    return camera_id


def _sync_update_zone_config(data: dict, zone_counter) -> str:
    """同步执行区域配置更新与持久化（供 asyncio.to_thread 使用）"""
    camera_id = data.get("camera_id")
    roi = []
    if data.get("roi"):
        if isinstance(data["roi"], str):
            parts = [int(x.strip()) for x in data["roi"].split(",") if x.strip().isdigit()]
            if len(parts) == 4:
                roi = parts
        elif isinstance(data["roi"], list) and len(data["roi"]) == 4:
            roi = [int(x) for x in data["roi"]]

    zcfg = ZoneConfig(
        camera_id=camera_id,
        name=data.get("name", camera_id),
        roi=roi,
        enable_vlm_review=data.get("enable_vlm_review", True),
    )
    zone_counter.register_camera(zcfg)

    import config as app_config
    cameras = app_config.load_confined_cameras()
    updated = False
    for cam in cameras:
        if cam.get("camera_id") == camera_id:
            cam["roi"] = roi
            cam["enable_vlm_review"] = zcfg.enable_vlm_review
            if data.get("name"):
                cam["name"] = data["name"]
            updated = True
            break
    if updated:
        app_config.save_confined_cameras(cameras)
    return camera_id


# -- 摄像头/区域管理（一摄像头一区域，扁平化）--

@router.get("/cameras")
async def list_cameras(request: Request):
    """列出有限空间摄像头（含区域参数与实时状态）"""
    zone_counter = _get_zone_counter(request)
    camera_manager = _get_camera_manager(request)
    if zone_counter is None:
        return {"cameras": []}

    cameras_cfg = {}
    try:
        import config as app_config
        for c in app_config.load_confined_cameras():
            cameras_cfg[c.get("camera_id")] = c
    except Exception:
        pass

    states = zone_counter.list_cameras()
    for s in states:
        cid = s.get("camera_id")
        cfg = cameras_cfg.get(cid, {})
        s["source"] = cfg.get("source", "")
        s["source_type"] = cfg.get("source_type", "auto")
        s["enabled"] = cfg.get("enabled", True)
        s["width"] = cfg.get("width", 640)
        s["height"] = cfg.get("height", 480)
        s["fps"] = cfg.get("fps", 15)
        if camera_manager:
            st = camera_manager.get_camera_status(cid)
            if st:
                s["connected"] = st.get("connected", False)
                s["frame_count"] = st.get("frame_count", 0)
    return {"cameras": states}


@router.post("/cameras")
async def create_camera(data: dict, request: Request):
    """添加/更新有限空间摄像头（一摄像头一区域，同时注册到 CameraManager 和 ZoneCounter）"""
    camera_manager = _get_camera_manager(request)
    zone_counter = _get_zone_counter(request)
    if camera_manager is None or zone_counter is None:
        return JSONResponse({"error": "Not initialized"}, status_code=500)

    camera_id = str(data.get("camera_id", "")).strip()
    source = str(data.get("source", "")).strip()
    if not camera_id or not source:
        return JSONResponse({"error": "camera_id 和 source 必填"}, status_code=400)

    try:
        new_id = await asyncio.to_thread(_sync_create_camera, data, camera_manager, zone_counter)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return {"success": True, "camera_id": new_id}


@router.get("/cameras/{camera_id}")
async def get_camera(camera_id: str, request: Request):
    """获取单个摄像头详情"""
    zone_counter = _get_zone_counter(request)
    if zone_counter is None:
        return JSONResponse({"error": "Not initialized"}, status_code=500)
    state = zone_counter.get_camera_state(camera_id)
    if not state:
        return JSONResponse({"error": "Camera not found"}, status_code=404)
    return state


@router.delete("/cameras/{camera_id}")
async def delete_camera(camera_id: str, request: Request):
    """删除摄像头并注销区域"""
    camera_manager = _get_camera_manager(request)
    zone_counter = _get_zone_counter(request)
    if camera_manager is None or zone_counter is None:
        return JSONResponse({"error": "Not initialized"}, status_code=500)

    camera_manager.unregister_camera(camera_id)
    zone_counter.unregister_camera(camera_id)

    try:
        from video_stream import get_stream_server
        get_stream_server().unregister_camera(camera_id)
    except Exception:
        pass

    _remove_camera_from_config(camera_id)
    return {"success": True, "camera_id": camera_id}


@router.post("/cameras/{camera_id}/config")
async def update_camera_config(camera_id: str, data: dict, request: Request):
    """更新摄像头配置（含区域参数）"""
    data["camera_id"] = camera_id
    return await create_camera(data, request)


@router.post("/cameras/{camera_id}/zone-config")
async def update_zone_config(camera_id: str, data: dict, request: Request):
    """只更新区域配置（不重启视频流）"""
    zone_counter = _get_zone_counter(request)
    if zone_counter is None:
        return JSONResponse({"error": "Not initialized"}, status_code=500)

    data["camera_id"] = camera_id
    try:
        await asyncio.to_thread(_sync_update_zone_config, data, zone_counter)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return {"success": True, "camera_id": camera_id}


@router.get("/cameras/{camera_id}/status")
async def camera_status(camera_id: str, request: Request):
    """获取摄像头实时状态"""
    zone_counter = _get_zone_counter(request)
    if zone_counter is None:
        return JSONResponse({"error": "Not initialized"}, status_code=500)
    state = zone_counter.get_camera_state(camera_id)
    if not state:
        return JSONResponse({"error": "Camera not found"}, status_code=404)
    return {
        "camera_id": camera_id,
        "name": state["name"],
        "vlm_review_pending": state["vlm_review_pending"],
        "last_review_result": state["last_review_result"],
    }


@router.get("/cameras/{camera_id}/events")
async def camera_events(camera_id: str, limit: int = 20, request: Request = None):
    """获取摄像头事件历史"""
    zone_counter = _get_zone_counter(request)
    if zone_counter is None:
        return JSONResponse({"error": "Not initialized"}, status_code=500)
    state = zone_counter.get_camera_state(camera_id)
    if not state:
        return JSONResponse({"error": "Camera not found"}, status_code=404)
    events = state.get("event_history", [])
    return {"events": events[-limit:], "total": len(events)}


# -- 兼容旧前端：zones 端点 alias 到 cameras --

@router.get("/zones")
async def list_zones_alias(request: Request):
    """兼容旧前端：/api/zones 映射到 /api/cameras"""
    res = await list_cameras(request)
    cameras = res.get("cameras", [])
    return {"zones": cameras}


@router.get("/zones/{zone_id}")
async def get_zone_alias(zone_id: str, request: Request):
    return await get_camera(zone_id, request)


@router.get("/zones/{zone_id}/status")
async def zone_status_alias(zone_id: str, request: Request):
    return await camera_status(zone_id, request)


@router.get("/zones/{zone_id}/events")
async def zone_events_alias(zone_id: str, limit: int = 20, request: Request = None):
    return await camera_events(zone_id, limit, request)


@router.delete("/zones/{zone_id}")
async def delete_zone_alias(zone_id: str, request: Request):
    return await delete_camera(zone_id, request)


# -- 记录查询 --

@router.get("/records")
async def list_records(
    page: int = 1,
    size: int = 20,
    zone_id: Optional[str] = None,
    event_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    request: Request = None,
):
    """全局记录查询（分页）"""
    storage = _get_storage(request)
    if storage is None:
        return JSONResponse({"error": "Storage not initialized"}, status_code=500)
    start_time = start_date + "T00:00:00" if start_date else None
    end_time = end_date + "T23:59:59" if end_date else None
    return storage.get_records_paginated(
        page=page, size=size, zone_id=zone_id, event_type=event_type,
        start_time=start_time, end_time=end_time,
    )


@router.get("/records/stats")
async def records_stats(request: Request):
    """记录统计"""
    storage = _get_storage(request)
    if storage is None:
        return JSONResponse({"error": "Storage not initialized"}, status_code=500)
    return storage.get_stats()


@router.delete("/records")
async def clear_all_records(request: Request):
    """清空所有有限空间记录及关联图片"""
    storage = _get_storage(request)
    if storage is None:
        return JSONResponse({"error": "Storage not initialized"}, status_code=500)
    try:
        storage.clear_records()
        return {"success": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/records/{record_id}/frames")
async def get_record_frames(record_id: str, count: int = 30, request: Request = None):
    """获取记录帧序列（base64 列表）"""
    storage = _get_storage(request)
    if storage is None:
        return JSONResponse({"error": "Storage not initialized"}, status_code=500)
    frames = storage.load_frames(record_id, count)
    return {"frames": frames, "count": len(frames)}


# -- 文件上传 --

@router.post("/upload/video")
async def upload_video(file: UploadFile = File(...)):
    """上传本地视频文件,返回服务端路径以供摄像头 source 使用"""
    upload_dir = Path(__file__).resolve().parent.parent.parent / "uploads" / "videos"
    upload_dir.mkdir(parents=True, exist_ok=True)

    allowed_exts = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm"}
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed_exts:
        return JSONResponse(
            {"error": f"不支持的文件格式: {ext}，仅支持 {', '.join(sorted(allowed_exts))}"},
            status_code=400,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp}_{file.filename}"
    dest_path = upload_dir / safe_name
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        return JSONResponse({"error": f"保存失败: {e}"}, status_code=500)

    return {
        "success": True,
        "filename": safe_name,
        "path": str(dest_path.absolute()),
    }


# -- 全局配置 --

@router.get("/settings")
async def get_settings():
    """获取全局配置"""
    import config as app_config
    return app_config.load_confined_settings()


@router.post("/settings")
async def save_settings(data: dict, request: Request):
    """保存全局配置"""
    import config as app_config
    current = app_config.load_confined_settings()
    current.update({k: v for k, v in data.items() if v is not None})
    app_config.save_confined_settings(current)
    # 同步更新内存中的全局配置，使推理间隔等参数无需重启即生效
    gs = getattr(request.app.state, "global_settings", None)
    if gs is not None:
        gs.update(current)
    return {"success": True, "settings": current}


# -- 测试端点 --

@router.post("/zones/{zone_id}/test-event")
async def test_event(zone_id: str, data: dict, request: Request):
    """手动注入测试事件"""
    zone_counter = _get_zone_counter(request)
    storage = _get_storage(request)
    if zone_counter is None or storage is None:
        return JSONResponse({"error": "Not initialized"}, status_code=500)

    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    state = zone_counter.get_zone_state(zone_id)
    if not state:
        zone_counter.register_zone(
            ZoneConfig(
                zone_id=zone_id,
                camera_id=data.get("camera_id", "test-cam"),
                name=data.get("name", zone_id),
                roi=[],
            )
        )

    event_type = data.get("event_type", "enter")
    count = data.get("count", 1)

    with zone_counter._lock:
        state_obj = zone_counter._zones.get(zone_id)
        if not state_obj:
            return JSONResponse({"error": "Zone not found"}, status_code=404)

    diff = count if event_type == "enter" else -count
    event = {
        "event_id": f"cs-test-{uuid.uuid4().hex[:8]}",
        "zone_id": zone_id,
        "zone_name": state_obj.config.name,
        "camera_id": state_obj.config.camera_id,
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "monitor_mode": "test",
        "diff": diff,
        "description": f"测试事件：{event_type} {count} 人",
    }
    storage.add_record(event)

    return {"success": True, "zone_id": zone_id, "event": event}


@router.get("/prompt")
async def get_prompt():
    """获取有限空间提示词"""
    import config as app_config
    return {"prompt": app_config.load_confined_prompt()}


@router.post("/prompt")
async def save_prompt(data: dict):
    """保存有限空间提示词"""
    import config as app_config
    text = data.get("prompt", "")
    if app_config.save_confined_prompt(text):
        return {"success": True}
    return JSONResponse({"error": "保存失败"}, status_code=500)
