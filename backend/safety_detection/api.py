"""
安全检测业务 API 路由
包含检测器状态、模型管理、测试告警注入等安全检测特有端点
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from backend.detection_registry import registry

router = APIRouter(tags=["safety"])


@router.get("/detector/types")
async def list_detection_types():
    """获取所有检测类型定义"""
    return {"types": registry.to_api_list()}


@router.get("/detector/types/{dtype}")
async def get_detection_type(dtype: str):
    """获取单个检测类型定义"""
    type_def = registry.get(dtype)
    if type_def is None:
        return JSONResponse({"error": f"Unknown detection type: {dtype}"}, status_code=404)
    return {
        "key": dtype,
        "label": type_def.get("label", dtype),
        "color": type_def.get("color", "#888888"),
        "icon": type_def.get("icon", ""),
        "post_process": type_def.get("post_process", "yolo_box"),
        "defaults": type_def.get("defaults", {}),
    }


@router.put("/detector/types/{dtype}")
async def update_detection_type(dtype: str, data: dict):
    """更新检测类型的默认运行参数"""
    type_def = registry.get(dtype)
    if type_def is None:
        return JSONResponse({"error": f"Unknown detection type: {dtype}"}, status_code=404)

    allowed_keys = {"enabled", "interval", "threshold", "consecutive_required",
                    "cooldown", "use_vlm", "min_box_count", "max_box_count"}
    defaults_update = {k: v for k, v in data.items() if k in allowed_keys}

    if not defaults_update:
        return JSONResponse({"error": "No valid fields to update"}, status_code=400)

    registry.update_defaults(dtype, defaults_update)
    return {"success": True, "dtype": dtype, "defaults": registry.get_defaults(dtype)}


@router.get("/detector/status")
async def detector_status(request: Request):
    """获取检测器状态"""
    safety_detector = getattr(request.app.state, "safety_detector", None)
    if safety_detector is None:
        return {"error": "Detector not initialized"}
    return {
        "loaded_models": safety_detector.loaded_models,
        "model_status": safety_detector.get_model_status(),
        "mode": getattr(safety_detector, "device", "cpu").upper(),
        "npu_cores": getattr(safety_detector, "_npu_cores", 0),
    }


@router.get("/detector/models")
async def list_models(request: Request):
    """获取已加载模型列表"""
    safety_detector = getattr(request.app.state, "safety_detector", None)
    if safety_detector is None:
        return {"models": []}
    return {"models": safety_detector.get_model_status()}


@router.post("/cameras/{camera_id}/test-alert")
async def test_alert(camera_id: str, data: dict, request: Request):
    """手动注入测试告警（用于验证检测流程和 UI）"""
    multi_detector = getattr(request.app.state, "multi_detector", None)
    if multi_detector is None:
        return JSONResponse({"error": "Not initialized"}, status_code=500)

    dtype = data.get("dtype", "fire")
    confidence = data.get("confidence", 0.95)

    simulated = {
        "detected": True,
        "confidence": confidence,
        "boxes": [[100, 100, 200, 200]],
        "scores": [confidence],
    }

    multi_detector.inject_detection(camera_id, dtype, simulated)
    return {
        "success": True,
        "camera_id": camera_id,
        "dtype": dtype,
        "confidence": confidence,
    }
