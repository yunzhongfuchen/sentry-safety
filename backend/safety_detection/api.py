"""
安全检测业务 API 路由
包含检测器状态、模型管理、测试告警注入等安全检测特有端点
"""

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

import math

from backend.detection_registry import registry

router = APIRouter(tags=["safety"])


def _validate_default_value(key: str, value):
    """校验 defaults 字段类型和范围，非法时返回错误信息，合法时返回 None"""
    if key in ("enabled", "use_vlm"):
        if not isinstance(value, bool):
            return f"{key} must be a boolean"
        return None

    if key in ("min_box_count", "max_box_count"):
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return f"{key} must be a non-negative integer or null"
        return None

    if key == "consecutive_required":
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            return f"{key} must be an integer >= 1"
        return None

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{key} must be a number"

    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return f"{key} must be a finite number"

    if key == "interval" and value <= 0:
        return f"{key} must be a positive number"

    if key == "threshold" and not (0 <= value <= 1):
        return f"{key} must be a number between 0 and 1"

    if key == "cooldown" and value < 0:
        return f"{key} must be a non-negative number"

    return None


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
    """更新检测类型（结构性字段 + defaults）"""
    type_def = registry.get(dtype)
    if type_def is None:
        return JSONResponse({"error": f"Unknown detection type: {dtype}"}, status_code=404)

    structural_fields = {"label", "color", "icon", "model_path", "npu_model_path",
                        "post_process", "classes", "model_confidence", "vlm_prompt_key", "inspection_label"}
    structural_update = {k: v for k, v in data.items() if k in structural_fields}
    defaults_update = {k: v for k, v in data.items() if k not in structural_fields}

    try:
        if structural_update:
            registry.update_type(dtype, structural_update)
        if defaults_update:
            allowed_keys = {"enabled", "interval", "threshold", "consecutive_required",
                            "cooldown", "use_vlm", "min_box_count", "max_box_count"}
            for k, v in defaults_update.items():
                if k not in allowed_keys:
                    continue
                error = _validate_default_value(k, v)
                if error:
                    return JSONResponse({"error": error}, status_code=400)
            registry.update_defaults(dtype, defaults_update)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    return {"success": True, "dtype": dtype, "type": registry.get(dtype)}


@router.post("/detector/types")
async def create_detection_type(data: dict):
    """新增检测类型"""
    try:
        key = registry.add_type(data)
        type_def = registry.get(key)
        return {
            "key": key,
            "label": type_def.get("label", key),
            "color": type_def.get("color", "#888888"),
            "icon": type_def.get("icon", ""),
            "post_process": type_def.get("post_process", "yolo_box"),
            "defaults": type_def.get("defaults", {}),
        }
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@router.delete("/detector/types/{dtype}")
async def delete_detection_type(dtype: str, request: Request):
    """删除检测类型（检查摄像头引用）"""
    if registry.get(dtype) is None:
        return JSONResponse({"error": f"Unknown detection type: {dtype}"}, status_code=404)
    # 检查摄像头引用
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if camera_manager is not None:
        for cam_id, cam in camera_manager._cameras.items():
            if dtype in cam.config.detection_types:
                return JSONResponse({"error": f"Type '{dtype}' is referenced by camera '{cam_id}'"}, status_code=409)
    registry.delete_type(dtype)
    return {"success": True, "dtype": dtype}


@router.post("/detector/types/{dtype}/model")
async def upload_model(dtype: str, file: UploadFile = File(...)):
    """上传模型文件"""
    if registry.get(dtype) is None:
        return JSONResponse({"error": f"Unknown detection type: {dtype}"}, status_code=404)
    filename = file.filename
    if not filename.endswith((".pt", ".rknn")):
        return JSONResponse({"error": "Only .pt and .rknn files are allowed"}, status_code=400)
    content = await file.read()
    registry.save_model(filename, content)
    # 自动填入对应路径字段
    type_def = registry.get(dtype)
    if filename.endswith(".pt"):
        type_def["model_path"] = filename
    else:
        type_def["npu_model_path"] = filename
    registry._save(registry._types)
    return {"success": True, "model_path": filename, "dtype": dtype}


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
    """获取已加载模型列表（覆盖 safety_detector 与 GPU scheduler）"""
    safety_detector = getattr(request.app.state, "safety_detector", None)
    gpu_scheduler = getattr(request.app.state, "gpu_scheduler", None)

    if safety_detector is None and gpu_scheduler is None:
        return {"models": []}

    models = []
    if safety_detector is not None:
        models = safety_detector.get_model_status()

    if gpu_scheduler is not None:
        loaded_types = set(gpu_scheduler.model_configs.keys())
        models_by_type = {m["type"]: m for m in models}
        for dtype in loaded_types:
            cfg = gpu_scheduler.model_configs[dtype]
            entry = models_by_type.get(dtype)
            if entry is None:
                entry = {"type": dtype, "loaded": False}
                models.append(entry)
            entry["backend"] = "gpu"
            entry["device"] = cfg.device
            entry["loaded"] = True

    return {"models": models}


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
