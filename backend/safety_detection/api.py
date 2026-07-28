"""
安全检测业务 API 路由
包含检测器状态、模型管理、测试告警注入等安全检测特有端点
"""

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

import math

from backend.detection_registry import registry
from backend.model_registry import model_registry, parse_model_metadata

router = APIRouter(tags=["safety"])


def _validate_default_value(key: str, value):
    """校验 defaults 字段类型和范围，非法时返回错误信息，合法时返回 None"""
    if key in ("enabled", "use_vlm", "static_filter"):
        if not isinstance(value, bool):
            return f"{key} must be a boolean"
        return None

    if key == "box_count_mode":
        if value is None:
            return None
        if value not in ("gte", "lte", "between", "outside"):
            return f"{key} must be one of 'gte', 'lte', 'between', 'outside' or null"
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
        "model_path": type_def.get("model_path"),
        "post_process": type_def.get("post_process", "yolo_box"),
        "classes": type_def.get("classes"),
        "model_confidence": type_def.get("model_confidence", 0.5),
        "vlm_prompt": type_def.get("vlm_prompt", ""),
        "inspection_label": type_def.get("inspection_label", type_def.get("label", dtype)),
        "defaults": type_def.get("defaults", {}),
    }


@router.put("/detector/types/{dtype}")
async def update_detection_type(dtype: str, data: dict):
    """更新检测类型（结构性字段 + defaults）"""
    type_def = registry.get(dtype)
    if type_def is None:
        return JSONResponse({"error": f"Unknown detection type: {dtype}"}, status_code=404)

    structural_fields = {"label", "color", "model_key",
                        "post_process", "classes", "model_confidence", "vlm_prompt", "inspection_label"}
    allowed_defaults = {"enabled", "interval", "threshold", "consecutive_required",
                        "cooldown", "use_vlm", "min_box_count", "max_box_count", "box_count_mode",
                        "static_filter", "static_diff_threshold"}
    structural_update = {k: v for k, v in data.items() if k in structural_fields}
    defaults_update = {k: v for k, v in data.items() if k not in structural_fields and k in allowed_defaults}

    if not structural_update and not defaults_update:
        return JSONResponse({"error": "No valid fields to update"}, status_code=400)

    try:
        if structural_update:
            registry.update_type(dtype, structural_update)
        if defaults_update:
            for k, v in defaults_update.items():
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
            "model_path": type_def.get("model_path"),
            "post_process": type_def.get("post_process", "yolo_box"),
            "classes": type_def.get("classes"),
            "model_confidence": type_def.get("model_confidence", 0.5),
            "vlm_prompt": type_def.get("vlm_prompt", ""),
            "inspection_label": type_def.get("inspection_label", type_def.get("label", key)),
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
        referencing_cameras = camera_manager.get_camera_ids_with_type(dtype)
        if referencing_cameras:
            cam_id = referencing_cameras[0]
            return JSONResponse({"error": f"Type '{dtype}' is referenced by camera '{cam_id}'"}, status_code=409)
    registry.delete_type(dtype)
    return {"success": True, "dtype": dtype}


@router.post("/detector/types/{dtype}/model")
async def upload_model(dtype: str, file: UploadFile = File(...)):
    """上传模型文件（兼容端点）：创建/复用模型条目并把算法指过去"""
    if registry.get(dtype) is None:
        return JSONResponse({"error": f"Unknown detection type: {dtype}"}, status_code=404)
    filename = file.filename or ""
    if not filename.endswith((".pt", ".rknn")):
        return JSONResponse({"error": "Only .pt and .rknn files are allowed"}, status_code=400)
    content = await file.read()
    path = model_registry.save_model_file(filename, content)
    # 复用同文件名的已有模型条目，否则新建
    model_key = None
    for m in model_registry.to_api_list():
        if m["file"] == path.name:
            model_key = m["key"]
            break
    if model_key is None:
        meta = parse_model_metadata(path)
        model_key = model_registry.add_model(
            file=path.name, name=path.stem,
            post_process=meta.get("post_process", "yolo_box"),
            class_names=meta.get("class_names", {}),
            file_size=len(content),
        )
    registry.update_type(dtype, {"model_key": model_key})
    return {"success": True, "model_key": model_key, "dtype": dtype}


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


# ---------- 模型管理 ----------

@router.get("/models")
async def list_model_entries():
    """模型列表（含被引用算法数）"""
    in_use = registry.get_model_keys_in_use()
    models = []
    for m in model_registry.to_api_list():
        m["used_by"] = 1 if m["key"] in in_use else 0
        models.append(m)
    return {"models": models}


@router.post("/models/upload")
async def upload_model_file(file: UploadFile = File(...), name: str = ""):
    """上传模型文件并创建条目，.pt 自动解析元数据"""
    filename = file.filename or ""
    if not filename.endswith((".pt", ".rknn")):
        return JSONResponse({"error": "Only .pt and .rknn files are allowed"}, status_code=400)
    content = await file.read()
    path = model_registry.save_model_file(filename, content)
    meta = parse_model_metadata(path)
    key = model_registry.add_model(
        file=path.name,
        name=name or path.stem,
        post_process=meta.get("post_process", "yolo_box"),
        class_names=meta.get("class_names", {}),
        file_size=len(content),
    )
    entry = model_registry.get(key)
    return {"success": True, "key": key,
            "post_process": entry["post_process"],
            "class_names": entry["class_names"]}


@router.put("/models/{key}")
async def update_model_entry(key: str, data: dict):
    """更新模型名称/后处理/类别清单（.rknn 手填用）"""
    if model_registry.get(key) is None:
        return JSONResponse({"error": f"Unknown model: {key}"}, status_code=404)
    updates = {k: v for k, v in data.items() if k in ("name", "post_process", "class_names")}
    if not updates:
        return JSONResponse({"error": "No valid fields to update"}, status_code=400)
    if "post_process" in updates and updates["post_process"] not in ("yolo_box", "yolo_pose"):
        return JSONResponse({"error": "post_process must be yolo_box or yolo_pose"}, status_code=400)
    model_registry.update_model(key, updates)
    return {"success": True, "key": key}


@router.delete("/models/{key}")
async def delete_model_entry(key: str):
    """删除模型（被算法引用时 409）"""
    if model_registry.get(key) is None:
        return JSONResponse({"error": f"Unknown model: {key}"}, status_code=404)
    if key in registry.get_model_keys_in_use():
        return JSONResponse({"error": f"Model '{key}' is referenced by algorithms"}, status_code=409)
    model_registry.delete_model(key)
    return {"success": True, "key": key}


# ---------- 算法管理 ----------

def _algo_to_response(key: str, td: dict) -> dict:
    return {
        "key": key,
        "label": td.get("label", key),
        "color": td.get("color", "#888888"),
        "model_key": td.get("model_key"),
        "model_path": td.get("model_path"),
        "post_process": td.get("post_process", "yolo_box"),
        "classes": td.get("classes"),
        "model_confidence": td.get("model_confidence", 0.5),
        "vlm_prompt": td.get("vlm_prompt", ""),
        "inspection_label": td.get("inspection_label", td.get("label", key)),
        "alarm_description": td.get("alarm_description", ""),
        "defaults": td.get("defaults", {}),
    }


@router.get("/algorithms")
async def list_algorithms():
    return {"algorithms": [_algo_to_response(t["key"], t) for t in registry.to_api_list()]}


@router.post("/algorithms")
async def create_algorithm(data: dict):
    data = dict(data)
    data.pop("model_path", None)  # model_path 由 model_key 解析，不接受直传
    try:
        key = registry.add_type(data)
        return _algo_to_response(key, registry.get(key))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@router.put("/algorithms/{key}")
async def update_algorithm(key: str, data: dict):
    if registry.get(key) is None:
        return JSONResponse({"error": f"Unknown algorithm: {key}"}, status_code=404)
    structural_fields = {"label", "color", "model_key", "classes", "model_confidence",
                         "vlm_prompt", "inspection_label", "alarm_description"}
    allowed_defaults = {"enabled", "interval", "threshold", "consecutive_required",
                        "cooldown", "use_vlm", "min_box_count", "max_box_count",
                        "box_count_mode", "static_filter", "static_diff_threshold"}
    structural_update = {k: v for k, v in data.items() if k in structural_fields}
    defaults_update = {k: v for k, v in data.items() if k not in structural_fields and k in allowed_defaults}
    if not structural_update and not defaults_update:
        return JSONResponse({"error": "No valid fields to update"}, status_code=400)
    try:
        if structural_update:
            registry.update_type(key, structural_update)
        if defaults_update:
            for k, v in defaults_update.items():
                error = _validate_default_value(k, v)
                if error:
                    return JSONResponse({"error": error}, status_code=400)
            registry.update_defaults(key, defaults_update)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"success": True, "key": key, "algorithm": _algo_to_response(key, registry.get(key))}


@router.delete("/algorithms/{key}")
async def delete_algorithm(key: str, request: Request):
    if registry.get(key) is None:
        return JSONResponse({"error": f"Unknown algorithm: {key}"}, status_code=404)
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if camera_manager is not None:
        referencing = camera_manager.get_camera_ids_with_type(key)
        if referencing:
            return JSONResponse(
                {"error": f"Algorithm '{key}' is referenced by camera '{referencing[0]}'"},
                status_code=409)
    registry.delete_type(key)
    return {"success": True, "key": key}
