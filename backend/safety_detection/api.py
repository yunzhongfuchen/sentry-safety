"""
安全检测业务 API 路由
包含检测器状态、模型管理、测试告警注入等安全检测特有端点
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["safety"])


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
