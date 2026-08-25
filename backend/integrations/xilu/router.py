"""
西艾氟 (Xilu) OpenAPI 路由定义
严格提供 /cvApi/open/api/cv/* 规范路由，并支持兼容别名
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response

import performance_storage as storage
from backend.integrations.xilu.auth import verify_token
from backend.integrations.xilu.schemas import wrap_response
from backend.integrations.xilu.service import XiluApiService

router = APIRouter(tags=["Xilu_CV_OpenAPI"])


async def _parse_request_data(request: Request) -> dict:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            return await request.json()
        except Exception:
            return {}
    try:
        form = await request.form()
        return dict(form)
    except Exception:
        return dict(request.query_params)


async def handle_find_model_page(request: Request):
    data = await _parse_request_data(request)
    size = int(data.get("size", 10))
    current = int(data.get("current", 1))
    res = XiluApiService.get_model_page(current=current, size=size)
    return wrap_response(res)


async def handle_find_warning_page(request: Request):
    data = await _parse_request_data(request)
    size = int(data.get("size", 10))
    current = int(data.get("current", 1))
    res = XiluApiService.get_warning_page(
        current=current,
        size=size,
        begin_time=data.get("beginTime"),
        end_time=data.get("endTime"),
        clear_begin_time=data.get("clearBeginTime"),
        clear_end_time=data.get("clearEndTime"),
        warning_state=data.get("warningState"),
        camera_name=data.get("cameraName"),
        camera_id_list=data.get("cameraIdList"),
        warning_type_list=data.get("warningTypeList"),
    )
    return wrap_response(res)


async def handle_find_warning_number(request: Request):
    data = await _parse_request_data(request)
    tree_id = data.get("treeId")
    res = XiluApiService.get_warning_number(tree_id=tree_id)
    return wrap_response(res)


async def handle_warning_image(record_id: str):
    safe_id = record_id.replace(".jpg", "").replace(".jpeg", "")
    frames_dir = storage.FRAMES_DIR
    target_path = frames_dir / f"{safe_id}_snapshot.jpg"
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Warning image not found")
    return Response(content=target_path.read_bytes(), media_type="image/jpeg")


# ── 注册主路径 (/cvApi/open/api/cv/...) ──

router.add_api_route(
    "/cvApi/open/api/cv/findModelPage",
    handle_find_model_page,
    methods=["POST", "GET"],
    dependencies=[Depends(verify_token)],
)
router.add_api_route(
    "/cvApi/open/api/cv/findCvWarningPage",
    handle_find_warning_page,
    methods=["POST", "GET"],
    dependencies=[Depends(verify_token)],
)
router.add_api_route(
    "/cvApi/open/api/cv/findCvWarningNumber",
    handle_find_warning_number,
    methods=["POST", "GET"],
    dependencies=[Depends(verify_token)],
)
router.add_api_route(
    "/cvApi/open/api/cv/warning/image/{record_id:path}",
    handle_warning_image,
    methods=["GET"],
    dependencies=[Depends(verify_token)],
)


# ── 注册兼容别名路径 (/open/api/cv/...) ──

router.add_api_route(
    "/open/api/cv/findModelPage",
    handle_find_model_page,
    methods=["POST", "GET"],
    dependencies=[Depends(verify_token)],
)
router.add_api_route(
    "/open/api/cv/findCvWarningPage",
    handle_find_warning_page,
    methods=["POST", "GET"],
    dependencies=[Depends(verify_token)],
)
router.add_api_route(
    "/open/api/cv/findCvWarningNumber",
    handle_find_warning_number,
    methods=["POST", "GET"],
    dependencies=[Depends(verify_token)],
)
router.add_api_route(
    "/open/api/cv/warning/image/{record_id:path}",
    handle_warning_image,
    methods=["GET"],
    dependencies=[Depends(verify_token)],
)
