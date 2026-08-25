# 西艾氟 (CV_OpenAPI) 接口提供层实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为外部系统（西艾氟）构建标准 OpenAPI 接口提供层，严格对齐文档中定义的模型查询、报警列表查询、报警数量统计以及图片访问 4 个 API，仅返回业务必要字段。

**Architecture:** 在 `backend/cvapi` 下构建模块化的防腐层：
- `mappings.py` 定义类型/状态/代码映射
- `schemas.py` 严格定义入参与精简出参模型
- `auth.py` 静态 Token 校验依赖
- `service.py` 封装数据查询与结构转换
- `router.py` 注册 `/cvApi/open/api/cv/*` 及兼容路由 `/open/api/cv/*`
- 在 `backend/main_multi.py` 挂载路由

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, Pytest

**Spec:** `docs/superpowers/specs/2026-08-25-cvapi-provisioning-layer-design.md`

## Global Constraints
- API 地址绝对严格一致，支持 `/cvApi/open/api/cv/findModelPage`、`/cvApi/open/api/cv/findCvWarningPage`、`/cvApi/open/api/cv/findCvWarningNumber` 和 `/cvApi/open/api/cv/warning/image/{record_id}`
- 返回字段严格精简，只返回西艾氟需要的业务有效字段，不含冗余字段
- 统一返回信封 `{requestId, code: 0, state: 200, msg: null, timestamp, data, success: true}`
- 必须支持 `application/x-www-form-urlencoded` 以及 `application/json` 两种请求体

---

### Task 1: 映射与数据契约定义 (`mappings.py` & `schemas.py`)

**Files:**
- Create: `backend/cvapi/__init__.py`
- Create: `backend/cvapi/mappings.py`
- Create: `backend/cvapi/schemas.py`
- Test: `tests/test_cvapi_schemas.py`

**Interfaces:**
- Produces: 
  - `mappings.DTYPE_TO_MODEL_TYPE`: dict
  - `mappings.DTYPE_TO_POLICE_TYPE`: dict
  - `mappings.get_warning_type(dtype: str) -> str`
  - `mappings.get_police_type(dtype: str) -> str`
  - `schemas.wrap_response(data: Any, code: int = 0, msg: Optional[str] = None) -> dict`
  - `schemas.FindModelPageRequest`, `schemas.FindWarningPageRequest`, `schemas.FindWarningNumberRequest`
  - `schemas.ModelItem`, `schemas.WarningItem`, `schemas.WarningNumberData`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cvapi_schemas.py
from backend.cvapi.mappings import get_warning_type, get_police_type
from backend.cvapi.schemas import wrap_response, ModelItem, WarningItem, WarningNumberData

def test_mappings():
    assert get_warning_type("fire") == "fire_recog"
    assert get_warning_type("unknown_xyz") == "unknown_xyz_recog"
    assert get_police_type("fire") == "SP008"

def test_wrap_response():
    res = wrap_response({"test": 123})
    assert res["code"] == 0
    assert res["state"] == 200
    assert res["success"] is True
    assert res["data"] == {"test": 123}
    assert "requestId" in res
    assert "timestamp" in res

def test_model_item_minimal_fields():
    item = ModelItem(
        modelId="fire",
        modelName="明火检测",
        modelDes="检测明火",
        modelType="fire_recog",
        modelColour="#ef4444",
        modelState="1",
        number=2,
        modelUrl=""
    )
    dumped = item.model_dump()
    assert set(dumped.keys()) == {
        "modelId", "modelName", "modelDes", "modelType",
        "modelColour", "modelState", "number", "modelUrl"
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cvapi_schemas.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'backend.cvapi')

- [ ] **Step 3: Implement mappings and schemas**

```python
# backend/cvapi/__init__.py
# (empty)

# backend/cvapi/mappings.py
from typing import Dict

DTYPE_TO_WARNING_TYPE: Dict[str, str] = {
    "fire": "fire_recog",
    "smoke": "smoke_recog",
    "uniform": "uniform_recog",
    "mask": "mask_recog",
    "cigarette": "cigarette_recog",
    "sleep": "sleep_recog",
}

DTYPE_TO_POLICE_TYPE: Dict[str, str] = {
    "fire": "SP008",
    "smoke": "SP008",
    "uniform": "SP001",
    "mask": "SP002",
    "cigarette": "SP003",
    "sleep": "SP004",
}

def get_warning_type(dtype: str) -> str:
    return DTYPE_TO_WARNING_TYPE.get(dtype, f"{dtype}_recog")

def get_police_type(dtype: str) -> str:
    return DTYPE_TO_POLICE_TYPE.get(dtype, "SP008")


# backend/cvapi/schemas.py
import time
import uuid
from typing import Any, List, Optional, Union
from pydantic import BaseModel, Field

def wrap_response(data: Any, code: int = 0, msg: Optional[str] = None, state: int = 200) -> dict:
    return {
        "requestId": str(uuid.uuid4()),
        "code": code,
        "state": state,
        "msg": msg,
        "timestamp": str(int(time.time() * 1000)),
        "data": data,
        "success": code == 0,
    }

class ModelItem(BaseModel):
    modelId: str
    modelName: str
    modelDes: str
    modelType: str
    modelColour: str
    modelState: str = "1"
    number: int = 0
    modelUrl: str = ""

class ModelPageData(BaseModel):
    total: int
    size: int
    current: int
    pages: int
    orders: List[Any] = []
    searchCount: bool = True
    records: List[ModelItem]

class WarningItem(BaseModel):
    id: str
    cameraId: str
    cameraCode: str
    cameraName: str
    warningType: str
    warningContent: str
    warningTime: str
    warningTimeEnd: str
    warningState: str
    clearTime: Optional[str] = None
    imgUrl: str
    policeType: str
    policeLeave: str = "2"
    warningValue: str = "1"
    warningNumber: int = 1
    warningRange: str = "[]"
    warningPatrolType: str = "0"

class WarningPageData(BaseModel):
    total: int
    size: int
    current: int
    pages: int
    orders: List[Any] = []
    searchCount: bool = True
    records: List[WarningItem]

class WarningNumberData(BaseModel):
    todayWarningNumber: int = 0
    weekWarningNumber: int = 0
    monthWarningNumber: int = 0
    quarterWarningNumber: int = 0
    yearWarningNumber: int = 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cvapi_schemas.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/cvapi/__init__.py backend/cvapi/mappings.py backend/cvapi/schemas.py tests/test_cvapi_schemas.py
git commit -m "feat(cvapi): add mappings and clean pydantic schemas"
```

---

### Task 2: 业务查询与转换服务 (`service.py`)

**Files:**
- Create: `backend/cvapi/service.py`
- Test: `tests/test_cvapi_service.py`

**Interfaces:**
- Produces:
  - `CvApiService.get_model_page(current: int, size: int) -> dict`
  - `CvApiService.get_warning_page(params: dict) -> dict`
  - `CvApiService.get_warning_number(tree_id: Optional[str] = None) -> dict`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cvapi_service.py
import pytest
from backend.cvapi.service import CvApiService

def test_get_model_page():
    data = CvApiService.get_model_page(current=1, size=10)
    assert "records" in data
    assert "total" in data
    assert len(data["records"]) > 0
    first = data["records"][0]
    assert "modelId" in first
    assert "modelType" in first
    assert "tenantId" not in first  # 确保无冗余字段

def test_get_warning_number():
    data = CvApiService.get_warning_number()
    assert "todayWarningNumber" in data
    assert "weekWarningNumber" in data
    assert "monthWarningNumber" in data
    assert "quarterWarningNumber" in data
    assert "yearWarningNumber" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cvapi_service.py -v`
Expected: FAIL

- [ ] **Step 3: Implement CvApiService**

```python
# backend/cvapi/service.py
import datetime
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import config as app_config
import performance_storage as storage
from backend.detection_registry import registry
from backend.cvapi.mappings import get_warning_type, get_police_type

logger = logging.getLogger(__name__)

class CvApiService:

    @staticmethod
    def _get_camera_name_map() -> Dict[str, str]:
        try:
            cameras = app_config.load_camera_configs()
            return {c.get("camera_id"): c.get("name", c.get("camera_id")) for c in cameras}
        except Exception:
            return {}

    @staticmethod
    def _get_camera_algorithm_usage() -> Dict[str, int]:
        counts = {}
        try:
            cameras = app_config.load_camera_configs()
            for cam in cameras:
                algos = cam.get("algorithms", cam.get("detection_types", {}))
                for dtype, cfg in algos.items():
                    if isinstance(cfg, dict) and cfg.get("enabled"):
                        counts[dtype] = counts.get(dtype, 0) + 1
        except Exception:
            pass
        return counts

    @classmethod
    def get_model_page(cls, current: int = 1, size: int = 10) -> dict:
        all_types = registry.all_types()
        usage_counts = cls._get_camera_algorithm_usage()
        items = []

        for dtype in all_types:
            td = registry.get(dtype) or {}
            items.append({
                "modelId": dtype,
                "modelName": td.get("label", dtype),
                "modelDes": td.get("alarm_description") or f"{td.get('label', dtype)}检测算法",
                "modelType": get_warning_type(dtype),
                "modelColour": td.get("color", "#52CCA3"),
                "modelState": "1",
                "number": usage_counts.get(dtype, 0),
                "modelUrl": "",
            })

        total = len(items)
        if size == -1 or size <= 0:
            paged_items = items
            pages = 1
        else:
            pages = max(1, (total + size - 1) // size)
            start = (current - 1) * size
            paged_items = items[start:start + size]

        return {
            "total": total,
            "size": size,
            "current": current,
            "pages": pages,
            "orders": [],
            "searchCount": True,
            "records": paged_items,
        }

    @classmethod
    def get_warning_page(
        cls,
        current: int = 1,
        size: int = 10,
        begin_time: Optional[str] = None,
        end_time: Optional[str] = None,
        clear_begin_time: Optional[str] = None,
        clear_end_time: Optional[str] = None,
        warning_state: Optional[str] = None,
        camera_name: Optional[str] = None,
        camera_id_list: Optional[Union[List[str], str]] = None,
        warning_type_list: Optional[Union[List[str], str]] = None,
    ) -> dict:
        records = storage.load_records()
        cam_names = cls._get_camera_name_map()

        # 解析 ID 列表
        target_camera_ids = None
        if camera_id_list:
            if isinstance(camera_id_list, str):
                target_camera_ids = {c.strip() for c in camera_id_list.split(",") if c.strip()}
            elif isinstance(camera_id_list, list):
                target_camera_ids = {str(c).strip() for c in camera_id_list if str(c).strip()}

        # 解析 WarningType 列表
        target_warning_types = None
        if warning_type_list:
            if isinstance(warning_type_list, str):
                target_warning_types = {w.strip() for w in warning_type_list.split(",") if w.strip()}
            elif isinstance(warning_type_list, list):
                target_warning_types = {str(w).strip() for w in warning_type_list if str(w).strip()}

        filtered = []
        for r in records:
            r_time = r.get("time", "")
            if begin_time and r_time < begin_time:
                continue
            if end_time and r_time > (end_time if " " in end_time else f"{end_time} 23:59:59"):
                continue

            # 销警状态过滤
            status = r.get("status", "pending")
            is_cleared = status in ("confirmed", "false_positive")
            state_str = "0" if is_cleared else "1"
            if warning_state is not None and warning_state != "" and str(warning_state) != state_str:
                continue

            cid = r.get("camera_id", "")
            if target_camera_ids and cid not in target_camera_ids:
                continue

            cname = cam_names.get(cid, cid)
            if camera_name and camera_name not in cname:
                continue

            dtype = r.get("detection_type", "")
            wtype = get_warning_type(dtype)
            if target_warning_types and wtype not in target_warning_types and dtype not in target_warning_types:
                continue

            # 构造精简记录
            rid = r.get("id", "")
            boxes = r.get("small_model", {}).get("boxes", []) if isinstance(r.get("small_model"), dict) else []
            confidence = r.get("confidence", 0.0)
            warning_range_str = json.dumps([boxes, [confidence] * len(boxes)] if boxes else [])

            td = registry.get(dtype) or {}
            content = td.get("alarm_description") or r.get("reason") or f"检测到{td.get('label', dtype)}报警"

            filtered.append({
                "id": rid,
                "cameraId": cid,
                "cameraCode": cid,
                "cameraName": cname,
                "warningType": wtype,
                "warningContent": content,
                "warningTime": r_time,
                "warningTimeEnd": r_time,
                "warningState": state_str,
                "clearTime": r_time if is_cleared else None,
                "imgUrl": f"/cvApi/open/api/cv/warning/image/{rid}.jpg",
                "policeType": get_police_type(dtype),
                "policeLeave": "2",
                "warningValue": str(len(boxes)) if boxes else "1",
                "warningNumber": len(boxes) if boxes else 1,
                "warningRange": warning_range_str,
                "warningPatrolType": "0",
            })

        total = len(filtered)
        if size == -1 or size <= 0:
            paged_items = filtered
            pages = 1
        else:
            pages = max(1, (total + size - 1) // size)
            start = (current - 1) * size
            paged_items = filtered[start:start + size]

        return {
            "total": total,
            "size": size,
            "current": current,
            "pages": pages,
            "orders": [],
            "searchCount": True,
            "records": paged_items,
        }

    @classmethod
    def get_warning_number(cls, tree_id: Optional[str] = None) -> dict:
        records = storage.load_records()
        now = datetime.datetime.now()
        
        today_start = now.strftime("%Y-%m-%d 00:00:00")
        week_start = (now - datetime.timedelta(days=now.weekday())).strftime("%Y-%m-%d 00:00:00")
        month_start = now.strftime("%Y-%m-01 00:00:00")
        quarter_month = (now.month - 1) // 3 * 3 + 1
        quarter_start = f"{now.year}-{quarter_month:02d}-01 00:00:00"
        year_start = f"{now.year}-01-01 00:00:00"

        c_today, c_week, c_month, c_quarter, c_year = 0, 0, 0, 0, 0
        for r in records:
            t = r.get("time", "")
            if not t:
                continue
            if t >= today_start:
                c_today += 1
            if t >= week_start:
                c_week += 1
            if t >= month_start:
                c_month += 1
            if t >= quarter_start:
                c_quarter += 1
            if t >= year_start:
                c_year += 1

        return {
            "todayWarningNumber": c_today,
            "weekWarningNumber": c_week,
            "monthWarningNumber": c_month,
            "quarterWarningNumber": c_quarter,
            "yearWarningNumber": c_year,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cvapi_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/cvapi/service.py tests/test_cvapi_service.py
git commit -m "feat(cvapi): implement CvApiService for models, warning page and stats"
```

---

### Task 3: 鉴权与 API 路由 (`auth.py` & `router.py`)

**Files:**
- Create: `backend/cvapi/auth.py`
- Create: `backend/cvapi/router.py`
- Test: `tests/test_cvapi_router.py`

**Interfaces:**
- Produces:
  - `auth.require_api_token`
  - `router.router`: APIRouter (挂载所有 open api 路由与图片路由)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cvapi_router.py
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.cvapi.router import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)

def test_find_model_page_endpoint():
    res = client.post("/cvApi/open/api/cv/findModelPage", data={"size": 10, "current": 1})
    assert res.status_code == 200
    json_data = res.json()
    assert json_data["code"] == 0
    assert json_data["success"] is True
    assert "records" in json_data["data"]

def test_find_warning_page_endpoint():
    res = client.post("/cvApi/open/api/cv/findCvWarningPage", data={"size": 5, "current": 1})
    assert res.status_code == 200
    json_data = res.json()
    assert json_data["code"] == 0
    assert "records" in json_data["data"]

def test_find_warning_number_endpoint():
    res = client.post("/cvApi/open/api/cv/findCvWarningNumber", data={})
    assert res.status_code == 200
    json_data = res.json()
    assert json_data["code"] == 0
    assert "todayWarningNumber" in json_data["data"]

def test_compat_route_without_cvapi():
    # 测试兼容路径 /open/api/cv/findModelPage
    res = client.post("/open/api/cv/findModelPage", data={"size": -1})
    assert res.status_code == 200
    assert res.json()["code"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cvapi_router.py -v`
Expected: FAIL

- [ ] **Step 3: Implement auth and router**

```python
# backend/cvapi/auth.py
import os
from typing import Optional
from fastapi import Header, HTTPException, Query, Request, status

def verify_token(
    authorization: Optional[str] = Header(None),
    x_token: Optional[str] = Header(None, alias="X-Token"),
    query_token: Optional[str] = Query(None, alias="token"),
) -> bool:
    expected_token = os.getenv("CVAPI_TOKEN", "").strip()
    if not expected_token:
        # 未配置 Token 则免鉴权，方便本地/内网调试
        return True

    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    elif x_token:
        token = x_token.strip()
    elif query_token:
        token = query_token.strip()

    if token != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Token"
        )
    return True


# backend/cvapi/router.py
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, Depends, Form, Request, Response, HTTPException
import performance_storage as storage
from backend.cvapi.auth import verify_token
from backend.cvapi.schemas import wrap_response
from backend.cvapi.service import CvApiService

router = APIRouter(tags=["CV_OpenAPI"])

async def _parse_body_or_form(request: Request) -> dict:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            return await request.json()
        except Exception:
            return {}
    form = await request.form()
    return dict(form)

# 统一处理器
async def handle_find_model_page(request: Request):
    data = await _parse_body_or_form(request)
    size = int(data.get("size", 10))
    current = int(data.get("current", 1))
    res = CvApiService.get_model_page(current=current, size=size)
    return wrap_response(res)

async def handle_find_warning_page(request: Request):
    data = await _parse_body_or_form(request)
    size = int(data.get("size", 10))
    current = int(data.get("current", 1))
    res = CvApiService.get_warning_page(
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
    data = await _parse_body_or_form(request)
    tree_id = data.get("treeId")
    res = CvApiService.get_warning_number(tree_id=tree_id)
    return wrap_response(res)

async def handle_warning_image(record_id: str):
    # 剥离可能带有的 .jpg 扩展名
    safe_id = record_id.replace(".jpg", "").replace(".jpeg", "")
    snapshot_bytes = None
    frames_dir = storage.FRAMES_DIR
    target_path = frames_dir / f"{safe_id}_snapshot.jpg"
    if target_path.exists():
        snapshot_bytes = target_path.read_bytes()
    if not snapshot_bytes:
        raise HTTPException(status_code=404, detail="Warning image not found")
    return Response(content=snapshot_bytes, media_type="image/jpeg")


# 注册主路径 (/cvApi/open/api/cv/...)
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
)

# 注册兼容路径 (/open/api/cv/...)
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
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cvapi_router.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/cvapi/auth.py backend/cvapi/router.py tests/test_cvapi_router.py
git commit -m "feat(cvapi): add auth and FastAPI router with dual path support"
```

---

### Task 4: 主应用挂载与全量回归测试 (`main_multi.py`)

**Files:**
- Modify: `backend/main_multi.py`
- Test: `tests/test_main_multi_cvapi.py`

**Interfaces:**
- Consumes: `backend.cvapi.router.router`

- [ ] **Step 1: Write integration test**

```python
# tests/test_main_multi_cvapi.py
from fastapi.testclient import TestClient
from backend.main_multi import app

client = TestClient(app)

def test_main_app_has_cvapi_routes():
    res = client.post("/cvApi/open/api/cv/findModelPage", data={"size": 10})
    assert res.status_code == 200
    assert res.json()["code"] == 0

    res2 = client.post("/cvApi/open/api/cv/findCvWarningPage", data={"size": 5})
    assert res2.status_code == 200
    assert res2.json()["code"] == 0

    res3 = client.post("/cvApi/open/api/cv/findCvWarningNumber", data={})
    assert res3.status_code == 200
    assert res3.json()["code"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_main_multi_cvapi.py -v`
Expected: FAIL (404 Not Found)

- [ ] **Step 3: Mount cvapi router in `main_multi.py`**

In `backend/main_multi.py`:
```python
from backend.cvapi.router import router as cvapi_router

# 挂载开放接口
app.include_router(cvapi_router)
```

- [ ] **Step 4: Run full test suite to ensure no regression**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/main_multi.py tests/test_main_multi_cvapi.py
git commit -m "feat(cvapi): mount cvapi router to main_multi app and verify all tests pass"
```
