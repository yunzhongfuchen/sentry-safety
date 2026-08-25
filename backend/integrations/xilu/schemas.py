"""
西艾氟 OpenAPI 统一信封与精简数据模型
"""

import time
import uuid
from typing import Any, List, Optional
from pydantic import BaseModel


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
