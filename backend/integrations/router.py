"""
外部系统 API 统一聚合路由
挂载所有对接公司的外部 API 接口
"""

from fastapi import APIRouter
from backend.integrations.xilu.router import router as xilu_router

router = APIRouter()

# 挂载西艾氟公司 OpenAPI
router.include_router(xilu_router)

# 后续若有其它公司开放 API，在此继续 include_router 即可
