"""
西艾氟 OpenAPI 类型与警情编码映射
"""

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
