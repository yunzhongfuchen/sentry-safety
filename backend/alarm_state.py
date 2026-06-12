import time
from typing import Any, Dict, Optional


def create_record(camera_id: str, dtype: str, result: Dict[str, Any], record_id: Optional[str] = None) -> Dict[str, Any]:
    """根据小模型检测结果创建一条新告警记录"""
    if record_id is None:
        record_id = f"{camera_id}_{dtype}_{int(time.time() * 1000)}"
    trigger_time = time.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "id": record_id,
        "camera_id": camera_id,
        "detection_type": dtype,
        "level": "small_model_alarm",
        "status": "pending",
        "time": trigger_time,
        "confidence": result.get("max_confidence", result.get("confidence", 0)),
        "reason": result.get("reason", ""),
        "small_model": {
            "detected": result.get("detected", False),
            "confidence": result.get("max_confidence", 0),
            "boxes": result.get("boxes", []),
        },
        "vlm_review": None,
        "source": result.get("source", "small_model"),
        "frame_count": 0,
    }


def apply_vlm_review(record: Dict[str, Any], vlm_result: Dict[str, Any]) -> None:
    """把 VLM 复核结果应用到记录：只改 level 和 vlm_review，不改 status"""
    confirmed = vlm_result.get("confirmed", False)
    conf = vlm_result.get("confidence", 0)
    reason = vlm_result.get("reason", "")
    record["vlm_review"] = {
        "confirmed": confirmed,
        "confidence": conf,
        "reason": reason,
    }
    if confirmed:
        record["level"] = "vlm_alarm"
        record["reason"] = f"[VLM 确认] {reason}" if reason else "[VLM 确认] 复核通过"
    else:
        record["level"] = "vlm_ignore"
        record["reason"] = f"[VLM 已排除] {reason}" if reason else "[VLM 已排除] 复核未通过"


def confirm_alarm(record: Dict[str, Any]) -> None:
    """人工确认报警"""
    record["status"] = "confirmed"


def confirm_false_positive(record: Dict[str, Any]) -> None:
    """人工确认误报"""
    record["status"] = "false_positive"
