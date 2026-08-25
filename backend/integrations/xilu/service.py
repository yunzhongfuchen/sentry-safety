"""
西艾氟 OpenAPI 服务层
"""

import datetime
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import config as app_config
import performance_storage as storage
from backend.detection_registry import registry
from backend.integrations.xilu.mappings import get_warning_type, get_police_type

logger = logging.getLogger(__name__)


class XiluApiService:

    @staticmethod
    def _get_camera_name_map() -> Dict[str, str]:
        try:
            cameras = app_config.load_camera_configs()
            return {c.get("camera_id"): c.get("name", c.get("camera_id")) for c in cameras}
        except Exception:
            return {}

    @staticmethod
    def _get_camera_algorithm_usage() -> Dict[str, int]:
        counts: Dict[str, int] = {}
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

        target_camera_ids = None
        if camera_id_list:
            if isinstance(camera_id_list, str):
                target_camera_ids = {c.strip() for c in camera_id_list.split(",") if c.strip()}
            elif isinstance(camera_id_list, list):
                target_camera_ids = {str(c).strip() for c in camera_id_list if str(c).strip()}

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

            status = r.get("status", "pending")
            is_cleared = status in ("confirmed", "false_positive")
            state_str = "0" if is_cleared else "1"
            if warning_state is not None and warning_state != "" and str(warning_state) != state_str:
                continue

            if is_cleared:
                if clear_begin_time and r_time < clear_begin_time:
                    continue
                if clear_end_time and r_time > clear_end_time:
                    continue
            elif clear_begin_time or clear_end_time:
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

            rid = r.get("id", "")
            boxes = r.get("small_model", {}).get("boxes", []) if isinstance(r.get("small_model"), dict) else []
            confidence = float(r.get("confidence", 0.0))

            if boxes:
                range_obj = [boxes, [confidence] * len(boxes)]
            else:
                range_obj = []
            warning_range_str = json.dumps(range_obj)

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


# 别名兼容
CvApiService = XiluApiService
