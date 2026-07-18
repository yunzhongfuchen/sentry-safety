import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.detection_registry import registry
from dotenv import load_dotenv

# 找到项目根目录的.env文件
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# 提示词文件路径
PROMPT_FILE = Path(__file__).parent.parent / 'data' / 'prompt.txt'

# 确保ARK_API_KEY环境变量已设置
if not os.getenv('ARK_API_KEY'):
    os.environ['ARK_API_KEY'] = os.getenv('ARK_API_KEY', '')

# ==================== VLM 提供商配置 ====================
# 自动判断：如果 BAILIAN_API_KEY 有值则使用百炼，否则使用 Ark
ARK_API_KEY = os.getenv("ARK_API_KEY", "")
VLM_ENDPOINT = os.getenv("VLM_ENDPOINT", "")

# Backward-compatible: ARK_MODEL defaults to VLM_ENDPOINT so existing deployments keep working.
# New deployments should set ARK_MODEL to the actual Ark model ID (e.g. doubao-vision-pro-32k).
ARK_MODEL = os.getenv("ARK_MODEL", VLM_ENDPOINT)

BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY", "")
BAILIAN_ENDPOINT = os.getenv("BAILIAN_ENDPOINT", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
BAILIAN_MODEL = os.getenv("BAILIAN_MODEL", "qwen3.5-plus")

# ==================== CV检测配置 ====================
# 检测模型
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")

# 检测置信度
DETECTION_CONFIDENCE = float(os.getenv("DETECTION_CONFIDENCE", "0.5"))

# 检测设备: cpu | cuda | npu (RK3588)
DETECTION_DEVICE = os.getenv("DETECTION_DEVICE", "cpu")

# RKNN 模型路径 (NPU 模式使用)
RKNN_MODEL = os.getenv("RKNN_MODEL", "models/yolov8n.rknn")

# RKNN 核心分配模式: auto | manual
RKNN_CORE_MODE = os.getenv("RKNN_CORE_MODE", "auto")

# ==================== GPU 动态调度器配置 ====================
USE_GPU_SCHEDULER = os.getenv("USE_GPU_SCHEDULER", "false").lower() == "true"
GPU_SCHEDULER_NUM_QUEUES = int(os.getenv("GPU_SCHEDULER_NUM_QUEUES", "0"))  # 0 = 自动（一模型一队列）
GPU_SCHEDULER_INTERVAL = float(os.getenv("GPU_SCHEDULER_INTERVAL", "0.5"))
GPU_SCHEDULER_HALF = os.getenv("GPU_SCHEDULER_HALF", "false").lower() == "true"

# ==================== 抽帧配置 ====================
FRAME_COUNT = int(os.getenv("FRAME_COUNT", "15"))
FRAME_INTERVAL = float(os.getenv("FRAME_INTERVAL", "0.5"))

# ==================== 触发冷却配置 ====================
TRIGGER_COOLDOWN = int(os.getenv("TRIGGER_COOLDOWN", "3"))

# ==================== 视频源配置 ====================
# 单摄像头模式使用
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")

# 多摄像头配置 (环境变量方式，格式: id:source:id2:source2)
CAMERAS_ENV = os.getenv("CAMERAS", "")

# ==================== 服务配置 ====================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ==================== 多摄像头配置 ====================
# 最大并发分析数
MAX_CONCURRENT_ANALYSIS = int(os.getenv("MAX_CONCURRENT_ANALYSIS", "3"))

# 摄像头默认配置
DEFAULT_CAMERA_WIDTH = int(os.getenv("DEFAULT_CAMERA_WIDTH", "640"))
DEFAULT_CAMERA_HEIGHT = int(os.getenv("DEFAULT_CAMERA_HEIGHT", "480"))
DEFAULT_CAMERA_FPS = int(os.getenv("DEFAULT_CAMERA_FPS", "15"))

# ==================== 日志配置 ====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")

# ==================== 存储配置 ====================
DATA_DIR = Path(__file__).parent.parent / "data"
RECORDS_FILE = DATA_DIR / "records.json"
FRAMES_DIR = DATA_DIR / "frames"

# ==================== 配置文件路径 ====================
CONFIG_DIR = Path(__file__).parent.parent / "config"
GLOBAL_CONFIG_FILE = CONFIG_DIR / "global.json"
CAMERAS_CONFIG_FILE = CONFIG_DIR / "cameras.json"

# ==================== 默认检测类型配置 ====================
def get_default_type_config() -> dict:
    """从注册表动态生成默认检测类型配置（运行参数部分）"""
    dtc = {}
    for dtype in registry.all_types():
        defaults = registry.get_defaults(dtype)
        dtc[dtype] = {
            "enabled": defaults.get("enabled", False),
            "interval": defaults.get("interval", 1),
            "threshold": defaults.get("threshold", 0.5),
            "consecutive_required": defaults.get("consecutive_required", 3),
            "cooldown": defaults.get("cooldown", 60),
            "use_vlm": defaults.get("use_vlm", False),
            "min_box_count": defaults.get("min_box_count"),
            "max_box_count": defaults.get("max_box_count"),
            "box_count_mode": defaults.get("box_count_mode"),
            "static_filter": defaults.get("static_filter", False),
            "static_diff_threshold": defaults.get("static_diff_threshold", 0.02),
        }
    return dtc


# 向后兼容：模块级变量，首次访问时从注册表生成
try:
    DEFAULT_TYPE_CONFIG = get_default_type_config()
except Exception:
    DEFAULT_TYPE_CONFIG = {
        "fire": {"enabled": False, "interval": 1, "threshold": 0.6, "consecutive_required": 3, "cooldown": 60, "use_vlm": False, "min_box_count": 1, "max_box_count": None},
        "smoke": {"enabled": False, "interval": 1, "threshold": 0.55, "consecutive_required": 3, "cooldown": 60, "use_vlm": False, "min_box_count": 1, "max_box_count": None},
        "uniform": {"enabled": False, "interval": 1, "threshold": 0.5, "consecutive_required": 3, "cooldown": 60, "use_vlm": False, "min_box_count": 1, "max_box_count": None},
        "mask": {"enabled": False, "interval": 1, "threshold": 0.5, "consecutive_required": 3, "cooldown": 60, "use_vlm": False, "min_box_count": 1, "max_box_count": None},
        "cigarette": {"enabled": False, "interval": 1, "threshold": 0.5, "consecutive_required": 3, "cooldown": 60, "use_vlm": False, "min_box_count": 1, "max_box_count": None},
        "sleep": {"enabled": False, "interval": 60, "threshold": 0.7, "consecutive_required": 3, "cooldown": 60, "use_vlm": False, "min_box_count": 1, "max_box_count": None},
    }

# ==================== 默认全局配置 ====================
def get_default_global_settings() -> dict:
    """动态生成全局默认设置，display_detection_types 从注册表读取"""
    try:
        ddt = {dtype: True for dtype in registry.all_types()}
    except Exception:
        ddt = {"fire": True, "smoke": True, "uniform": True, "mask": True, "cigarette": True, "sleep": True}
    return {
        "vlm_max_concurrent": 3,
        "vlm_inspection_interval": 30,
        "max_records": 100000,
        "max_storage_mb": 500,
        "memory_threshold_percent": 80,
        "emergency_cleanup_ratio": 0.2,
        "snapshot_quality": 70,
        "frame_quality": 60,
        "detection_resolution": [640, 480],
        "use_gpu_scheduler": False,
        "gpu_scheduler_num_queues": 0,
        "gpu_scheduler_interval": 0.5,
        "gpu_scheduler_half": False,
        "display_detection_types": ddt,
        "display_detection_interval": 1.0,
        "save_image_timestamp": True,
    }


# 向后兼容
try:
    DEFAULT_GLOBAL_SETTINGS = get_default_global_settings()
except Exception:
    DEFAULT_GLOBAL_SETTINGS = {
        "vlm_max_concurrent": 3,
        "vlm_inspection_interval": 30,
        "max_records": 100000,
        "max_storage_mb": 500,
        "memory_threshold_percent": 80,
        "emergency_cleanup_ratio": 0.2,
        "snapshot_quality": 70,
        "frame_quality": 60,
        "detection_resolution": [640, 480],
        "use_gpu_scheduler": False,
        "gpu_scheduler_num_queues": 0,
        "gpu_scheduler_interval": 0.5,
        "gpu_scheduler_half": False,
        "display_detection_types": {"fire": True, "smoke": True, "uniform": True, "mask": True, "cigarette": True, "sleep": True},
        "display_detection_interval": 1.0,
        "save_image_timestamp": True,
    }

# ==================== 默认摄像头参数（全局模板） ====================
try:
    DEFAULT_CAMERA_GLOBALS = {
        "width": 640,
        "height": 480,
        "fps": 15,
        "source_type": "auto",
        "detection_types": get_default_type_config(),
    }
except Exception:
    DEFAULT_CAMERA_GLOBALS = {
        "width": 640,
        "height": 480,
        "fps": 15,
        "source_type": "auto",
        "detection_types": dict(DEFAULT_TYPE_CONFIG),
    }


def load_camera_globals() -> dict:
    """加载摄像头全局默认参数，不存在则创建。

    detection_types 默认值由检测类型注册表（config/detection_types.json）提供，
    此函数不再返回 detection_types 字段（旧文件中的该字段会被忽略）。
    """
    base = {k: v for k, v in DEFAULT_CAMERA_GLOBALS.items() if k != "detection_types"}
    globals_file = CONFIG_DIR / "camera_globals.json"
    if globals_file.exists():
        try:
            with open(globals_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 忽略旧的 detection_types 字段，统一由注册表提供
            data.pop("detection_types", None)
            merged = dict(base)
            merged.update(data)
            return merged
        except Exception:
            pass
    save_camera_globals(base)
    return dict(base)


def save_camera_globals(globals_data: dict) -> bool:
    """保存摄像头全局默认参数"""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_DIR / "camera_globals.json", "w", encoding="utf-8") as f:
            json.dump(globals_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def apply_camera_globals(cam_config: dict, globals_data: dict = None) -> dict:
    """将全局默认值应用到摄像头配置（不覆盖已有值）"""
    if globals_data is None:
        globals_data = load_camera_globals()

    result = dict(cam_config)

    # 基础参数：若缺失则填充全局默认值
    for key in ("width", "height", "fps", "source_type"):
        if result.get(key) is None:
            result[key] = globals_data.get(key, DEFAULT_CAMERA_GLOBALS.get(key))

    # detection_types：若缺失或为空则填充注册表默认值
    dt = result.get("detection_types")
    default_dtc = get_default_type_config()
    if not dt:
        result["detection_types"] = {
            k: dict(v) for k, v in default_dtc.items()
        }
    else:
        # 逐个类型深合并：摄像头已有字段保留，缺失字段用注册表默认值填充
        merged_dt = {}
        for dtype, default_cfg in default_dtc.items():
            cam_cfg = dt.get(dtype, {})
            merged_cfg = dict(default_cfg)
            for k, v in cam_cfg.items():
                merged_cfg[k] = v
            merged_dt[dtype] = merged_cfg
        result["detection_types"] = merged_dt

    return result

# ==================== 默认提示词 ====================
DEFAULT_PROMPT = """你是一个工业安全监控系统。请分析以下视频帧序列，判断是否有人进入或离开电梯。

请返回JSON格式的分析结果：
{
    "action": "enter" / "leave" / "none",
    "confidence": 0.0-1.0,
    "reason": "判断理由简要说明"
}

判断标准：
- action: "enter"=进入, "leave"=离开, "none"=无动作
- 观察人体运动方向：从画面外进入画面内=进入，从画面内走出画面外=离开
- 如果人体在画面内但没有进出行为，返回 "none"
- 电梯门开关状态也需要考虑"""

DEFAULT_QUESTION = "请分析以上视频帧，判断是否有人进入或离开电梯，返回JSON格式结果。"


def load_prompt() -> tuple:
    """加载提示词，返回 (prompt, question)"""
    if PROMPT_FILE.exists():
        try:
            content = PROMPT_FILE.read_text(encoding='utf-8')
            parts = content.split('\n\n---\n\n')
            if len(parts) >= 2:
                return parts[0].strip(), parts[1].strip()
            return content.strip(), DEFAULT_QUESTION
        except Exception:
            pass
    return DEFAULT_PROMPT, DEFAULT_QUESTION


def save_prompt(prompt: str, question: str) -> bool:
    """保存提示词，如果为空则删除文件恢复默认"""
    try:
        PROMPT_FILE.parent.mkdir(parents=True, exist_ok=True)

        # 如果为空或只有空格，删除文件恢复默认
        if not prompt.strip():
            if PROMPT_FILE.exists():
                PROMPT_FILE.unlink()
            return True

        # 保存提示词
        content = f"{prompt.strip()}\n\n---\n\n{question.strip() if question else DEFAULT_QUESTION}"
        PROMPT_FILE.write_text(content, encoding='utf-8')
        return True
    except Exception:
        return False


# ==================== VLM 复核提示词配置 ====================
VLM_PROMPTS_FILE = CONFIG_DIR / "vlm_prompts.json"

DEFAULT_VLM_PROMPTS = {
    "fire_review": """你正在复核一个工业安全监控系统的火焰检测结果。
请仔细查看图片，判断画面中是否真的有明火。
注意排除以下误判情况：
- 红色灯光、红色物体反光
- 夕阳、晚霞
- 橙色安全帽或衣服

请以 JSON 格式返回：
{"confirmed": true/false, "confidence": 0.0-1.0, "reason": "判断理由"}""",

    "smoke_review": """你正在复核一个工业安全监控系统的烟雾检测结果。
请仔细查看图片，判断画面中是否真的有烟雾。
注意排除以下误判情况：
- 水蒸气、雾气
- 灰尘扬起
- 白色墙壁反光

请以 JSON 格式返回：
{"confirmed": true/false, "confidence": 0.0-1.0, "reason": "判断理由"}""",

    "mask_review": """你正在复核一个工业安全监控系统的口罩佩戴检测结果。
请仔细查看图片，判断画面中是否真的有未佩戴口罩的人员。
注意排除以下情况：
- 人员正在喝水或用餐（暂时摘下）
- 人员手持物品遮挡面部
- 距离太远看不清

请以 JSON 格式返回：
{"confirmed": true/false, "confidence": 0.0-1.0, "reason": "判断理由"}""",

    "cigarette_review": """你正在复核一个工业安全监控系统的吸烟行为检测结果。
请仔细查看图片，判断画面中是否真的有人正在吸烟。
注意排除以下情况：
- 手持笔、筷子等细长物体
- 人员只是在摸嘴或吃东西
- 画面模糊无法确认

请以 JSON 格式返回：
{"confirmed": true/false, "confidence": 0.0-1.0, "reason": "判断理由"}""",

    "uniform_review": """你正在复核一个工业安全监控系统的工服/反光背心检测结果。
请仔细查看图片，判断画面中是否真的有未穿工服或反光背心的人员。
注意：
- 不同岗位工服颜色可能不同
- 只需判断是否有"未穿"的情况

请以 JSON 格式返回：
{"confirmed": true/false, "confidence": 0.0-1.0, "reason": "判断理由"}""",

    "sleep_review": """你正在复核一个工业安全监控系统的睡岗/打盹检测结果。
请仔细查看图片，判断画面中是否真的有人正在睡岗或打盹。
注意排除以下情况：
- 人员只是低头看手机或文件
- 人员闭目休息但时间很短
- 画面模糊无法确认

请以 JSON 格式返回：
{"confirmed": true/false, "confidence": 0.0-1.0, "reason": "判断理由"}""",

    "inspection": """你正在执行工业安全监控巡检。请仔细检查监控画面，判断是否存在以下安全隐患：
{enabled_types_desc}

请以 JSON 格式返回，不要其他内容：
{{
  "detections": {{
{detections_json}
  }}
}}

注意：
- 只检查上述列出的类型，不要自行扩展
- confidence 范围 0.0-1.0
- 如果没有发现任何异常，所有 detected 都返回 false""",
}


def load_vlm_prompts() -> Dict[str, str]:
    """加载 VLM 复核提示词配置，缺失键自动补全默认值"""
    if VLM_PROMPTS_FILE.exists():
        try:
            with open(VLM_PROMPTS_FILE, "r", encoding="utf-8") as f:
                stored = json.load(f)
            merged = dict(DEFAULT_VLM_PROMPTS)
            merged.update(stored)
            # 自动把新增模板写回文件
            if merged != stored:
                save_vlm_prompts(merged)
            return merged
        except Exception:
            pass
    save_vlm_prompts(DEFAULT_VLM_PROMPTS)
    return dict(DEFAULT_VLM_PROMPTS)


def save_vlm_prompts(prompts: Dict[str, str]) -> bool:
    """保存 VLM 复核提示词配置到 config/vlm_prompts.json"""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(VLM_PROMPTS_FILE, "w", encoding="utf-8") as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


# ==================== 全局配置管理 ====================

def load_global_settings() -> Dict[str, Any]:
    """加载全局配置（global.json），不存在则创建默认值"""
    if GLOBAL_CONFIG_FILE.exists():
        try:
            with open(GLOBAL_CONFIG_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)
            # 合并默认值（处理新增字段）
            merged = dict(DEFAULT_GLOBAL_SETTINGS)
            merged.update(settings)
            return merged
        except Exception:
            pass
    # 创建默认文件
    save_global_settings(DEFAULT_GLOBAL_SETTINGS)
    return dict(DEFAULT_GLOBAL_SETTINGS)


def save_global_settings(settings: Dict[str, Any]) -> bool:
    """保存全局配置到 global.json"""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(GLOBAL_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


# ==================== 摄像头配置管理 ====================

def load_camera_configs() -> List[Dict[str, Any]]:
    """加载摄像头配置（cameras.json），支持旧格式自动迁移"""
    if not CAMERAS_CONFIG_FILE.exists():
        return []

    try:
        with open(CAMERAS_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    cameras = data.get("cameras", [])
    migrated = False

    for cam in cameras:
        # 旧格式迁移：没有 detection_types 字段时注入默认配置
        if "detection_types" not in cam:
            cam["detection_types"] = get_default_type_config()
            # 根据旧版的 detection_enabled 调整
            if cam.get("detection_enabled") is False:
                for dtype in cam["detection_types"]:
                    cam["detection_types"][dtype]["enabled"] = False
            migrated = True

        # 确保每个 detection_type 都有 use_vlm（向后兼容）
        for dtype, cfg in cam.get("detection_types", {}).items():
            if "use_vlm" not in cfg:
                cfg["use_vlm"] = False
                migrated = True

        # 旧配置迁移：移除 level，补充 cooldown
        for dtype, cfg in cam.get("detection_types", {}).items():
            if "level" in cfg:
                del cfg["level"]
                migrated = True
            if "cooldown" not in cfg:
                cfg["cooldown"] = get_default_type_config().get(dtype, {}).get("cooldown", 3)
                migrated = True

        # 确保新字段存在
        cam.setdefault("source_type", "auto")

    if migrated:
        save_camera_configs(cameras)

    return cameras


def save_camera_configs(cameras: List[Dict[str, Any]]) -> bool:
    """保存摄像头配置到 cameras.json"""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CAMERAS_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"cameras": cameras}, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def ensure_dirs():
    """确保必要的目录存在"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (Path(__file__).parent.parent / "models").mkdir(parents=True, exist_ok=True)


# 初始化目录
ensure_dirs()

