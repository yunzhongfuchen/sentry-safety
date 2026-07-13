import logging
import json
import base64
import re
from typing import Any, Dict, List, Optional
import numpy as np
import cv2
import requests
from backend import config

logger = logging.getLogger(__name__)


# 动态加载提示词（支持运行时修改）
def _get_prompt_and_question():
    return config.load_prompt()


def _get_vlm_prompts() -> Dict[str, str]:
    """加载 VLM 复核提示词模板（config/vlm_prompts.json）"""
    return config.load_vlm_prompts()


# ==================== 多类型安全检测 Prompt 模板 ====================

# 保留本地兜底模板，避免配置文件异常时无法运行
_FALLBACK_PROMPT_TEMPLATES = {
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


def _get_prompt_template(prompt_type: str) -> str:
    """优先从配置文件读取模板，失败则使用本地兜底"""
    try:
        prompts = _get_vlm_prompts()
    except Exception:
        prompts = {}
    return prompts.get(prompt_type, _FALLBACK_PROMPT_TEMPLATES.get(prompt_type, _FALLBACK_PROMPT_TEMPLATES["fire_review"]))


class VideoUnderstander:
    """VLM 分析器（支持火山引擎 Ark / 阿里云百炼 等多提供商 OpenAI 兼容 API）"""

    def __init__(self):
        # 自动判断提供商：百炼 key 有值则优先用百炼，否则回退到 Ark
        if config.BAILIAN_API_KEY:
            self.provider = "bailian"
            self.api_key = config.BAILIAN_API_KEY
            self.model = config.BAILIAN_MODEL
            self.endpoint = config.BAILIAN_ENDPOINT
        else:
            self.provider = "ark"
            self.api_key = config.ARK_API_KEY
            self.model = config.ARK_MODEL
            self.endpoint = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        self._initialized = False

    def _ensure_initialized(self):
        """确保模型已初始化"""
        if self._initialized:
            return True
        if not self.api_key or not self.model:
            logger.warning(f"[{self.provider}] No API key or model configured, VLM will run in mock mode")
            return False
        self._initialized = True
        return True

    def _encode_frame(self, frame: np.ndarray) -> str:
        """将帧编码为base64"""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer.tobytes()).decode('utf-8')

    def _call_api(self, messages: List[dict]) -> Optional[str]:
        """调用 OpenAI 兼容 API（百炼启用流式避免网关超时）"""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 1000,
            }
            is_stream = self.provider == "bailian"
            if is_stream:
                payload["stream"] = True
                payload["stream_options"] = {"include_usage": True}
            if self.provider == "bailian":
                payload["enable_thinking"] = False

            resp = requests.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
                stream=is_stream,
            )
            resp.raise_for_status()

            if not is_stream:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                print(f"[VLM_RESPONSE] provider={self.provider} model={self.model} content={content[:200]}... usage={usage}")
                return content

            # 流式读取 SSE
            content_parts = []
            usage = {}
            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if not line_str.startswith("data: "):
                    continue
                data_str = line_str[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                if chunk.get("usage"):
                    usage = chunk["usage"]

            full_content = "".join(content_parts)
            print(f"[VLM_RESPONSE] provider={self.provider} model={self.model} content={full_content[:200]}... usage={usage}")
            return full_content
        except Exception as e:
            logger.error(f"[{self.provider}] API error: {e}")
            return None

    # ------------------------------------------------------------------
    # 旧接口（兼容原有电梯检测逻辑）
    # ------------------------------------------------------------------

    def _build_messages(self, frames: List[np.ndarray]):
        """构建发送给模型的messages"""
        prompt, question = _get_prompt_and_question()

        content = []
        for frame in frames:
            b64_img = self._encode_frame(frame)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
            })
        content.append({"type": "text", "text": question})

        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ]

    def analyze(self, frames: List[np.ndarray]) -> Optional[dict]:
        """分析视频帧（旧接口，兼容电梯检测）"""
        if not frames:
            logger.error("No frames to analyze")
            return None

        if not self._ensure_initialized():
            return self._mock_analyze(frames)

        messages = self._build_messages(frames)
        content = self._call_api(messages)
        if content is None:
            return self._mock_analyze(frames)

        logger.info(f"Model response: {content}")
        return self._parse_response(content)

    # ------------------------------------------------------------------
    # 新接口（多类型安全检测）
    # ------------------------------------------------------------------

    def analyze_multi(
        self,
        frames: List[np.ndarray],
        prompt_type: str = "review",
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        多图分析接口（安全检测专用）

        Args:
            frames: 图片帧列表
            prompt_type: prompt 模板名称（fire_review / smoke_review / mask_review / cigarette_review / uniform_review / sleep_review / inspection 等）
            extra_context: 额外上下文，用于填充模板变量

        Returns:
            解析后的 JSON dict
        """
        if not frames:
            return {"error": "No frames provided"}

        # 构建 prompt
        if prompt_type == "inspection":
            template = self._build_inspection_prompt(extra_context)
        else:
            template = _get_prompt_template(prompt_type)

        if not self._ensure_initialized():
            return self._mock_analyze_multi(frames, prompt_type)

        messages = self._build_multi_messages(frames, template)
        content = self._call_api(messages)
        if content is None:
            return self._mock_analyze_multi(frames, prompt_type)

        logger.info(f"VLM [{prompt_type}] response: {content[:200]}...")
        return self._parse_safety_response(content, prompt_type)

    def _build_multi_messages(self, frames: List[np.ndarray], prompt: str):
        """构建多图分析的 messages"""
        content = []
        for frame in frames:
            b64_img = self._encode_frame(frame)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
            })
        content.append({"type": "text", "text": prompt})

        return [
            {"role": "system", "content": "你是一个工业安全监控专家。请根据用户提供的图片和问题进行判断。"},
            {"role": "user", "content": content},
        ]

    def _build_inspection_prompt(self, extra_context: dict) -> str:
        """动态构建巡检 prompt"""
        types = extra_context.get("enabled_types", [])
        type_desc = {
            "fire": "明火",
            "smoke": "烟雾",
            "uniform": "未穿工服",
            "mask": "未戴口罩",
            "cigarette": "吸烟",
            "sleep": "睡岗/打盹",
        }
        checks = [f"- {type_desc.get(t, t)}" for t in types]
        checks_str = "\n".join(checks)
        detections_json = "\n".join(
            [f'    "{t}": {{"detected": true/false, "confidence": 0.0-1.0, "reason": "判断理由"}}' for t in types]
        )

        template = _get_prompt_template("inspection")
        return template.format(
            enabled_types_desc=checks_str,
            detections_json=detections_json,
        )

    # ------------------------------------------------------------------
    # 响应解析
    # ------------------------------------------------------------------

    def _parse_response(self, response: str) -> dict:
        """旧接口解析（enter/leave/none）"""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                action = result.get('action', 'none')
                if 'entry' in result and 'action' not in result:
                    action = 'enter' if result.get('entry') else 'none'
                confidence = result.get('confidence', 0.0)
                confidence = min(max(confidence, 0.0), 1.0)
                return {
                    'action': action,
                    'confidence': confidence,
                    'reason': result.get('reason', '')
                }
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")

        return {
            'action': 'none',
            'confidence': 0.0,
            'reason': f'解析失败: {response[:100]}'
        }

    def _parse_safety_response(self, response: str, prompt_type: str) -> dict:
        """
        安全检测响应解析（四层 fallback）
        1. 直接 JSON 解析
        2. 从 markdown 代码块中提取
        3. 提取第一个 {...} 块
        4. 按字段正则提取
        """
        # Level 1: 直接解析
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            pass

        # Level 2: 从 markdown 代码块提取
        code_block = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if code_block:
            try:
                result = json.loads(code_block.group(1))
                return result
            except json.JSONDecodeError:
                pass

        # Level 3: 提取第一个 { ... } 块
        brace_block = re.search(r'\{[\s\S]*?\}', response)
        if brace_block:
            try:
                result = json.loads(brace_block.group())
                return result
            except json.JSONDecodeError:
                pass

        # Level 4: 按 prompt_type 正则提取关键字段
        result = {}
        if "confirmed" in response.lower() or prompt_type.endswith("_confirm") or prompt_type.endswith("_review"):
            confirmed_match = re.search(r'"confirmed"\s*[:=]\s*(true|false)', response, re.IGNORECASE)
            if confirmed_match:
                result["confirmed"] = confirmed_match.group(1).lower() == "true"
        conf_match = re.search(r'"confidence"\s*[:=]\s*(\d+\.?\d*)', response)
        if conf_match:
            result["confidence"] = float(conf_match.group(1))
        reason_match = re.search(r'"reason"\s*[:=]\s*"([^"]*)"', response)
        if reason_match:
            result["reason"] = reason_match.group(1)

        if result:
            result["_parse_fallback"] = True
            return result

        return {
            "error": "Failed to parse VLM response",
            "raw": response[:200],
        }

    # ------------------------------------------------------------------
    # Mock 模式
    # ------------------------------------------------------------------

    def _mock_analyze(self, frames: List[np.ndarray]) -> dict:
        """Mock分析结果（旧接口）"""
        import random
        actions = ['enter', 'leave', 'none']
        result = {
            'action': random.choice(actions),
            'confidence': round(random.uniform(0.7, 0.99), 2),
            'reason': 'Mock分析结果 - 实际使用需配置火山引擎API'
        }
        logger.info(f"Mock result: {result}")
        return result

    def _mock_analyze_multi(self, frames: List[np.ndarray], prompt_type: str) -> dict:
        """Mock分析结果（新接口）"""
        import random
        if prompt_type == "inspection":
            result = {
                "detections": {},
            }
        else:
            result = {
                "confirmed": random.choice([True, False]),
                "confidence": round(random.uniform(0.6, 0.95), 2),
                "reason": f"Mock: {prompt_type} 复核结果",
            }
        logger.info(f"Mock multi result [{prompt_type}]: {result}")
        return result
