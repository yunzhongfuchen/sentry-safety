import time

import cv2
import numpy as np


def draw_timestamp_on_frame(frame: np.ndarray, timestamp: float) -> np.ndarray:
    """在帧右上角绘制时间戳（白字黑边）。"""
    h, w = frame.shape[:2]
    text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(w, h) / 800.0)
    thickness = max(1, int(min(w, h) / 400))
    margin = max(8, int(min(w, h) * 0.015))

    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = w - text_w - margin
    y = text_h + margin

    cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def encode_frame_to_jpg(frame: np.ndarray, quality: int, draw_ts: bool, timestamp: float) -> bytes:
    """把帧编码为 JPEG 字节，按需叠加时间戳。"""
    img = frame.copy()
    if draw_ts:
        img = draw_timestamp_on_frame(img, timestamp)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()
