import time
import cv2
import numpy as np
from backend.frame_utils import draw_timestamp_on_frame, encode_frame_to_jpg


def test_draw_timestamp_on_frame_adds_text():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    ts = time.mktime(time.strptime("2026-07-13 12:00:00", "%Y-%m-%d %H:%M:%S"))
    out = draw_timestamp_on_frame(frame, ts)
    assert out.shape == frame.shape


def test_encode_frame_to_jpg_without_timestamp():
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    data = encode_frame_to_jpg(frame, quality=80, draw_ts=False, timestamp=time.time())
    assert isinstance(data, bytes)
    assert len(data) > 0
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None


def test_encode_frame_to_jpg_with_timestamp():
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    ts = time.mktime(time.strptime("2026-07-13 12:00:00", "%Y-%m-%d %H:%M:%S"))
    data = encode_frame_to_jpg(frame, quality=80, draw_ts=True, timestamp=ts)
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
