"""睡岗检测模块 - 基于 YOLOv8-pose 的人体姿态 + 睡姿分析

从 infer.py 提取的核心逻辑，适配 engine.py 的多线程架构。
"""

import logging

import numpy as np
from ultralytics import YOLO

# 关键点置信度阈值
KPT_CONF_THRESHOLD = 0.4

# 睡姿分类标签
POSTURE_LABELS = {
    'face_up': '仰卧',
    'face_down': '俯卧',
    'side': '侧卧',
    'standing/sitting': '站立/坐姿',
    'not sleeping': '未睡眠',
}


def analyze_sleep(keypoints, bbox):
    """根据 YOLOv8-pose 关键点分析睡姿。

    Args:
        keypoints: numpy array (17, 3) of [x, y, confidence]
        bbox: [x1, y1, x2, y2] bounding box

    Returns:
        dict with is_sleeping, sleep_confidence, posture, posture_confidence, etc.
    """
    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    aspect_ratio = bbox_w / max(bbox_h, 1)

    # 1. 躺姿判断 - 宽高比
    if aspect_ratio > 1.5:
        lying_score = min(aspect_ratio / 2.0, 1.0)
    elif aspect_ratio > 1.2:
        lying_score = 0.6
    else:
        lying_score = max(0, aspect_ratio / 1.2) * 0.3

    # 2. 头部可见度
    face_kpts = keypoints[0:5, 2]
    head_conf = float(np.mean(face_kpts))

    # 3. 面部/眼睛可见性
    face_vis = float(np.mean(keypoints[[0, 1, 2], 2]))
    eye_vis = float(np.mean(keypoints[[1, 2], 2]))

    # 4. 肩膀/髋部对称性（仅用于睡姿分类）
    l_s, r_s = float(keypoints[5, 2]), float(keypoints[6, 2])
    l_h, r_h = float(keypoints[11, 2]), float(keypoints[12, 2])

    sh_y_vals, sh_x_vals = [], []
    if l_s > KPT_CONF_THRESHOLD:
        sh_y_vals.append(float(keypoints[5, 1]))
        sh_x_vals.append(float(keypoints[5, 0]))
    if r_s > KPT_CONF_THRESHOLD:
        sh_y_vals.append(float(keypoints[6, 1]))
        sh_x_vals.append(float(keypoints[6, 0]))

    hp_y_vals, hp_x_vals = [], []
    if l_h > KPT_CONF_THRESHOLD:
        hp_y_vals.append(float(keypoints[11, 1]))
        hp_x_vals.append(float(keypoints[11, 0]))
    if r_h > KPT_CONF_THRESHOLD:
        hp_y_vals.append(float(keypoints[12, 1]))
        hp_x_vals.append(float(keypoints[12, 0]))

    shoulder_level = 0.5
    shoulder_sym = 0.5
    hip_sym = 0.5

    if len(sh_y_vals) == 2:
        shoulder_y_diff = abs(sh_y_vals[0] - sh_y_vals[1]) / max(bbox_w, 1)
        shoulder_level = 1.0 - min(shoulder_y_diff, 1.0)
    if len(sh_x_vals) == 2:
        shoulder_sym = 1.0 - abs(sh_x_vals[0] - sh_x_vals[1]) / max(bbox_w, 1)
    if len(hp_x_vals) == 2:
        hip_sym = 1.0 - abs(hp_x_vals[0] - hp_x_vals[1]) / max(bbox_w, 1)

    # 5. 头部相对于肩膀的位置
    head_kpt_indices = [0, 1, 2, 3, 4]  # nose, eyes, ears
    head_below_shoulder = False
    head_drop = 0.0
    head_visible_count = 0
    head_kpt_low_conf = 0

    if len(sh_y_vals) > 0:
        avg_sh_y = np.mean(sh_y_vals)
        min_sh_y = min(sh_y_vals)
        drops = []
        for idx in head_kpt_indices:
            conf = float(keypoints[idx, 2])
            if conf > KPT_CONF_THRESHOLD:
                head_visible_count += 1
                drop = (float(keypoints[idx, 1]) - avg_sh_y) / max(bbox_h, 1)
                drop_min = (float(keypoints[idx, 1]) - min_sh_y) / max(bbox_h, 1)
                drops.append(drop)
                drops.append(drop_min)
            elif conf > 0.15:
                drop = (float(keypoints[idx, 1]) - avg_sh_y) / max(bbox_h, 1)
                drop_min = (float(keypoints[idx, 1]) - min_sh_y) / max(bbox_h, 1)
                if drop > 0:
                    head_kpt_low_conf += 1
                if drop_min > 0:
                    drops.append(drop)
                    drops.append(drop_min)
        if drops:
            head_drop = max(drops)
            head_below_shoulder = head_drop > 0.0

    # 6. 头埋臂弯信号
    head_hidden = head_visible_count == 0 and len(sh_y_vals) > 0

    # ===== 综合判断: 是否在睡觉 =====
    # 办公室场景：看不到眼睛，只通过身体姿态判断
    # 目标：趴桌(头部明显下沉) 或 躺下(身体横卧) 才算异常
    # 排除：轻微低头(看手机/写字)、正常坐姿办公

    # 头部下沉深度（相对 bbox 高度）
    head_drop_ratio = head_drop if head_below_shoulder else 0.0

    # 1. 躺下判定：bbox 明显横宽（人横卧）
    #    通用阈值 1.5：多摄像头、不可定制场景下的保守策略
    #    - 真实监控中正常人最大 AR ≈ 1.42，1.5 基本零误报
    #    - 加 head_drop <= 0.1 排除弯腰/蹲着捡东西等极端 case
    if aspect_ratio > 1.5 and head_drop <= 0.1:
        sleep_score = min(aspect_ratio / 2.0, 1.0)
        is_sleeping = True
        sleep_reason = 'lying'
    # 2. 趴桌判定：头部明显下沉（>8% bbox 高度）且姿态大致竖直或略前倾
    #    正常低头看手机/写字时 head_drop_ratio 通常 < 0.12
    #    趴桌时 head_drop_ratio 通常 > 0.20
    #    阈值从 0.18 放宽到 0.08，覆盖更多轻度趴桌/头靠手臂场景
    #    AR>0.7 排除躺着的人（AR 通常很高），但不设上限，避免趴桌且身体横宽时被漏掉
    elif head_drop_ratio > 0.08 and aspect_ratio > 0.7:
        sleep_score = min(0.5 + head_drop_ratio, 1.0)
        is_sleeping = True
        sleep_reason = 'head_on_desk'
    # 3. 其他情况都不算睡岗（包括轻微低头、正常办公、站立）
    else:
        sleep_score = 0.0
        is_sleeping = False
        sleep_reason = 'none'

    # ===== 睡姿分类 =====
    if is_sleeping:
        face_down_s = (1.0 - face_vis) * 0.5 + (1.0 - eye_vis) * 0.5
        face_up_s = face_vis * 0.4 + shoulder_sym * 0.3 + shoulder_level * 0.3
        side_s = (1.0 - shoulder_sym) * 0.4 + (1.0 - hip_sym) * 0.3 + (1.0 - shoulder_level) * 0.3
        total = face_down_s + face_up_s + side_s
        if total > 0:
            face_down_s /= total
            face_up_s /= total
            side_s /= total
        scores = {'face_down': face_down_s, 'face_up': face_up_s, 'side': side_s}
        posture = max(scores, key=scores.get)
        posture_conf = scores[posture]
    else:
        posture = 'standing/sitting' if aspect_ratio < 0.8 else 'not sleeping'
        posture_conf = max(1.0 - sleep_score, 0.1)
        scores = {}

    sleep_reason = 'lying (AR)' if aspect_ratio > 1.5 and head_drop <= 0.1 else \
                   'head_drop' if head_below_shoulder else \
                   'head_hidden' if head_hidden else 'none'

    return {
        'is_sleeping': is_sleeping,
        'sleep_confidence': min(sleep_score, 1.0),
        'posture': posture,
        'posture_label': POSTURE_LABELS.get(posture, posture),
        'posture_confidence': posture_conf,
        'posture_scores': scores,
        'aspect_ratio': aspect_ratio,
        'head_conf': head_conf,
        'head_below_shoulder': head_below_shoulder,
        'head_hidden': head_hidden,
        'head_drop': head_drop,
        'sleep_reason': sleep_reason,
        'shoulder_sym': shoulder_sym,
        'hip_sym': hip_sym,
    }


def process_frame(pose_model, frame, conf=0.25):
    """对单帧进行睡岗检测推理。

    Args:
        pose_model: YOLOv8-pose 模型实例
        frame: BGR numpy 帧
        conf: 置信度阈值

    Returns:
        list of dict, one per detected person, with:
            box, score, sleeping, posture_label, sleep_confidence, keypoints
    """
    results = []
    try:
        r = pose_model.predict(frame, conf=conf, verbose=False)
        if r and r[0].boxes is not None:
            for i in range(len(r[0].boxes)):
                bbox = r[0].boxes.xyxy[i].cpu().numpy()
                kp = r[0].keypoints.data[i].cpu().numpy() if r[0].keypoints is not None else None
                score = float(r[0].boxes.conf[i])
                if kp is not None and len(kp) >= 17:
                    info = analyze_sleep(kp, bbox)
                    results.append({
                        'box': bbox.tolist(),
                        'score': score,
                        'sleeping': info['is_sleeping'],
                        'posture_label': info['posture_label'],
                        'sleep_confidence': info['sleep_confidence'],
                        'keypoints': kp,
                        '_info': info,
                    })
    except Exception as e:
        logging.getLogger(__name__).warning(f"Sleep detection inference error: {e}")
    return results
