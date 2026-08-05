"""睡岗检测模块 - 基于 YOLOv8-pose 的人体姿态 + 睡姿分析

从 infer.py 提取的核心逻辑，适配 engine.py 的多线程架构。
"""

import logging
import time

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

    # 7. 头-肩二维位移（俯视/高角度下头与肩是"前后"关系，head_drop 的垂直假设失效）
    head_shoulder_disp = 0.0
    head_shoulder_dy = 0.0
    if head_visible_count > 0 and len(sh_y_vals) > 0:
        head_pts_2d = np.array([keypoints[i, :2] for i in head_kpt_indices
                                if keypoints[i, 2] > KPT_CONF_THRESHOLD])
        head_c = head_pts_2d.mean(axis=0)
        sh_c = np.array([np.mean(sh_x_vals), np.mean(sh_y_vals)])
        head_shoulder_disp = float(np.linalg.norm(head_c - sh_c) / max(bbox_h, 1))
        head_shoulder_dy = float((head_c[1] - sh_c[1]) / max(bbox_h, 1))

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
    # 3. 头埋臂弯判定：头部关键点全部不可见但肩膀可见（趴睡时脸被手臂/桌面遮挡），
    #    AR>0.7 排除背对摄像头站立/端坐的竖直框
    elif head_hidden and aspect_ratio > 0.7:
        sleep_score = 0.5
        is_sleeping = True
        sleep_reason = 'head_hidden'
    # 4. 头侧靠/侧埋判定：头部关键点仅 1 个可见（侧趴时耳朵可见、脸被头发/手臂遮挡），
    #    头不在肩膀上方（未抬头），框非竖直 → 侧趴睡
    #    清醒坐姿可见头部关键点 ≥2，可见数是最稳的区分信号
    elif (head_visible_count == 1 and len(sh_y_vals) > 0
          and head_drop > -0.05 and aspect_ratio > 0.7):
        sleep_score = 0.5
        is_sleeping = True
        sleep_reason = 'head_side'
    # 5. 趴伏判定（俯视/高角度兜底）：头相对肩明显前移（二维位移 ≥0.13 倍框高），
    #    且头几乎不低于肩膀水平线（dy ≥ -0.14），头部关键点 1-3 个可见。
    #    dy 阈值 -0.14 的取舍（睡岗6_1 实测，低频抽帧场景宁漏勿误）：
    #    - 高角度端坐者头在图像中仍明显高于肩（dy ≤ -0.16），被排除；
    #    - 放宽到 -0.15 与清醒样本余量仅 0.01，姿态抖动即误报；
    #    - 代价：浅趴睡者（dy -0.13~-0.15）命中减半，由其他规则部分兜底。
    #    conf 0.5 低于告警阈值 0.7，仅进显示不单独告警。
    elif (1 <= head_visible_count <= 3 and len(sh_y_vals) > 0
          and aspect_ratio > 0.7
          and head_shoulder_disp >= 0.13 and head_shoulder_dy >= -0.14):
        sleep_score = 0.5
        is_sleeping = True
        sleep_reason = 'slumped'
    # 6. 其他情况都不算睡岗（包括轻微低头、正常办公、站立）
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

    # 旧版 reason 归一化：只对原有分支生效，不覆盖新分支的 reason
    if sleep_reason in ('lying', 'head_on_desk', 'none'):
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


def _upper_body_quality(kp):
    """上半身关键点质量：头(0-4)+肩(5,6) 置信度之和"""
    return float(np.sum(kp[:7, 2]))


def _inter_area(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_iou(a, b):
    inter = _inter_area(a, b)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


class SleepDisplayTracker:
    """睡岗显示时序防抖：跨帧跟踪每个人，原始判定需持续一段时间才翻转显示状态。

    单帧硬判决在阈值附近会逐帧翻转（画面闪烁）。本跟踪器做时间迟滞：
    进入睡眠需持续 ENTER_SECONDS，退出需持续 EXIT_SECONDS。
    这是机制级防抖，与具体判定阈值无关，对任何临界目标都稳定。
    """

    ENTER_SECONDS = 0.4
    EXIT_SECONDS = 1.0
    MAX_MISS_SECONDS = 1.5
    MATCH_IOU = 0.3

    def __init__(self, now=time.monotonic):
        self._tracks = []
        self._now = now

    def update(self, subjects):
        """输入本帧 subject 列表（含 box/sleeping/sleep_confidence），原地稳定后返回"""
        now = self._now()
        unmatched = list(range(len(self._tracks)))
        for s in subjects:
            box = s['box']
            best_i, best_iou = None, self.MATCH_IOU
            for i in unmatched:
                iou = _box_iou(box, self._tracks[i]['box'])
                if iou > best_iou:
                    best_i, best_iou = i, iou
            raw = bool(s.get('sleeping'))
            if best_i is None:
                track = {'box': box, 'display': False, 'last_raw': raw,
                         'streak_start': now, 'last_conf': 0.0, 'last_seen': now}
                self._tracks.append(track)
            else:
                unmatched.remove(best_i)
                track = self._tracks[best_i]
                track['box'] = box
                track['last_seen'] = now
                if raw != track['last_raw']:
                    track['streak_start'] = now
                    track['last_raw'] = raw
            if raw and s.get('sleep_confidence'):
                track['last_conf'] = s['sleep_confidence']
            # 时间迟滞：状态翻转要求原始判定持续足够时长（epsilon 容差避免浮点累加误差）
            if not track['display'] and raw and now - track['streak_start'] >= self.ENTER_SECONDS - 1e-6:
                track['display'] = True
            elif track['display'] and not raw and now - track['streak_start'] >= self.EXIT_SECONDS - 1e-6:
                track['display'] = False
            # 用稳定后的显示状态覆盖本帧输出
            s['sleeping'] = track['display']
            if track['display']:
                if not raw:
                    s['sleep_confidence'] = track['last_conf']
            else:
                s['sleep_confidence'] = 0.0
        self._tracks = [t for t in self._tracks if now - t['last_seen'] <= self.MAX_MISS_SECONDS]
        return subjects


def dedupe_subjects(subjects, iou_thr=0.5, contain_thr=0.7):
    """同一人重复框去重：IoU 高或一方被包含时，保留上半身关键点质量更高的框。

    趴睡姿态下姿态模型可能对同一人输出"上半身"和"全身"两个框，
    默认 NMS(iou=0.7) 拦不住，这里按上半身信息质量二次去重。
    """
    keep = []
    ordered = sorted(subjects, key=lambda s: _upper_body_quality(s['keypoints']), reverse=True)
    for s in ordered:
        dup = False
        for k in keep:
            inter = _inter_area(s['box'], k['box'])
            if inter <= 0:
                continue
            area_s = (s['box'][2] - s['box'][0]) * (s['box'][3] - s['box'][1])
            area_k = (k['box'][2] - k['box'][0]) * (k['box'][3] - k['box'][1])
            iou = inter / (area_s + area_k - inter + 1e-6)
            containment = inter / (min(area_s, area_k) + 1e-6)
            if iou >= iou_thr or containment >= contain_thr:
                dup = True
                break
        if not dup:
            keep.append(s)
    return keep


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
        # iou=0.5 收紧模型自带 NMS（默认 0.7 对趴睡姿态的重复框太宽松）
        r = pose_model.predict(frame, conf=conf, iou=0.5, verbose=False)
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
    return dedupe_subjects(results)
