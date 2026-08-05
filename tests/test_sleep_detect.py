"""睡岗检测核心逻辑测试：head_hidden/head_side/slumped 判睡与重复框去重"""

import numpy as np

from safety_detection.sleep_detect import analyze_sleep, dedupe_subjects, SleepDisplayTracker


def _make_kp(head_conf=0.9, head_y=60, shoulder_y=100):
    """构造 17 点姿态关键点 [x, y, conf]，默认头可见、正常坐姿"""
    kp = np.zeros((17, 3))
    for i in range(5):  # nose, eyes, ears
        kp[i] = [130, head_y, head_conf]
    kp[5] = [100, shoulder_y, 0.9]   # left shoulder
    kp[6] = [160, shoulder_y, 0.9]   # right shoulder
    kp[11] = [105, 200, 0.9]         # left hip
    kp[12] = [155, 200, 0.9]         # right hip
    return kp


# ---------------------------------------------------------------------------
# analyze_sleep: head_hidden（头埋臂弯）判睡
# ---------------------------------------------------------------------------

def test_head_hidden_on_desk_is_sleeping():
    """头部关键点全部不可见 + 肩膀可见 + 框非竖直 → 趴桌头埋臂弯，判睡"""
    kp = _make_kp(head_conf=0.0)
    bbox = [50, 50, 250, 250]  # AR = 1.0
    info = analyze_sleep(kp, bbox)
    assert info['is_sleeping'] is True
    assert info['sleep_reason'] == 'head_hidden'
    assert info['sleep_confidence'] >= 0.5


def test_head_hidden_standing_back_not_sleeping():
    """背对摄像头站立：头不可见但框竖直（AR<0.7）→ 不判睡"""
    kp = _make_kp(head_conf=0.0)
    bbox = [50, 50, 130, 330]  # AR ≈ 0.29
    info = analyze_sleep(kp, bbox)
    assert info['is_sleeping'] is False


def test_normal_sitting_not_sleeping():
    """正常坐姿办公：头可见无下沉 → 不判睡（回归）"""
    kp = _make_kp(head_conf=0.9, head_y=60, shoulder_y=100)
    bbox = [50, 40, 150, 240]  # AR = 0.5
    info = analyze_sleep(kp, bbox)
    assert info['is_sleeping'] is False


def test_head_drop_still_detected():
    """头部明显下沉的趴桌 → 仍按原有 head_on_desk 规则判睡（回归）"""
    kp = _make_kp(head_conf=0.9, head_y=140, shoulder_y=100)
    bbox = [50, 40, 200, 240]  # AR = 0.75, head_drop = 40/200 = 0.2
    info = analyze_sleep(kp, bbox)
    assert info['is_sleeping'] is True
    assert info['sleep_reason'] == 'head_drop'


def test_head_side_resting_is_sleeping():
    """头侧靠/侧埋：仅 1 个头部关键点可见 + 头与肩膀齐平 + 框非竖直 → 判睡

    真实漏检签名：侧趴时耳朵可见(0.9+)、其余头部关键点被头发/手臂遮挡(<0.3)。
    """
    kp = np.zeros((17, 3))
    kp[0] = [130, 90, 0.11]
    kp[1] = [130, 90, 0.01]
    kp[2] = [130, 90, 0.18]
    kp[3] = [130, 90, 0.17]
    kp[4] = [130, 102, 0.92]  # 唯一可见的耳朵，与肩膀齐平
    kp[5] = [100, 100, 0.9]
    kp[6] = [160, 100, 0.9]
    kp[11] = [105, 200, 0.9]
    kp[12] = [155, 200, 0.9]
    bbox = [50, 50, 200, 220]  # AR = 150/170 ≈ 0.88
    info = analyze_sleep(kp, bbox)
    assert info['is_sleeping'] is True
    assert info['sleep_reason'] == 'head_side'
    assert info['sleep_confidence'] >= 0.5


def test_awake_two_visible_head_kpts_not_sleeping():
    """清醒坐姿：2 个头部关键点可见、头在肩膀上方 → 不判睡（真实清醒基线签名）"""
    kp = np.zeros((17, 3))
    kp[0] = [130, 40, 0.39]
    kp[1] = [130, 40, 0.05]
    kp[2] = [130, 40, 0.48]
    kp[3] = [130, 40, 0.16]
    kp[4] = [130, 40, 0.94]
    kp[5] = [100, 100, 0.9]
    kp[6] = [160, 100, 0.9]
    kp[11] = [105, 200, 0.9]
    kp[12] = [155, 200, 0.9]
    bbox = [50, 40, 200, 240]  # AR = 0.75，与判睡区间重叠，靠可见数排除
    info = analyze_sleep(kp, bbox)
    assert info['is_sleeping'] is False


# ---------------------------------------------------------------------------
# analyze_sleep: slumped（俯视/高角度趴伏）判睡
# ---------------------------------------------------------------------------

def test_slumped_topdown_is_sleeping():
    """俯视角度趴睡：头前探（二维位移 0.25 框高、dy=-0.11）+ 2 个头部关键点可见 → 判睡

    真实漏检签名（睡岗6_1 俯视场景）：侧脸/后脑对镜头，仅眼+耳可见，
    头在图像中位于肩膀"前方"（横向偏移为主），head_drop 垂直规则全部失效。
    """
    kp = np.zeros((17, 3))
    kp[0] = [90, 130, 0.10]
    kp[1] = [90, 130, 0.06]
    kp[2] = [90, 130, 0.65]  # 可见眼
    kp[3] = [90, 130, 0.14]
    kp[4] = [90, 130, 0.97]  # 可见耳
    kp[5] = [100, 150, 0.9]
    kp[6] = [160, 150, 0.9]
    kp[11] = [105, 220, 0.9]
    kp[12] = [155, 220, 0.9]
    bbox = [50, 50, 210, 230]  # AR = 160/180 ≈ 0.89
    info = analyze_sleep(kp, bbox)
    assert info['is_sleeping'] is True
    assert info['sleep_reason'] == 'slumped'
    assert info['sleep_confidence'] == 0.5


def test_slumped_upright_head_high_not_sleeping():
    """端坐：头高居肩膀上方（dy=-0.28 < -0.14），即使位移大也不判睡"""
    kp = np.zeros((17, 3))
    for i in range(5):
        kp[i] = [130, 50, 0.1 if i in (1, 3) else 0.8]  # 3 个可见
    kp[5] = [100, 100, 0.9]
    kp[6] = [160, 100, 0.9]
    kp[11] = [105, 200, 0.9]
    kp[12] = [155, 200, 0.9]
    bbox = [50, 40, 210, 220]  # AR = 160/180 ≈ 0.89, disp=0.28, dy=-0.28
    info = analyze_sleep(kp, bbox)
    assert info['is_sleeping'] is False


def test_slumped_topdown_upright_typing_not_sleeping():
    """俯视角度坐着打字（清醒）：头在肩前上方（disp=0.20, dy=-0.20），不判睡

    真实误报签名（睡岗6_1 右下角）：高角度坐姿的 dy≈-0.16~-0.18，
    与趴睡（dy≈-0.08~-0.12）之间取 -0.14 分界，余量 0.02 防姿态抖动。
    """
    kp = np.zeros((17, 3))
    kp[4] = [124, 114, 0.9]  # 仅 1 个耳可见
    kp[5] = [100, 150, 0.9]
    kp[6] = [160, 150, 0.9]
    kp[11] = [105, 220, 0.9]
    kp[12] = [155, 220, 0.9]
    bbox = [50, 50, 210, 230]  # AR≈0.89, bh=180: disp=|{6,-36}|/180≈0.20, dy=-36/180=-0.20
    info = analyze_sleep(kp, bbox)
    assert info['is_sleeping'] is False


def test_slumped_face_visible_not_sleeping():
    """脸朝向镜头（4 个头部关键点可见）：非趴伏姿态 → 不判睡"""
    kp = np.zeros((17, 3))
    kp[0] = [90, 130, 0.8]
    kp[1] = [90, 130, 0.7]
    kp[2] = [90, 130, 0.65]
    kp[3] = [90, 130, 0.14]
    kp[4] = [90, 130, 0.97]
    kp[5] = [100, 150, 0.9]
    kp[6] = [160, 150, 0.9]
    kp[11] = [105, 220, 0.9]
    kp[12] = [155, 220, 0.9]
    bbox = [50, 50, 210, 230]
    info = analyze_sleep(kp, bbox)
    assert info['is_sleeping'] is False


# ---------------------------------------------------------------------------
# dedupe_subjects: 同一人重复框去重
# ---------------------------------------------------------------------------

def _subject(box, head_conf):
    kp = _make_kp(head_conf=head_conf)
    return {'box': box, 'score': 0.9, 'sleeping': False, 'keypoints': kp}


def test_dedupe_contained_boxes_keeps_better_upper_body():
    """包含关系的重复框 → 保留上半身关键点质量更高的"""
    good = _subject([100, 100, 300, 400], head_conf=0.9)   # 大框，头质量好
    poor = _subject([120, 100, 280, 220], head_conf=0.1)   # 被包含的上半身框
    kept = dedupe_subjects([poor, good])
    assert len(kept) == 1
    assert kept[0]['box'] == good['box']


def test_dedupe_high_iou_boxes():
    """高 IoU 重复框 → 去重"""
    a = _subject([100, 100, 300, 400], head_conf=0.9)
    b = _subject([110, 105, 305, 395], head_conf=0.2)
    kept = dedupe_subjects([a, b])
    assert len(kept) == 1
    assert kept[0]['box'] == a['box']


def test_dedupe_distinct_persons_kept():
    """不同人的框不重叠 → 都保留"""
    a = _subject([0, 0, 100, 200], head_conf=0.9)
    b = _subject([300, 0, 400, 200], head_conf=0.5)
    kept = dedupe_subjects([a, b])
    assert len(kept) == 2


# ---------------------------------------------------------------------------
# SleepDisplayTracker: 显示时序防抖
# ---------------------------------------------------------------------------

class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt=0.2):
        self.t += dt


def _subj(box, sleeping, conf=0.6):
    kp = np.zeros((17, 3))
    return {
        'box': box,
        'sleeping': sleeping,
        'sleep_confidence': conf if sleeping else 0.0,
        'keypoints': kp,
    }


BOX_A = [100, 100, 300, 400]
BOX_B = [500, 100, 700, 400]


def test_tracker_flicker_stabilized():
    """原始判定在阈值附近逐帧翻转 → 进入睡眠后单帧漏判不退出，显示稳定"""
    clock = _FakeClock()
    tracker = SleepDisplayTracker(now=clock)
    raws = [True, True, True, False, True, False, True, True]
    outputs = []
    for raw in raws:
        outputs.append(tracker.update([_subj(BOX_A, raw)])[0]['sleeping'])
        clock.advance()
    # 前 0.4s（2 帧）进入期中不显示，之后持续显示（单帧 False 不退出）
    assert outputs[:2] == [False, False]
    assert outputs[2:] == [True] * 6


def test_tracker_single_blip_not_shown():
    """单帧误判为睡（毛刺）→ 未达到进入时长，不显示"""
    clock = _FakeClock()
    tracker = SleepDisplayTracker(now=clock)
    raws = [False, False, True, False, False]
    outputs = []
    for raw in raws:
        outputs.append(tracker.update([_subj(BOX_A, raw)])[0]['sleeping'])
        clock.advance()
    assert outputs == [False] * 5


def test_tracker_exits_after_sustained_awake():
    """持续清醒超过退出时长 → 退出睡眠显示"""
    clock = _FakeClock()
    tracker = SleepDisplayTracker(now=clock)
    for _ in range(3):  # 进入睡眠
        tracker.update([_subj(BOX_A, True)])
        clock.advance()
    assert tracker.update([_subj(BOX_A, True)])[0]['sleeping'] is True
    for _ in range(6):  # 1.2s 持续清醒
        clock.advance()
        out = tracker.update([_subj(BOX_A, False)])[0]['sleeping']
    assert out is False


def test_tracker_keeps_last_confidence_during_blip():
    """睡眠显示期间遇到单帧漏判 → 置信度沿用最近一次有效值，不闪 0.00"""
    clock = _FakeClock()
    tracker = SleepDisplayTracker(now=clock)
    for _ in range(3):
        tracker.update([_subj(BOX_A, True, conf=0.58)])
        clock.advance()
    out = tracker.update([_subj(BOX_A, False)])[0]
    assert out['sleeping'] is True
    assert out['sleep_confidence'] == 0.58


def test_tracker_stale_track_dropped():
    """人消失超过最大丢失时长后，原位置出现新人 → 重新判定，不继承旧状态"""
    clock = _FakeClock()
    tracker = SleepDisplayTracker(now=clock)
    for _ in range(3):
        tracker.update([_subj(BOX_A, True)])
        clock.advance()
    assert tracker.update([_subj(BOX_A, True)])[0]['sleeping'] is True
    clock.advance(2.0)  # 消失 2s（无 update 调用）
    tracker.update([])
    out = tracker.update([_subj(BOX_A, True)])[0]  # 新人，重新进入期
    assert out['sleeping'] is False


def test_tracker_two_persons_independent():
    """两个人按框位置独立跟踪：一个闪烁稳定显示，一个持续清醒不显示"""
    clock = _FakeClock()
    tracker = SleepDisplayTracker(now=clock)
    raws_a = [True, True, True, False, True]
    for raw in raws_a:
        outs = tracker.update([_subj(BOX_A, raw), _subj(BOX_B, False)])
        clock.advance()
    assert outs[0]['sleeping'] is True
    assert outs[1]['sleeping'] is False
