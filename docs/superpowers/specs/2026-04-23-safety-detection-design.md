# Sentry 安全检测扩展设计文档

## 1. 背景与目标

在现有 sentry-rk3588-v1.0.0 基础上，重构为综合安全检测平台，支持以下场景：

- **烟雾** (smoke)
- **明火** (fire)
- **穿工作服** (uniform / PPE)
- **戴口罩** (mask)
- **抽烟** (cigarette)
- **睡岗** (sleep)

原有人员进出检测（person）功能移除。

### 核心约束

- **边缘设备**：RK3588，NPU 推理优先，CPU 回退
- **当前规模**：3 路摄像头测试，架构预留 50 路扩展
- **线程极简**：边缘设备不能开大量线程
- **帧覆盖机制**：检测慢时不积压，只读最新帧
- **小模型 + 大模型结合**：边缘端小模型实时检测，火山引擎 VLM 二次确认

---

## 2. 架构总览

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    视频流层 (每路独立线程)                     │
│  CameraManager: 每路摄像头一个线程拉流，帧覆盖 current_frame   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ 每秒读取最新帧
┌─────────────────────────────────────────────────────────────┐
│              检测层 (策略可插拔 DetectionStrategy)            │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ 当前策略: SerialStrategy（串行轮询，RK3588 低资源模式）    ││
│  │ 未来策略: ThreadPoolStrategy（线程池并行，高配设备模式）   ││
│  │                                                         ││
│  │  fire/smoke ──→ P0 告警（立即告警，异步 VLM 复核）       ││
│  │  uniform/mask/cigarette ──→ P1 告警                      ││
│  │    （抽帧 → VLM 确认 → 告警）                            ││
│  │  sleep ──→ P1 告警（60s 间隔，CV+VLM 同一人判定）        ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              VLM 层 (单线程队列 + 信号量控并发)               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  VLMQueue: 消费复核/确认任务，max_concurrent 限制并发数   ││
│  │  - P0 fire/smoke 复核任务                                ││
│  │  - P1 uniform 窗口合规确认任务                           ││
  │  - P1 mask/cigarette 确认任务                            ││
│  │  - sleep VLM 判断任务                                    ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              记录层 (performance_storage.py 扩展)             │
│  - 统一记录格式，兼容 P0/P1                                  │
│  - 后端保存线程异步落盘                                      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 线程模型

整个系统仅 4 类线程：

| 线程类型 | 数量 | 说明 |
|---|---|---|
| 视频拉流线程 | 每路 1 个 | `camera_manager` 已有，独立拉流，帧覆盖 |
| **检测循环线程** | **全局 1 个** | 串行轮询，执行所有模型推理 |
| **VLM 队列线程** | **全局 1 个** | 消费 VLM 任务，`Semaphore` 控制并发 |
| **VLM 巡检线程** | **全局 1 个** | 每 30 秒触发一次全类型 VLM 综合巡检 |

**禁止**：线程池、每路检测线程、每模型独立线程。

---

## 3. 检测类型与模型映射

### 3.1 模型来源

参考 SafeVision demo 的模型选择：

| 检测类型 | 模型 | 来源 | 推理方式 |
|---|---|---|---|
| `fire` | `fire.pt` (YOLO) | SafeVision | ultralytics YOLO，CPU/NPU |
| `smoke` | `fire.pt` 复用 或 `smoke.pt` | SafeVision | ultralytics YOLO |
| `cigarette` | `iic/cv_tinynas_object-detection_damoyolo_cigarette` | ModelScope | `pipeline()` 调用 |
| `mask` | `iic/cv_tinynas_object-detection_damoyolo_facemask` | ModelScope | `pipeline()` 调用 |
| `uniform` | `melihuzunoglu/ppe-detection` (YOLO) | HuggingFace | ultralytics YOLO |
| `sleep` | `yolov8s-pose.pt` + `sleep_detect.py` | SafeVision | YOLOv8-pose + 姿态分析 |

### 3.2 懒加载策略

`SafetyDetector` 按需加载模型，但 **CPU 模型**和 **NPU 模型**的加载方式不同：

**CPU/PyTorch 模型**：全局单例，所有线程共享同一份模型对象。

```python
# CPU 模型：内存里只有 1 份，多线程共享
self._cpu_models: Dict[str, Any] = {}

if "uniform" in enabled_types and "uniform" not in self._cpu_models:
    self._cpu_models["uniform"] = YOLO("models/uniform.pt")
```

**NPU/RKNN 模型**：每个 NPU 核心需要独立的 RKNN 实例。同一种模型在 NPU 内存中存在 `npu_cores` 份拷贝。

```python
# NPU 模型：每种模型 × NPU核心数 个实例
self._npu_models: Dict[str, Dict[int, Any]] = {}  # {dtype: {core_id: rknn_instance}}

npu_cores = detect_npu_cores()  # RK3588 返回 3

if "fire" in enabled_types and "fire" not in self._npu_models:
    self._npu_models["fire"] = {}
    for core_id in range(npu_cores):
        rknn = RKNNLite(verbose=False)
        rknn.load_rknn("models/fire.rknn")
        rknn.init_runtime(core_mask=core_id)  # 绑定到指定核心
        self._npu_models["fire"][core_id] = rknn
```

**为什么 NPU 模型必须每核心一份？**

NPU 核心是独立的硬件单元，每个核心有自己的显存和计算管线，**不能共享同一个模型上下文**。就像 3 张显卡不能共享同一份显存里的模型。

```
NPU Core 0 ── 持有 fire 实例 #0
NPU Core 1 ── 持有 fire 实例 #1
NPU Core 2 ── 持有 fire 实例 #2

三个核心同时推理，互不干扰
```

**内存控制**：
- 未启用的检测类型不加载模型
- CPU 内存占用 = 启用模型数 × 1 份
- NPU 内存占用 = 启用模型数 × NPU核心数 份
- 例：RK3588（3 核）全开 6 种 NPU 模型 → NPU 上驻留 6 × 3 = 18 个 RKNN 实例

---

## 4. 两级告警机制

### 4.1 告警分级

| 级别 | 检测类型 | 触发方式 | 响应速度 | 冷却 |
|---|---|---|---|---|
| **P0 - 实时告警** | `fire`, `smoke` | 小模型超阈值立即告警，VLM 仅异步复核 | 秒级 | 10s |
| **P1 - 确认告警** | `uniform`, `mask`, `cigarette`, `sleep` | 小模型检测 → 抽帧 → VLM 确认 → 告警 | 5-10s | 3s |

### 4.2 P0 告警流程

```
小模型检测 fire/smoke 超阈值
    │
    ├──→ 立即创建 P0 记录 (status="alerted")
    │    前端实时推送红色告警
    │    主流程结束（用户已收到告警）
    │
    └──→ 异步提交 VLM 复核任务（不阻塞）
              │
              ▼
         VLM 分析触发帧 + 后续 2-3 帧
              │
              ├── 真实火情 → 更新记录: vlm_confirmed=true, status="confirmed"
              └── 误报 → 更新记录: vlm_confirmed=false, status="false_positive"
```

**复核 Prompt 设计**：

```
你是一位消防安全复核专家。请仔细查看提供的图片，判断画面中是否真实存在明火/烟雾。
注意区分真实火焰/烟雾与误报（红色物体、灯光、屏幕反光、水汽等）。
请用 JSON 格式回答：{"is_real": true/false, "confidence": 0-1, "reason": "..."}
```

**误报记录保留**：不删除，标记为 `false_positive`，用于后续统计和阈值调优。

### 4.3 P1 告警流程

```
小模型检测到 uniform/mask/cigarette 超阈值
    │
    ├──→ 保存快照，创建记录 (status="pending")
    │
    └──→ 抽帧 (15 帧 / 0.5s 间隔)
              │
              ▼
         VLM 场景确认（专用 prompt）
              │
              ├── 确认违规 → status="confirmed", 前端展示告警
              └── 判断正常 → status="rejected", 记录保留
```

---

## 5. SafetyDetector 模块设计

### 5.1 类定义

```python
class SafetyDetector:
    def __init__(self, npu_cores: int = 0):
        self._cpu_models: Dict[str, Any] = {}           # CPU 模型：每种 1 份
        self._npu_models: Dict[str, Dict[int, Any]] = {} # NPU 模型：每种 × npu_cores 份
        self._model_lock = threading.RLock()
        self._npu_cores = npu_cores

    def ensure_models_loaded(self, detection_types: List[str]):
        """懒加载指定类型所需的模型（区分 CPU/NPU）"""

    def detect(self, frame: np.ndarray, detection_types: List[str], core_id: int = 0) -> Dict[str, Any]:
        """
        对单帧执行所有启用的检测
        
        Returns:
            {
                "fire": {"detected": False, "boxes": [], "max_confidence": 0.0},
                "sleep": {"detected": True, "subjects": [...], "count": 1},
                ...
            }
        """
```

### 5.2 睡岗检测特殊处理

睡岗是唯一需要姿态分析的检测类型：

```python
def _detect_sleep(self, frame: np.ndarray) -> dict:
    # 1. YOLOv8-pose 检测人体关键点
    results = self._models["sleep_pose"](frame, verbose=False)
    
    sleeping_subjects = []
    for result in results:
        if result.keypoints is None:
            continue
        for kpts, box in zip(result.keypoints.data, result.boxes):
            # 2. 自定义姿态分析
            analysis = sleep_detect.analyze_sleep(
                kpts.cpu().numpy(),
                box.xyxy[0].cpu().numpy()
            )
            if analysis["is_sleeping"]:
                sleeping_subjects.append({
                    "box": box.xyxy[0].tolist(),
                    "posture": analysis["posture"],
                    "confidence": analysis["sleep_confidence"]
                })
    
    return {
        "detected": len(sleeping_subjects) > 0,
        "subjects": sleeping_subjects,
        "count": len(sleeping_subjects)
    }
```

### 5.3 阈值配置

```python
DETECTION_THRESHOLDS = {
    "fire": 0.6,
    "smoke": 0.55,
    "cigarette": 0.5,
    "mask": 0.5,        # 注意：模型检测的是"未戴口罩的人"
    "uniform": 0.5,     # 注意：模型检测的是"未穿工作服的人"
    "sleep": 0.7,
}
```

---

## 6. 睡岗状态机

### 6.1 设计目标

睡岗检测的核心难点是**多人交替打盹的误报**：帧1人员A打盹、帧2人员B打盹、帧3人员C打盹，按帧累计会误触发告警，但实际上没有同一人连续睡岗达到阈值。

解决方案：**不做边缘端人员追踪，改为收集 `consecutive_required` 张睡岗帧后，一次性提交 VLM 做多图同一人判定**。

- 每 **60 秒** 检测一次（通过 `interval` 配置）
- **CV 小模型检测睡岗姿态**：检测到姿态后保存该帧，不立即提交 VLM
- **`consecutive_required` 张帧凑齐后**：将这 `consecutive_required` 张图一起发给 VLM，让 VLM 判断"是否是同一个人在连续睡岗"
- 任一一次 CV 未检测到睡岗姿态：清空已收集的帧和计数器
- `consecutive_required` 完全可配置（默认 3，可通过 `cameras.json` 修改）

### 6.2 状态流转

```
状态: idle ──60s 触发──→ cv_check
                              │
                              ├── cv=False ──→ 清空历史帧 + count=0 ──→ idle
                              │
                              └── cv=True ──→ 保存帧到 history_frames
                                                      │
                                                      ├── count < consecutive_required ──→ idle
                                                      │
                                                      └── count >= consecutive_required
                                                                │
                                                                ▼
                                                          提交 VLM 同一人判定
                                                                │
                                                                ├── VLM: 是同一人 ──→ ALERT!
                                                                │
                                                                └── VLM: 不是同一人 ──→ 清空 + idle
```

### 6.3 实现

```python
def _handle_sleep_detection(camera_id: str, frame: np.ndarray, schedule: TypeSchedule):
    """睡岗检测：收集 consecutive_required 张睡岗帧后，触发 VLM 同一人判定"""
    
    # 1. CV 检测睡岗姿态
    cv_result = safety_detector._detect_sleep(frame)
    if not cv_result["detected"]:
        # CV 未通过：清空历史帧和计数
        schedule.consecutive_count = 0
        schedule.history_frames.clear()
        return
    
    # 2. CV 通过：保存当前帧（使用 deque 限制最大长度，防止内存泄漏）
    schedule.consecutive_count += 1
    schedule.history_frames.append(frame)
    
    # 3. 达到配置阈值，触发 VLM 同一人判定
    if schedule.consecutive_count >= schedule.consecutive_required:
        vlm_queue.submit(ReviewTask(
            record_id=...,
            frames=list(schedule.history_frames),
            prompt=SLEEP_IDENTITY_PROMPT,
            callback=lambda result: _on_sleep_identity_result(camera_id, schedule, result)
        ))

def _on_sleep_identity_result(camera_id: str, schedule: TypeSchedule, vlm_result: dict):
    """VLM 同一人判定回调"""
    is_same = vlm_result.get("same_person", False)
    confidence = vlm_result.get("confidence", 0)
    
    if is_same and confidence > 0.7:
        create_alert(
            camera_id, "sleep", level="P1",
            detail=f"连续 {schedule.consecutive_required} 次检测到同一人睡岗"
        )
    else:
        logger.info(
            f"Sleep identity check failed on {camera_id}: "
            f"same_person={is_same}, conf={confidence:.2f}"
        )
    
    # 无论 VLM 是否确认，判定后都清空状态，等待下一轮
    schedule.consecutive_count = 0
    schedule.history_frames.clear()
```

### 6.4 TypeSchedule 扩展

睡岗需要在 `TypeSchedule` 中增加历史帧缓存：

```python
@dataclass
class TypeSchedule:
    dtype: str
    interval: float
    threshold: float
    consecutive_required: int      # 可配置：触发 VLM 判定的帧数阈值
    consecutive_count: int = 0
    last_detection_time: float = 0
    pending_vlm: bool = False
    history_frames: deque = field(default_factory=lambda: deque(maxlen=10))  # 睡岗专用：缓存历史帧
    
    def is_due(self, now: float) -> bool:
        return now - self.last_detection_time >= self.interval
```

### 6.5 内存清理

摄像头被动态删除时（`DELETE /cameras/{id}`），必须同步清理对应的 `TypeSchedule` 状态，避免 `history_frames` 中的图片数组长期占用内存：

```python
def unregister_camera_schedules(camera_id: str):
    """摄像头注销时清理所有检测类型的连续计数和历史帧缓存"""
    if camera_id in self._schedules:
        for schedule in self._schedules[camera_id].values():
            schedule.history_frames.clear()
        del self._schedules[camera_id]
```

### 6.6 VLM Prompt

```
你正在查看同一摄像头的 {consecutive_required} 张监控截图，拍摄时间间隔约 {interval} 秒。

请仔细判断：这 {consecutive_required} 张图中，睡岗/打盹的是否是同一个特定的人？
注意排除以下情况：
- 不同的人轮流打盹
- 同一个人只是短暂低头后恢复正常
- 画面中有多人，但睡岗的人换了

请用 JSON 格式回答：
{
  "same_person": true/false,
  "confidence": 0-1,
  "reason": "简要说明判断依据"
}
```

---

## 7. VLM 异步队列

### 7.1 VLMQueue 设计

```python
class VLMQueue:
    def __init__(self, understander, max_concurrent: int = 3):
        self.understander = understander
        self.semaphore = threading.Semaphore(max_concurrent)
        self.queue = deque()
        self._running = False
        
    def start(self):
        self._running = True
        threading.Thread(target=self._consume, daemon=True).start()
        
    def submit(self, task: ReviewTask):
        self.queue.append(task)
        
    def _consume(self):
        while self._running:
            if not self.queue:
                time.sleep(0.1)
                continue
            
            task = self.queue.popleft()
            self.semaphore.acquire()
            
            # 启动工作线程执行 VLM 请求
            threading.Thread(
                target=self._run_vlm,
                args=(task,),
                daemon=True
            ).start()
    
    def _run_vlm(self, task: ReviewTask):
        try:
            result = self.understander.analyze(task.frames)
            # 回调在独立工作线程中执行，需加锁保护共享状态
            with task.lock:
                task.callback(result)
        finally:
            self.semaphore.release()
```

### 7.2 队列优先级与上限

VLM 任务分两类：
- **P0 复核任务**（fire/smoke）：已产生实时告警，需尽快复核确认，**优先级高**
- **P1 确认任务**（uniform/mask/cigarette）：决定告警是否产生，**优先级低**

实现：使用两个队列，`p0_queue` 优先于 `p1_queue` 消费。

**队列上限限制**：
- `p0_queue` 上限：50（P0 任务少但紧急，上限宽松）
- `p1_queue` 上限：100（P1 任务多，上限严格）
- 超过上限时，丢弃最旧的同优先级任务，并记录日志警告

```python
class VLMQueue:
    def __init__(self, understander, max_concurrent: int = 3):
        self.understander = understander
        self.semaphore = threading.Semaphore(max_concurrent)
        self.p0_queue = deque(maxlen=50)   # fire/smoke 复核，高优先级
        self.p1_queue = deque(maxlen=100)  # uniform/mask/cigarette 确认，低优先级
        self._running = False
        
    def submit(self, task: ReviewTask):
        if task.level == "P0":
            if len(self.p0_queue) >= self.p0_queue.maxlen:
                logger.warning(f"P0 queue full, dropping oldest review task")
            self.p0_queue.append(task)
        else:
            if len(self.p1_queue) >= self.p1_queue.maxlen:
                logger.warning(f"P1 queue full, dropping oldest confirm task")
            self.p1_queue.append(task)
        
    def _consume(self):
        while self._running:
            task = None
            
            # 优先消费 P0 队列
            if self.p0_queue:
                task = self.p0_queue.popleft()
            elif self.p1_queue:
                task = self.p1_queue.popleft()
            else:
                time.sleep(0.1)
                continue
            
            self.semaphore.acquire()
            threading.Thread(target=self._run_vlm, args=(task,), daemon=True).start()
```

---

## 8. VLM 全类型巡检

### 8.1 设计目标

每 **30 秒**（可配置）触发一次 VLM 综合巡检，**一次对话**判断所有**已启用**的检测类型是否存在异常。

**与小模型检测的关系**：
- **补充校验**：小模型负责高频实时检测（1fps），VLM 巡检负责低频综合复核（0.033fps）
- **交叉验证**：如果 VLM 巡检发现某类型异常，但小模型未告警，可触发补充告警或调高该类型敏感度

### 8.2 巡检流程

```
每 30 秒触发
    │
    ├──→ 收集所有摄像头当前帧
    ├──→ 收集所有摄像头已启用的检测类型（去重）
    ├──→ 构建综合 prompt
    │
    └──→ 提交 VLM 一次对话
              │
              ▼
    {
      "fire": {"detected": false, "confidence": 0.1},
      "smoke": {"detected": true, "confidence": 0.85, "description": "画面左上角有烟雾"},
      "uniform": {"detected": false},
      "mask": {"detected": true, "confidence": 0.92, "description": "人员未佩戴口罩"},
      "cigarette": {"detected": false},
      "sleep": {"detected": false}
    }
              │
              ▼
    对比小模型检测结果：
    - VLM 和小模型一致 → 正常
    - VLM 检测到，小模型漏检 → 触发补充告警
    - 小模型检测到，VLM 漏检 → 记录日志（用于后续优化小模型阈值）
```

### 8.3 Prompt 设计

```
你是一位安全监控综合研判专家。请仔细查看提供的监控画面，判断是否存在以下安全隐患：

需要检查的类型（只判断以下已启用的类型）：
{enabled_types}

对每种类型，请判断画面中是否真实存在：
- fire（明火）：真实火焰、燃烧
- smoke（烟雾）：真实烟雾，非水汽/灰尘
- uniform（工服）：是否存在未穿工作服的人员
- mask（口罩）：是否存在未戴口罩的人员
- cigarette（抽烟）：是否存在正在抽烟的人员
- sleep（睡岗）：是否存在正在睡觉/打盹的人员

请用 JSON 格式回答，只包含用户要求的类型：
{
  "fire": {"detected": true/false, "confidence": 0-1, "description": "..."},
  "smoke": {"detected": true/false, "confidence": 0-1, "description": "..."},
  ...
}
```

### 8.4 实现

```python
class VLMInspector:
    def __init__(self, detector, interval: float = 30.0):
        self.detector = detector
        self.interval = interval
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._inspection_loop, daemon=True).start()

    def _inspection_loop(self):
        while self._running:
            time.sleep(self.interval)
            self._run_inspection()

    def _run_inspection(self):
        # 1. 收集所有摄像头当前帧和已启用类型
        frames = []
        enabled_types = set()

        for camera_id in camera_manager.get_camera_ids():
            frame = camera_manager.get_frame(camera_id)
            if frame is not None:
                frames.append((camera_id, frame))
                types = get_enabled_types(camera_id)
                enabled_types.update(types)

        if not frames or not enabled_types:
            return

        # 2. 构建综合 prompt
        prompt = build_inspection_prompt(enabled_types)

        # 3. 一次 VLM 调用（多图输入）
        result = self.understander.analyze_multi(frames, prompt)

        # 4. 处理巡检结果
        for camera_id, camera_result in result.items():
            for dtype, detection in camera_result.items():
                if detection["detected"]:
                    self._handle_inspection_result(
                        camera_id, dtype, detection
                    )

    def _handle_inspection_result(self, camera_id: str, dtype: str, detection: dict):
        """处理单条巡检结果，执行四重去重后注入检测流"""

        # 四重去重检查（避免重复处理已知的异常）
        # 1. 当前已有活跃告警
        if self.detector.has_active_alert(camera_id, dtype):
            return

        # 2. 该类型正在等待 VLM 复核/确认
        if self.detector.is_pending_vlm(camera_id, dtype):
            return

        # 3. 该类型处于告警冷却期
        if self.detector.is_in_cooldown(camera_id, dtype):
            return

        # 4. 睡岗特殊状态：已有待复核的睡岗检测
        if dtype == "sleep" and self.detector.sleep_has_pending_vlm(camera_id):
            return

        # 通过去重检查：VLM 巡检发现漏检，注入小模型检测流（补课）
        logger.warning(
            f"VLM inspection found {dtype} on {camera_id} "
            f"(conf={detection.get('confidence', 0):.2f}), "
            f"injecting into detection flow"
        )

        # 构建一个模拟的小模型检测结果，注入正常告警流程
        inject_detection = {
            "camera_id": camera_id,
            "type": dtype,
            "confidence": detection.get("confidence", 0.8),
            "description": detection.get("description", ""),
            "source": "vlm_inspection",  # 标记来源为巡检补课
        }

        # 高置信度巡检结果（>0.85）可跳过二次 VLM 复核，直接走确认流程
        if detection.get("confidence", 0) > 0.85:
            inject_detection["vlm_pre_confirmed"] = True

        # 特殊处理睡岗：巡检不缩短 60s 检测间隔，仅作为一次独立命中计数
        if dtype == "sleep":
            inject_detection["skip_interval_check"] = True

        # 注入检测器的主处理流程（与真实小模型检测等效）
        self.detector.inject_detection(inject_detection)
```

**去重规则说明**：

| 去重层级 | 检查内容 | 目的 |
|---------|---------|------|
| 第 1 重 | `has_active_alert` | 避免对已告警的状态重复注入 |
| 第 2 重 | `is_pending_vlm` | 避免与正在 VLM 队列中的任务冲突 |
| 第 3 重 | `is_in_cooldown` | 遵守告警冷却期，防止频繁重复告警 |
| 第 4 重 | `sleep_has_pending_vlm` | 睡岗特殊状态保护，避免重复提交复核 |

**补课注入逻辑**：
- 巡检发现的漏检会被包装为模拟小模型检测结果，注入正常的告警判定流程
- 高置信度（>0.85）结果标记 `vlm_pre_confirmed=true`，可跳过 P1 类型的二次 VLM 确认，直接产生告警
- 睡岗检测保持 60s 间隔机制，巡检命中只计入累计次数，不触发额外抽帧

### 8.5 资源控制

- **频率可配置**：`vlm_inspection_interval`（默认 30 秒，可设为 60s/120s/关闭）
- **帧数限制**：每次最多取 3 路摄像头的帧，避免 VLM 输入过大
- **超时控制**：VLM 巡检超时 10 秒则放弃本次，等待下一轮

---

## 9. 检测调度循环

### 9.1 设计原则：NPU 核心自适应 + 策略可插拔

检测调度层的核心约束是**检测线程数必须与 NPU 核心数匹配**，而非摄像头数：

- **RK3588（3 NPU 核心）**：3 个检测线程，每线程绑定 1 个 NPU 核心
- **后续 8 NPU 设备**：8 个检测线程，每线程绑定 1 个 NPU 核心
- **纯 CPU 设备**：回退到 `SerialStrategy`（单线程串行）

采用**策略模式（Strategy Pattern）**，将"如何调度检测"与"检测业务逻辑"解耦，支持运行时根据设备自动选择最优策略。

```python
from abc import ABC, abstractmethod

class DetectionStrategy(ABC):
    """检测调度策略抽象基类"""

    @abstractmethod
    def run(self, detector: "MultiDetector") -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass
```

### 9.2 NPU 核心自适应检测

**`NPUDetector`** 负责动态探测 NPU 核心数，并创建对应数量的 RKNN 实例：

```python
import os
from typing import List, Dict, Optional

def detect_npu_cores() -> int:
    """
    自动检测可用 NPU 核心数。
    优先级：环境变量 > RKNN 查询 > 默认值
    """
    # 1. 环境变量覆盖（便于测试和跨平台适配）
    env_cores = os.getenv("NPU_CORES")
    if env_cores:
        return int(env_cores)

    # 2. RKNN 运行时查询（RK3588 返回 3）
    try:
        from rknnlite.api import RKNNLite
        # RKNNLite 不直接暴露核心数，通过枚举 mask 判断
        masks = [
            RKNNLite.NPU_CORE_0,
            RKNNLite.NPU_CORE_1,
            RKNNLite.NPU_CORE_2,
        ]
        valid = 0
        for mask in masks:
            rknn = RKNNLite(verbose=False)
            # 尝试初始化，成功则计数
            if rknn.init_runtime(core_mask=mask) == 0:
                valid += 1
                rknn.release()
            else:
                break
        if valid > 0:
            return valid
    except Exception:
        pass

    # 3. 默认回退（无 NPU 时返回 0）
    return 0


def group_cameras(camera_ids: List[str], n_cores: int) -> List[List[str]]:
    """
    将摄像头平均分配到 NPU 核心组。
    例：50 路 + 8 核 → [7,7,6,6,6,6,6,6]
    """
    if n_cores <= 0:
        return [camera_ids]

    groups: List[List[str]] = [[] for _ in range(n_cores)]
    for i, cam_id in enumerate(camera_ids):
        groups[i % n_cores].append(cam_id)
    return groups
```

### 9.3 CorePinnedStrategy（NPU 设备推荐）

**适用场景**：RK3588（3 核）或后续多 NPU 核心设备。

**核心设计**：
- 检测线程数 = NPU 核心数（自动探测）
- 摄像头按数量平均分配到各核心
- 每线程**固定绑定**一个 NPU 核心，独占该核心上的 RKNN 实例
- 实现真正的多核并行推理

```python
class CorePinnedStrategy(DetectionStrategy):
    """
    NPU 核心绑定策略：
    - 线程数 = 探测到的 NPU 核心数
    - 摄像头按核心数分组，每组一个 worker 线程
    - 每个 worker 固定使用一个 NPU 核心（core_id）
    """

    def __init__(self):
        self._running = False
        self._threads: List[threading.Thread] = []
        self._npu_cores = detect_npu_cores()

    def run(self, detector: "MultiDetector"):
        self._running = True

        # 摄像头分组：按 NPU 核心数平均分配
        camera_groups = group_cameras(detector.camera_ids, self._npu_cores)
        logger.info(
            f"CorePinnedStrategy: {self._npu_cores} NPU cores, "
            f"{len(detector.camera_ids)} cameras grouped as "
            f"{[len(g) for g in camera_groups]}"
        )

        # 为每个核心启动一个 worker 线程
        for core_id, cam_group in enumerate(camera_groups):
            if not cam_group:
                continue
            t = threading.Thread(
                target=self._worker_loop,
                args=(detector, cam_group, core_id),
                name=f"npu-core-{core_id}",
                daemon=True
            )
            t.start()
            self._threads.append(t)

    def _worker_loop(self, detector: "MultiDetector", camera_ids: List[str], core_id: int):
        """单个 NPU 核心的工作线程：串行处理分配给它的摄像头"""
        while self._running:
            cycle_start = time.time()

            for camera_id in camera_ids:
                if not self._running:
                    break

                frame = detector.camera_manager.get_frame(camera_id)
                if frame is None:
                    continue

                due_types = detector.get_due_types(camera_id)
                if not due_types:
                    continue

                # 关键：固定使用当前线程绑定的 NPU 核心
                results = detector.safety_detector.detect(
                    frame, due_types, core_id=core_id
                )

                # 结果处理（与 SerialStrategy 完全一致）
                for dtype, result in results.items():
                    schedule = detector.schedules[camera_id][dtype]

                    if dtype == "uniform":
                        detector.handle_uniform_window(camera_id, result, schedule)
                    elif dtype == "sleep":
                        detector.handle_sleep_detection(camera_id, frame, result, schedule)
                    else:
                        if result["detected"]:
                            schedule.consecutive_count += 1
                            if schedule.consecutive_count >= schedule.consecutive_required:
                                detector.handle_trigger(camera_id, dtype, frame, result)
                        else:
                            schedule.consecutive_count = 0

            # 自适应 yield
            yield_time = min(0.05, len(camera_ids) * 0.002)
            time.sleep(yield_time)

            # 控制轮询周期
            elapsed = time.time() - cycle_start
            sleep_time = max(0, detector.cycle_interval - elapsed)
            time.sleep(sleep_time)

    def stop(self):
        self._running = False
        for t in self._threads:
            t.join(timeout=3)
```

### 9.4 SerialStrategy（无 NPU 回退）

**适用场景**：纯 CPU 设备、NPU 驱动不可用、或调试模式。

**特点**：单线程串行轮询所有摄像头，不依赖 NPU。

```python
class SerialStrategy(DetectionStrategy):
    """串行轮询策略：单线程依次处理所有摄像头（CPU 回退）"""

    def __init__(self):
        self._running = False

    def run(self, detector: "MultiDetector"):
        self._running = True
        while self._running:
            cycle_start = time.time()

            for camera_id in detector.camera_ids:
                if not self._running:
                    break
                self._process_camera(detector, camera_id)
                yield_time = min(0.05, len(detector.camera_ids) * 0.002)
                time.sleep(yield_time)

            elapsed = time.time() - cycle_start
            sleep_time = max(0, detector.cycle_interval - elapsed)
            time.sleep(sleep_time)

    def _process_camera(self, detector: "MultiDetector", camera_id: str):
        """处理单路摄像头的检测"""
        frame = detector.camera_manager.get_frame(camera_id)
        if frame is None:
            return

        due_types = detector.get_due_types(camera_id)
        if not due_types:
            return

        results = detector.safety_detector.detect(frame, due_types)

        for dtype, result in results.items():
            schedule = detector.schedules[camera_id][dtype]

            if dtype == "uniform":
                detector.handle_uniform_window(camera_id, result, schedule)
            elif dtype == "sleep":
                detector.handle_sleep_detection(camera_id, frame, result, schedule)
            else:
                if result["detected"]:
                    schedule.consecutive_count += 1
                    if schedule.consecutive_count >= schedule.consecutive_required:
                        detector.handle_trigger(camera_id, dtype, frame, result)
                else:
                    schedule.consecutive_count = 0

    def stop(self):
        self._running = False
```

### 9.5 策略自动选择

启动时自动探测硬件，选择最优策略，无需手动配置：

```python
def auto_select_strategy() -> DetectionStrategy:
    """根据硬件自动选择检测策略"""
    npu_cores = detect_npu_cores()

    if npu_cores >= 2:
        logger.info(f"Detected {npu_cores} NPU cores, using CorePinnedStrategy")
        return CorePinnedStrategy()
    else:
        logger.info("No NPU detected or only 1 core, using SerialStrategy")
        return SerialStrategy()


# 启动代码
detector = MultiDetector(
    camera_manager=cam_mgr,
    safety_detector=safety,
    strategy=auto_select_strategy()  # 自动选择
)
detector.start()
```

### 9.6 按类型独立调度

每个摄像头维护各检测类型的调度状态：

```python
@dataclass
class TypeSchedule:
    enabled: bool
    interval: float       # 检测间隔（秒）
    threshold: float
    last_run: float = 0.0

    # 异常帧数判定：连续 N 次检测到才算异常，避免单帧误判
    # 注：uniform 类型使用 compliance_window_seconds 而非本字段
    consecutive_required: int = 1   # 默认 1 次（即单帧即告警）
    consecutive_count: int = 0      # 当前连续命中计数
    pending_vlm: bool = False       # 是否已提交 VLM 等待中

    # 工服专用：合规观察窗口（秒），窗口内检测到 vest 即合规
    compliance_window_seconds: float = 30.0

    # 睡岗专用：缓存检测到睡岗姿态的历史帧，凑齐 consecutive_required 张后提交 VLM 同一人判定
    history_frames: deque = field(default_factory=lambda: deque(maxlen=10))

    def is_due(self, now: float) -> bool:
        return now - self.last_detection_time >= self.interval
```

### 9.7 帧覆盖机制

**不会积压**：`camera_manager.get_frame()` 始终返回最新帧，旧帧被覆盖。检测慢只会导致轮询周期变长，每路实际检测间隔增加，但永远不会内存泄漏。

**策略无关性**：无论使用 `CorePinnedStrategy` 还是 `SerialStrategy`，帧覆盖机制都保持不变——每个摄像头线程独立覆盖自己的 `current_frame`，检测线程只读取最新帧。

**NPU 核心隔离性**：`CorePinnedStrategy` 中，每个 worker 线程只处理分配给它的摄像头，帧数据完全隔离，不会出现跨核心混淆。
---

## 10. 记录系统

### 10.1 统一记录格式

```json
{
  "id": "cam_01_1713849600000",
  "camera_id": "cam_01",
  "camera_name": "车间入口",
  "time": "2026-04-23 14:30:00",
  "detection_type": "fire",
  "level": "P0",
  "status": "confirmed",
  
  "small_model": {
    "detected": true,
    "confidence": 0.91,
    "boxes": [[100, 200, 300, 400]]
  },
  
  "vlm_review": {
    "confirmed": true,
    "confidence": 0.95,
    "reason": "画面中央有明火燃烧",
    "review_time": "2026-04-23 14:30:02"
  },
  
  "snapshot": "base64...",
  "frames": [],
  "timing": {
    "total_seconds": 2.1
  }
}
```

### 10.2 状态流转

| 阶段 | P0 (fire/smoke) | P1 (其他) |
|---|---|---|
| 初始 | `alerted` | `pending` |
| VLM 确认后 | `confirmed` / `false_positive` | `confirmed` / `rejected` |

### 10.3 存储限制与内存回收

边缘设备内存有限，需防止 `data/frames/` 下的图片文件无限增长。

**三级回收策略：**

| 策略 | 触发条件 | 行为 |
|---|---|---|
| **数量限制** | 单条记录默认最多保存 15 帧（抽帧数），超出不存 | 已有，无需改动 |
| **记录总数限制** | `MAX_RECORDS = 100`，超出时删除最旧记录及其图片 | 已有，无需改动 |
| **存储空间限制** | `MAX_STORAGE_MB = 500`，超出时按时间删除旧记录 | **新增** |
| **内存紧急回收** | 系统内存使用 > 80% 时，立即清理最旧的 20% 记录 | **新增** |

**后台清理线程：**

```python
def _storage_cleanup_loop():
    """每小时检查一次存储空间，所有阈值从 global_settings 读取"""
    while True:
        time.sleep(3600)
        
        settings = load_global_settings()
        
        # 1. 检查存储空间
        total_mb = get_storage_size_mb(DATA_DIR)
        if total_mb > settings["max_storage_mb"]:
            logger.warning(f"Storage {total_mb}MB > limit {settings['max_storage_mb']}MB, cleanup triggered")
            cleanup_old_records(target_mb=settings["max_storage_mb"] * 0.8)
        
        # 2. 检查内存使用
        mem_percent = psutil.virtual_memory().percent
        if mem_percent > settings["memory_threshold_percent"]:
            logger.warning(f"Memory usage {mem_percent}% > threshold {settings['memory_threshold_percent']}%, emergency cleanup")
            cleanup_oldest_records(ratio=settings["emergency_cleanup_ratio"])
```

**图片压缩策略：**

- 快照：`quality=70`（已有）
- 抽帧：`quality=60`（已有）
- 可配置：通过环境变量 `SNAPSHOT_QUALITY`、`FRAME_QUALITY` 调整

---

## 11. 配置格式

### 11.1 cameras.json（摄像头列表）

```json
{
  "cameras": [
    {
      "camera_id": "cam_01",
      "source": "0",
      "source_type": "auto",
      "name": "车间入口",
      "enabled": true,
      "width": 640,
      "height": 480,
      "video_loop": true,
      "video_playback_speed": 1.0,
      "detection_types": {
        "fire": {"enabled": false, "interval": 1, "threshold": 0.6, "consecutive_required": 2},
        "smoke": {"enabled": false, "interval": 1, "threshold": 0.55, "consecutive_required": 2},
        "uniform": {"enabled": true, "interval": 1, "threshold": 0.5, "compliance_window_seconds": 30},
        "mask": {"enabled": true, "interval": 1, "threshold": 0.5, "consecutive_required": 1},
        "cigarette": {"enabled": false, "interval": 1, "threshold": 0.5, "consecutive_required": 1},
        "sleep": {"enabled": false, "interval": 60, "threshold": 0.7, "consecutive_required": 3}
      }
    }
  ]
}
```

**字段说明：**

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `camera_id` | string | - | 摄像头唯一标识 |
| `source` | string | - | 视频源：摄像头索引（如 `"0"`）、RTSP 地址或本地视频文件路径 |
| `source_type` | string | `"auto"` | 源类型：`"camera"`、`"rtsp"`、`"video"` 或 `"auto"`（自动识别） |
| `name` | string | `""` | 摄像头显示名称 |
| `enabled` | bool | `true` | 是否启用该摄像头 |
| `width` / `height` | int | 640 / 480 | 视频流分辨率 |
| `video_loop` | bool | `true` | 视频文件输入时是否循环播放 |
| `video_playback_speed` | float | `1.0` | 视频文件播放倍速，范围 `0.5` ~ `2.0` |
| `detection_types` | object | - | 各检测类型的开关及参数 |

### 11.2 global.json（全局配置）

全局参数单独存放，便于统一管理和热更新：

```json
{
  "detection_interval": 1.0,
  "sleep_interval": 60.0,
  "vlm_max_concurrent": 3,
  "detection_resolution": [640, 480],
  "p0_alert_cooldown": 10,
  "p1_alert_cooldown": 3,
  "max_records": 100,
  "max_storage_mb": 500,
  "memory_threshold_percent": 80,
  "emergency_cleanup_ratio": 0.2,
  "snapshot_quality": 70,
  "frame_quality": 60,
  "vlm_inspection_interval": 30
}
```

### 11.3 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DETECTION_INTERVAL` | 1.0 | 通用检测间隔（秒） |
| `SLEEP_INTERVAL` | 60.0 | 睡岗检测间隔（秒） |
| `VLM_MAX_CONCURRENT` | 3 | VLM 最大并发数 |
| `DETECTION_RESOLUTION` | 640,480 | 检测用帧分辨率 |
| `MAX_RECORDS` | 100 | 最大保留告警记录数 |
| `MAX_STORAGE_MB` | 500 | 最大存储空间（MB） |
| `MEMORY_THRESHOLD_PERCENT` | 80 | 内存紧急回收阈值（%） |
| `SNAPSHOT_QUALITY` | 70 | 快照图片质量（1-100） |
| `FRAME_QUALITY` | 60 | 抽帧图片质量（1-100） |
| `VLM_INSPECTION_INTERVAL` | 30 | VLM 全类型巡检间隔（秒），0 表示关闭 |

---

## 12. API 端点扩展

| 方法 | 端点 | 功能 |
|---|---|---|
| `GET` | `/cameras` | 增加 `detection_types` 配置字段 |
| `POST` | `/cameras/{id}/config` | 动态修改单路摄像头检测类型配置 |
| `POST` | `/cameras/batch-config` | 批量修改多路摄像头配置 |
| `GET` | `/alerts` | 获取告警记录（支持 `?level=P0&P1`、`?type=fire` 过滤） |
| `GET` | `/alerts/stats` | 今日告警统计（按类型/级别聚合） |
| `POST` | `/alerts/{id}/ignore` | 标记告警为误报 |
| `GET` | `/detector/models` | 获取已加载模型列表及状态 |
| `GET` | `/settings` | 获取全局配置（读取 global.json） |
| `POST` | `/settings` | 修改全局配置，实时生效并持久化到 global.json |
| `POST` | `/cameras/{id}/source` | 切换视频源（摄像头索引 / RTSP / 本地视频文件路径） |
| `POST` | `/cameras/{id}/playback/control` | 视频播放控制：播放/暂停/seek/倍速/循环开关 |
| `GET` | `/cameras/{id}/playback/status` | 获取视频播放状态（当前帧/总帧/进度/是否循环） |

---

## 13. 性能策略

### 13.1 测试计划（待执行）

当前处于设计阶段，以下为预期测试项：

- 3 路摄像头全类型开启，验证 RK3588 NPU 负载和帧率稳定性
- 调整 `DETECTION_INTERVAL`（0.5s ~ 2s），找到延迟和准确率的平衡点
- 50 路模拟压力测试（可用视频文件循环播放模拟）

### 13.2 50 路扩展预留

| 优化手段 | 效果 |
|---|---|
| 模型懒加载 | 未启用的类型不加载模型 |
| 模型全局共享 | 50 路共用一套模型实例 |
| 检测分辨率可调 | 320×240 比 640×480 快 2-3 倍 |
| 每路只开 1-2 个类型 | 减少单帧推理负载 |
| NPU 批量推理 | RKNN 一次 inference 送多帧 |
| fire/smoke 合并 | 如果模型同时输出，只需一次推理 |
| VLM 并发控制 | `max_concurrent` 根据网络带宽调整 |

### 13.3 睡岗资源消耗

- 60 秒间隔对总负载几乎无影响
- pose 模型只在该路启用睡岗时加载
- VLM 判断只需单帧，不抽 15 帧

---

## 14. 前端概要（后续详细设计）

后端功能实现后，前端将扩展：

- **配置面板**：每路摄像头检测类型开关、间隔设置、连续判定次数
- **批量配置**：选择多个摄像头，统一设置检测类型/间隔/阈值/连续次数
- **视频源选择器**：手动选择本地视频文件作为输入源，支持播放/暂停/进度拖动/倍速/循环
- **全局设置**：存储限制（记录数/空间/MB）、图片质量、内存回收阈值
- **实时告警条**：顶部 P0 红色浮动告警
- **告警列表**：侧边栏按级别分色展示
- **视频流标注**：MJPEG 画面上不同检测类型画不同颜色框
- **统计面板**：今日各类型告警数量

---

## 15. 需要改动的文件清单

### 15.1 新增文件

| 文件 | 说明 |
|---|---|
| `backend/safety_detector.py` | **核心**：SafetyDetector 类，加载所有小模型并执行推理 |
| `backend/vlm_queue.py` | VLMQueue 类，异步队列 + 信号量控并发 |
| `backend/vlm_inspector.py` | VLMInspector 类，每 30 秒全类型综合巡检 |

### 15.2 修改文件

| 文件 | 改动范围 |
|---|---|
| `main_multi.py` | ① `init_components()` 中初始化 `SafetyDetector`、`VLMQueue`、`VLMInspector`、存储清理线程；② `startup` 事件启动检测循环线程和巡检线程；③ 新增 `/cameras/{id}/config`、`/cameras/batch-config`、`/alerts` 等 API 端点；④ `shutdown` 释放模型资源 |
| `understander.py` | ① 新增各检测类型的专用 prompt（fire/smoke/uniform/mask/cigarette/sleep）；② 新增全类型综合巡检 prompt 和多图分析接口 `analyze_multi()`；③ `VideoUnderstander` 支持按类型选择 prompt |
| `config.py` | ① 新增 `DETECTION_THRESHOLDS`、`SLEEP_INTERVAL`、`VLM_MAX_CONCURRENT`、`VLM_INSPECTION_INTERVAL`、`DETECTION_RESOLUTION`、`MAX_STORAGE_MB`、`MEMORY_THRESHOLD_PERCENT`、`SNAPSHOT_QUALITY`、`FRAME_QUALITY`；② 支持从 `cameras.json` 读取 `detection_types` 配置（含 `consecutive_required`/`compliance_window_seconds`）；③ 支持从 `global.json` 读取全局参数，运行时修改并持久化 |
| `camera_manager.py` | `CameraConfig` 增加 `source_type`、`video_loop`、`video_playback_speed` 字段；支持视频源切换、播放控制 |
| `multi_detector.py` | 重构检测循环：引入 `SafetyDetector`，按类型独立调度，移除原有的 person-only 逻辑 |
| `performance_storage.py` | ① 记录格式扩展为统一告警记录（兼容 P0/P1）；② 增加按类型/级别查询接口；③ 新增存储清理函数 `cleanup_old_records`、`cleanup_oldest_records`、`get_storage_size_mb` |
| `cameras.json` | 配置格式升级，只保留摄像头列表和 `detection_types` |
| `global.json` | **新增**：全局参数配置文件，独立存放所有可手动设置的阈值/限制 |
| `.env.default` | 新增 `DETECTION_INTERVAL`、`SLEEP_INTERVAL`、`VLM_MAX_CONCURRENT`、`VLM_INSPECTION_INTERVAL` 环境变量 |

### 15.3 移除/废弃

- `backend/detector.py`：原有单路人员检测器，功能被 `SafetyDetector` 覆盖
- `backend/rk3588_detector.py`：保留 `HybridDetector` 作为底层推理工具，`SafetyDetector` 内部调用
- `backend/extractor.py`：抽帧逻辑整合进 `MultiDetector`

---

## 16. 风险与缓解

| 风险 | 缓解措施 |
|---|---|
| DamoYOLO 在 aarch64 上兼容性未知 | 测试阶段优先验证，不通过则改用 YOLO 替代模型 |
| 多模型同时加载内存不足 | 懒加载 + 按摄像头启用类型控制 |
| 50 路串行检测轮询周期过长 | 降低每路类型数 + NPU batch + 降分辨率 |
| VLM 网络不稳定导致 P1 告警延迟 | 设置 VLM 超时，超时则按小模型结果告警 |
