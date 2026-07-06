# Sentry Glass-Clay UI Redesign

> 生成时间：2026-06-30  
> 分支：`ui/hud-redesign-2026-06-29`  
> 范围：重新设计前端监控 / 记录 / 设置三个页面，统一为 Glassmorphism + Claymorphism 风格

---

## 1. 背景与目标

当前项目 `sentry-safety` 存在两个监控入口：

- `/multi` → `frontend/safety_detection/multi.html`（传统多摄像头控制台）
- `/hud` → `frontend/safety_detection/hud.html`（深色科幻 HUD 监控中心）

本次 redesign 目标：

1. **统一监控入口**：新增 `/monitor` 替代 `/multi` 和 `/hud`
2. **统一视觉风格**：三个前端页面采用 Glassmorphism + Claymorphism（毛玻璃 + 粘土）风格
3. **保留现有功能**：records 和 settings 保留现有业务能力，只换视觉与布局
4. **删除旧资产**：直接删除 `multi.html`、`hud.html` 及对应后端路由

---

## 2. 范围

### 2.1 页面范围

| 页面 | 文件 | 说明 |
|------|------|------|
| 监控页 | `frontend/safety_detection/monitor.html` | 新增，唯一监控入口 |
| 记录页 | `frontend/safety_detection/records.html` | 重写，glass-clay 风格 |
| 设置页 | `frontend/safety_detection/settings.html` | 重写，合并为 3 个 Tab |

### 2.2 公共样式

| 文件 | 说明 |
|------|------|
| `frontend/safety_detection/styles/glass-clay.css` | 设计 token + 公共组件类 |
| `frontend/safety_detection/shared.js` | 已有工具函数，继续复用并适当扩展 |

### 2.3 删除文件

| 文件 | 说明 |
|------|------|
| `frontend/safety_detection/multi.html` | 被 `monitor.html` 替代 |
| `frontend/safety_detection/hud.html` | 被 `monitor.html` 替代 |

### 2.4 后端路由

- **新增**：`GET /monitor` → 返回 `monitor.html`
- **删除**：`GET /multi`、`GET /hud`
- **保留**：`GET /records.html`、`GET /settings.html`（仅更新文件内容）
- **复用现有 API**：`/cameras`、`/status`、`/records`、`/alerts/*`、`/settings`、`/overlay`、`/system/*` 等

---

## 3. 设计系统

### 3.1 设计理念

**"工业实验室精确感 + 温暖粘土材质"**

避免模板化的奶油色 + 亮琥珀默认风格，采用克制的冷灰白背景与焦橙主色，让监控系统既专业又不过于冰冷。

### 3.2 颜色系统

```css
:root {
    --bg-base: #f0f2ee;           /* 冷灰白，实验室工作台感 */
    --bg-soft: #f7f8f6;           /* 抬升表面 */
    --bg-elevated: #ffffff;

    --glass-bg: rgba(255, 255, 255, 0.78);
    --glass-border: rgba(255, 255, 255, 0.9);
    --glass-edge: rgba(15, 23, 42, 0.06);
    --glass-shadow:
        0 2px 4px rgba(15, 23, 42, 0.04),
        0 12px 24px rgba(15, 23, 42, 0.06),
        inset 0 1px 0 rgba(255, 255, 255, 0.95);

    --clay-shadow:
        4px 4px 10px rgba(181, 189, 177, 0.5),
        -4px -4px 10px rgba(255, 255, 255, 0.85);
    --clay-shadow-inset:
        inset 3px 3px 6px rgba(181, 189, 177, 0.45),
        inset -3px -3px 6px rgba(255, 255, 255, 0.9);
    --clay-shadow-pressed:
        inset 4px 4px 8px rgba(181, 189, 177, 0.5),
        inset -4px -4px 8px rgba(255, 255, 255, 0.85);

    --text-primary: #1a211c;
    --text-secondary: #5f6b63;
    --text-muted: #949f96;

    --accent: #e05a18;            /* 焦橙色 */
    --accent-soft: rgba(224, 90, 24, 0.10);
    --accent-hover: #c44c12;

    --success: #0d9f6e;
    --warning: #e05a18;
    --danger: #dc2626;
    --info: #2563eb;
}
```

### 3.3 字体

- 标题/显示：`Space Grotesk`
- 正文：`Noto Sans SC`
- 数据/时间/代码：`JetBrains Mono`

### 3.4 圆角

- 大卡片：`18px`
- 按钮/输入框：`10px`
- 标签/徽章：`6px`

### 3.5 公共组件类

| 类名 | 用途 |
|------|------|
| `.glass-card` | 毛玻璃卡片 |
| `.clay-button` | 粘土按钮（凸起 + 按压效果） |
| `.clay-button.primary` | 焦橙主按钮 |
| `.clay-input` / `.clay-select` | 粘土输入框/下拉框 |
| `.status-dot` | 在线/离线/告警状态点 |
| `.status-dot.alert` | 告警状态点带柔和脉冲 |
| `.type-badge` | 检测类型标签 |
| `.type-badge.fire` / `.smoke` / `.uniform` / `.mask` / `.cigarette` / `.sleep` | 各类型配色 |
| `.nav-header` | 三页统一顶部导航 |

### 3.6 动效

| 动效 | 实现 |
|------|------|
| 卡片 hover | `translateY(-2px)` + 阴影加深，250ms ease |
| 按钮按下 | `box-shadow` 切换为 inset，150ms ease |
| 告警状态点 | `scale` + `opacity` 柔和脉冲，2s infinite |
| 页面切换/加载 | 淡入 200ms |
| 减弱动画 | 全部支持 `prefers-reduced-motion: reduce` |

---

## 4. 页面设计

### 4.1 监控页（Monitor）

#### 布局

```
┌─────────────────────────────────────────────────────────┐
│  SENTRY          在线 3 / 5    告警 12    14:32:08 UTC  │  ← 顶部状态栏
├────────────────────────────────────┬────────────────────┤
│                                    │  [● Camera 1]      │
│                                    │  [○ Camera 2]      │
│         主画面 MJPEG               │  [● Camera 3]      │  ← 右侧摄像头列表
│                                    │  [○ Camera 4]      │
│                                    │  [● Camera 5]      │
├────────────────────────────────────┴────────────────────┤
│  [14:32:01] Camera 1 · 明火 (P0)                        │  ← 底部告警条
│  [14:31:45] Camera 3 · 睡岗 (P1)                        │
└─────────────────────────────────────────────────────────┘
```

#### 顶部状态栏

- 左侧：SENTRY logo + 标题
- 中间：在线摄像头数 / 总数、今日告警总数、UTC 时间
- 右侧：页面导航（监控 / 记录 / 设置）

#### 中央主画面

- 只拉取当前选中摄像头的 `/cameras/{id}/stream`
- 显示摄像头名称 + 状态点
- 当前告警类型标签（如果有）
- 视频文件源时显示播放控制（复用 `/cameras/{id}/playback/control`）

#### 右侧摄像头列表

- 只显示：名称 + 在线/离线状态点
- 不推流，节约带宽
- 点击切换主画面

#### 底部告警条

- 最近 6 条告警
- 每条显示：时间、摄像头、检测类型、级别（P0/P1）
- 告警类型使用 `.type-badge` 样式

#### 行为

- URL 同步：`/monitor?camera=xxx`
- 默认选中第一个在线摄像头
- 切换摄像头时只拉取新选中摄像头的流

### 4.2 记录页（Records）

#### 布局

```
┌─────────────────────────────────────────────────────────┐
│  SENTRY  ·  检测记录                                      │
├─────────────────────────────────────────────────────────┤
│  [今日告警 12]  [待确认 3]  [已确认 5]  [误报 4]         │  ← 统计概览
├─────────────────────────────────────────────────────────┤
│  筛选：日期 [____]  摄像头 [全部 ▼]  类型 [全部 ▼]       │
├─────────────────────────────────────────────────────────┤
│  时间          摄像头      类型      级别    置信度      │
│  14:32:01      Camera 1   明火      P0      96%         │
│  14:31:45      Camera 3   睡岗      P1      82%         │
└─────────────────────────────────────────────────────────┘
```

#### 统计概览

- 今日告警总数
- 待确认、已确认、误报
- 各检测类型分布（火情/睡岗/吸烟/其他）

#### 筛选器

- 日期范围
- 摄像头（来自 `/cameras`）
- 检测类型（fire/smoke/uniform/mask/cigarette/sleep）
- 告警级别（小模型/大模型/大模型忽略）
- 状态（待确认/已确认/误报）
- 每页条数

#### 记录表格

- 列：时间、摄像头、类型、级别、状态、置信度、说明
- 点击行打开详情弹窗

#### 记录详情弹窗

- 基本信息 + 检测详情
- 触发快照
- 视频帧序列播放器
- 操作：确认报警 / 确认误报

#### 分页

- 上一页 / 下一页 / 页码信息

### 4.3 设置页（Settings）

统一为 **3 个 Tab**。

#### Tab 1：摄像头

- 摄像头列表（ID、名称、源地址、类型、分辨率、启用状态）
- 添加 / 编辑 / 删除摄像头
- 批量配置检测类型
- 单摄像头检测类型配置（启用、间隔、阈值、连续、冷却、VLM）

#### Tab 2：检测配置

- 检测类型默认值卡片（6 种类型各自的启用/间隔/阈值/连续/冷却/VLM）
- VLM 最大并发
- VLM 巡检间隔

#### Tab 3：系统设置

- 当前检测设备 / 运行模式
- 已加载模型列表
- GPU 动态调度器（开关、队列数、调度周期、FP16）
- 最大记录数、存储上限、内存阈值、清理比例
- 快照质量、帧质量
- 重启检测服务
- 实时日志

---

## 5. 数据流与 API

### 5.1 复用 API 列表

| 接口 | 用途 |
|------|------|
| `GET /cameras` | 摄像头列表 |
| `GET /cameras/{id}/stream` | 单路视频流 |
| `GET /cameras/{id}/playback/status` | 视频文件播放状态 |
| `POST /cameras/{id}/playback/control` | 视频文件播放控制 |
| `GET /status` | 系统状态、最近告警、运行时长 |
| `GET /overlay` / `POST /overlay` | 画框类型配置 |
| `GET /records` | 历史记录（分页/筛选） |
| `GET /records/summary` | 记录统计 |
| `GET /record/{id}` | 记录详情 |
| `GET /record/{id}/snapshot` | 快照 |
| `GET /record/{id}/frames` | 视频帧 |
| `POST /alerts/{id}/confirm` | 确认报警 |
| `POST /alerts/{id}/ignore` | 确认误报 |
| `GET /alerts/stats` | 告警统计 |
| `GET /settings` / `POST /settings` | 系统设置 |
| `POST /cameras/add` | 添加摄像头 |
| `POST /cameras/{id}/config` | 更新摄像头配置 |
| `DELETE /cameras/{id}` | 删除摄像头 |
| `POST /cameras/batch-config` | 批量配置 |
| `POST /cameras/{id}/reset-config` | 重置为默认值 |
| `POST /upload/video` | 上传本地视频 |
| `GET /detector/models` | 已加载模型 |
| `GET /detector/status` | 检测器状态 |
| `GET /system/mode` | 当前运行设备 |
| `POST /system/restart` | 重启检测服务 |

### 5.2 数据刷新策略

- Monitor 页：摄像头与状态每 2 秒轮询，UTC 时间每秒更新
- Records 页：手动刷新 + 切换筛选/分页时加载
- Settings 页：进入时加载，保存后局部刷新

---

## 6. 技术约束

1. **Vue 3 CDN**：继续沿用现有模式，不引入构建工具
2. **单文件 HTML**：每个页面为独立 HTML 文件，内联 CSS + Vue 逻辑
3. **公共 CSS**：`styles/glass-clay.css` 统一设计 token 和组件类
4. **公共 JS**：`shared.js` 复用 safeFetch、formatTime、debounce 等工具
5. **删除旧文件**：实现完成后删除 `multi.html` 和 `hud.html`
6. **后端改动**：仅新增 `/monitor` 路由，删除 `/multi` 和 `/hud` 路由

---

## 7. 成功标准

- [ ] `/monitor` 正常打开，显示 glass-clay 风格监控页
- [ ] `/multi` 和 `/hud` 返回 404（已删除）
- [ ] 监控页只拉取主画面一路流，右侧列表不推流
- [ ] 点击摄像头切换主画面，URL 同步更新
- [ ] 记录页统计、筛选、表格、详情弹窗、分页功能正常
- [ ] 设置页 3 个 Tab 内容完整，保存成功
- [ ] 三页均支持 `prefers-reduced-motion`
- [ ] 旧文件 `multi.html`、`hud.html` 已从仓库移除
