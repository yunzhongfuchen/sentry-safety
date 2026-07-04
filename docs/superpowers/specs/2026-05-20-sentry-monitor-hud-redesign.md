# Sentry 监控中心 HUD 风格重设计

## 背景与目标

为 `frontend/safety_detection/` 新增 `hud.html` 页面，打造**科幻终端 HUD 风格**的监控中心，要求：
- 高级感：克制、精致、不花哨
- 未来感：全息投影式界面、终端数据流、HUD 装饰元素
- 沉浸感：视频是绝对视觉中心，数据环绕视频呈现

## 改动范围

- **新增** `frontend/safety_detection/hud.html`：全新的 HUD 风格监控页面
- **保留** `frontend/safety_detection/multi.html`：原页面不动，作为备选入口
- **后端** `backend/main_multi.py`：新增 `/hud` 路由，指向 `hud.html`
- `records.html` 和 `settings.html` 保持现有样式不变

---

## 1. 设计系统

### 1.1 色彩

```css
:root {
  --bg: #030308;
  --surface: rgba(10, 10, 18, 0.6);
  --surface-solid: #0A0A12;
  --border: #1A1A2E;
  --border-glow: #00E5FF;

  --text-primary: #E8E8EC;
  --text-secondary: #7A7A8A;
  --text-muted: #4A4A5A;

  --accent: #00E5FF;
  --accent-dim: rgba(0, 229, 255, 0.15);
  --warning: #FFB347;
  --danger: #FF4D6A;
  --success: #00FF9D;
}
```

- 背景不用纯黑，使用 `#030308`（极暗蓝黑），避免 OLED 死黑感
- 主 accent 为冰蓝 `#00E5FF`，在暗背景上自带发光感
- 状态色：成功（在线）用青绿，警告用琥珀，告警用红

### 1.2 字体

```css
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&family=Noto+Sans+SC:wght@300;400;500;700&display=swap');
```

| 用途 | 字体 | 字重 | 备注 |
|------|------|------|------|
| 英文标题 / 大数字 | Rajdhani | 600-700 | 窄长科技感 |
| 英文数据 / 终端日志 | JetBrains Mono | 400-500 | 等宽精确 |
| 中文 / 正文 | Noto Sans SC | 400-500 | 清晰可读 |

### 1.3 间距与圆角

- 全局无圆角（`border-radius: 0`），所有面板和按钮使用直角，强化终端感
- 间距系统：4px 基线（4 / 8 / 12 / 16 / 24 / 32）
- 面板内边距：16px

---

## 2. 布局架构

全屏 `100vh`，无页面滚动，CSS Grid 三行三列：

```
┌─────────────────────────────────────────────────────────────┐
│  顶部状态栏 (48px)                                            │
├──────────┬──────────────────────────────┬───────────────────┤
│          │                              │                   │
│ 左侧终端  │      中央主视频区             │   右侧仪表盘       │
│ 面板      │      (带 HUD 边框)            │   (环形/弧形)      │
│ (260px)  │                              │   (260px)         │
│          │                              │                   │
├──────────┴──────────────────────────────┴───────────────────┤
│  底部摄像头状态条 (64px)                                       │
└─────────────────────────────────────────────────────────────┘
```

```css
#app {
  height: 100vh;
  display: grid;
  grid-template-rows: 48px 1fr 64px;
  grid-template-columns: 260px 1fr 260px;
  grid-template-areas:
    "header header header"
    "left main right"
    "footer footer footer";
  gap: 1px;
  background: var(--bg);
}
```

---

## 3. 各区域详细设计

### 3.1 顶部状态栏（grid-area: header）

- 高度 48px，背景 `var(--surface-solid)`
- 底边框 1px `var(--border)`，带扫描光效动画（5s 循环）
- **左侧**：Logo + "SENTRY"（Rajdhani 600，14px，字间距 0.15em）
- **中间**：UTC 时间 + 系统运行时长（JetBrains Mono，12px，`var(--text-secondary)`）
- **右侧**：在线摄像头数 + 今日告警数（数字 Rajdhani 700，20px，在线数用 `var(--success)`，告警数用 `var(--danger)`）

**扫描光效动画**：
```css
.header::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0;
  width: 120px; height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
  animation: scanline 5s linear infinite;
}
@keyframes scanline {
  0% { left: -120px; }
  100% { left: 100%; }
}
```

### 3.2 中央主视频区（grid-area: main）

- 背景 `#000`，内部视频 `object-fit: contain`
- 四边 HUD 角括号装饰（CSS 伪元素，1px `var(--accent)` 线，24px 长）：
```css
.video-hud::before,
.video-hud::after {
  content: '';
  position: absolute;
  width: 24px; height: 24px;
  border: 1px solid var(--accent);
}
/* 四角定位省略，实际用四个独立伪元素或 span */
```
- 扫描线覆盖层（全区域，`repeating-linear-gradient`，opacity 0.03，不阻挡交互）
- 检测到异常时：边框从暗色渐变为琥珀/红色，`box-shadow` 脉冲动画（2s loop）
- 右下角叠加信息：帧率 + 分辨率（JetBrains Mono 11px，`var(--text-secondary)`）
- 视频文件播放控件：悬浮于视频左下角，半透明黑底 + 青色的播放/暂停/循环按钮

### 3.3 左侧终端面板（grid-area: left）

**默认折叠状态**：宽度 48px，只显示垂直排列的图标/缩写
- 鼠标悬停展开到 260px，过渡 300ms ease

**展开后内容**：
- 标题：`// ALERT_LOG`（JetBrains Mono 10px，`var(--text-muted)`）
- 报警日志列表（最多显示 8 条）：
  - 格式：`[14:32:07] CAM_01 > FIRE (P0)`
  - 字体 JetBrains Mono 11px
  - P0 用 `var(--danger)`，P1 用 `var(--warning)`
  - 新日志从底部滑入（`translateY(8px) → 0` + opacity，200ms）
- 分割线下方：摄像头列表（只显示名称和状态圆点）
  - 在线：青绿色圆点 + 微弱脉冲
  - 离线：灰色圆点
  - 点击摄像头名称可将该摄像头设为主画面

### 3.4 右侧仪表盘面板（grid-area: right）

三个核心指标，使用 **SVG 弧形进度条**（半圆或 3/4 圆）：

1. **GPU 负载**：0-100%，青色渐变
2. **检测 FPS**：实时帧率，琥珀色
3. **内存占用**：百分比，<70% 青，70-85% 琥珀，>85% 红

每个仪表盘：
- SVG 描边宽度 3px，底色 `var(--border)`，进度色随值变化
- 中心显示数值：Rajdhani 700，28px
- 下方标签：JetBrains Mono 10px，`var(--text-muted)`

**面板底部**：模型状态列表
- 小模型 / VLM 复核 / GPU 调度器
- 每项左侧圆点指示器：在线 `var(--success)`，离线 `var(--text-muted)`
- 字体 12px，Noto Sans SC

**画框开关组**（从底部栏移过来）：
- 6 个检测类型切换按钮（明火/烟雾/工服/口罩/吸烟/睡岗）
- 样式：1px 边框按钮，激活时填充对应颜色 + 文字变深
- 排列：2 列 3 行网格

### 3.5 底部摄像头状态条（grid-area: footer）

**重要调整：子摄像头不再显示实时画面，仅显示状态信息。**

- 高度 64px，背景 `var(--surface-solid)`
- 横向排列所有摄像头（超出时横向滚动）
- 每个摄像头卡片：
  - 宽度 160px，高度 48px，1px 边框
  - 主画面摄像头边框高亮（`var(--accent)` + 微弱外发光 `box-shadow: 0 0 8px var(--accent-dim)`）
  - **无视频预览**，只显示：
    - 摄像头名称（Noto Sans SC 12px）
    - 状态圆点（在线/离线）
    - 激活的检测类型标签（小标签，9px，对应检测类型颜色）
  - Hover：扫描线从上到下扫过（CSS mask + translateY 动画）
  - Click：切换主画面，150ms crossfade

---

## 4. 动效设计

| 动效 | 实现 | 参数 |
|------|------|------|
| 顶部扫描线 | CSS animation | 5s linear infinite |
| 视频扫描线覆盖 | `repeating-linear-gradient` + 微妙 translateY | opacity 0.03，持续循环 |
| 告警边框脉冲 | `box-shadow` 颜色变化 + scale | 2s ease-in-out infinite |
| 日志滑入 | transform + opacity | 200ms ease-out |
| 面板展开 | width transition | 300ms ease，内容 stagger 30ms |
| 摄像头切换 | crossfade | opacity 150ms |
| 底部卡片扫描 | mask-image + translateY | 1.5s ease-in-out |
| 状态圆点脉冲 | opacity + scale | 2s ease-in-out infinite（仅在线摄像头） |

**全部支持 `prefers-reduced-motion: reduce`**。

---

## 5. 功能变更对照

| 功能 | 当前 multi.html / hud.html | 新设计 |
|------|-----------------|--------|
| 主视频流 | 中央显示 | 中央显示，增加 HUD 边框和扫描线 |
| 子摄像头画面 | 右侧 6 宫格实时视频 | **移除**，改为底部状态条（无画面） |
| 最近报警 | 左侧 220px 固定面板 | 左侧终端面板（可折叠），终端日志风格 |
| 底部统计栏 | 在线/检测中/总告警/运行时间 | 移到顶部状态栏 + 右侧仪表盘 |
| 画框开关 | 底部栏右侧 | 移到右侧仪表盘面板 |
| 播放控制 | 底部栏（仅视频文件） | 视频区左下角悬浮控件 |
| 系统指标 | 无 | 新增 GPU/FPS/内存 仪表盘 |
| 模型状态 | 无 | 新增右侧模型状态列表 |
| 摄像头切换 | 点击右侧子画面 | 点击底部状态条或左侧列表 |

---

## 6. 技术实现要点

1. **Vue 3 保留**：继续使用 Vue 3（CDN 引入），不引入构建工具
2. **纯 CSS 实现**：所有动效、HUD 装饰、扫描线、仪表盘均用 CSS/SVG 实现，不引入 Canvas/WebGL
3. **字体加载**：Google Fonts CDN，增加 `font-display: swap`
4. **性能**：
   - 扫描线使用 `pointer-events: none`
   - 动画元素使用 `will-change: transform, opacity`
   - 子摄像头不加载视频流，大幅减少带宽和 CPU
5. **响应式**：最小支持 1280×720，低于此分辨率时左侧/右侧面板自动折叠
6. **无障碍**：所有交互元素有 `aria-label`，支持键盘导航和 `prefers-reduced-motion`

---

## 7. 文件变更

| 文件 | 操作 |
|------|------|
| `frontend/safety_detection/hud.html` | **新增** HUD 风格监控页面 |
| `frontend/safety_detection/multi.html` | **保留** 不变，作为原入口 |
| `backend/main_multi.py` | **新增** `/hud` 路由 |
| `frontend/safety_detection/style.css` | 不影响 |
| `frontend/safety_detection/app.js` | 不影响 |
