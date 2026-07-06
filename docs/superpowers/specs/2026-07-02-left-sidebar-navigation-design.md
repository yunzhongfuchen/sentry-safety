# 左侧多级导航改造设计

**日期**：2026-07-02  
**状态**：已确认，待实现  
**范围**：将现有页面右上角标签式导航改为左侧边栏多级导航

## 背景

当前前端为多页面结构（`monitor.html` / `records.html` / `settings.html`），顶部右侧通过简单标签按钮在页面间跳转。随着设置页承载三个 tab（摄像头 / 检测配置 / 系统设置），当前导航层级已经不够表达信息结构：
- 监控、记录是独立一级页面
- 设置是一级分组，其下还有三个功能子页（实际上是 `settings.html` 内部 tab）

因此需要把全局导航改为**左侧边栏 + 多级结构**，使信息架构与实际页面结构一致。

## 目标

- 以左侧边栏替换右上角标签页按钮。
- 将 **监控**、**记录** 作为一级导航项。
- 将 **设置** 作为一级父项，其下固定显示三个二级导航项：
  - 摄像头
  - 检测配置
  - 系统设置
- 点击一级 **设置** 仅负责**展开/收起子项，不跳转**。
- 点击二级子项跳转到 `settings.html`，并切换到对应 tab。
- 保持现有 glass-clay / Cool Slate 视觉体系，不引入新框架，不改单页架构。

## 非目标

- 不改页面业务逻辑。
- 不改 `settings.html` 现有 tab 内容与结构，只改它的入口方式。
- 不把多页面改造成 SPA。
- 不引入构建工具、组件框架或路由库。

## 信息架构

### 一级导航
- 监控 → `monitor.html`
- 记录 → `records.html`
- 设置 → 展开/收起二级子项，不直接跳转

### 二级导航（设置）
- 摄像头 → `settings.html?tab=cameras`
- 检测配置 → `settings.html?tab=detection`
- 系统设置 → `settings.html?tab=system`

## 交互设计

### 1. 导航布局
- 页面整体改为 **左侧边栏 + 右侧内容区**。
- 左侧边栏在 `monitor.html`、`records.html`、`settings.html` 三页中保持统一位置、宽度与样式。
- 右上角现有页面切换标签删除。

### 2. 一级项行为
- **监控 / 记录**：点击即跳转页面。
- **设置**：点击切换展开/收起状态；本身不跳转。
- 设置子项默认**可展开/收起**，不是始终展开。

### 3. 二级项行为
- 二级项点击进入 `settings.html`，并通过 query 参数激活对应 tab。
- 在 `settings.html` 内，如果当前 tab 属于某个子项，则：
  - 左侧“设置”父项保持展开
  - 当前子项高亮

### 4. 高亮规则
- 当前页面是 `monitor.html` → “监控”高亮。
- 当前页面是 `records.html` → “记录”高亮。
- 当前页面是 `settings.html?tab=...` → “设置”父项高亮/展开，且对应子项高亮。
- 若进入 `settings.html` 没有 `tab` 参数，默认视为 `cameras`。

## 技术方案

采用**公共逻辑 + 各页复用**方案，而不是引入真正的组件系统：

### 1. CSS
在共享样式中新增边栏布局与多级导航样式：
- 页面外层 shell（sidebar + content）
- 一级项样式
- 二级项缩进与选中态
- 设置父项展开/收起态
- 响应式处理（至少保证当前桌面布局可用）

### 2. JS
在 `shared.js` 中新增轻量导航辅助逻辑，例如：
- 识别当前页面和 query 参数
- 生成/控制“设置”展开状态
- 统一高亮当前一级/二级导航项
- 提供 `settings.html` tab 与 URL query 的同步

### 3. HTML 页面改动
分别调整：
- `monitor.html`
- `records.html`
- `settings.html`

改动内容：
- 删除原顶部右侧标签导航
- 新增左侧导航 DOM
- 把原内容区包进统一 content 容器

## 页面级影响

### monitor.html
- 从“顶部导航 + 主内容”改为“左侧导航 + 主内容”。
- 原监控大屏核心布局（视频区 / 侧栏 / 告警区）保留，仅放入新的内容区。

### records.html
- 左侧导航固定存在。
- 记录筛选、统计、表格区域保持原逻辑与顺序。

### settings.html
- 原页面内部三 tab 保留。
- 左侧新增全局导航；页面内部 tab 仍作为内容区的局部切换控件存在。
- 从二级导航进入时，根据 `?tab=` 选中对应 tab。

## 方案权衡

### 方案 A（推荐）：左侧全局导航 + settings 保留内部 tab
**优点**：
- 改动最小，复用现有 `settings.html` 结构
- 信息层级清晰
- 不需要拆成三个独立设置页面

**缺点**：
- “设置”在左侧是二级入口，进入后内容区里仍有 tab，存在轻微重复感

### 方案 B：把 settings 的三个 tab 彻底拆成三个独立 HTML 页面
**优点**：
- 导航层级最纯粹
- URL 与页面结构完全一致

**缺点**：
- 改动面大，重复代码多，不符合本次最小改动目标

### 方案 C：左侧导航仅作为视觉壳，settings 父项点击直接跳转默认 tab
**优点**：
- 交互简单

**缺点**：
- 与用户明确要求冲突（用户要求“设置”父项只展开/收起，不跳转）

**结论**：采用 **方案 A**。

## 错误处理与边界情况

- `settings.html` 上若 `tab` 参数非法，回退到 `cameras`。
- 从非 settings 页面点击设置子项时，直接带 query 跳转到目标 tab。
- 若用户在 `settings.html` 内切换本地 tab，URL 也应同步更新，避免刷新后丢失当前子项状态。

## 验证标准

1. `monitor.html` / `records.html` / `settings.html` 三页都显示左侧导航。
2. 右上角旧标签导航已移除。
3. 点击“设置”仅展开/收起，不跳转。
4. 点击设置二级项可进入 `settings.html` 且切到正确 tab。
5. `settings.html` 刷新后仍保留当前 tab（依赖 URL query）。
6. 当前页面/当前设置子项高亮正确。
7. 不影响监控、记录、设置页现有业务功能。

## 影响文件（预估）

- `frontend/safety_detection/monitor.html`
- `frontend/safety_detection/records.html`
- `frontend/safety_detection/settings.html`
- `frontend/safety_detection/shared.js`
- `frontend/safety_detection/styles/glass-clay.css`
