# Glass-Clay Cool Slate 精修 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留 glass-clay 玻璃+黏土设计系统的前提下，把暖橙配色换成 Cool Slate 冷静蓝灰，并把拟态阴影调到"平衡"档。

**Architecture:** 设计系统集中在 `frontend/safety_detection/styles/glass-clay.css` 的 `:root` CSS 令牌，所有页面通过 `var(--...)` 继承。因此改动 = 重写一段 `:root` 令牌 + 修一处 HTML 硬编码颜色。无 JS、无布局、无结构改动。

**Tech Stack:** 原生 HTML + CSS 自定义属性（CSS variables）+ Vue 3 全局构建（不涉及）。

## Global Constraints

- 不改布局、组件 DOM 结构、字体（保留 Space Grotesk + Noto Sans SC）。
- 不改 `--bg-elevated` `--glass-bg` `--glass-border` `--radius-*` `--font-*`。
- 不动 monitor.html / records.html 中的 `#0f1210` 视频/图片近黑底。
- 主色统一为 `#2563eb`；告警红保留 `#dc2626`。
- 验证以人工目视为准（无 CSS 单元测试框架），辅以 grep 残留色排查。

---

### Task 1: 重写 glass-clay.css 的 :root 配色与阴影令牌

**Files:**
- Modify: `frontend/safety_detection/styles/glass-clay.css:3-46`（`:root` 块）

**Interfaces:**
- Consumes: 无（叶子改动）。
- Produces: 更新后的 CSS 令牌（`--bg-base` `--accent` `--clay-shadow` 等），供所有页面 `var(--...)` 继承。令牌名保持不变，只改值。

- [ ] **Step 1: 替换背景与玻璃令牌**

将 `glass-clay.css` 第 4-14 行（`--bg-base` 到 `--glass-shadow`）替换为：

```css
    --bg-base: #eef1f4;
    --bg-soft: #f4f6f9;
    --bg-elevated: #ffffff;

    --glass-bg: rgba(255, 255, 255, 0.78);
    --glass-border: rgba(255, 255, 255, 0.9);
    --glass-edge: rgba(30, 41, 59, 0.06);
    --glass-shadow:
        0 1px 2px rgba(30, 41, 59, 0.05),
        0 10px 22px rgba(30, 41, 59, 0.07),
        inset 0 1px 0 rgba(255, 255, 255, 0.95);
```

- [ ] **Step 2: 替换黏土阴影令牌**

将第 16-24 行（`--clay-shadow` / `--clay-shadow-inset` / `--clay-shadow-pressed`）替换为：

```css
    --clay-shadow:
        3px 3px 8px rgba(148, 163, 184, 0.4),
        -3px -3px 8px rgba(255, 255, 255, 0.9);
    --clay-shadow-inset:
        inset 2px 2px 5px rgba(148, 163, 184, 0.35),
        inset -2px -2px 5px rgba(255, 255, 255, 0.95);
    --clay-shadow-pressed:
        inset 3px 3px 6px rgba(148, 163, 184, 0.4),
        inset -3px -3px 6px rgba(255, 255, 255, 0.9);
```

- [ ] **Step 3: 替换文字、主色、状态色令牌**

将第 26-37 行（`--text-primary` 到 `--info`）替换为：

```css
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --text-muted: #94a3b8;

    --accent: #2563eb;
    --accent-soft: rgba(37, 99, 235, 0.10);
    --accent-hover: #1d4ed8;

    --success: #0d9488;
    --warning: #d97706;
    --danger: #dc2626;
    --info: #0284c7;
```

- [ ] **Step 4: grep 验证令牌已无暖橙残留**

Run: `grep -nE "#e05a18|#c44c12|224, ?90, ?24|181, ?189, ?177|#f0f2ee|#1a211c" frontend/safety_detection/styles/glass-clay.css`
Expected: 无输出（exit code 1）。若有输出说明某处旧值未替换。

- [ ] **Step 5: 提交**

```bash
git add frontend/safety_detection/styles/glass-clay.css
git commit -m "feat: retheme glass-clay to Cool Slate palette and balanced shadows"
```

---

### Task 2: 修 index.html 硬编码链接颜色

**Files:**
- Modify: `frontend/safety_detection/index.html:14`

**Interfaces:**
- Consumes: Task 1 的 `--accent`（`#2563eb`）。
- Produces: index.html 链接颜色统一到主色变量。

- [ ] **Step 1: 把 `#e94560` 换成 `var(--accent)`**

`index.html` 第 14 行的内联样式中，`color: #e94560` 和 `border: 1px solid #e94560` 分别改为 `color: var(--accent)` 和 `border: 1px solid var(--accent)`。行内其它样式不动。

- [ ] **Step 2: grep 验证无残留 `#e94560`**

Run: `grep -n "#e94560" frontend/safety_detection/index.html`
Expected: 无输出（exit code 1）。

- [ ] **Step 3: 提交**

```bash
git add frontend/safety_detection/index.html
git commit -m "fix: use accent token for index links instead of hardcoded red"
```

---

### Task 3: 全局残留色排查与逐页目视验证

**Files:**
- 只读检查：`frontend/safety_detection/*.html`

**Interfaces:**
- Consumes: Task 1 + Task 2 的成果。
- Produces: 验证结论（无回归）。

- [ ] **Step 1: 全局 grep 排查暖橙/遗留红残留**

Run: `grep -rnE "#e05a18|#c44c12|#e94560" frontend/safety_detection --include="*.html" --include="*.css"`
Expected: 仅可能命中 `demo-glass-clay.html`（旧演示稿，非线上页面）；monitor/records/settings/index/console/prompt 均无命中。若线上页面命中，需改用对应变量。

- [ ] **Step 2: 确认近黑底保留**

Run: `grep -rn "#0f1210" frontend/safety_detection --include="*.html"`
Expected: monitor.html 与 records.html 仍保留 `#0f1210`（视频/图片底，故意保留）。

- [ ] **Step 3: 逐页目视验证**

启动后端（按项目 `start.sh` / `start_all.sh`），浏览器依次打开：
- `/monitor`（monitor.html）
- `/records.html`
- `/settings.html`
- `/`（index.html）

逐页确认：
1. 底色为冷灰 `#eef1f4`、主色为蓝 `#2563eb`、无残留暖橙。
2. 玻璃卡投影收薄、黏土内凹块转冷，观感干净不吵。
3. 告警/危险态红色 `#dc2626` 在冷底上仍醒目。
4. 正文（`#1e293b` on `#eef1f4`）与次要文字（`#64748b`）对比度可读。
5. 按钮 hover / pressed 交互态阴影正常。

- [ ] **Step 4: 无需提交**

本任务仅验证；如目视发现某页有硬编码颜色遗漏，回到对应页面用变量修复后单独提交。

---

## Self-Review

**Spec coverage：**
- 配色令牌改动 → Task 1 Step 1、3 ✓
- 阴影令牌改动 → Task 1 Step 2 ✓
- index.html 硬编码 `#e94560` → Task 2 ✓
- `#0f1210` 保留不动 → Task 3 Step 2 ✓
- 验证标准（无残留暖橙、主色统一、告警醒目、对比度、交互态）→ Task 3 Step 3 ✓
- 字体/间距/布局不动 → Global Constraints ✓

**Placeholder scan：** 无 TBD/TODO；每个改动步骤给出确切色值与确切文件行。

**Type/命名一致性：** 令牌名全程不变（仅改值），`--accent` 在 Task 1 定义、Task 2 引用一致。

无遗漏。
