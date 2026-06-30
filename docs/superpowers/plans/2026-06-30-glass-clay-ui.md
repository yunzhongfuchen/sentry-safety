# Glass-Clay UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 Sentry 前端三页面（monitor / records / settings）的 Glassmorphism + Claymorphism 风格重设计，新增 `/monitor` 路由，删除旧 `/multi` 和 `/hud` 入口。

**Architecture:** 每个页面为独立 HTML 文件，内联 Vue 3 逻辑；公共设计 token 与组件类抽离到 `styles/glass-clay.css`；公共工具函数复用 `shared.js`；后端仅新增 `/monitor` 路由并删除旧路由。保留现有 API 不变。

**Tech Stack:** HTML5, CSS3, Vue 3 (CDN), Python, FastAPI

## Global Constraints

- 使用 Vue 3 CDN（`vue3.global.prod.js`），不引入构建工具
- 每个页面为单文件 HTML（内联 CSS + JS）
- 公共样式统一引用 `styles/glass-clay.css`
- 公共工具函数复用 `shared.js`
- 主色焦橙 `#e05a18`，背景冷灰白 `#f0f2ee`
- 删除 `multi.html` 和 `hud.html`
- 后端新增 `/monitor`，删除 `/multi` 和 `/hud`

---

## File Structure

| 文件 | 操作 | 职责 |
|------|------|------|
| `frontend/safety_detection/styles/glass-clay.css` | 创建 | 设计 token + 公共组件类 |
| `frontend/safety_detection/monitor.html` | 创建 | 新监控页 |
| `frontend/safety_detection/records.html` | 重写 | glass-clay 风格记录页 |
| `frontend/safety_detection/settings.html` | 重写 | glass-clay 风格设置页（3 Tab） |
| `frontend/safety_detection/shared.js` | 扩展 | 补充页面级工具函数 |
| `frontend/safety_detection/multi.html` | 删除 | 被 monitor.html 替代 |
| `frontend/safety_detection/hud.html` | 删除 | 被 monitor.html 替代 |
| `backend/main_multi.py` | 修改 | 新增 `/monitor`，删除 `/multi` 和 `/hud` |

---

## Task 1: Create Shared Design System CSS

**Files:**
- Create: `frontend/safety_detection/styles/glass-clay.css`
- Test: parse CSS syntax via Python

**Interfaces:**
- Produces: CSS variables and component classes used by all three pages

- [ ] **Step 1: Create glass-clay.css with design tokens and components**

Create `frontend/safety_detection/styles/glass-clay.css`:

```css
/* Glass-Clay Design System for Sentry */

:root {
    --bg-base: #f0f2ee;
    --bg-soft: #f7f8f6;
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

    --accent: #e05a18;
    --accent-soft: rgba(224, 90, 24, 0.10);
    --accent-hover: #c44c12;

    --success: #0d9f6e;
    --warning: #e05a18;
    --danger: #dc2626;
    --info: #2563eb;

    --radius-lg: 18px;
    --radius-md: 10px;
    --radius-sm: 6px;

    --font-display: 'Space Grotesk', 'Noto Sans SC', sans-serif;
    --font-body: 'Noto Sans SC', 'Space Grotesk', sans-serif;
    --font-mono: 'JetBrains Mono', 'Noto Sans SC', monospace;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: var(--font-body);
    background: var(--bg-base);
    color: var(--text-primary);
    min-height: 100vh;
    line-height: 1.5;
}

/* Glass card */
.glass-card {
    background: var(--glass-bg);
    border: 1px solid var(--glass-edge);
    border-radius: var(--radius-lg);
    box-shadow: var(--glass-shadow);
    backdrop-filter: blur(24px) saturate(140%);
    -webkit-backdrop-filter: blur(24px) saturate(140%);
    padding: 22px;
    transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1),
                box-shadow 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-card:hover {
    transform: translateY(-2px);
    box-shadow:
        0 4px 8px rgba(15, 23, 42, 0.05),
        0 18px 32px rgba(15, 23, 42, 0.08),
        inset 0 1px 0 rgba(255, 255, 255, 0.95);
}

/* Clay button */
.clay-button {
    border: none;
    background: var(--bg-soft);
    color: var(--text-primary);
    font-family: var(--font-display);
    font-size: 13px;
    font-weight: 500;
    padding: 10px 20px;
    border-radius: var(--radius-md);
    box-shadow: var(--clay-shadow);
    cursor: pointer;
    transition: all 0.15s ease;
}

.clay-button:hover { transform: translateY(-1px); }

.clay-button:active {
    transform: translateY(1px);
    box-shadow: var(--clay-shadow-pressed);
}

.clay-button.primary {
    background: var(--accent);
    color: #fff;
    box-shadow:
        4px 4px 10px rgba(224, 90, 24, 0.25),
        -4px -4px 10px rgba(255, 255, 255, 0.5);
}

.clay-button.primary:hover { background: var(--accent-hover); }

.clay-button.ghost {
    background: transparent;
    box-shadow: none;
    color: var(--text-secondary);
}

.clay-button.ghost:hover { color: var(--accent); }

.clay-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

/* Clay input */
.clay-input,
.clay-select {
    border: none;
    background: var(--bg-base);
    color: var(--text-primary);
    font-family: var(--font-body);
    font-size: 14px;
    padding: 12px 16px;
    border-radius: var(--radius-md);
    box-shadow: var(--clay-shadow-inset);
    outline: none;
    transition: box-shadow 0.2s ease;
}

.clay-input:focus,
.clay-select:focus {
    box-shadow:
        var(--clay-shadow-inset),
        0 0 0 3px var(--accent-soft);
}

.clay-select {
    appearance: none;
    -webkit-appearance: none;
    padding-right: 36px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%235f6b63' d='M6 8L1 3h10z'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 14px center;
}

/* Status dot */
.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}

.status-dot.online {
    background: var(--success);
    box-shadow: 0 0 0 3px rgba(13, 159, 110, 0.15);
}

.status-dot.offline {
    background: var(--text-muted);
}

.status-dot.alert {
    background: var(--danger);
    box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.12);
    animation: alert-pulse 2s ease-in-out infinite;
}

@keyframes alert-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.75; transform: scale(1.15); }
}

/* Type badges */
.type-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: var(--radius-sm);
    font-size: 12px;
    font-weight: 500;
    font-family: var(--font-body);
    background: var(--bg-elevated);
    color: var(--text-secondary);
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}

.type-badge.fire { background: #fff1f0; color: #991b1b; }
.type-badge.smoke { background: #fff7ed; color: #9a3412; }
.type-badge.uniform { background: #f0fdf4; color: #166534; }
.type-badge.mask { background: #ecfeff; color: #155e75; }
.type-badge.cigarette { background: #faf5ff; color: #6b21a8; }
.type-badge.sleep { background: #fefce8; color: #854d0e; }

/* Navigation header */
.nav-header {
    height: 64px;
    background: var(--glass-bg);
    border-bottom: 1px solid var(--glass-edge);
    box-shadow: var(--glass-shadow);
    backdrop-filter: blur(24px) saturate(140%);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 28px;
    position: sticky;
    top: 0;
    z-index: 100;
}

.nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    font-family: var(--font-display);
    font-size: 20px;
    font-weight: 700;
    color: var(--text-primary);
}

.nav-brand .logo {
    width: 34px;
    height: 34px;
    background: var(--accent);
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-size: 16px;
    box-shadow: 0 2px 8px rgba(224, 90, 24, 0.25);
}

.nav-links {
    display: flex;
    gap: 6px;
}

.nav-links a {
    padding: 9px 18px;
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 14px;
    font-weight: 500;
    transition: all 0.2s ease;
}

.nav-links a:hover {
    color: var(--text-primary);
    background: rgba(15, 23, 42, 0.04);
}

.nav-links a.active {
    color: var(--accent);
    background: var(--accent-soft);
}

/* Tables */
.gc-table-wrap {
    background: var(--glass-bg);
    border: 1px solid var(--glass-edge);
    border-radius: var(--radius-lg);
    box-shadow: var(--glass-shadow);
    overflow: hidden;
}

.gc-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}

.gc-table th {
    text-align: left;
    padding: 14px 18px;
    background: rgba(15, 23, 42, 0.03);
    border-bottom: 1px solid var(--glass-edge);
    font-weight: 600;
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.gc-table td {
    padding: 14px 18px;
    border-bottom: 1px solid var(--glass-edge);
    vertical-align: middle;
}

.gc-table tr:hover td { background: rgba(15, 23, 42, 0.02); }
.gc-table tr:last-child td { border-bottom: none; }

/* Tabs */
.gc-tabs {
    display: flex;
    gap: 6px;
    margin-bottom: 24px;
    border-bottom: 1px solid var(--glass-edge);
}

.gc-tab {
    padding: 12px 20px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    color: var(--text-secondary);
    border-bottom: 2px solid transparent;
    transition: all 0.2s ease;
    margin-bottom: -1px;
}

.gc-tab:hover { color: var(--text-primary); }
.gc-tab.active { color: var(--accent); border-bottom-color: var(--accent); }

/* Form grid */
.gc-form-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 18px;
    margin-bottom: 24px;
}

.gc-form-field {
    display: flex;
    flex-direction: column;
    gap: 7px;
}

.gc-form-field label {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
}

/* Modal */
.gc-modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(15, 23, 42, 0.45);
    backdrop-filter: blur(6px);
    display: flex;
    align-items: flex-start;
    justify-content: center;
    z-index: 1000;
    padding: 40px 20px;
}

.gc-modal-box {
    background: var(--bg-soft);
    border: 1px solid var(--glass-edge);
    border-radius: var(--radius-lg);
    box-shadow: var(--glass-shadow);
    width: 100%;
    max-width: 880px;
    max-height: calc(100vh - 80px);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.gc-modal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 24px;
    border-bottom: 1px solid var(--glass-edge);
}

.gc-modal-body { padding: 24px; flex: 1; overflow-y: auto; }
.gc-modal-footer {
    display: flex;
    gap: 10px;
    justify-content: flex-end;
    padding: 16px 24px;
    border-top: 1px solid var(--glass-edge);
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(15, 23, 42, 0.15); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(15, 23, 42, 0.25); }

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
```

- [ ] **Step 2: Verify CSS syntax**

Run:
```bash
python3 -c "
import re
with open('frontend/safety_detection/styles/glass-clay.css') as f:
    css = f.read()
# Basic brace balance check
assert css.count('{') == css.count('}'), 'Unbalanced braces'
print('CSS syntax OK')
"
```

Expected: `CSS syntax OK`

- [ ] **Step 3: Commit**

```bash
git add frontend/safety_detection/styles/glass-clay.css
git commit -m "feat: add glass-clay shared design system CSS

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Extend shared.js with Page Utilities

**Files:**
- Modify: `frontend/safety_detection/shared.js`
- Test: run a quick Node/JS check

**Interfaces:**
- Consumes: existing utility functions
- Produces: shared helpers used by monitor/records/settings pages

- [ ] **Step 1: Add page-level utility functions**

Append to `frontend/safety_detection/shared.js`:

```javascript
// Type labels and colors shared across pages
const DETECTION_TYPES = [
    { key: 'fire', label: '明火', color: '#ef4444' },
    { key: 'smoke', label: '烟雾', color: '#f97316' },
    { key: 'uniform', label: '工服', color: '#22c55e' },
    { key: 'mask', label: '口罩', color: '#0ea5e9' },
    { key: 'cigarette', label: '吸烟', color: '#a855f7' },
    { key: 'sleep', label: '睡岗', color: '#eab308' },
];

function getTypeLabel(type) {
    return DETECTION_TYPES.find(t => t.key === type)?.label || type || '未知';
}

function getTypeColor(type) {
    return DETECTION_TYPES.find(t => t.key === type)?.color || '#94a3b8';
}

function formatDateTime(iso) {
    if (!iso) return '-';
    const d = new Date(iso);
    if (isNaN(d)) return iso;
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`;
}

function formatTimeOnly(iso) {
    if (!iso) return '--:--:--';
    const d = new Date(iso);
    if (isNaN(d)) return iso;
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`;
}

function calculateUptime(startedAt) {
    const start = new Date(startedAt);
    if (isNaN(start)) return '-';
    const diff = Math.floor((Date.now() - start) / 1000);
    const h = Math.floor(diff / 3600);
    const m = Math.floor((diff % 3600) / 60);
    return `${h}h${m}m`;
}

function getLevelLabel(level) {
    const map = {
        small_model_alarm: '小模型报警',
        vlm_alarm: '大模型报警',
        vlm_ignore: '大模型忽略',
        P0: 'P0',
        P1: 'P1'
    };
    return map[level] || level || '-';
}

function getStatusLabel(status) {
    const map = { pending: '待确认', confirmed: '已确认', false_positive: '误报' };
    return map[status] || status || '-';
}

// Default detection type configuration structure
function defaultDetectionTypes() {
    return {
        fire: { enabled: false, interval: 1, threshold: 0.6, consecutive_required: 2, cooldown: 10, use_vlm: false },
        smoke: { enabled: false, interval: 1, threshold: 0.55, consecutive_required: 2, cooldown: 10, use_vlm: false },
        uniform: { enabled: false, interval: 1, threshold: 0.5, consecutive_required: 1, compliance_window_seconds: 30, cooldown: 3, use_vlm: false },
        mask: { enabled: false, interval: 1, threshold: 0.5, consecutive_required: 1, cooldown: 3, use_vlm: false },
        cigarette: { enabled: false, interval: 1, threshold: 0.5, consecutive_required: 1, cooldown: 3, use_vlm: false },
        sleep: { enabled: false, interval: 60, threshold: 0.7, consecutive_required: 3, cooldown: 30, use_vlm: false },
    };
}
```

- [ ] **Step 2: Verify syntax with Node**

Run:
```bash
node --check frontend/safety_detection/shared.js
```

Expected: no output (success)

- [ ] **Step 3: Commit**

```bash
git add frontend/safety_detection/shared.js
git commit -m "feat: extend shared.js with page-level detection and formatting helpers

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Create monitor.html Base Structure

**Files:**
- Create: `frontend/safety_detection/monitor.html`
- Test: Python HTML parser

**Interfaces:**
- Consumes: `glass-clay.css`, `shared.js`, Vue 3 CDN
- Produces: complete monitor page

- [ ] **Step 1: Create monitor.html with base structure**

Create `frontend/safety_detection/monitor.html`:

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentry 监控中心</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Noto+Sans+SC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="/static/styles/glass-clay.css">
    <script src="/static/vue3.global.prod.js"></script>
    <script src="/static/shared.js"></script>
    <style>
        .monitor-layout {
            height: calc(100vh - 64px);
            display: grid;
            grid-template-columns: 1fr 280px;
            grid-template-rows: 1fr 160px;
            grid-template-areas:
                "main sidebar"
                "alerts alerts";
            gap: 18px;
            padding: 18px;
        }

        .main-area {
            grid-area: main;
            display: flex;
            flex-direction: column;
            min-width: 0;
        }

        .video-card {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 0;
            overflow: hidden;
        }

        .video-header {
            padding: 14px 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--glass-edge);
        }

        .video-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 600;
            font-size: 15px;
        }

        .video-body {
            flex: 1;
            position: relative;
            background: #0f1210;
            min-height: 0;
        }

        .video-body img {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }

        .video-placeholder {
            position: absolute;
            inset: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
            gap: 10px;
        }

        .video-controls {
            position: absolute;
            bottom: 14px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 10px;
            z-index: 10;
        }

        .sidebar-area {
            grid-area: sidebar;
            display: flex;
            flex-direction: column;
            gap: 12px;
            min-width: 0;
        }

        .cam-list {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .cam-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px 14px;
            border-radius: var(--radius-md);
            background: var(--bg-soft);
            border: 1px solid transparent;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .cam-item:hover {
            border-color: var(--accent);
            box-shadow: var(--clay-shadow);
        }

        .cam-item.active {
            border-color: var(--accent);
            background: var(--accent-soft);
        }

        .cam-name {
            flex: 1;
            font-weight: 500;
            font-size: 14px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .cam-types {
            display: flex;
            gap: 4px;
            flex-wrap: wrap;
        }

        .cam-type-mini {
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 4px;
            background: var(--bg-elevated);
            color: var(--text-secondary);
        }

        .alerts-area {
            grid-area: alerts;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .alerts-list {
            flex: 1;
            display: flex;
            gap: 12px;
            overflow-x: auto;
            padding: 4px 2px;
            align-items: stretch;
        }

        .alert-card {
            flex-shrink: 0;
            width: 260px;
            padding: 14px 16px;
            display: flex;
            flex-direction: column;
            gap: 6px;
            cursor: pointer;
        }

        .alert-time {
            font-family: var(--font-mono);
            font-size: 12px;
            color: var(--text-muted);
        }

        .alert-main {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
        }

        .alert-camera {
            font-weight: 600;
            font-size: 14px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .stat-pill {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 20px;
            background: var(--bg-soft);
            font-size: 13px;
            font-weight: 500;
        }

        .nav-stats {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        @media (max-width: 1100px) {
            .monitor-layout {
                grid-template-columns: 1fr;
                grid-template-rows: 1fr 200px 160px;
                grid-template-areas:
                    "main"
                    "sidebar"
                    "alerts";
            }
        }
    </style>
</head>
<body>
    <div id="app">
        <header class="nav-header">
            <div class="nav-brand">
                <div class="logo">S</div>
                <span>Sentry</span>
            </div>
            <div class="nav-stats">
                <div class="stat-pill">
                    <span class="status-dot online"></span>
                    在线 {{ onlineCount }} / {{ cameras.length }}
                </div>
                <div class="stat-pill">
                    <span class="status-dot alert"></span>
                    告警 {{ alertCount }}
                </div>
                <div class="stat-pill" style="font-family: var(--font-mono);">
                    {{ utcTime }}
                </div>
            </div>
            <nav class="nav-links">
                <a href="/monitor" class="active">监控</a>
                <a href="/records.html">记录</a>
                <a href="/settings.html">设置</a>
            </nav>
        </header>

        <div class="monitor-layout">
            <!-- Main video -->
            <div class="main-area">
                <div class="glass-card video-card">
                    <div class="video-header">
                        <div class="video-title">
                            <span :class="['status-dot', mainCameraStatus === 'connected' ? 'online' : 'offline']"></span>
                            {{ mainCameraName }}
                        </div>
                        <span v-if="mainAlertType" class="type-badge" :class="mainAlertType">
                            {{ getTypeLabel(mainAlertType) }}
                        </span>
                    </div>
                    <div class="video-body">
                        <img v-if="mainCameraId && mainCameraStatus === 'connected' && streamVisible"
                             :src="`/cameras/${mainCameraId}/stream`"
                             @error="onVideoError(mainCameraId)" />
                        <div v-else class="video-placeholder">
                            <div style="font-size: 32px; opacity: 0.4;">📷</div>
                            <div>{{ mainCameraId ? getStatusText(mainCameraStatus) : '无可用摄像头' }}</div>
                        </div>

                        <div class="video-controls" v-if="playbackStatus.is_video_file">
                            <button class="clay-button" @click="controlPlayback(playbackStatus.playing ? 'pause' : 'play')">
                                {{ playbackStatus.playing ? '⏸ 暂停' : '▶ 播放' }}
                            </button>
                            <button class="clay-button" @click="controlPlayback('loop', { loop: !playbackStatus.loop })">
                                {{ playbackStatus.loop ? '🔁 循环' : '➡️ 单次' }}
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Sidebar camera list -->
            <div class="sidebar-area">
                <div class="glass-card" style="flex: 1; display: flex; flex-direction: column; padding: 16px;">
                    <div style="font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">
                        摄像头列表
                    </div>
                    <div class="cam-list">
                        <div v-for="cam in cameras" :key="cam.camera_id"
                             :class="['cam-item', cam.camera_id === mainCameraId ? 'active' : '']"
                             @click="selectCamera(cam.camera_id)">
                            <span :class="['status-dot', cam.status === 'connected' ? 'online' : 'offline']"></span>
                            <span class="cam-name">{{ cam.name || cam.camera_id }}</span>
                            <div class="cam-types">
                                <span v-for="(cfg, type) in cam.detection_types" :key="type"
                                      v-if="cfg.enabled"
                                      class="cam-type-mini">
                                    {{ getTypeLabel(type) }}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Bottom alerts -->
            <div class="alerts-area">
                <div class="glass-card" style="flex: 1; display: flex; flex-direction: column; padding: 16px;">
                    <div style="font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">
                        最近告警
                    </div>
                    <div class="alerts-list">
                        <div v-for="alert in recentAlerts" :key="alert.id" class="alert-card glass-card">
                            <div class="alert-time">{{ alert.time }}</div>
                            <div class="alert-main">
                                <span class="alert-camera">{{ alert.cameraName }}</span>
                                <span class="type-badge" :class="alert.detection_type">{{ getTypeLabel(alert.detection_type) }}</span>
                            </div>
                            <div style="font-size: 12px; color: var(--text-secondary);">
                                级别: {{ getLevelLabel(alert.level) }}
                            </div>
                        </div>
                        <div v-if="!recentAlerts.length" style="color: var(--text-muted); padding: 20px;">
                            暂无告警
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const { createApp, ref, computed, watch, onMounted, onUnmounted } = Vue;

        createApp({
            setup() {
                const cameras = ref([]);
                const cameraOrder = ref([]);
                const recentAlerts = ref([]);
                const overlayTypes = ref([]);
                const playbackStatus = ref({});
                const streamVisible = ref({});
                const utcTime = ref('--:--:--');
                let refreshTimer = null;
                let utcTimer = null;

                const mainCameraId = computed(() => cameraOrder.value[0] || null);
                const mainCameraObj = computed(() => cameras.value.find(c => c.camera_id === mainCameraId.value));
                const mainCameraName = computed(() => mainCameraObj.value?.name || mainCameraId.value || '未选择');
                const mainCameraStatus = computed(() => mainCameraObj.value?.status || 'idle');

                const onlineCount = computed(() => cameras.value.filter(c => c.status === 'connected').length);
                const alertCount = computed(() => recentAlerts.value.filter(a =>
                    a.level === 'small_model_alarm' || a.level === 'vlm_alarm' || a.level === 'P0'
                ).length);

                const mainAlertType = computed(() => {
                    const cam = mainCameraObj.value;
                    if (!cam || !cam.detection) return null;
                    for (const t of DETECTION_TYPES.map(x => x.key)) {
                        if (cam.detection[t]?.alert) return t;
                    }
                    return null;
                });

                function getStatusText(status) {
                    const map = { idle: '空闲', connecting: '连接中', connected: '已连接', error: '错误', reconnecting: '重连中' };
                    return map[status] || status;
                }

                function selectCamera(cid) {
                    const idx = cameraOrder.value.indexOf(cid);
                    if (idx > 0) {
                        const newOrder = [...cameraOrder.value];
                        [newOrder[0], newOrder[idx]] = [newOrder[idx], newOrder[0]];
                        cameraOrder.value = newOrder;
                    } else if (idx === -1) {
                        cameraOrder.value = [cid, ...cameraOrder.value.filter(id => id !== cid)];
                    }
                    updateUrl();
                }

                function updateUrl() {
                    if (!mainCameraId.value) return;
                    const url = new URL(window.location.href);
                    url.searchParams.set('camera', mainCameraId.value);
                    window.history.replaceState({}, '', url);
                }

                function onVideoError(cid) {
                    streamVisible.value[cid] = false;
                    setTimeout(() => { streamVisible.value[cid] = true; }, 3000);
                }

                async function fetchCameras() {
                    try {
                        const data = await safeFetch('/cameras');
                        if (data.cameras) {
                            cameras.value = data.cameras;
                            const enabledIds = data.cameras.filter(c => c.enabled !== false).map(c => c.camera_id);
                            const url = new URL(window.location.href);
                            const urlCam = url.searchParams.get('camera');

                            let newOrder;
                            if (cameraOrder.value.length === 0) {
                                newOrder = urlCam && enabledIds.includes(urlCam)
                                    ? [urlCam, ...enabledIds.filter(id => id !== urlCam)]
                                    : [...enabledIds];
                            } else {
                                const existing = new Set(cameraOrder.value);
                                const newIds = enabledIds.filter(id => !existing.has(id));
                                newOrder = cameraOrder.value.filter(id => enabledIds.includes(id));
                                newOrder.push(...newIds);
                            }
                            cameraOrder.value = newOrder;
                        }
                    } catch (e) { console.error('获取摄像头失败:', e); }
                }

                async function fetchStatus() {
                    try {
                        const data = await safeFetch('/status');
                        if (data.recent_records) {
                            recentAlerts.value = data.recent_records.slice(0, 6).map(r => {
                                const cam = cameras.value.find(c => c.camera_id === r.camera_id);
                                return {
                                    id: r.id,
                                    time: formatDateTime(r.time),
                                    cameraName: cam?.name || r.camera_id,
                                    detection_type: r.detection_type || r.action,
                                    level: r.level || 'small_model_alarm',
                                };
                            });
                        }
                    } catch (e) { console.error('获取状态失败:', e); }
                }

                async function fetchOverlay() {
                    try {
                        const data = await safeFetch('/overlay');
                        if (data.overlay_types) overlayTypes.value = data.overlay_types;
                    } catch (e) { console.error('获取画框配置失败:', e); }
                }

                async function fetchPlaybackStatus() {
                    if (!mainCameraId.value) return;
                    try {
                        const data = await safeFetch(`/cameras/${mainCameraId.value}/playback/status`);
                        playbackStatus.value = data || {};
                    } catch (e) { playbackStatus.value = {}; }
                }

                async function controlPlayback(action, extra = {}) {
                    if (!mainCameraId.value) return;
                    try {
                        const data = await safeFetch(`/cameras/${mainCameraId.value}/playback/control`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ action, ...extra })
                        });
                        if (data.status) playbackStatus.value = data.status;
                    } catch (e) { console.error('播放控制失败:', e); }
                }

                function updateUtc() {
                    utcTime.value = new Date().toISOString().slice(11, 19);
                }

                watch(mainCameraId, () => {
                    playbackStatus.value = {};
                    fetchPlaybackStatus();
                    updateUrl();
                });

                onMounted(() => {
                    fetchCameras();
                    fetchStatus();
                    fetchOverlay();
                    fetchPlaybackStatus();
                    updateUtc();
                    refreshTimer = setInterval(() => { fetchCameras(); fetchStatus(); fetchPlaybackStatus(); }, 2000);
                    utcTimer = setInterval(updateUtc, 1000);
                });

                onUnmounted(() => {
                    if (refreshTimer) clearInterval(refreshTimer);
                    if (utcTimer) clearInterval(utcTimer);
                });

                return {
                    cameras, recentAlerts, overlayTypes, playbackStatus, streamVisible, utcTime,
                    mainCameraId, mainCameraName, mainCameraStatus, mainAlertType,
                    onlineCount, alertCount,
                    getTypeLabel, getLevelLabel, getStatusText,
                    selectCamera, onVideoError, controlPlayback
                };
            }
        }).mount('#app');
    </script>
</body>
</html>
```

- [ ] **Step 2: Verify HTML syntax**

Run:
```bash
python3 -c "from html.parser import HTMLParser; HTMLParser().feed(open('frontend/safety_detection/monitor.html', encoding='utf-8').read()); print('HTML OK')"
```

Expected: `HTML OK`

- [ ] **Step 3: Commit**

```bash
git add frontend/safety_detection/monitor.html
git commit -m "feat: add glass-clay monitor page

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Create records.html

**Files:**
- Modify: `frontend/safety_detection/records.html`
- Test: Python HTML parser

**Interfaces:**
- Consumes: `glass-clay.css`, `shared.js`, Vue 3 CDN
- Produces: complete records page

- [ ] **Step 1: Replace records.html with glass-clay version**

Replace the entire content of `frontend/safety_detection/records.html` with the following:

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentry 检测记录</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Noto+Sans+SC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="/static/styles/glass-clay.css">
    <script src="/static/vue3.global.prod.js"></script>
    <script src="/static/shared.js"></script>
    <style>
        .records-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }

        .stat-card {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .stat-value {
            font-family: var(--font-display);
            font-size: 32px;
            font-weight: 700;
            color: var(--accent);
            line-height: 1;
        }

        .stat-label {
            font-size: 13px;
            color: var(--text-secondary);
        }

        .filter-bar {
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            align-items: flex-end;
            margin-bottom: 24px;
        }

        .filter-field {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .filter-field label {
            font-size: 12px;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .type-distribution {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 24px;
        }

        .pagination {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 12px;
            padding: 18px;
        }

        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-muted);
        }

        .snapshot-img {
            width: 100%;
            max-height: 360px;
            object-fit: contain;
            border-radius: var(--radius-md);
            background: #0f1210;
            display: block;
        }

        .frame-player {
            background: #0f1210;
            border-radius: var(--radius-md);
            overflow: hidden;
        }

        .frame-display {
            position: relative;
            height: 280px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .frame-display img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }

        .frame-controls {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 16px;
            background: var(--bg-soft);
            border-top: 1px solid var(--glass-edge);
        }

        .frame-slider {
            flex: 1;
            -webkit-appearance: none;
            height: 6px;
            border-radius: 3px;
            background: var(--glass-edge);
            outline: none;
        }

        .frame-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: var(--accent);
            cursor: pointer;
        }

        .thumbnails {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
            gap: 8px;
            padding: 12px 16px;
            max-height: 150px;
            overflow-y: auto;
            background: var(--bg-soft);
        }

        .thumbnail {
            aspect-ratio: 16/10;
            background: #0f1210;
            border-radius: var(--radius-sm);
            overflow: hidden;
            cursor: pointer;
            border: 2px solid transparent;
        }

        .thumbnail.active { border-color: var(--accent); }
        .thumbnail img { width: 100%; height: 100%; object-fit: cover; }
    </style>
</head>
<body>
    <div id="app">
        <header class="nav-header">
            <div class="nav-brand">
                <div class="logo">S</div>
                <span>Sentry · 检测记录</span>
            </div>
            <nav class="nav-links">
                <a href="/monitor">监控</a>
                <a href="/records.html" class="active">记录</a>
                <a href="/settings.html">设置</a>
            </nav>
        </header>

        <div class="records-container">
            <!-- Stats -->
            <div class="stats-grid">
                <div class="stat-card glass-card">
                    <div class="stat-value">{{ stats.total }}</div>
                    <div class="stat-label">今日告警</div>
                </div>
                <div class="stat-card glass-card">
                    <div class="stat-value" style="color: var(--warning);">{{ stats.pending }}</div>
                    <div class="stat-label">待确认</div>
                </div>
                <div class="stat-card glass-card">
                    <div class="stat-value" style="color: var(--danger);">{{ stats.confirmed }}</div>
                    <div class="stat-label">已确认</div>
                </div>
                <div class="stat-card glass-card">
                    <div class="stat-value" style="color: var(--success);">{{ stats.false_positive }}</div>
                    <div class="stat-label">误报</div>
                </div>
            </div>

            <!-- Type distribution -->
            <div class="type-distribution">
                <span v-for="(count, type) in typeDistribution" :key="type" class="type-badge" :class="type">
                    {{ getTypeLabel(type) }} {{ count }}
                </span>
            </div>

            <!-- Filters -->
            <div class="glass-card filter-bar">
                <div class="filter-field">
                    <label>日期</label>
                    <input type="date" class="clay-input" v-model="filters.date" />
                </div>
                <div class="filter-field">
                    <label>摄像头</label>
                    <select class="clay-select" v-model="filters.camera_id">
                        <option value="">全部</option>
                        <option v-for="cam in cameras" :key="cam.camera_id" :value="cam.camera_id">
                            {{ cam.name || cam.camera_id }}
                        </option>
                    </select>
                </div>
                <div class="filter-field">
                    <label>检测类型</label>
                    <select class="clay-select" v-model="filters.detection_type">
                        <option value="">全部</option>
                        <option v-for="t in DETECTION_TYPES" :key="t.key" :value="t.key">{{ t.label }}</option>
                    </select>
                </div>
                <div class="filter-field">
                    <label>告警级别</label>
                    <select class="clay-select" v-model="filters.level">
                        <option value="">全部</option>
                        <option value="small_model_alarm">小模型报警</option>
                        <option value="vlm_alarm">大模型报警</option>
                        <option value="vlm_ignore">大模型忽略</option>
                    </select>
                </div>
                <div class="filter-field">
                    <label>状态</label>
                    <select class="clay-select" v-model="filters.status">
                        <option value="">全部</option>
                        <option value="pending">待确认</option>
                        <option value="confirmed">已确认</option>
                        <option value="false_positive">误报</option>
                    </select>
                </div>
                <div class="filter-field">
                    <label>每页</label>
                    <select class="clay-select" v-model.number="pagination.page_size">
                        <option :value="10">10</option>
                        <option :value="20">20</option>
                        <option :value="50">50</option>
                    </select>
                </div>
                <button class="clay-button primary" @click="loadRecords">刷新</button>
            </div>

            <!-- Table -->
            <div class="gc-table-wrap">
                <div v-if="loading" class="empty-state">加载中...</div>
                <table v-else class="gc-table">
                    <thead>
                        <tr>
                            <th>时间</th>
                            <th>摄像头</th>
                            <th>类型</th>
                            <th>级别</th>
                            <th>状态</th>
                            <th>置信度</th>
                            <th>说明</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-for="r in records" :key="r.id" @click="showDetail(r.id)" style="cursor: pointer;">
                            <td style="white-space: nowrap;">{{ formatDateTime(r.time) }}</td>
                            <td>{{ r.camera_name }}</td>
                            <td>
                                <span class="type-badge" :class="r.detection_type">
                                    {{ getTypeLabel(r.detection_type) }}
                                </span>
                            </td>
                            <td>{{ getLevelLabel(r.level) }}</td>
                            <td>{{ getStatusLabel(r.status) }}</td>
                            <td>{{ r.confidence != null ? (r.confidence * 100).toFixed(0) + '%' : '-' }}</td>
                            <td style="max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" :title="r.reason">
                                {{ r.reason || '-' }}
                            </td>
                        </tr>
                    </tbody>
                </table>
                <div v-if="!loading && records.length === 0" class="empty-state">暂无检测记录</div>

                <!-- Pagination -->
                <div class="pagination">
                    <button class="clay-button" @click="prevPage" :disabled="pagination.page <= 1">上一页</button>
                    <span style="color: var(--text-secondary);">第 {{ pagination.page }} 页 / 共 {{ totalPages }} 页（共 {{ pagination.total }} 条）</span>
                    <button class="clay-button" @click="nextPage" :disabled="pagination.page >= totalPages">下一页</button>
                </div>
            </div>
        </div>

        <!-- Detail Modal -->
        <div v-if="detail" class="gc-modal-overlay" @click.self="detail = null">
            <div class="gc-modal-box">
                <div class="gc-modal-header">
                    <h3 style="font-size: 16px; font-weight: 700;">记录详情</h3>
                    <button class="clay-button ghost" @click="detail = null">&times;</button>
                </div>
                <div class="gc-modal-body">
                    <div class="gc-form-grid">
                        <div class="glass-card">
                            <div style="font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-bottom: 12px;">基本信息</div>
                            <div class="gc-form-field"><label>记录ID</label><input class="clay-input" :value="detail.id" disabled /></div>
                            <div class="gc-form-field"><label>检测时间</label><input class="clay-input" :value="formatDateTime(detail.time)" disabled /></div>
                            <div class="gc-form-field"><label>摄像头</label><input class="clay-input" :value="detail.camera_id" disabled /></div>
                            <div class="gc-form-field"><label>检测类型</label><input class="clay-input" :value="getTypeLabel(detail.detection_type)" disabled /></div>
                        </div>
                        <div class="glass-card">
                            <div style="font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-bottom: 12px;">检测详情</div>
                            <div class="gc-form-field"><label>告警级别</label><input class="clay-input" :value="getLevelLabel(detail.level)" disabled /></div>
                            <div class="gc-form-field"><label>状态</label><input class="clay-input" :value="getStatusLabel(detail.status)" disabled /></div>
                            <div class="gc-form-field"><label>置信度</label><input class="clay-input" :value="detail.confidence != null ? (detail.confidence * 100).toFixed(1) + '%' : '-'" disabled /></div>
                            <div class="gc-form-field"><label>说明</label><input class="clay-input" :value="detail.reason || '-'" disabled /></div>
                        </div>
                    </div>

                    <div style="margin-bottom: 20px;">
                        <div style="font-size: 14px; font-weight: 600; margin-bottom: 12px;">触发快照</div>
                        <img v-if="snapshot" :src="snapshot" class="snapshot-img" />
                        <div v-else class="empty-state" style="padding: 30px;">无快照</div>
                    </div>

                    <div>
                        <div style="font-size: 14px; font-weight: 600; margin-bottom: 12px;">视频帧序列</div>
                        <div class="frame-player">
                            <div class="frame-display">
                                <img v-if="frames[currentFrame]" :src="frames[currentFrame]" />
                                <span v-else style="color: var(--text-muted);">无视频帧</span>
                            </div>
                            <div class="frame-controls">
                                <button class="clay-button" @click="togglePlay">{{ playing ? '暂停' : '播放' }}</button>
                                <button class="clay-button" @click="toggleReverse">{{ reverse ? '正序' : '倒序' }}</button>
                                <input type="range" class="frame-slider" min="0" :max="frames.length - 1" v-model.number="currentFrame" />
                                <span style="font-family: var(--font-mono); font-size: 13px;">{{ currentFrame + 1 }} / {{ frames.length }}</span>
                            </div>
                            <div class="thumbnails">
                                <div v-for="(f, i) in frames" :key="i" :class="['thumbnail', i === currentFrame ? 'active' : '']" @click="currentFrame = i">
                                    <img :src="f" />
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="gc-modal-footer">
                    <button class="clay-button" @click="detail = null">关闭</button>
                    <button v-if="detail.status !== 'confirmed'" class="clay-button primary" @click="confirmAlert">确认报警</button>
                    <button v-if="detail.status !== 'false_positive'" class="clay-button" @click="ignoreAlert" style="color: var(--danger);">确认误报</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        const { createApp, ref, computed, watch, onMounted } = Vue;

        createApp({
            setup() {
                const cameras = ref([]);
                const records = ref([]);
                const stats = ref({ total: 0, pending: 0, confirmed: 0, false_positive: 0 });
                const typeDistribution = ref({});
                const loading = ref(false);
                const filters = ref({ date: '', camera_id: '', detection_type: '', level: '', status: '' });
                const pagination = ref({ page: 1, page_size: 20, total: 0 });
                const detail = ref(null);
                const snapshot = ref('');
                const frames = ref([]);
                const currentFrame = ref(0);
                const playing = ref(false);
                const reverse = ref(false);
                let playTimer = null;

                const totalPages = computed(() => Math.ceil(pagination.value.total / pagination.value.page_size) || 1);

                watch(() => pagination.value.page_size, () => { pagination.value.page = 1; loadRecords(); });

                async function loadCameras() {
                    try {
                        const data = await safeFetch('/cameras');
                        cameras.value = data.cameras || [];
                    } catch (e) { console.error('加载摄像头失败:', e); }
                }

                async function loadStats() {
                    try {
                        const data = await safeFetch('/records/summary');
                        stats.value = {
                            total: data.total || 0,
                            pending: data.by_status?.pending || 0,
                            confirmed: data.by_status?.confirmed || 0,
                            false_positive: data.by_status?.false_positive || 0
                        };
                        typeDistribution.value = data.by_type || {};
                    } catch (e) { console.error('加载统计失败:', e); }
                }

                async function loadRecords() {
                    loading.value = true;
                    try {
                        const params = new URLSearchParams({
                            page: pagination.value.page,
                            page_size: pagination.value.page_size
                        });
                        if (filters.value.camera_id) params.append('camera_id', filters.value.camera_id);
                        if (filters.value.detection_type) params.append('type', filters.value.detection_type);
                        if (filters.value.level) params.append('level', filters.value.level);
                        if (filters.value.status) params.append('status', filters.value.status);
                        if (filters.value.date) {
                            params.append('date_from', filters.value.date);
                            params.append('date_to', filters.value.date);
                        }

                        const data = await safeFetch(`/alerts?${params}`);
                        records.value = (data.records || []).map(r => ({
                            ...r,
                            camera_name: cameras.value.find(c => c.camera_id === r.camera_id)?.name || r.camera_id
                        }));
                        pagination.value.total = data.total || 0;
                    } catch (e) {
                        console.error('加载记录失败:', e);
                        records.value = [];
                    } finally { loading.value = false; }
                }

                function prevPage() { if (pagination.value.page > 1) { pagination.value.page--; loadRecords(); } }
                function nextPage() { if (pagination.value.page < totalPages.value) { pagination.value.page++; loadRecords(); } }

                async function showDetail(recordId) {
                    detail.value = null;
                    snapshot.value = '';
                    frames.value = [];
                    currentFrame.value = 0;
                    stopPlay();
                    try {
                        const data = await safeFetch(`/record/${encodeURIComponent(recordId)}?include_frames=false`);
                        detail.value = data;
                        if (data.snapshot) {
                            snapshot.value = `data:image/jpeg;base64,${data.snapshot}`;
                        } else {
                            const snapData = await safeFetch(`/record/${encodeURIComponent(recordId)}/snapshot`);
                            if (snapData.snapshot) snapshot.value = `data:image/jpeg;base64,${snapData.snapshot}`;
                        }
                        const frameData = await safeFetch(`/record/${encodeURIComponent(recordId)}/frames?start=0&count=30`);
                        frames.value = (frameData.frames || []).map(f => `data:image/jpeg;base64,${f}`);
                    } catch (e) { console.error('加载详情失败:', e); }
                }

                async function confirmAlert() {
                    if (!detail.value) return;
                    try {
                        await safeFetch(`/alerts/${encodeURIComponent(detail.value.id)}/confirm`, { method: 'POST' });
                        detail.value.status = 'confirmed';
                        loadRecords();
                        loadStats();
                    } catch (e) { console.error('确认失败:', e); }
                }

                async function ignoreAlert() {
                    if (!detail.value) return;
                    try {
                        await safeFetch(`/alerts/${encodeURIComponent(detail.value.id)}/ignore`, { method: 'POST' });
                        detail.value.status = 'false_positive';
                        loadRecords();
                        loadStats();
                    } catch (e) { console.error('误报确认失败:', e); }
                }

                function togglePlay() {
                    if (playing.value) { stopPlay(); return; }
                    if (!frames.value.length) return;
                    playing.value = true;
                    playNext();
                }

                function playNext() {
                    if (!playing.value) return;
                    if (!reverse.value && currentFrame.value >= frames.value.length - 1) {
                        currentFrame.value = 0;
                    } else if (reverse.value && currentFrame.value <= 0) {
                        currentFrame.value = frames.value.length - 1;
                    } else {
                        currentFrame.value += reverse.value ? -1 : 1;
                    }
                    playTimer = setTimeout(playNext, 300);
                }

                function stopPlay() {
                    playing.value = false;
                    if (playTimer) { clearTimeout(playTimer); playTimer = null; }
                }

                function toggleReverse() {
                    reverse.value = !reverse.value;
                }

                onMounted(() => {
                    loadCameras().then(() => {
                        loadRecords();
                    });
                    loadStats();
                });

                return {
                    cameras, records, stats, typeDistribution, loading, filters, pagination,
                    detail, snapshot, frames, currentFrame, playing, reverse, totalPages,
                    DETECTION_TYPES,
                    getTypeLabel, getLevelLabel, getStatusLabel, formatDateTime,
                    loadRecords, prevPage, nextPage, showDetail, confirmAlert, ignoreAlert,
                    togglePlay, toggleReverse
                };
            }
        }).mount('#app');
    </script>
</body>
</html>
```

- [ ] **Step 2: Verify HTML syntax**

Run:
```bash
python3 -c "from html.parser import HTMLParser; HTMLParser().feed(open('frontend/safety_detection/records.html', encoding='utf-8').read()); print('HTML OK')"
```

Expected: `HTML OK`

- [ ] **Step 3: Commit**

```bash
git add frontend/safety_detection/records.html
git commit -m "feat: rewrite records page in glass-clay style

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Create settings.html

**Files:**
- Modify: `frontend/safety_detection/settings.html`
- Test: Python HTML parser

**Interfaces:**
- Consumes: `glass-clay.css`, `shared.js`, Vue 3 CDN
- Produces: complete settings page with 3 tabs

- [ ] **Step 1: Replace settings.html with glass-clay version**

Replace the entire content of `frontend/safety_detection/settings.html` with the following:

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentry 系统设置</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Noto+Sans+SC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="/static/styles/glass-clay.css">
    <script src="/static/vue3.global.prod.js"></script>
    <script src="/static/shared.js"></script>
    <style>
        .settings-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 24px;
        }

        .type-cards-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }

        @media (max-width: 1000px) { .type-cards-grid { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 640px) { .type-cards-grid { grid-template-columns: 1fr; } }

        .type-card {
            border-top: 3px solid;
        }

        .type-card-header {
            font-size: 15px;
            font-weight: 700;
            margin-bottom: 12px;
        }

        .type-card-checkboxes {
            display: flex;
            gap: 16px;
            margin-bottom: 14px;
        }

        .type-card-checkbox {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
            color: var(--text-secondary);
            cursor: pointer;
        }

        .type-card-checkbox input {
            width: 16px;
            height: 16px;
            accent-color: var(--accent);
            cursor: pointer;
        }

        .type-card-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }

        .type-card-field {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .type-card-field label {
            font-size: 12px;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .type-card-field input {
            border: none;
            background: var(--bg-base);
            color: var(--text-primary);
            font-size: 13px;
            padding: 8px 10px;
            border-radius: var(--radius-sm);
            box-shadow: var(--clay-shadow-inset);
            outline: none;
        }

        .model-card {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .model-card .name { font-weight: 600; font-size: 15px; }
        .model-card .meta { font-size: 13px; color: var(--text-secondary); margin-top: 2px; }

        .toast {
            position: fixed;
            top: 80px;
            right: 24px;
            padding: 12px 22px;
            border-radius: var(--radius-md);
            font-size: 14px;
            font-weight: 500;
            z-index: 2000;
            animation: toastin 0.3s ease;
            box-shadow: var(--glass-shadow);
        }

        .toast.success { background: var(--success); color: white; }
        .toast.error { background: var(--danger); color: white; }

        @keyframes toastin {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .logs-panel {
            max-height: 360px;
            overflow-y: auto;
            font-family: var(--font-mono);
            font-size: 12px;
        }

        .log-entry {
            padding: 5px 0;
            border-bottom: 1px solid var(--glass-edge);
        }

        .log-entry:last-child { border-bottom: none; }
        .log-time { color: var(--text-muted); margin-right: 10px; }
    </style>
</head>
<body>
    <div id="app">
        <header class="nav-header">
            <div class="nav-brand">
                <div class="logo">S</div>
                <span>Sentry · 系统设置</span>
            </div>
            <nav class="nav-links">
                <a href="/monitor">监控</a>
                <a href="/records.html">记录</a>
                <a href="/settings.html" class="active">设置</a>
            </nav>
        </header>

        <div class="settings-container">
            <div class="glass-card" style="padding: 0; overflow: hidden;">
                <div class="gc-tabs" style="padding: 0 24px; margin-bottom: 0; border-bottom: 1px solid var(--glass-edge);">
                    <div :class="['gc-tab', { active: tab === 'cameras' }]" @click="tab = 'cameras'">摄像头</div>
                    <div :class="['gc-tab', { active: tab === 'detection' }]" @click="tab = 'detection'">检测配置</div>
                    <div :class="['gc-tab', { active: tab === 'system' }]" @click="tab = 'system'">系统设置</div>
                </div>

                <div style="padding: 24px;">
                    <!-- Cameras Tab -->
                    <div v-if="tab === 'cameras'">
                        <div style="display: flex; gap: 10px; margin-bottom: 18px;">
                            <button class="clay-button primary" @click="openCameraDialog()">+ 添加摄像头</button>
                            <button class="clay-button" v-if="selectedCameras.length" @click="openBatchDialog">批量配置 ({{ selectedCameras.length }})</button>
                        </div>

                        <div class="gc-table-wrap">
                            <table class="gc-table">
                                <thead>
                                    <tr>
                                        <th><input type="checkbox" @change="toggleSelectAll" /></th>
                                        <th>ID</th>
                                        <th>名称</th>
                                        <th>源地址</th>
                                        <th>类型</th>
                                        <th>分辨率</th>
                                        <th>启用</th>
                                        <th>操作</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr v-for="cam in cameras" :key="cam.camera_id">
                                        <td><input type="checkbox" :value="cam.camera_id" v-model="selectedCameras" /></td>
                                        <td>{{ cam.camera_id }}</td>
                                        <td>{{ cam.name || '-' }}</td>
                                        <td style="max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{{ cam.source }}</td>
                                        <td>{{ cam.source_type || 'auto' }}</td>
                                        <td>{{ cam.width }}x{{ cam.height }}</td>
                                        <td>
                                            <span :style="{ color: cam.enabled !== false ? 'var(--success)' : 'var(--danger)', fontWeight: 600 }">
                                                {{ cam.enabled !== false ? '是' : '否' }}
                                            </span>
                                        </td>
                                        <td>
                                            <button class="clay-button" style="padding: 6px 12px; margin-right: 6px;" @click="openCameraDialog(cam)">编辑</button>
                                            <button class="clay-button" style="padding: 6px 12px; color: var(--danger);" @click="deleteCamera(cam.camera_id)">删除</button>
                                        </td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <!-- Detection Tab -->
                    <div v-if="tab === 'detection'">
                        <div style="font-size: 13px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-bottom: 16px;">VLM 配置</div>
                        <div class="gc-form-grid">
                            <div class="gc-form-field">
                                <label>VLM 最大并发</label>
                                <input type="number" class="clay-input" v-model.number="settings.vlm_max_concurrent" min="1" />
                            </div>
                            <div class="gc-form-field">
                                <label>VLM 巡检间隔 (s, 0=关闭)</label>
                                <input type="number" class="clay-input" v-model.number="settings.vlm_inspection_interval" min="0" />
                            </div>
                        </div>
                        <button class="clay-button primary" @click="saveVlmSettings">保存 VLM 配置</button>

                        <div style="font-size: 13px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin: 28px 0 16px;">检测类型默认值</div>
                        <div class="type-cards-grid">
                            <div v-for="t in DETECTION_TYPES" :key="t.key" class="glass-card type-card" :style="{ borderTopColor: t.color }">
                                <div class="type-card-header" :style="{ color: t.color }">{{ t.label }}</div>
                                <div class="type-card-checkboxes">
                                    <label class="type-card-checkbox"><input type="checkbox" v-model="settings.default_detection_types[t.key].enabled" /> 启用</label>
                                    <label class="type-card-checkbox"><input type="checkbox" v-model="settings.default_detection_types[t.key].use_vlm" /> VLM</label>
                                </div>
                                <div class="type-card-grid">
                                    <div class="type-card-field"><label>间隔</label><input type="number" v-model.number="settings.default_detection_types[t.key].interval" /></div>
                                    <div class="type-card-field"><label>阈值</label><input type="number" step="0.1" v-model.number="settings.default_detection_types[t.key].threshold" /></div>
                                    <div class="type-card-field"><label>连续</label><input type="number" v-model.number="settings.default_detection_types[t.key].consecutive_required" /></div>
                                    <div class="type-card-field"><label>冷却</label><input type="number" v-model.number="settings.default_detection_types[t.key].cooldown" min="0" /></div>
                                </div>
                                <div v-if="t.key === 'uniform'" class="type-card-grid" style="margin-top: 10px;">
                                    <div class="type-card-field" style="grid-column: span 2;">
                                        <label>合规窗口 (s)</label>
                                        <input type="number" v-model.number="settings.default_detection_types[t.key].compliance_window_seconds" />
                                    </div>
                                </div>
                            </div>
                        </div>
                        <button class="clay-button primary" @click="saveDefaultTypes">保存检测默认值</button>
                    </div>

                    <!-- System Tab -->
                    <div v-if="tab === 'system'">
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 24px;">
                            <div class="glass-card">
                                <div style="font-size: 13px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-bottom: 14px;">运行信息</div>
                                <div class="gc-form-field"><label>当前检测设备</label><input class="clay-input" :value="systemMode" disabled /></div>
                                <div class="gc-form-field"><label>运行模式</label><input class="clay-input" :value="detectorInfo.mode || 'CPU'" disabled /></div>
                                <div class="gc-form-field"><label>视频解码后端</label><input class="clay-input" :value="detectorInfo.decoder_backend || 'none'" disabled /></div>
                            </div>

                            <div class="glass-card">
                                <div style="font-size: 13px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-bottom: 14px;">GPU 动态调度器</div>
                                <div class="gc-form-field">
                                    <label>启用 GPU 调度器</label>
                                    <select class="clay-select" v-model="settings.use_gpu_scheduler">
                                        <option :value="true">是</option>
                                        <option :value="false">否</option>
                                    </select>
                                </div>
                                <div class="gc-form-field"><label>并行队列数 (0=自动)</label><input type="number" class="clay-input" v-model.number="settings.gpu_scheduler_num_queues" min="0" /></div>
                                <div class="gc-form-field"><label>调度周期 (s)</label><input type="number" class="clay-input" step="0.1" v-model.number="settings.gpu_scheduler_interval" min="0.1" /></div>
                                <div class="gc-form-field">
                                    <label>FP16 半精度推理</label>
                                    <select class="clay-select" v-model="settings.gpu_scheduler_half">
                                        <option :value="true">是</option>
                                        <option :value="false">否</option>
                                    </select>
                                </div>
                            </div>

                            <div class="glass-card">
                                <div style="font-size: 13px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-bottom: 14px;">存储与清理</div>
                                <div class="gc-form-field"><label>最大记录数</label><input type="number" class="clay-input" v-model.number="settings.max_records" min="10" /></div>
                                <div class="gc-form-field"><label>存储上限 (MB, 0=无限制)</label><input type="number" class="clay-input" v-model.number="settings.max_storage_mb" min="0" /></div>
                                <div class="gc-form-field"><label>内存阈值 (%)</label><input type="number" class="clay-input" v-model.number="settings.memory_threshold_percent" min="1" max="100" /></div>
                                <div class="gc-form-field"><label>紧急清理比例 (0-1)</label><input type="number" class="clay-input" step="0.1" v-model.number="settings.emergency_cleanup_ratio" min="0" max="1" /></div>
                            </div>

                            <div class="glass-card">
                                <div style="font-size: 13px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-bottom: 14px;">图像质量</div>
                                <div class="gc-form-field"><label>快照质量 (1-100)</label><input type="number" class="clay-input" v-model.number="settings.snapshot_quality" min="1" max="100" /></div>
                                <div class="gc-form-field"><label>帧质量 (1-100)</label><input type="number" class="clay-input" v-model.number="settings.frame_quality" min="1" max="100" /></div>
                            </div>
                        </div>
                        <button class="clay-button primary" @click="saveSystemSettings">保存系统设置</button>

                        <div style="font-size: 13px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin: 28px 0 16px;">已加载模型</div>
                        <div v-for="m in models" :key="m.type" class="glass-card model-card" style="margin-bottom: 10px;">
                            <div>
                                <div class="name">{{ m.type }}</div>
                                <div class="meta">{{ m.backend }} / {{ m.device }}</div>
                            </div>
                            <span class="type-badge" :class="m.loaded ? 'uniform' : ''">{{ m.loaded ? '已加载' : '未加载' }}</span>
                        </div>

                        <div style="font-size: 13px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin: 28px 0 16px;">系统操作</div>
                        <button class="clay-button primary" @click="restartSystem">重启检测服务</button>
                        <p style="margin-top: 10px; font-size: 13px; color: var(--text-secondary);">重启将重新加载所有模型和摄像头配置</p>

                        <div style="font-size: 13px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin: 28px 0 16px;">实时日志</div>
                        <div class="glass-card logs-panel">
                            <div v-for="(log, idx) in logs" :key="idx" class="log-entry">
                                <span class="log-time">{{ log.time }}</span>
                                <span :style="{ color: log.level === 'error' ? 'var(--danger)' : log.level === 'warning' ? 'var(--warning)' : 'inherit' }">{{ log.message }}</span>
                            </div>
                            <div v-if="!logs.length" style="color: var(--text-muted); padding: 8px 0;">暂无日志</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Camera Modal -->
        <div v-if="cameraDialog" class="gc-modal-overlay" @click.self="cameraDialog = null">
            <div class="gc-modal-box">
                <div class="gc-modal-header">
                    <h3 style="font-size: 16px; font-weight: 700;">{{ cameraDialog._existing ? '编辑摄像头' : '添加摄像头' }}</h3>
                    <button class="clay-button ghost" @click="cameraDialog = null">&times;</button>
                </div>
                <div class="gc-modal-body">
                    <div class="gc-form-grid">
                        <div class="gc-form-field"><label>摄像头 ID</label><input class="clay-input" v-model="cameraDialog.camera_id" :disabled="cameraDialog._existing" /></div>
                        <div class="gc-form-field"><label>名称</label><input class="clay-input" v-model="cameraDialog.name" /></div>
                        <div class="gc-form-field" style="grid-column: span 2;">
                            <label>源地址</label>
                            <input class="clay-input" v-model="cameraDialog.source" placeholder="rtsp://... 或 /path/to/video.mp4" />
                            <input type="file" accept="video/*" style="display: none;" ref="videoFileInput" @change="onVideoFileSelected" />
                            <button class="clay-button" style="margin-top: 8px;" @click="$refs.videoFileInput.click()">上传本地视频</button>
                            <span v-if="uploading" style="font-size: 12px; color: var(--text-secondary); margin-left: 8px;">上传中...</span>
                        </div>
                        <div class="gc-form-field">
                            <label>源类型</label>
                            <select class="clay-select" v-model="cameraDialog.source_type">
                                <option value="auto">自动</option>
                                <option value="rtsp">RTSP</option>
                                <option value="video">视频文件</option>
                                <option value="camera">摄像头</option>
                            </select>
                        </div>
                        <div class="gc-form-field"><label>宽</label><input type="number" class="clay-input" v-model.number="cameraDialog.width" /></div>
                        <div class="gc-form-field"><label>高</label><input type="number" class="clay-input" v-model.number="cameraDialog.height" /></div>
                        <div class="gc-form-field">
                            <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                                <input type="checkbox" v-model="cameraDialog.enabled" /> 启用摄像头
                            </label>
                        </div>
                    </div>

                    <div style="font-size: 13px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin: 20px 0 12px;">检测类型配置</div>
                    <div class="gc-table-wrap" style="overflow-x: auto;">
                        <table class="gc-table">
                            <thead>
                                <tr>
                                    <th>类型</th>
                                    <th>启用</th>
                                    <th>间隔</th>
                                    <th>阈值</th>
                                    <th>连续</th>
                                    <th>冷却</th>
                                    <th>VLM</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr v-for="t in DETECTION_TYPES" :key="t.key">
                                    <td :style="{ color: t.color, fontWeight: 600 }">{{ t.label }}</td>
                                    <td><input type="checkbox" v-model="cameraDialog.detection_types[t.key].enabled" /></td>
                                    <td><input type="number" class="clay-input" v-model.number="cameraDialog.detection_types[t.key].interval" style="width: 70px; padding: 6px 8px;" /></td>
                                    <td><input type="number" step="0.1" class="clay-input" v-model.number="cameraDialog.detection_types[t.key].threshold" style="width: 70px; padding: 6px 8px;" /></td>
                                    <td><input type="number" class="clay-input" v-model.number="cameraDialog.detection_types[t.key].consecutive_required" style="width: 70px; padding: 6px 8px;" /></td>
                                    <td><input type="number" class="clay-input" v-model.number="cameraDialog.detection_types[t.key].cooldown" min="0" style="width: 70px; padding: 6px 8px;" /></td>
                                    <td><input type="checkbox" v-model="cameraDialog.detection_types[t.key].use_vlm" /></td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="gc-modal-footer">
                    <button class="clay-button" @click="cameraDialog = null">取消</button>
                    <button class="clay-button" @click="resetCameraTypes">恢复默认</button>
                    <button class="clay-button primary" @click="saveCamera">保存</button>
                </div>
            </div>
        </div>

        <!-- Batch Config Modal -->
        <div v-if="batchDialog" class="gc-modal-overlay" @click.self="batchDialog = false">
            <div class="gc-modal-box">
                <div class="gc-modal-header">
                    <h3 style="font-size: 16px; font-weight: 700;">批量配置 ({{ selectedCameras.length }} 个摄像头)</h3>
                    <button class="clay-button ghost" @click="batchDialog = false">&times;</button>
                </div>
                <div class="gc-modal-body">
                    <div class="gc-table-wrap" style="overflow-x: auto;">
                        <table class="gc-table">
                            <thead>
                                <tr><th>类型</th><th>启用</th><th>间隔</th><th>阈值</th><th>连续</th><th>冷却</th><th>VLM</th></tr>
                            </thead>
                            <tbody>
                                <tr v-for="t in DETECTION_TYPES" :key="t.key">
                                    <td :style="{ color: t.color, fontWeight: 600 }">{{ t.label }}</td>
                                    <td><input type="checkbox" v-model="batchTypes[t.key].enabled" /></td>
                                    <td><input type="number" class="clay-input" v-model.number="batchTypes[t.key].interval" style="width: 70px;" /></td>
                                    <td><input type="number" step="0.1" class="clay-input" v-model.number="batchTypes[t.key].threshold" style="width: 70px;" /></td>
                                    <td><input type="number" class="clay-input" v-model.number="batchTypes[t.key].consecutive_required" style="width: 70px;" /></td>
                                    <td><input type="number" class="clay-input" v-model.number="batchTypes[t.key].cooldown" min="0" style="width: 70px;" /></td>
                                    <td><input type="checkbox" v-model="batchTypes[t.key].use_vlm" /></td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="gc-modal-footer">
                    <button class="clay-button" @click="batchDialog = false">取消</button>
                    <button class="clay-button" @click="batchTypes = JSON.parse(JSON.stringify(settings.default_detection_types || defaultDetectionTypes()))">恢复默认</button>
                    <button class="clay-button primary" @click="saveBatchConfig">应用</button>
                </div>
            </div>
        </div>

        <div v-if="toast" :class="['toast', toast.type]">{{ toast.message }}</div>
    </div>

    <script>
        const { createApp, ref, onMounted, onUnmounted } = Vue;

        createApp({
            setup() {
                const tab = ref('cameras');
                const cameras = ref([]);
                const settings = ref({});
                const models = ref([]);
                const detectorInfo = ref({});
                const selectedCameras = ref([]);
                const cameraDialog = ref(null);
                const batchDialog = ref(false);
                const batchTypes = ref(defaultDetectionTypes());
                const toast = ref(null);
                const uploading = ref(false);
                const logs = ref([]);
                const systemMode = ref('');
                let logTimer = null;

                function showToast(msg, type = 'success') {
                    toast.value = { message: msg, type };
                    setTimeout(() => toast.value = null, 2500);
                }

                async function loadCameras() {
                    try {
                        const data = await safeFetch('/cameras');
                        cameras.value = data.cameras || [];
                    } catch (e) { console.error('加载摄像头失败:', e); }
                }

                async function loadSettings() {
                    try {
                        const data = await safeFetch('/settings');
                        settings.value = data;
                        const defs = defaultDetectionTypes();
                        const merged = {};
                        for (const key of Object.keys(defs)) {
                            merged[key] = { ...defs[key], ...(data.default_detection_types?.[key] || {}) };
                        }
                        settings.value.default_detection_types = merged;
                        batchTypes.value = JSON.parse(JSON.stringify(merged));
                        if (settings.value.use_gpu_scheduler === undefined) settings.value.use_gpu_scheduler = false;
                        if (settings.value.gpu_scheduler_num_queues === undefined) settings.value.gpu_scheduler_num_queues = 0;
                        if (settings.value.gpu_scheduler_interval === undefined) settings.value.gpu_scheduler_interval = 0.5;
                        if (settings.value.gpu_scheduler_half === undefined) settings.value.gpu_scheduler_half = false;
                        if (settings.value.emergency_cleanup_ratio === undefined) settings.value.emergency_cleanup_ratio = 0.5;
                    } catch (e) { console.error('加载设置失败:', e); }
                }

                async function loadModels() {
                    try {
                        const data = await safeFetch('/detector/models');
                        models.value = data.models || [];
                    } catch (e) { console.error('加载模型失败:', e); }
                    try {
                        const data = await safeFetch('/detector/status');
                        if (!data.error) detectorInfo.value = { mode: data.mode || 'CPU', npu_cores: data.npu_cores || 0 };
                    } catch (e) {}
                    try {
                        const data = await safeFetch('/status');
                        if (data.decoder_backends) {
                            const backends = Object.values(data.decoder_backends);
                            if (backends.length === 1) detectorInfo.value.decoder_backend = backends[0];
                            else if (backends.length > 1) {
                                const gpu = backends.filter(b => b === 'gpu').length;
                                const cpu = backends.filter(b => b === 'cpu').length;
                                detectorInfo.value.decoder_backend = `GPU:${gpu} CPU:${cpu}`;
                            }
                        }
                        if (data.detector?.mode === 'gpu_scheduler') {
                            detectorInfo.value.mode = 'GPU调度器';
                            detectorInfo.value.gpu_queues = data.detector.queues;
                        }
                    } catch (e) {}
                }

                async function saveVlmSettings() {
                    try {
                        const payload = {
                            vlm_max_concurrent: settings.value.vlm_max_concurrent,
                            vlm_inspection_interval: settings.value.vlm_inspection_interval
                        };
                        const res = await fetch('/settings', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(payload)
                        });
                        showToast(res.ok ? '保存成功' : '保存失败', res.ok ? 'success' : 'error');
                    } catch (e) { showToast('保存失败', 'error'); }
                }

                async function saveDefaultTypes() {
                    try {
                        const res = await fetch('/settings', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ default_detection_types: settings.value.default_detection_types })
                        });
                        showToast(res.ok ? '检测默认值保存成功' : '保存失败', res.ok ? 'success' : 'error');
                    } catch (e) { showToast('保存失败', 'error'); }
                }

                async function saveSystemSettings() {
                    try {
                        const payload = {
                            max_records: settings.value.max_records,
                            max_storage_mb: settings.value.max_storage_mb,
                            memory_threshold_percent: settings.value.memory_threshold_percent,
                            emergency_cleanup_ratio: settings.value.emergency_cleanup_ratio,
                            snapshot_quality: settings.value.snapshot_quality,
                            frame_quality: settings.value.frame_quality,
                            use_gpu_scheduler: settings.value.use_gpu_scheduler,
                            gpu_scheduler_num_queues: settings.value.gpu_scheduler_num_queues,
                            gpu_scheduler_interval: settings.value.gpu_scheduler_interval,
                            gpu_scheduler_half: settings.value.gpu_scheduler_half
                        };
                        const res = await fetch('/settings', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(payload)
                        });
                        showToast(res.ok ? '保存成功' : '保存失败', res.ok ? 'success' : 'error');
                    } catch (e) { showToast('保存失败', 'error'); }
                }

                function openCameraDialog(cam = null) {
                    const baseTypes = settings.value.default_detection_types
                        ? JSON.parse(JSON.stringify(settings.value.default_detection_types))
                        : defaultDetectionTypes();
                    if (cam) {
                        cameraDialog.value = {
                            ...JSON.parse(JSON.stringify(cam)),
                            detection_types: { ...baseTypes, ...(cam.detection_types || {}) },
                            _existing: true
                        };
                    } else {
                        cameraDialog.value = {
                            camera_id: '', name: '', source: '', source_type: 'auto',
                            width: 640, height: 480, enabled: true,
                            detection_types: baseTypes
                        };
                    }
                }

                function resetCameraTypes() {
                    if (!cameraDialog.value) return;
                    cameraDialog.value.detection_types = settings.value.default_detection_types
                        ? JSON.parse(JSON.stringify(settings.value.default_detection_types))
                        : defaultDetectionTypes();
                }

                async function saveCamera() {
                    const d = cameraDialog.value;
                    if (!d.camera_id || !d.source) { showToast('ID 和源地址必填', 'error'); return; }
                    try {
                        const url = d._existing ? `/cameras/${d.camera_id}/config` : '/cameras/add';
                        const payload = d._existing ? {
                            name: d.name, source: d.source, source_type: d.source_type,
                            width: d.width, height: d.height, enabled: d.enabled,
                            detection_types: d.detection_types
                        } : d;
                        const res = await fetch(url, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(payload)
                        });
                        if (res.ok) {
                            showToast('保存成功');
                            cameraDialog.value = null;
                            loadCameras();
                        } else showToast('保存失败', 'error');
                    } catch (e) { showToast('保存失败', 'error'); }
                }

                async function deleteCamera(id) {
                    if (!confirm('确定删除摄像头 ' + id + '？')) return;
                    try {
                        const res = await fetch('/cameras/' + id, { method: 'DELETE' });
                        showToast(res.ok ? '删除成功' : '删除失败', res.ok ? 'success' : 'error');
                        if (res.ok) loadCameras();
                    } catch (e) { showToast('删除失败', 'error'); }
                }

                function openBatchDialog() {
                    batchTypes.value = settings.value.default_detection_types
                        ? JSON.parse(JSON.stringify(settings.value.default_detection_types))
                        : defaultDetectionTypes();
                    batchDialog.value = true;
                }

                async function saveBatchConfig() {
                    try {
                        const res = await fetch('/cameras/batch-config', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ camera_ids: selectedCameras.value, detection_types: batchTypes.value })
                        });
                        showToast(res.ok ? '批量配置成功' : '批量配置失败', res.ok ? 'success' : 'error');
                        if (res.ok) {
                            batchDialog.value = false;
                            selectedCameras.value = [];
                            loadCameras();
                        }
                    } catch (e) { showToast('批量配置失败', 'error'); }
                }

                function toggleSelectAll(e) {
                    selectedCameras.value = e.target.checked ? cameras.value.map(c => c.camera_id) : [];
                }

                async function onVideoFileSelected(event) {
                    const file = event.target.files[0];
                    if (!file) return;
                    uploading.value = true;
                    try {
                        const formData = new FormData();
                        formData.append('file', file);
                        const res = await fetch('/upload/video', { method: 'POST', body: formData });
                        const data = await res.json();
                        if (data.success) {
                            cameraDialog.value.source = data.path;
                            cameraDialog.value.source_type = 'video';
                            showToast('上传成功: ' + data.filename);
                        } else showToast(data.error || '上传失败', 'error');
                    } catch (e) { showToast('上传失败', 'error'); }
                    finally { uploading.value = false; event.target.value = ''; }
                }

                async function restartSystem() {
                    if (!confirm('确定重启检测服务？')) return;
                    try {
                        const res = await fetch('/system/restart', { method: 'POST' });
                        showToast(res.ok ? '重启指令已发送' : '重启失败', res.ok ? 'success' : 'error');
                    } catch (e) { showToast('重启失败', 'error'); }
                }

                async function fetchSystemMode() {
                    try {
                        const data = await safeFetch('/system/mode');
                        const dev = (data.device || 'cpu').toUpperCase();
                        systemMode.value = dev === 'NPU' && data.npu_cores > 0 ? `${dev} (${data.npu_cores}核)` : dev;
                    } catch (e) {}
                }

                async function fetchLogs() {
                    try {
                        const data = await safeFetch('/status');
                        if (data.logs) logs.value = data.logs.slice(-50);
                    } catch (e) {}
                }

                onMounted(() => {
                    loadCameras();
                    loadSettings();
                    loadModels();
                    fetchSystemMode();
                    fetchLogs();
                    logTimer = setInterval(fetchLogs, 3000);
                });

                onUnmounted(() => {
                    if (logTimer) clearInterval(logTimer);
                });

                return {
                    tab, cameras, settings, models, detectorInfo, selectedCameras,
                    cameraDialog, batchDialog, batchTypes, toast, uploading, logs, systemMode,
                    DETECTION_TYPES,
                    loadCameras, loadSettings, loadModels,
                    saveVlmSettings, saveDefaultTypes, saveSystemSettings,
                    openCameraDialog, resetCameraTypes, saveCamera, deleteCamera,
                    openBatchDialog, saveBatchConfig, toggleSelectAll,
                    onVideoFileSelected, restartSystem
                };
            }
        }).mount('#app');
    </script>
</body>
</html>
```

- [ ] **Step 2: Verify HTML syntax**

Run:
```bash
python3 -c "from html.parser import HTMLParser; HTMLParser().feed(open('frontend/safety_detection/settings.html', encoding='utf-8').read()); print('HTML OK')"
```

Expected: `HTML OK`

- [ ] **Step 3: Commit**

```bash
git add frontend/safety_detection/settings.html
git commit -m "feat: rewrite settings page in glass-clay style with 3 tabs

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Update Backend Routes

**Files:**
- Modify: `backend/main_multi.py`
- Modify: `backend/performance_storage.py`
- Test: `python3 -m py_compile backend/main_multi.py backend/performance_storage.py`

**Interfaces:**
- Consumes: existing FastAPI app, HTMLResponse, storage pagination
- Produces: `/monitor` route; removes `/multi` and `/hud`; date-filtered `/alerts`

- [ ] **Step 1: Add /monitor route and remove /multi /hud**

In `backend/main_multi.py`, find the `/multi` route around line 543 and the `/hud` route around line 552. Replace both with the following single route:

```python
@app.get("/monitor")
async def monitor_view():
    """Glass-clay 风格监控中心"""
    fp = Path(__file__).parent.parent / "frontend" / "safety_detection" / "monitor.html"
    if fp.exists():
        return HTMLResponse(fp.read_text(encoding="utf-8"))
    return {"error": "Monitor page not found"}
```

Remove the existing `/multi` and `/hud` route functions entirely.

- [ ] **Step 2: Update storage.py to support date filtering**

In `backend/performance_storage.py`, find `get_records_paginated` around line 170. Add date filtering logic after the existing filters:

```python
    if date_from:
        filtered = [r for r in filtered if r.get("time", "") >= date_from]
    if date_to:
        # Include the entire day up to 23:59:59
        filtered = [r for r in filtered if r.get("time", "") <= date_to + "T23:59:59"]
```

Insert these lines after the status filter (around line 203) and before the `total = len(filtered)` line.

- [ ] **Step 3: Update /alerts endpoint to accept date params**

In `backend/main_multi.py`, find the `/alerts` route around line 865. Update the function signature and call:

```python
@app.get("/alerts")
async def get_alerts(
    camera_id: Optional[str] = None,
    level: Optional[str] = None,
    type: Optional[str] = None,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
):
    """分页获取告警记录，支持过滤"""
    records, total = storage.get_records_paginated(
        page=page, page_size=page_size,
        camera_id=camera_id, level=level, dtype=type, status=status,
        date_from=date_from, date_to=date_to,
    )
    return {"records": records, "total": total, "page": page, "page_size": page_size}
```

- [ ] **Step 4: Verify Python syntax**

Run:
```bash
python3 -m py_compile backend/main_multi.py backend/performance_storage.py
```

Expected: no output (success)

- [ ] **Step 5: Commit**

```bash
git add backend/main_multi.py backend/performance_storage.py
git commit -m "feat: add /monitor route, remove /multi and /hud, add date filtering to /alerts

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Delete Old Pages

**Files:**
- Delete: `frontend/safety_detection/multi.html`
- Delete: `frontend/safety_detection/hud.html`
- Test: verify files are gone

**Interfaces:**
- Produces: removal of obsolete entry points

- [ ] **Step 1: Delete old HTML files**

Run:
```bash
git rm frontend/safety_detection/multi.html frontend/safety_detection/hud.html
```

Expected: both files staged for deletion

- [ ] **Step 2: Verify deletion**

Run:
```bash
ls frontend/safety_detection/multi.html frontend/safety_detection/hud.html 2>&1
```

Expected: both paths report "No such file or directory"

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: remove obsolete multi.html and hud.html

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: Integration Verification

**Files:**
- Test via running server

**Interfaces:**
- Consumes: all newly created/modified files
- Produces: verification that the redesign works end-to-end

- [ ] **Step 1: Start backend server**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate py312
python -m uvicorn backend.main_multi:app --host 0.0.0.0 --port 8000 --reload
```

Wait for "Uvicorn running on http://0.0.0.0:8000".

- [ ] **Step 2: Verify routes**

Open in browser:
- `http://localhost:8000/monitor` → should load glass-clay monitor page
- `http://localhost:8000/records.html` → should load glass-clay records page
- `http://localhost:8000/settings.html` → should load glass-clay settings page
- `http://localhost:8000/multi` → should return 404
- `http://localhost:8000/hud` → should return 404

- [ ] **Step 3: Monitor page checks**

- Page loads without console errors
- Top header shows SENTRY, online count, alert count, UTC time
- Main video area loads `/cameras/{id}/stream`
- Right sidebar shows camera list (no video streams)
- Clicking camera switches main video
- URL updates to `/monitor?camera=xxx`
- Bottom alert bar shows recent alerts

- [ ] **Step 4: Records page checks**

- Stats cards load
- Filters work
- Table loads records
- Pagination works
- Clicking row opens detail modal
- Snapshot/frame player loads
- Confirm/ignore buttons update status

- [ ] **Step 5: Settings page checks**

- 3 tabs switch correctly
- Cameras tab loads list
- Add/edit/delete camera works
- Detection config tab loads defaults
- System tab loads models and logs
- Save buttons work

- [ ] **Step 6: Stop server**

Press Ctrl+C to stop uvicorn.

- [ ] **Step 7: Final status check**

Run:
```bash
git status
```

Expected: clean working tree.

---

## Self-Review

### Spec Coverage

| Spec Requirement | Implementing Task |
|------------------|-------------------|
| 新增 `/monitor` 入口 | Task 6 |
| 删除 `/multi`、`/hud` | Task 6 + Task 7 |
| glass-clay 设计系统 | Task 1 |
| monitor 布局与行为 | Task 3 |
| records 统计/筛选/表格/详情 | Task 4 |
| settings 3 个 Tab | Task 5 |
| 右侧摄像头列表不推流 | Task 3 |
| URL 同步 camera | Task 3 |
| prefers-reduced-motion | Task 1 CSS |
| 复用现有 API | All tasks |

### Placeholder Scan

- No TBD/TODO
- No vague instructions
- All file paths exact
- All code blocks contain actual code

### Type Consistency

- `DETECTION_TYPES` constant reused across shared.js, monitor, records, settings
- `defaultDetectionTypes()` function reused in shared.js and settings
- API endpoints match existing backend routes

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-30-glass-clay-ui.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach would you like to use?
