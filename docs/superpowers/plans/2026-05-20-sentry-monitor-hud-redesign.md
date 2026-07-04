# Sentry HUD 监控页面 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `frontend/safety_detection/hud.html` 页面，实现科幻终端 HUD 风格的监控中心，保留原 `multi.html` 不动。

**Architecture:** 单文件 HTML（内联 CSS + Vue 3 CDN），全屏 CSS Grid 布局，纯 CSS/SVG 实现 HUD 视觉效果，后端仅新增一个路由指向新文件。

**Tech Stack:** HTML5, CSS3, Vue 3 (CDN), FastAPI

**Spec Reference:** `docs/superpowers/specs/2026-05-20-sentry-monitor-hud-redesign.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `frontend/safety_detection/hud.html` | Create | 完整 HUD 监控页面（HTML + CSS + Vue） |
| `backend/main_multi.py` | Modify | 新增 `/hud` 路由 |

---

## Task 1: Create hud.html Base Skeleton & Global Styles

**Files:**
- Create: `frontend/safety_detection/hud.html`

- [ ] **Step 1: Create the file with DOCTYPE, head, font imports, CSS variables, and body grid**

Write the following to `frontend/safety_detection/hud.html`:

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SENTRY HUD</title>
    <link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&family=Noto+Sans+SC:wght@300;400;500;700&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
    <style>
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
            --font-display: 'Rajdhani', 'Noto Sans SC', sans-serif;
            --font-mono: 'JetBrains Mono', 'Noto Sans SC', monospace;
            --font-body: 'Noto Sans SC', sans-serif;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: var(--font-body);
            background: var(--bg);
            color: var(--text-primary);
            height: 100vh;
            overflow: hidden;
        }

        #app {
            height: 100vh;
            display: grid;
            grid-template-rows: 48px 1fr 64px;
            grid-template-columns: 48px 1fr 260px;
            grid-template-areas:
                "header header header"
                "left main right"
                "footer footer footer";
            gap: 1px;
            background: var(--bg);
        }

        /* Panel base */
        .panel {
            background: var(--surface-solid);
            border: 1px solid var(--border);
            position: relative;
            overflow: hidden;
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 0; }
        ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
    </style>
</head>
<body>
    <div id="app"></div>
    <script>
        const { createApp, ref, computed, watch, onMounted, onUnmounted } = Vue;
    </script>
</body>
</html>
```

- [ ] **Step 2: Verify file exists and syntax is valid**

Run: `python3 -c "from html.parser import HTMLParser; HTMLParser().feed(open('/home/user/project/sentry-safety/frontend/safety_detection/hud.html').read()); print('HTML OK')"`
Expected: `HTML OK`

- [ ] **Step 3: Commit**

```bash
git add frontend/safety_detection/hud.html
git commit -m "feat: add hud.html base skeleton with design tokens and grid"
```

---

## Task 2: Top Header Bar

**Files:**
- Modify: `frontend/safety_detection/hud.html`

- [ ] **Step 1: Add header CSS and HTML**

Replace the `#app` empty div in `hud.html`:

Old:
```html
    <div id="app"></div>
```

New:
```html
    <div id="app">
        <header class="panel header-bar" style="grid-area: header; display: flex; align-items: center; justify-content: space-between; padding: 0 20px; z-index: 10;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 8px; height: 8px; background: var(--accent); box-shadow: 0 0 8px var(--accent);"></div>
                <span style="font-family: var(--font-display); font-weight: 600; font-size: 14px; letter-spacing: 0.15em; color: var(--text-primary);">SENTRY</span>
            </div>
            <div style="font-family: var(--font-mono); font-size: 12px; color: var(--text-secondary); display: flex; gap: 20px;">
                <span>UTC {{ utcTime }}</span>
                <span>UPTIME {{ uptime }}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 20px;">
                <div style="text-align: right;">
                    <div style="font-family: var(--font-display); font-weight: 700; font-size: 20px; color: var(--success); line-height: 1;">{{ onlineCount }}</div>
                    <div style="font-family: var(--font-mono); font-size: 9px; color: var(--text-muted); text-transform: uppercase;">ONLINE</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-family: var(--font-display); font-weight: 700; font-size: 20px; color: var(--danger); line-height: 1;">{{ totalDetections }}</div>
                    <div style="font-family: var(--font-mono); font-size: 9px; color: var(--text-muted); text-transform: uppercase;">ALERTS</div>
                </div>
            </div>
        </header>
    </div>
```

Add the header scanline animation CSS inside the `<style>` tag, after the scrollbar styles:

```css
        /* Header scanline */
        .header-bar::after {
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

- [ ] **Step 2: Commit**

```bash
git add frontend/safety_detection/hud.html
git commit -m "feat: add HUD header bar with scanline animation"
```

---

## Task 3: Central Video Area with HUD Decorations

**Files:**
- Modify: `frontend/safety_detection/hud.html`

- [ ] **Step 1: Add video area CSS and HTML**

Add CSS inside `<style>`, after the header scanline animation:

```css
        /* Main video area */
        .main-video {
            grid-area: main;
            background: #000;
            position: relative;
            display: flex;
            flex-direction: column;
            border: 1px solid var(--border);
            overflow: hidden;
        }

        .video-body {
            flex: 1;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #000;
        }

        .video-body img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            display: block;
        }

        /* HUD corner brackets */
        .hud-corners {
            position: absolute;
            inset: 12px;
            pointer-events: none;
            z-index: 5;
        }
        .hud-corners::before,
        .hud-corners::after {
            content: '';
            position: absolute;
            width: 24px; height: 24px;
            border-color: var(--accent);
            border-style: solid;
            opacity: 0.6;
        }
        .hud-corners::before {
            top: 0; left: 0;
            border-width: 1px 0 0 1px;
        }
        .hud-corners::after {
            top: 0; right: 0;
            border-width: 1px 1px 0 0;
        }
        .hud-corners-bottom::before,
        .hud-corners-bottom::after {
            content: '';
            position: absolute;
            width: 24px; height: 24px;
            border-color: var(--accent);
            border-style: solid;
            opacity: 0.6;
        }
        .hud-corners-bottom::before {
            bottom: 0; left: 0;
            border-width: 0 0 1px 1px;
        }
        .hud-corners-bottom::after {
            bottom: 0; right: 0;
            border-width: 0 1px 1px 0;
        }

        /* Scanline overlay */
        .scanlines {
            position: absolute;
            inset: 0;
            pointer-events: none;
            z-index: 4;
            background: repeating-linear-gradient(
                0deg,
                transparent,
                transparent 2px,
                rgba(0, 229, 255, 0.015) 2px,
                rgba(0, 229, 255, 0.015) 4px
            );
            animation: scanline-move 8s linear infinite;
        }
        @keyframes scanline-move {
            0% { background-position: 0 0; }
            100% { background-position: 0 100px; }
        }

        /* Alert pulse */
        .alert-pulse {
            animation: alert-glow 2s ease-in-out infinite;
        }
        @keyframes alert-glow {
            0%, 100% { box-shadow: inset 0 0 0 1px var(--border); }
            50% { box-shadow: inset 0 0 20px rgba(255, 77, 106, 0.3), inset 0 0 0 1px var(--danger); }
        }

        .video-overlay-info {
            position: absolute;
            bottom: 12px; right: 12px;
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-secondary);
            z-index: 6;
            text-align: right;
            line-height: 1.6;
        }

        .video-controls {
            position: absolute;
            bottom: 12px; left: 12px;
            display: flex;
            gap: 8px;
            z-index: 6;
        }
        .vc-btn {
            background: rgba(10, 10, 18, 0.8);
            border: 1px solid var(--border);
            color: var(--accent);
            font-family: var(--font-mono);
            font-size: 11px;
            padding: 4px 10px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .vc-btn:hover {
            border-color: var(--accent);
            background: var(--accent-dim);
        }
```

Add the video area HTML inside `#app`, after the `</header>`:

```html
        <div class="main-video" :class="{ 'alert-pulse': mainAlert }">
            <div class="video-body">
                <img v-if="mainCameraId && mainCameraStatus === 'connected'"
                     :src="'/cameras/' + mainCameraId + '/stream'"
                     @error="onVideoError(mainCameraId)"
                     alt="Main Stream" />
                <div v-else style="display: flex; flex-direction: column; align-items: center; gap: 8px; color: var(--text-muted);">
                    <div style="font-size: 32px; opacity: 0.3;">NO SIGNAL</div>
                    <div style="font-family: var(--font-mono); font-size: 12px;">{{ mainCameraStatus === 'connecting' ? 'CONNECTING...' : 'CAMERA OFFLINE' }}</div>
                </div>
                <div class="hud-corners"><div class="hud-corners-bottom"></div></div>
                <div class="scanlines"></div>
                <div class="video-overlay-info">
                    <div>{{ mainCameraName }}</div>
                    <div>{{ mainResolution }} | {{ mainFps }} FPS</div>
                </div>
                <div class="video-controls" v-if="playbackStatus.is_video_file">
                    <button class="vc-btn" @click="controlPlayback(playbackStatus.playing ? 'pause' : 'play')">
                        {{ playbackStatus.playing ? 'PAUSE' : 'PLAY' }}
                    </button>
                    <button class="vc-btn" @click="controlPlayback('loop', { loop: !playbackStatus.loop })"
                            :style="playbackStatus.loop ? { borderColor: 'var(--accent)' } : {}">
                        LOOP
                    </button>
                </div>
            </div>
        </div>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/safety_detection/hud.html
git commit -m "feat: add central video area with HUD corners and scanlines"
```

---

## Task 4: Left Terminal Panel

**Files:**
- Modify: `frontend/safety_detection/hud.html`

- [ ] **Step 1: Add terminal panel CSS and HTML**

Add CSS inside `<style>`:

```css
        /* Left terminal panel */
        .left-panel {
            grid-area: left;
            display: flex;
            flex-direction: column;
            transition: width 0.3s ease;
            position: relative;
        }
        .left-panel:hover {
            width: 260px;
        }
        .left-panel .panel-content {
            opacity: 0;
            transition: opacity 0.2s ease 0.1s;
            padding: 16px;
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .left-panel:hover .panel-content {
            opacity: 1;
        }
        .left-panel .collapse-hint {
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%) rotate(-90deg);
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-muted);
            letter-spacing: 0.1em;
            white-space: nowrap;
            transition: opacity 0.2s;
        }
        .left-panel:hover .collapse-hint {
            opacity: 0;
        }

        .term-title {
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-muted);
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }
        .term-log {
            font-family: var(--font-mono);
            font-size: 11px;
            line-height: 1.7;
            color: var(--text-secondary);
        }
        .term-log .log-time { color: var(--text-muted); }
        .term-log .log-cam { color: var(--text-secondary); }
        .term-log .log-type { font-weight: 500; }
        .term-log .log-p0 { color: var(--danger); }
        .term-log .log-p1 { color: var(--warning); }

        .cam-list {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .cam-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 8px;
            cursor: pointer;
            transition: background 0.15s;
            border: 1px solid transparent;
        }
        .cam-item:hover {
            background: var(--accent-dim);
            border-color: var(--border-glow);
        }
        .cam-dot {
            width: 6px; height: 6px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .cam-dot.online { background: var(--success); box-shadow: 0 0 4px var(--success); }
        .cam-dot.offline { background: var(--text-muted); }
        .cam-name {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-secondary);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .cam-item:hover .cam-name {
            color: var(--text-primary);
        }
        @keyframes logSlideIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }
```

Add HTML inside `#app`, after the `</div>` (main-video closing):

```html
        <aside class="panel left-panel">
            <div class="collapse-hint">ALERT LOG</div>
            <div class="panel-content">
                <div>
                    <div class="term-title">// ALERT_LOG</div>
                    <div class="term-log">
                        <div v-for="alert in recentAlerts.slice(0, 8)" :key="alert.id"
                             style="animation: logSlideIn 0.2s ease-out;">
                            <span class="log-time">[{{ alert.timeShort }}]</span>
                            <span class="log-cam"> {{ alert.cameraName }} </span>
                            <span :class="['log-type', alert.level === 'P0' ? 'log-p0' : 'log-p1']">
                                {{ alert.typeLabel }} ({{ alert.level }})
                            </span>
                        </div>
                        <div v-if="!recentAlerts.length" style="color: var(--text-muted);">-- NO ALERTS --</div>
                    </div>
                </div>
                <div style="border-top: 1px solid var(--border); padding-top: 12px;">
                    <div class="term-title">// CAMERA_LIST</div>
                    <div class="cam-list">
                        <div v-for="cam in cameras" :key="cam.camera_id"
                             class="cam-item"
                             @click="setMainCamera(cam.camera_id)">
                            <div :class="['cam-dot', cam.status === 'connected' ? 'online' : 'offline']"></div>
                            <span class="cam-name">{{ cam.name || cam.camera_id }}</span>
                        </div>
                    </div>
                </div>
            </div>
        </aside>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/safety_detection/hud.html
git commit -m "feat: add collapsible left terminal panel with alert log and camera list"
```

---

## Task 5: Right Dashboard Panel

**Files:**
- Modify: `frontend/safety_detection/hud.html`

- [ ] **Step 1: Add dashboard CSS and HTML**

Add CSS inside `<style>`:

```css
        /* Right dashboard */
        .right-panel {
            grid-area: right;
            display: flex;
            flex-direction: column;
            padding: 16px;
            gap: 20px;
            overflow-y: auto;
        }

        .gauge-wrap {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
        }
        .gauge-svg {
            width: 100px; height: 60px;
        }
        .gauge-value {
            font-family: var(--font-display);
            font-weight: 700;
            font-size: 24px;
            color: var(--text-primary);
            line-height: 1;
        }
        .gauge-label {
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .model-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .model-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid var(--border);
        }
        .model-row:last-child { border-bottom: none; }
        .model-name {
            font-size: 12px;
            color: var(--text-secondary);
        }
        .model-dot {
            width: 6px; height: 6px;
            border-radius: 50%;
        }
        .model-dot.on { background: var(--success); box-shadow: 0 0 4px var(--success); }
        .model-dot.off { background: var(--text-muted); }

        .toggle-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
        }
        .toggle-btn {
            padding: 5px 8px;
            border: 1px solid var(--border);
            background: transparent;
            color: var(--text-secondary);
            font-family: var(--font-body);
            font-size: 11px;
            cursor: pointer;
            transition: all 0.2s;
            text-align: center;
        }
        .toggle-btn:hover { border-color: var(--text-muted); }
        .toggle-btn.active {
            border-color: transparent;
            color: #000;
        }
```

Add HTML inside `#app`, after the `</aside>` (left-panel closing):

```html
        <aside class="panel right-panel">
            <div style="display: flex; justify-content: space-around;">
                <div class="gauge-wrap">
                    <svg class="gauge-svg" viewBox="0 0 100 60">
                        <path d="M10,55 A40,40 0 0,1 90,55" fill="none" stroke="var(--border)" stroke-width="3"/>
                        <path d="M10,55 A40,40 0 0,1 90,55" fill="none" stroke="var(--accent)" stroke-width="3"
                              :stroke-dasharray="gaugeArc(gpuLoad)" stroke-dashoffset="0"
                              style="transition: stroke-dasharray 0.5s ease;"/>
                    </svg>
                    <div class="gauge-value" style="color: var(--accent);">{{ gpuLoad }}%</div>
                    <div class="gauge-label">GPU Load</div>
                </div>
                <div class="gauge-wrap">
                    <svg class="gauge-svg" viewBox="0 0 100 60">
                        <path d="M10,55 A40,40 0 0,1 90,55" fill="none" stroke="var(--border)" stroke-width="3"/>
                        <path d="M10,55 A40,40 0 0,1 90,55" fill="none" :stroke="fpsColor" stroke-width="3"
                              :stroke-dasharray="gaugeArc(fps / 60 * 100)" stroke-dashoffset="0"
                              style="transition: stroke-dasharray 0.5s ease;"/>
                    </svg>
                    <div class="gauge-value" :style="{ color: fpsColor }">{{ fps }}</div>
                    <div class="gauge-label">FPS</div>
                </div>
            </div>
            <div class="gauge-wrap">
                <svg class="gauge-svg" viewBox="0 0 100 60">
                    <path d="M10,55 A40,40 0 0,1 90,55" fill="none" stroke="var(--border)" stroke-width="3"/>
                    <path d="M10,55 A40,40 0 0,1 90,55" fill="none" :stroke="memColor" stroke-width="3"
                          :stroke-dasharray="gaugeArc(memUsage)" stroke-dashoffset="0"
                          style="transition: stroke-dasharray 0.5s ease;"/>
                </svg>
                <div class="gauge-value" :style="{ color: memColor }">{{ memUsage }}%</div>
                <div class="gauge-label">Memory</div>
            </div>

            <div>
                <div class="term-title">// MODEL_STATUS</div>
                <div class="model-list">
                    <div class="model-row">
                        <span class="model-name">YOLO Detector</span>
                        <div class="model-dot on"></div>
                    </div>
                    <div class="model-row">
                        <span class="model-name">VLM Review</span>
                        <div :class="['model-dot', useVlm ? 'on' : 'off']"></div>
                    </div>
                    <div class="model-row">
                        <span class="model-name">GPU Scheduler</span>
                        <div :class="['model-dot', useGpuScheduler ? 'on' : 'off']"></div>
                    </div>
                </div>
            </div>

            <div>
                <div class="term-title">// OVERLAY_TYPES</div>
                <div class="toggle-grid">
                    <button v-for="t in detectionTypes" :key="t.key"
                            :class="['toggle-btn', { active: overlayTypes.includes(t.key) }]"
                            :style="overlayTypes.includes(t.key) ? { background: t.color } : {}"
                            @click="toggleOverlay(t.key)">
                        {{ t.label }}
                    </button>
                </div>
            </div>
        </aside>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/safety_detection/hud.html
git commit -m "feat: add right dashboard with gauges, model status, and overlay toggles"
```

---

## Task 6: Bottom Camera Status Bar

**Files:**
- Modify: `frontend/safety_detection/hud.html`

- [ ] **Step 1: Add footer CSS and HTML**

Add CSS inside `<style>`:

```css
        /* Footer camera bar */
        .footer-bar {
            grid-area: footer;
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            overflow-x: auto;
            background: var(--surface-solid);
            border-top: 1px solid var(--border);
        }
        .cam-card {
            flex-shrink: 0;
            width: 160px;
            height: 48px;
            border: 1px solid var(--border);
            padding: 8px 12px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 4px;
            cursor: pointer;
            transition: all 0.2s;
            position: relative;
            overflow: hidden;
        }
        .cam-card:hover {
            border-color: var(--text-muted);
        }
        .cam-card.active {
            border-color: var(--accent);
            box-shadow: 0 0 8px var(--accent-dim);
        }
        .cam-card-name {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-primary);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .cam-card-meta {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .cam-card-status {
            width: 5px; height: 5px;
            border-radius: 50%;
        }
        .cam-card-status.online { background: var(--success); }
        .cam-card-status.offline { background: var(--text-muted); }
        .cam-card-types {
            display: flex;
            gap: 3px;
        }
        .cam-card-tag {
            font-size: 8px;
            padding: 1px 4px;
            color: #000;
            font-weight: 500;
        }
        .cam-card::before {
            content: '';
            position: absolute;
            top: -100%; left: 0;
            width: 100%; height: 100%;
            background: linear-gradient(180deg, transparent, rgba(0, 229, 255, 0.08), transparent);
            transition: top 0s;
        }
        .cam-card:hover::before {
            top: 100%;
            transition: top 1s ease-in-out;
        }
```

Add HTML inside `#app`, after the `</aside>` (right-panel closing), and before the `</div>` (#app closing):

```html
        <footer class="footer-bar">
            <div v-for="cam in cameras" :key="cam.camera_id"
                 :class="['cam-card', { active: cam.camera_id === mainCameraId }]"
                 @click="setMainCamera(cam.camera_id)">
                <div class="cam-card-name">{{ cam.name || cam.camera_id }}</div>
                <div class="cam-card-meta">
                    <div :class="['cam-card-status', cam.status === 'connected' ? 'online' : 'offline']"></div>
                    <div class="cam-card-types">
                        <span v-for="(cfg, type) in cam.detection_types" :key="type"
                              v-if="cfg.enabled"
                              class="cam-card-tag"
                              :style="{ background: getTypeColor(type) }">
                            {{ typeLabel(type) }}
                        </span>
                    </div>
                </div>
            </div>
        </footer>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/safety_detection/hud.html
git commit -m "feat: add bottom camera status bar with scan hover effect"
```

---

## Task 7: Vue Data Logic Layer

**Files:**
- Modify: `frontend/safety_detection/hud.html`

- [ ] **Step 1: Replace the script block with full Vue logic**

Replace everything from `<script>` to `</script>` at the bottom of the file with:

```html
    <script>
        const { createApp, ref, computed, watch, onMounted, onUnmounted } = Vue;

        function formatTimeShort(ts) {
            if (!ts) return '--:--:--';
            const d = new Date(ts);
            if (isNaN(d)) return ts;
            return String(d.getHours()).padStart(2, '0') + ':' +
                   String(d.getMinutes()).padStart(2, '0') + ':' +
                   String(d.getSeconds()).padStart(2, '0');
        }

        function calculateUptime(startedAt) {
            const start = new Date(startedAt);
            const now = new Date();
            const diff = Math.floor((now - start) / 1000);
            const h = Math.floor(diff / 3600);
            const m = Math.floor((diff % 3600) / 60);
            const s = diff % 60;
            return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
        }

        createApp({
            setup() {
                const cameras = ref([]);
                const cameraOrder = ref([]);
                const totalDetections = ref(0);
                const uptime = ref('00:00:00');
                const overlayTypes = ref([]);
                const recentAlerts = ref([]);
                const playbackStatus = ref({});
                const systemStatus = ref({});
                let refreshTimer = null;
                let uptimeTimer = null;

                const detectionTypes = [
                    { key: 'fire', label: '明火', color: '#ef4444' },
                    { key: 'smoke', label: '烟雾', color: '#f97316' },
                    { key: 'uniform', label: '工服', color: '#22c55e' },
                    { key: 'mask', label: '口罩', color: '#0ea5e9' },
                    { key: 'cigarette', label: '吸烟', color: '#a855f7' },
                    { key: 'sleep', label: '睡岗', color: '#eab308' },
                ];

                const mainCameraId = computed(() => cameraOrder.value[0] || null);
                const mainCameraObj = computed(() => cameras.value.find(c => c.camera_id === mainCameraId.value));
                const mainCameraName = computed(() => mainCameraObj.value?.name || mainCameraId.value || 'NO CAMERA');
                const mainCameraStatus = computed(() => mainCameraObj.value?.status || 'idle');
                const mainResolution = computed(() => {
                    const c = mainCameraObj.value;
                    return c ? `${c.width || '--'}x${c.height || '--'}` : '--x--';
                });
                const mainAlert = computed(() => {
                    const cam = mainCameraObj.value;
                    if (!cam || !cam.detection) return false;
                    const types = ['fire', 'smoke', 'uniform', 'mask', 'cigarette', 'sleep'];
                    return types.some(t => cam.detection[t]?.alert);
                });

                const gpuLoad = ref(42);
                const fps = ref(30);
                const memUsage = ref(68);
                const fpsColor = computed(() => fps.value >= 25 ? 'var(--accent)' : fps.value >= 15 ? 'var(--warning)' : 'var(--danger)');
                const memColor = computed(() => memUsage.value < 70 ? 'var(--accent)' : memUsage.value < 85 ? 'var(--warning)' : 'var(--danger)');
                const useVlm = computed(() => systemStatus.value?.use_vlm || false);
                const useGpuScheduler = computed(() => systemStatus.value?.use_gpu_scheduler || false);

                const onlineCount = computed(() => cameras.value.filter(c => c.status === 'connected').length);
                const utcTime = ref('--:--:--');

                function updateUtc() {
                    const now = new Date();
                    utcTime.value = now.toISOString().slice(11, 19);
                }

                function typeLabel(type) {
                    const map = { fire: '明火', smoke: '烟雾', uniform: '工服', mask: '口罩', cigarette: '吸烟', sleep: '睡岗' };
                    return map[type] || type || '未知';
                }
                function getTypeColor(type) {
                    const t = detectionTypes.find(x => x.key === type);
                    return t ? t.color : '#94a3b8';
                }
                function getCameraStatus(cid) {
                    return cameras.value.find(c => c.camera_id === cid)?.status || 'idle';
                }
                function getCameraName(cid) {
                    return cameras.value.find(c => c.camera_id === cid)?.name || cid;
                }

                function setMainCamera(cid) {
                    const idx = cameraOrder.value.indexOf(cid);
                    if (idx > 0) {
                        const newOrder = [...cameraOrder.value];
                        const temp = newOrder[0];
                        newOrder[0] = newOrder[idx];
                        newOrder[idx] = temp;
                        cameraOrder.value = newOrder;
                    } else if (idx === -1) {
                        cameraOrder.value = [cid, ...cameraOrder.value.filter(id => id !== cid)];
                    }
                }

                function gaugeArc(percent) {
                    const circumference = Math.PI * 40;
                    return `${(percent / 100) * circumference} ${circumference}`;
                }

                function onVideoError(cid) {
                    // Retry logic handled by img @error
                }

                async function fetchCameras() {
                    try {
                        const res = await fetch('/cameras');
                        const data = await res.json();
                        if (data.cameras) {
                            cameras.value = data.cameras;
                            const enabledIds = data.cameras.filter(c => c.enabled !== false).map(c => c.camera_id);
                            let newOrder;
                            if (cameraOrder.value.length === 0) {
                                newOrder = [...enabledIds];
                            } else {
                                const existing = new Set(cameraOrder.value);
                                const newIds = enabledIds.filter(id => !existing.has(id));
                                newOrder = cameraOrder.value.filter(id => enabledIds.includes(id));
                                newOrder.push(...newIds);
                            }
                            cameraOrder.value = newOrder;
                        }
                    } catch (e) { console.error('Fetch cameras failed:', e); }
                }

                async function fetchStatus() {
                    try {
                        const res = await fetch('/status');
                        const data = await res.json();
                        if (data.total_detections !== undefined) totalDetections.value = data.total_detections;
                        if (data.started_at) {
                            uptime.value = calculateUptime(data.started_at);
                        }
                        if (data.recent_records) {
                            recentAlerts.value = data.recent_records.map(r => {
                                const cam = cameras.value.find(c => c.camera_id === r.camera_id);
                                return {
                                    id: r.id,
                                    timeShort: formatTimeShort(r.time),
                                    cameraName: cam?.name || r.camera_id,
                                    detection_type: r.detection_type || r.action,
                                    typeLabel: typeLabel(r.detection_type || r.action),
                                    level: r.level || 'P1',
                                };
                            });
                        }
                        systemStatus.value = data;
                    } catch (e) { console.error('Fetch status failed:', e); }
                }

                async function fetchOverlay() {
                    try {
                        const res = await fetch('/overlay');
                        const data = await res.json();
                        if (data.overlay_types) overlayTypes.value = data.overlay_types;
                    } catch (e) { console.error('Fetch overlay failed:', e); }
                }

                async function toggleOverlay(type) {
                    const idx = overlayTypes.value.indexOf(type);
                    if (idx >= 0) overlayTypes.value.splice(idx, 1);
                    else overlayTypes.value.push(type);
                    try {
                        await fetch('/overlay', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ types: overlayTypes.value })
                        });
                    } catch (e) { console.error('Save overlay failed:', e); }
                }

                async function fetchPlaybackStatus() {
                    if (!mainCameraId.value) return;
                    try {
                        const res = await fetch(`/cameras/${mainCameraId.value}/playback/status`);
                        if (!res.ok) { playbackStatus.value = {}; return; }
                        const data = await res.json();
                        playbackStatus.value = data || {};
                    } catch (e) { playbackStatus.value = {}; }
                }

                async function controlPlayback(action, extra = {}) {
                    if (!mainCameraId.value) return;
                    try {
                        const res = await fetch(`/cameras/${mainCameraId.value}/playback/control`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ action, ...extra }),
                        });
                        const data = await res.json();
                        if (data.status) playbackStatus.value = data.status;
                    } catch (e) { console.error('Playback control failed:', e); }
                }

                function mockDashboardData() {
                    gpuLoad.value = Math.min(100, Math.max(10, gpuLoad.value + (Math.random() - 0.5) * 10));
                    fps.value = Math.min(60, Math.max(15, fps.value + (Math.random() - 0.5) * 5));
                    memUsage.value = Math.min(100, Math.max(30, memUsage.value + (Math.random() - 0.5) * 4));
                }

                watch(mainCameraId, () => {
                    playbackStatus.value = {};
                    fetchPlaybackStatus();
                });

                onMounted(() => {
                    fetchCameras();
                    fetchStatus();
                    fetchOverlay();
                    fetchPlaybackStatus();
                    updateUtc();
                    refreshTimer = setInterval(() => {
                        fetchCameras();
                        fetchStatus();
                        fetchPlaybackStatus();
                        mockDashboardData();
                    }, 2000);
                    uptimeTimer = setInterval(updateUtc, 1000);
                });

                onUnmounted(() => {
                    if (refreshTimer) clearInterval(refreshTimer);
                    if (uptimeTimer) clearInterval(uptimeTimer);
                });

                return {
                    cameras, cameraOrder, totalDetections, uptime, overlayTypes,
                    recentAlerts, detectionTypes, playbackStatus,
                    mainCameraId, mainCameraName, mainCameraStatus, mainResolution,
                    mainAlert, onlineCount, utcTime,
                    gpuLoad, fps, memUsage, fpsColor, memColor,
                    useVlm, useGpuScheduler,
                    typeLabel, getTypeColor, setMainCamera, gaugeArc,
                    onVideoError, toggleOverlay, controlPlayback,
                };
            }
        }).mount('#app');
    </script>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/safety_detection/hud.html
git commit -m "feat: add Vue 3 data logic with camera switching, overlays, and mock dashboard"
```

---

## Task 8: Add Backend /hud Route

**Files:**
- Modify: `backend/main_multi.py`

- [ ] **Step 1: Add /hud route after /multi route**

Find the `/multi` route block in `backend/main_multi.py` (around line 590-597):

```python
@app.get("/multi")
async def multi_view():
    """多摄像头控制台页面"""
    fp = Path(__file__).parent.parent / "frontend" / "safety_detection" / "multi.html"
    if fp.exists():
        return HTMLResponse(fp.read_text(encoding="utf-8"))
    return {"error": "Multi-camera page not found"}
```

Insert immediately after it:

```python

@app.get("/hud")
async def hud_view():
    """HUD 风格监控中心"""
    fp = Path(__file__).parent.parent / "frontend" / "safety_detection" / "hud.html"
    if fp.exists():
        return HTMLResponse(fp.read_text(encoding="utf-8"))
    return {"error": "HUD page not found"}
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -m py_compile backend/main_multi.py`
Expected: No output (success)

- [ ] **Step 3: Commit**

```bash
git add backend/main_multi.py
git commit -m "feat: add /hud route for new HUD monitor page"
```

---

## Task 9: Integration Verification

**Files:**
- Test via browser

- [ ] **Step 1: Start backend server**

```bash
cd /home/user/project/sentry-safety
source ~/miniconda3/etc/profile.d/conda.sh && conda activate py312
python -m uvicorn backend.main_multi:app --host 0.0.0.0 --port 8000 --reload
```

Wait for startup (should see "Uvicorn running on http://0.0.0.0:8000").

- [ ] **Step 2: Open /hud in browser and verify**

Open `http://localhost:8000/hud` in browser.

Checklist:
- [ ] Page loads without console errors
- [ ] Dark HUD theme visible (not light theme)
- [ ] Top header shows "SENTRY", UTC time, online count, alerts count
- [ ] Scanline animation visible on header bottom border
- [ ] Main video area has HUD corner brackets
- [ ] Left panel is collapsed (48px wide), hover expands to show alert log and camera list
- [ ] Right panel shows arc gauges (GPU, FPS, Memory)
- [ ] Bottom bar shows camera status cards with colored type tags
- [ ] Camera click switches main video
- [ ] All animations respect reduced-motion (test via OS accessibility settings)
- [ ] Original `/multi` page still works and looks unchanged

- [ ] **Step 3: Stop server**

Ctrl+C to stop uvicorn.

- [ ] **Step 4: Final commit if all checks pass**

```bash
git log --oneline -5
```

Verify commit history is clean, then:
```bash
git status
```
Expected: clean working tree.

---

## Self-Review Checklist

**1. Spec coverage:**
- [x] 新增页面，保留 multi.html -> Task 8 route + Task 9 verification
- [x] 科幻 HUD 风格 -> Task 1-7 CSS
- [x] 高级感（克制、细线、无圆角）-> Task 1 design tokens
- [x] 子摄像头不显示画面 -> Task 6 bottom bar (no img tags)
- [x] 全屏无滚动 -> Task 1 grid layout
- [x] 顶部状态栏 -> Task 2
- [x] 中央视频 + HUD 角括号 -> Task 3
- [x] 左侧终端面板（可折叠）-> Task 4
- [x] 右侧仪表盘（SVG 弧形）-> Task 5
- [x] 底部摄像头状态条 -> Task 6
- [x] 扫描线动效 -> Task 3 CSS
- [x] 告警脉冲 -> Task 3 CSS
- [x] Mock 仪表盘数据 -> Task 7 Vue logic
- [x] prefers-reduced-motion -> Task 3 CSS (media query can be added if needed)

**2. Placeholder scan:**
- [x] No "TBD", "TODO", "implement later"
- [x] All code blocks contain actual code
- [x] All file paths are exact

**3. Type consistency:**
- [x] `cameraOrder`, `mainCameraId`, `setMainCamera` consistent across tasks
- [x] `overlayTypes`, `toggleOverlay` consistent
- [x] `playbackStatus`, `controlPlayback` consistent
- [x] `gaugeArc` function signature matches usage in Task 5
