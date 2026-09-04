// Shared utilities for Sentry frontend pages
// Safe fetch wrapper with AbortController support and error handling
async function safeFetch(url, options = {}) {
    const controller = new AbortController();
    const timeout = options.timeout || 10000;
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
        const res = await fetch(url, { ...options, signal: controller.signal });
        clearTimeout(timer);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (e) {
        clearTimeout(timer);
        if (e.name === 'AbortError') return null;
        console.error(`safeFetch error: ${url}`, e);
        throw e;
    }
}

// Stable hash from string (replaces Math.random in data mapping)
function stableHash(str, max = 150) {
    let h = 0;
    for (let i = 0; i < str.length; i++) h = ((h << 5) - h + str.charCodeAt(i)) | 0;
    return Math.abs(h) % max;
}

// Format ISO timestamp to local display string
function formatTime(iso) {
    return iso.replace('T', ' ');
}

// Debounce function for search inputs
function debounce(fn, ms = 300) {
    let t;
    return (...args) => {
        clearTimeout(t);
        t = setTimeout(() => fn(...args), ms);
    };
}

// Throttle function for scroll/resize events
function throttle(fn, ms = 100) {
    let last = 0;
    return (...args) => {
        const now = Date.now();
        if (now - last >= ms) {
            last = now;
            fn(...args);
        }
    };
}

// Create managed interval that auto-cleans on page hide
function createManagedInterval(fn, ms) {
    let id = setInterval(fn, ms);
    function pause() { clearInterval(id); }
    function resume() { id = setInterval(fn, ms); }
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) pause(); else resume();
    });
    return { pause, resume };
}

// Type labels and colors shared across pages (loaded dynamically from API)
const _BUILTIN_DETECTION_TYPES = [
    { key: 'fire', label: '明火', color: '#ef4444' },
    { key: 'smoke', label: '烟雾', color: '#f97316' },
    { key: 'uniform', label: '工服', color: '#22c55e' },
    { key: 'mask', label: '口罩', color: '#0ea5e9' },
    { key: 'cigarette', label: '吸烟', color: '#a855f7' },
    { key: 'sleep', label: '睡岗', color: '#eab308' },
];

let DETECTION_TYPES = [..._BUILTIN_DETECTION_TYPES];

async function loadDetectionTypes() {
    try {
        const resp = await fetch('/detector/types');
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        if (data.types && data.types.length > 0) {
            DETECTION_TYPES = data.types.map(t => ({
                key: t.key,
                label: t.label,
                color: t.color,
                defaults: t.defaults || {},
            }));
        }
    } catch (e) {
        console.warn('Failed to load detection types from API, using builtin defaults:', e.message);
    }
    return DETECTION_TYPES;
}

function getTypeLabel(type) {
    return DETECTION_TYPES.find(t => t.key === type)?.label || type || '未知';
}

function getTypeColor(type) {
    return DETECTION_TYPES.find(t => t.key === type)?.color || '#94a3b8';
}

// 类型徽章动态样式：浅色背景（类型色 10% 透明度）+ 类型色文字
// 替代 glass-clay.css 中硬编码的 .type-badge.{type} 类，支持注册表自定义颜色和新类型
function typeBadgeStyle(type) {
    const color = getTypeColor(type);
    return {
        backgroundColor: color + '1A',
        color: color,
    };
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
    const result = {};
    for (const t of DETECTION_TYPES) {
        if (t.defaults) {
            result[t.key] = { ...t.defaults };
        } else {
            result[t.key] = {
                enabled: false, interval: 1, threshold: 0.5,
                consecutive_required: 3, cooldown: 60, use_vlm: false,
            };
        }
    }
    return result;
}


const SETTINGS_TABS = ['cameras', 'detection', 'system', 'push'];

function normalizeSettingsTab(raw) {
    return SETTINGS_TABS.includes(raw) ? raw : 'cameras';
}

function getSettingsTabFromQuery() {
    const params = new URLSearchParams(window.location.search);
    return normalizeSettingsTab(params.get('tab'));
}

function setSettingsTabQuery(tab) {
    const next = normalizeSettingsTab(tab);
    const url = new URL(window.location.href);
    url.searchParams.set('tab', next);
    window.history.replaceState({}, '', `${url.pathname}?${url.searchParams.toString()}`);
}

function getSidebarContext() {
    const path = window.location.pathname;
    if (path.includes('settings')) {
        return { page: 'settings', tab: getSettingsTabFromQuery(), settingsExpanded: true };
    }
    if (path.includes('models')) {
        return { page: 'models', tab: 'cameras', settingsExpanded: true };
    }
    if (path.includes('algorithms')) {
        return { page: 'algorithms', tab: 'cameras', settingsExpanded: true };
    }
    if (path.includes('types')) {
        return { page: 'types', tab: 'cameras', settingsExpanded: true };
    }
    if (path.includes('records')) {
        return { page: 'records', tab: 'cameras', settingsExpanded: true };
    }
    return { page: 'monitor', tab: 'cameras', settingsExpanded: true };
}

function renderSidebar(container, context) {
    container.innerHTML = `
        <aside class="app-sidebar">
            <div class="sidebar-brand">
                <div class="sidebar-brand-mark">S</div>
                <div class="sidebar-brand-copy">
                    <div class="sidebar-brand-title">视频诊断系统</div>
                    <div class="sidebar-brand-subtitle">安全检测平台</div>
                </div>
            </div>
            <nav class="sidebar-nav">
                <div class="nav-group">
                    <a href="/monitor" class="nav-item ${context.page === 'monitor' ? 'active' : ''}">
                        <span class="nav-item-label">监控</span>
                    </a>
                    <a href="/records.html" class="nav-item ${context.page === 'records' ? 'active' : ''}">
                        <span class="nav-item-label">记录</span>
                    </a>
                    <a href="/settings.html?tab=cameras" class="nav-item ${context.page === 'settings' && context.tab === 'cameras' ? 'active' : ''}">
                        <span class="nav-item-label">摄像头</span>
                    </a>
                    <a href="/models.html" class="nav-item ${context.page === 'models' ? 'active' : ''}">
                        <span class="nav-item-label">模型管理</span>
                    </a>
                    <a href="/algorithms.html" class="nav-item ${context.page === 'algorithms' ? 'active' : ''}">
                        <span class="nav-item-label">算法管理</span>
                    </a>
                    <a href="/settings.html?tab=push" class="nav-item ${context.page === 'settings' && context.tab === 'push' ? 'active' : ''}">
                        <span class="nav-item-label">告警推送</span>
                    </a>
                    <a href="/settings.html?tab=system" class="nav-item ${context.page === 'settings' && context.tab === 'system' ? 'active' : ''}">
                        <span class="nav-item-label">系统设置</span>
                    </a>
                </div>
            </nav>
        </aside>
    `;
}

/**
 * 在 canvas 上绘制 ROI 多边形
 * @param {HTMLCanvasElement} canvas
 * @param {Array<[number, number]>} points - 归一化坐标点 [[x1,y1], [x2,y2], ...]
 * @param {boolean} closed - 是否闭合
 */
function drawRoiOnCanvas(canvas, points, closed = false) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (points.length === 0) return;
    ctx.strokeStyle = '#22c55e';
    ctx.fillStyle = 'rgba(34, 197, 94, 0.2)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(points[0][0] * w, points[0][1] * h);
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i][0] * w, points[i][1] * h);
    }
    if (closed && points.length >= 3) {
        ctx.closePath();
        ctx.fill();
    }
    ctx.stroke();
    // 绘制顶点
    ctx.fillStyle = '#22c55e';
    for (const [x, y] of points) {
        ctx.beginPath();
        ctx.arc(x * w, y * h, 4, 0, Math.PI * 2);
        ctx.fill();
    }
}

function clamp01(v) {
    return Math.max(0, Math.min(1, v));
}

function squareToPolygon(x1, y1, x2, y2) {
    const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
    const half = Math.max(Math.abs(x2 - x1), Math.abs(y2 - y1)) / 2;
    return [
        [clamp01(cx - half), clamp01(cy - half)],
        [clamp01(cx + half), clamp01(cy - half)],
        [clamp01(cx + half), clamp01(cy + half)],
        [clamp01(cx - half), clamp01(cy + half)],
    ];
}

function diamondToPolygon(x1, y1, x2, y2) {
    const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
    const rx = Math.abs(x2 - x1) / 2;
    const ry = Math.abs(y2 - y1) / 2;
    return [
        [clamp01(cx), clamp01(cy - ry)],
        [clamp01(cx + rx), clamp01(cy)],
        [clamp01(cx), clamp01(cy + ry)],
        [clamp01(cx - rx), clamp01(cy)],
    ];
}

function ellipseToPolygon(x1, y1, x2, y2, segments = 32) {
    const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
    const rx = Math.abs(x2 - x1) / 2;
    const ry = Math.abs(y2 - y1) / 2;
    const pts = [];
    for (let i = 0; i < segments; i++) {
        const a = (i / segments) * Math.PI * 2;
        pts.push([clamp01(cx + rx * Math.cos(a)), clamp01(cy + ry * Math.sin(a))]);
    }
    return pts;
}

function shapeToPolygon(region) {
    if (region.shape === 'free') return region.points;
    if (!region.drag.start || !region.drag.end) return null;
    const [x1, y1] = region.drag.start;
    const [x2, y2] = region.drag.end;
    if (region.shape === 'square') return squareToPolygon(x1, y1, x2, y2);
    if (region.shape === 'diamond') return diamondToPolygon(x1, y1, x2, y2);
    if (region.shape === 'ellipse') return ellipseToPolygon(x1, y1, x2, y2);
    return null;
}

function drawPolygon(ctx, points, w, h) {
    ctx.beginPath();
    ctx.moveTo(points[0][0] * w, points[0][1] * h);
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i][0] * w, points[i][1] * h);
    }
    ctx.closePath();
}

function drawRegionPoints(ctx, region, w, h) {
    ctx.fillStyle = '#22c55e';
    if (region.shape === 'free') {
        for (const [x, y] of region.points) {
            ctx.beginPath();
            ctx.arc(x * w, y * h, 4, 0, Math.PI * 2);
            ctx.fill();
        }
    } else if (region.drag.start && region.drag.end) {
        [region.drag.start, region.drag.end].forEach(([x, y]) => {
            ctx.beginPath();
            ctx.arc(x * w, y * h, 5, 0, Math.PI * 2);
            ctx.fill();
        });
    }
}

/**
 * 根据当前 ROI 状态绘制形状（支持多个区域和反向框选）
 * @param {HTMLCanvasElement} canvas
 * @param {object} state - { regions: [{ shape, points, drag }], currentRegionIndex, invert }
 */
function drawRoiShape(canvas, state) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    const regions = state.regions || [];
    const current = state.currentRegionIndex ?? 0;
    ctx.clearRect(0, 0, w, h);

    const validRegions = regions.map((r, i) => ({ poly: shapeToPolygon(r), idx: i })).filter(r => r.poly && r.poly.length >= 3);
    if (validRegions.length === 0) return;

    ctx.fillStyle = 'rgba(34, 197, 94, 0.35)';
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 2;

    if (state.invert) {
        ctx.fillRect(0, 0, w, h);
        ctx.globalCompositeOperation = 'destination-out';
        validRegions.forEach(({ poly }) => {
            drawPolygon(ctx, poly, w, h);
            ctx.fill();
        });
        ctx.globalCompositeOperation = 'source-over';
        validRegions.forEach(({ poly, idx }) => {
            ctx.setLineDash(idx === current ? [] : [6, 4]);
            drawPolygon(ctx, poly, w, h);
            ctx.stroke();
        });
        ctx.setLineDash([]);
    } else {
        validRegions.forEach(({ poly, idx }) => {
            ctx.setLineDash(idx === current ? [] : [6, 4]);
            drawPolygon(ctx, poly, w, h);
            ctx.fill();
            ctx.stroke();
        });
        ctx.setLineDash([]);
    }

    validRegions.forEach(({ poly, idx }) => {
        drawRegionPoints(ctx, regions[idx], w, h);
    });
}

/** @deprecated 使用 shapeToPolygon(region) */
function _legacyShapeToPolygon(state) {
    return shapeToPolygon(state);
}
