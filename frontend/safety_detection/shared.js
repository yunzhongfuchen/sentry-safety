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
                icon: t.icon || '',
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
            result[t.key] = { box_count_mode: 'gte', ...t.defaults };
        } else {
            result[t.key] = {
                enabled: false, interval: 1, threshold: 0.5,
                consecutive_required: 3, cooldown: 60, use_vlm: false,
                min_box_count: 1, max_box_count: null,
                box_count_mode: 'gte',
            };
        }
    }
    return result;
}


const SETTINGS_TABS = ['cameras', 'detection', 'system'];

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
    if (path.includes('types')) {
        return { page: 'types', tab: 'cameras', settingsExpanded: true };
    }
    if (path.includes('records')) {
        return { page: 'records', tab: 'cameras', settingsExpanded: true };
    }
    return { page: 'monitor', tab: 'cameras', settingsExpanded: true };
}

function renderSidebar(container, context) {
    const expandedClass = context.settingsExpanded ? 'open' : '';
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
                </div>
                <div class="nav-group">
                    <button
                        type="button"
                        class="nav-item parent ${context.page === 'settings' ? 'active' : ''}"
                        aria-expanded="${String(context.settingsExpanded)}"
                        onclick="window.toggleSettingsNav && window.toggleSettingsNav()"
                    >
                        <span class="nav-item-label">设置</span>
                        <span class="nav-item-caret">›</span>
                    </button>
                    <div class="nav-children ${expandedClass}">
                        <a href="/settings.html?tab=cameras" class="nav-item child ${context.page === 'settings' && context.tab === 'cameras' ? 'active' : ''}">摄像头</a>
                        <a href="/settings.html?tab=detection" class="nav-item child ${context.page === 'settings' && context.tab === 'detection' ? 'active' : ''}">检测配置</a>
                        <a href="/types.html" class="nav-item child ${context.page === 'types' ? 'active' : ''}">类型管理</a>
                        <a href="/settings.html?tab=system" class="nav-item child ${context.page === 'settings' && context.tab === 'system' ? 'active' : ''}">系统设置</a>
                    </div>
                </div>
            </nav>
        </aside>
    `;
}

/**
 * 将人数条件模式转换为后端字段
 * @param {string} mode - 'gte' | 'lte' | 'between' | 'outside'
 * @param {number} a - 下界
 * @param {number} b - 上界（可选）
 * @returns {{min_box_count: number|null, max_box_count: number|null, box_count_mode: string}}
 */
function boxCountModeToFields(mode, a, b) {
    switch (mode) {
        case 'gte': return { min_box_count: a, max_box_count: null, box_count_mode: 'gte' };
        case 'lte': return { min_box_count: null, max_box_count: a, box_count_mode: 'lte' };
        case 'between': return { min_box_count: a, max_box_count: b, box_count_mode: 'between' };
        case 'outside': return { min_box_count: a, max_box_count: b, box_count_mode: 'outside' };
        default: return { min_box_count: null, max_box_count: null, box_count_mode: null };
    }
}

/**
 * 将后端字段转换为人数条件模式
 * @param {number|null} min - min_box_count
 * @param {number|null} max - max_box_count
 * @param {string|null} mode - box_count_mode
 * @returns {{mode: string, a: number|null, b: number|null}}
 */
function fieldsToBoxCountMode(min, max, mode) {
    if (mode === 'outside') return { mode: 'outside', a: min, b: max };
    if (min !== null && max !== null) return { mode: 'between', a: min, b: max };
    if (min !== null) return { mode: 'gte', a: min, b: null };
    if (max !== null) return { mode: 'lte', a: max, b: null };
    return { mode: 'gte', a: null, b: null };
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
