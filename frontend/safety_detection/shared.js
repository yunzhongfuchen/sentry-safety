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
        fire: { enabled: false, interval: 1, threshold: 0.6, consecutive_required: 3, cooldown: 60, use_vlm: false },
        smoke: { enabled: false, interval: 1, threshold: 0.55, consecutive_required: 3, cooldown: 60, use_vlm: false },
        uniform: { enabled: false, interval: 1, threshold: 0.5, consecutive_required: 3, compliance_window_seconds: 30, cooldown: 60, use_vlm: false },
        mask: { enabled: false, interval: 1, threshold: 0.5, consecutive_required: 3, cooldown: 60, use_vlm: false },
        cigarette: { enabled: false, interval: 1, threshold: 0.5, consecutive_required: 3, cooldown: 60, use_vlm: false },
        sleep: { enabled: false, interval: 60, threshold: 0.7, consecutive_required: 3, cooldown: 60, use_vlm: false },
    };
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
                    <div class="sidebar-brand-title">安全哨兵</div>
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
                        <a href="/settings.html?tab=system" class="nav-item child ${context.page === 'settings' && context.tab === 'system' ? 'active' : ''}">系统设置</a>
                    </div>
                </div>
            </nav>
        </aside>
    `;
}
