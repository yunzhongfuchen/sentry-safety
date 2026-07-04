# Left Sidebar Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current top-right page tabs with a left sidebar navigation that has first-level items for 监控 / 记录 / 设置 and second-level items under 设置 for 摄像头 / 检测配置 / 系统设置.

**Architecture:** Keep the current multi-page HTML + Vue 3 structure. Add one shared sidebar layout/style layer in `glass-clay.css`, one shared navigation/query-sync helper layer in `shared.js`, and small page-specific markup changes in `monitor.html`, `records.html`, and `settings.html`. Reuse the existing `settings.html` internal tab state instead of splitting settings into multiple HTML pages.

**Tech Stack:** Static HTML, Vue 3 global build, shared plain JavaScript helpers, shared CSS custom properties/layout utilities.

## Global Constraints

- Replace the current right-top nav buttons with a left sidebar on `monitor.html`, `records.html`, and `settings.html`.
- First-level nav items: `监控` -> `monitor.html`, `记录` -> `records.html`, `设置` -> expandable parent only.
- Second-level nav items under `设置`: `摄像头` -> `settings.html?tab=cameras`, `检测配置` -> `settings.html?tab=detection`, `系统设置` -> `settings.html?tab=system`.
- Clicking first-level `设置` must only expand/collapse children; it must not navigate.
- `settings.html` without `tab` query must default to `cameras`.
- On `settings.html`, the sidebar parent `设置` must be expanded and the current child item highlighted according to the active tab.
- Keep the current glass-clay / Cool Slate visual system.
- Do not convert the app into an SPA.
- Do not change business logic on monitor / records / settings pages.

---

## File Structure

- `frontend/safety_detection/styles/glass-clay.css`
  - Add reusable shell/sidebar/nav styles shared by all three pages.
  - Keep existing color tokens and component styles intact.
- `frontend/safety_detection/shared.js`
  - Add small navigation helpers for current-page detection, settings-tab parsing, query syncing, and sidebar state defaults.
- `frontend/safety_detection/monitor.html`
  - Replace top nav markup with shell + sidebar markup.
  - Wrap current monitor content in a new main content container.
- `frontend/safety_detection/records.html`
  - Replace top nav markup with shell + sidebar markup.
  - Wrap current records content in a new main content container.
- `frontend/safety_detection/settings.html`
  - Replace top nav markup with shell + sidebar markup.
  - Bind existing local `tab` state to URL query (`?tab=`) using shared helpers.
  - Keep existing internal tab content blocks.

## Task 1: Add shared sidebar shell and navigation styles

**Files:**
- Modify: `frontend/safety_detection/styles/glass-clay.css`
- Test: visual verification in `frontend/safety_detection/monitor.html`, `frontend/safety_detection/records.html`, `frontend/safety_detection/settings.html`

**Interfaces:**
- Consumes: existing design tokens in `:root`, existing `glass-card`, `clay-button`, `app-header` styling patterns.
- Produces:
  - `.app-shell` two-column page layout
  - `.app-sidebar`, `.sidebar-brand`, `.sidebar-nav`
  - `.nav-group`, `.nav-item`, `.nav-item.active`, `.nav-item.parent`, `.nav-item.child`
  - `.nav-children`, `.nav-children.open`
  - `.app-main`

- [ ] **Step 1: Read the current shared CSS near layout/navigation classes**

Run: `python - <<'PY'
from pathlib import Path
p = Path('frontend/safety_detection/styles/glass-clay.css')
for i, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
    if 1 <= i <= 260:
        print(f"{i}:{line}")
PY`
Expected: see the current token block, app header styles, and existing utility classes.

- [ ] **Step 2: Add the shared sidebar layout and nav styles**

Append the following CSS near the shared layout/navigation section in `frontend/safety_detection/styles/glass-clay.css`:

```css
.app-shell {
    min-height: 100vh;
    display: grid;
    grid-template-columns: 248px minmax(0, 1fr);
    background: var(--bg-base);
}

.app-sidebar {
    position: sticky;
    top: 0;
    height: 100vh;
    padding: 20px 16px;
    border-right: 1px solid var(--glass-edge);
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.74), rgba(255, 255, 255, 0.58));
    backdrop-filter: blur(18px);
    display: flex;
    flex-direction: column;
    gap: 18px;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 12px;
    border-radius: var(--radius-lg);
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    box-shadow: var(--glass-shadow);
}

.sidebar-brand-mark {
    width: 36px;
    height: 36px;
    border-radius: 14px;
    display: grid;
    place-items: center;
    font-family: var(--font-display);
    font-weight: 700;
    background: var(--accent-soft);
    color: var(--accent);
}

.sidebar-brand-copy {
    display: flex;
    flex-direction: column;
    min-width: 0;
}

.sidebar-brand-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
}

.sidebar-brand-subtitle {
    font-size: 12px;
    color: var(--text-secondary);
}

.sidebar-nav {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.nav-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.nav-item {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 12px 14px;
    border: none;
    border-radius: 14px;
    background: transparent;
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s ease, color 0.2s ease, box-shadow 0.2s ease;
}

.nav-item:hover {
    background: rgba(255, 255, 255, 0.56);
    color: var(--text-primary);
}

.nav-item.active {
    background: var(--glass-bg);
    color: var(--accent);
    border: 1px solid var(--glass-border);
    box-shadow: var(--glass-shadow);
}

.nav-item.parent {
    font-weight: 700;
}

.nav-item.child {
    margin-left: 14px;
    padding: 10px 12px;
    font-size: 13px;
    border-radius: 12px;
}

.nav-item-label {
    display: inline-flex;
    align-items: center;
    gap: 10px;
}

.nav-item-caret {
    font-size: 12px;
    color: var(--text-muted);
    transition: transform 0.2s ease;
}

.nav-item.parent[aria-expanded="true"] .nav-item-caret {
    transform: rotate(90deg);
}

.nav-children {
    display: none;
    flex-direction: column;
    gap: 6px;
}

.nav-children.open {
    display: flex;
}

.app-main {
    min-width: 0;
}

@media (max-width: 1100px) {
    .app-shell {
        grid-template-columns: 220px minmax(0, 1fr);
    }
}
```

- [ ] **Step 3: Run a syntax/sanity check on the CSS file**

Run: `python - <<'PY'
from pathlib import Path
text = Path('frontend/safety_detection/styles/glass-clay.css').read_text(encoding='utf-8')
assert '.app-shell' in text
assert '.app-sidebar' in text
assert '.nav-children.open' in text
print('sidebar css present')
PY`
Expected: `sidebar css present`

- [ ] **Step 4: Commit**

```bash
git add frontend/safety_detection/styles/glass-clay.css
git commit -m "feat: add shared sidebar navigation shell styles"
```

## Task 2: Add shared navigation and settings-tab URL helpers

**Files:**
- Modify: `frontend/safety_detection/shared.js`
- Test: helper behavior verified from browser console / page integration in later tasks

**Interfaces:**
- Consumes: `window.location`, `window.history.replaceState`, current `settings.html` `tab` values: `cameras | detection | system`.
- Produces:
  - `normalizeSettingsTab(raw)` -> `'cameras' | 'detection' | 'system'`
  - `getSettingsTabFromQuery()` -> `'cameras' | 'detection' | 'system'`
  - `setSettingsTabQuery(tab)` -> `void`
  - `getSidebarContext()` -> `{ page: 'monitor' | 'records' | 'settings', tab: 'cameras' | 'detection' | 'system', settingsExpanded: boolean }`

- [ ] **Step 1: Read the current shared.js helpers and naming style**

Run: `python - <<'PY'
from pathlib import Path
p = Path('frontend/safety_detection/shared.js')
for i, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
    print(f"{i}:{line}")
PY`
Expected: see the small helper-oriented coding style used by existing shared functions.

- [ ] **Step 2: Add minimal navigation/query helpers to shared.js**

Append the following code to `frontend/safety_detection/shared.js` after the existing shared helper functions:

```javascript
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
        return { page: 'records', tab: 'cameras', settingsExpanded: false };
    }
    return { page: 'monitor', tab: 'cameras', settingsExpanded: false };
}
```

- [ ] **Step 3: Run a lightweight helper verification**

Run: `python - <<'PY'
from pathlib import Path
text = Path('frontend/safety_detection/shared.js').read_text(encoding='utf-8')
for name in ['normalizeSettingsTab', 'getSettingsTabFromQuery', 'setSettingsTabQuery', 'getSidebarContext']:
    assert f'function {name}' in text
print('shared nav helpers present')
PY`
Expected: `shared nav helpers present`

- [ ] **Step 4: Commit**

```bash
git add frontend/safety_detection/shared.js
git commit -m "feat: add shared sidebar navigation helpers"
```

## Task 3: Replace the top nav in monitor.html and records.html with the shared left sidebar

**Files:**
- Modify: `frontend/safety_detection/monitor.html`
- Modify: `frontend/safety_detection/records.html`
- Test: browser verification on `monitor.html` and `records.html`

**Interfaces:**
- Consumes:
  - CSS classes from Task 1
  - `getSidebarContext()` from Task 2
- Produces:
  - `sidebar` reactive state in each page setup: `{ page, tab, settingsExpanded }`
  - `toggleSettingsNav()` method in each page setup
  - consistent shell/sidebar markup in both pages

- [ ] **Step 1: Replace the current top nav shell in `monitor.html`**

Change the current page-level wrapper so the body structure becomes:

```html
<div id="app" class="app-shell">
    <aside class="app-sidebar">
        <div class="sidebar-brand">
            <div class="sidebar-brand-mark">S</div>
            <div class="sidebar-brand-copy">
                <div class="sidebar-brand-title">Sentry</div>
                <div class="sidebar-brand-subtitle">安全检测平台</div>
            </div>
        </div>
        <nav class="sidebar-nav">
            <div class="nav-group">
                <a href="/monitor" :class="['nav-item', { active: sidebar.page === 'monitor' }]">
                    <span class="nav-item-label">监控</span>
                </a>
                <a href="/records.html" :class="['nav-item', { active: sidebar.page === 'records' }]">
                    <span class="nav-item-label">记录</span>
                </a>
            </div>
            <div class="nav-group">
                <button
                    type="button"
                    class="nav-item parent"
                    :class="{ active: sidebar.page === 'settings' }"
                    :aria-expanded="String(sidebar.settingsExpanded)"
                    @click="toggleSettingsNav"
                >
                    <span class="nav-item-label">设置</span>
                    <span class="nav-item-caret">›</span>
                </button>
                <div :class="['nav-children', { open: sidebar.settingsExpanded }]">
                    <a href="/settings.html?tab=cameras" :class="['nav-item', 'child', { active: sidebar.page === 'settings' && sidebar.tab === 'cameras' }]">摄像头</a>
                    <a href="/settings.html?tab=detection" :class="['nav-item', 'child', { active: sidebar.page === 'settings' && sidebar.tab === 'detection' }]">检测配置</a>
                    <a href="/settings.html?tab=system" :class="['nav-item', 'child', { active: sidebar.page === 'settings' && sidebar.tab === 'system' }]">系统设置</a>
                </div>
            </div>
        </nav>
    </aside>
    <main class="app-main">
        <!-- keep the existing monitor header/content here -->
    </main>
</div>
```

Then in the Vue `setup()` for `monitor.html`, add:

```javascript
const sidebar = ref(getSidebarContext());

function toggleSettingsNav() {
    sidebar.value.settingsExpanded = !sidebar.value.settingsExpanded;
}
```

And expose both in the returned setup object.

- [ ] **Step 2: Replace the current top nav shell in `records.html`**

Apply the same shell/sidebar markup pattern to `records.html`, using the same `sidebar` state and `toggleSettingsNav()` function in its `setup()` block.

Use exactly the same markup/classes as Task 3 Step 1 so the two pages stay in sync.

- [ ] **Step 3: Remove the old top-right nav markup only**

Delete only the old `<nav class="nav-links">...</nav>` block from both files. Do not change page business content inside the monitor or records main content areas.

- [ ] **Step 4: Run a targeted browser-free structural check**

Run: `python - <<'PY'
from pathlib import Path
for rel in ['frontend/safety_detection/monitor.html', 'frontend/safety_detection/records.html']:
    text = Path(rel).read_text(encoding='utf-8')
    assert 'class="app-shell"' in text
    assert 'class="app-sidebar"' in text
    assert 'toggleSettingsNav' in text
    assert 'nav-links' not in text
print('monitor/records sidebar markup present')
PY`
Expected: `monitor/records sidebar markup present`

- [ ] **Step 5: Verify in browser**

Run the app and manually verify:
- left sidebar is visible on monitor and records
- `监控` / `记录` highlight correctly per page
- clicking `设置` expands/collapses children without navigation
- clicking settings children opens the right settings URL

Expected: monitor and records business content still renders unchanged inside the new shell.

- [ ] **Step 6: Commit**

```bash
git add frontend/safety_detection/monitor.html frontend/safety_detection/records.html
git commit -m "feat: add left sidebar navigation to monitor and records"
```

## Task 4: Bind settings.html tab state to URL query and add the same left sidebar shell

**Files:**
- Modify: `frontend/safety_detection/settings.html`
- Test: browser verification on `settings.html`

**Interfaces:**
- Consumes:
  - CSS classes from Task 1
  - helper functions from Task 2: `getSettingsTabFromQuery`, `setSettingsTabQuery`, `getSidebarContext`
  - existing local `tab` state in `settings.html`
- Produces:
  - sidebar shell in settings page
  - `tab` initialized from query string
  - watch/sync behavior from local tab state back to query string

- [ ] **Step 1: Initialize the settings `tab` state from the URL query**

In `frontend/safety_detection/settings.html`, replace the current static/default tab initialization with:

```javascript
const tab = ref(getSettingsTabFromQuery());
const sidebar = ref(getSidebarContext());

function toggleSettingsNav() {
    sidebar.value.settingsExpanded = !sidebar.value.settingsExpanded;
}
```

Return `sidebar` and `toggleSettingsNav` from the setup object.

- [ ] **Step 2: Sync local tab changes back into `?tab=` and the sidebar highlight**

Add a `watch` in `settings.html`:

```javascript
watch(tab, (value) => {
    const next = normalizeSettingsTab(value);
    if (next !== value) {
        tab.value = next;
        return;
    }
    setSettingsTabQuery(next);
    sidebar.value.page = 'settings';
    sidebar.value.tab = next;
    sidebar.value.settingsExpanded = true;
});
```

This keeps refresh/bookmark/deep-link behavior stable.

- [ ] **Step 3: Replace the top nav shell with the shared left sidebar shell**

Apply the same `app-shell` / `app-sidebar` / `app-main` markup from Task 3 to `settings.html`.

Important differences for `settings.html`:
- the `设置` parent button should start active
- the children container should use `sidebar.settingsExpanded`
- child item active state must track `sidebar.tab`
- the existing internal tab strip (`gc-tab`) remains in the content area

Use this child-link pattern exactly:

```html
<a href="/settings.html?tab=cameras" :class="['nav-item', 'child', { active: sidebar.tab === 'cameras' }]">摄像头</a>
<a href="/settings.html?tab=detection" :class="['nav-item', 'child', { active: sidebar.tab === 'detection' }]">检测配置</a>
<a href="/settings.html?tab=system" :class="['nav-item', 'child', { active: sidebar.tab === 'system' }]">系统设置</a>
```

- [ ] **Step 4: Remove only the old top-right nav block**

Delete the old `<nav class="nav-links">...</nav>` block from `settings.html`. Keep the page header title and internal tab strip content.

- [ ] **Step 5: Run a targeted structural check**

Run: `python - <<'PY'
from pathlib import Path
text = Path('frontend/safety_detection/settings.html').read_text(encoding='utf-8')
assert 'class="app-shell"' in text
assert 'getSettingsTabFromQuery()' in text
assert 'setSettingsTabQuery' in text
assert 'toggleSettingsNav' in text
assert 'nav-links' not in text
print('settings sidebar + query sync present')
PY`
Expected: `settings sidebar + query sync present`

- [ ] **Step 6: Verify in browser**

Run the app and manually verify:
- opening `/settings.html` defaults to the 摄像头 tab
- opening `/settings.html?tab=detection` shows 检测配置
- opening `/settings.html?tab=system` shows 系统设置
- clicking the internal tab strip updates the URL query
- refreshing `settings.html?tab=...` keeps the same tab selected
- left sidebar `设置` stays expanded and highlights the current child item
- clicking `设置` parent collapses/expands children but does not navigate away

Expected: all existing settings forms keep working exactly as before.

- [ ] **Step 7: Commit**

```bash
git add frontend/safety_detection/settings.html
git commit -m "feat: sync settings tabs with sidebar navigation"
```

## Task 5: Final regression sweep for navigation behavior and layout integrity

**Files:**
- Verify only:
  - `frontend/safety_detection/monitor.html`
  - `frontend/safety_detection/records.html`
  - `frontend/safety_detection/settings.html`
  - `frontend/safety_detection/shared.js`
  - `frontend/safety_detection/styles/glass-clay.css`

**Interfaces:**
- Consumes: all outputs from Tasks 1-4.
- Produces: a verified branch ready for review.

- [ ] **Step 1: Run page-level static sanity checks**

Run: `python - <<'PY'
from pathlib import Path
checks = {
    'frontend/safety_detection/monitor.html': ['app-sidebar', '/settings.html?tab=cameras'],
    'frontend/safety_detection/records.html': ['app-sidebar', '/settings.html?tab=detection'],
    'frontend/safety_detection/settings.html': ['app-sidebar', 'setSettingsTabQuery', 'getSettingsTabFromQuery'],
    'frontend/safety_detection/shared.js': ['function normalizeSettingsTab', 'function getSidebarContext'],
    'frontend/safety_detection/styles/glass-clay.css': ['.app-shell', '.nav-item.child'],
}
for rel, needles in checks.items():
    text = Path(rel).read_text(encoding='utf-8')
    for needle in needles:
        assert needle in text, f'{needle} missing in {rel}'
print('static sidebar checks passed')
PY`
Expected: `static sidebar checks passed`

- [ ] **Step 2: Run the app and verify navigation end-to-end**

Manual verification checklist:
- `/monitor` loads with sidebar and correct highlight
- `/records.html` loads with sidebar and correct highlight
- `/settings.html` loads with sidebar, expanded 设置 section, and 摄像头 selected by default
- clicking `设置` parent from monitor/records only expands/collapses
- clicking each settings child from monitor/records opens the corresponding settings tab
- clicking each internal settings tab updates both content and `?tab=` query
- refreshing any `settings.html?tab=...` URL preserves sidebar highlight + internal tab content
- monitor layout, records table, and settings forms remain visually intact inside the new shell

Expected: no loss of existing business behavior.

- [ ] **Step 3: Review the diff for accidental scope creep**

Run: `git diff --stat HEAD~4..HEAD`
Expected: only `monitor.html`, `records.html`, `settings.html`, `shared.js`, and `glass-clay.css` are touched for implementation work.

- [ ] **Step 4: Commit any final navigation-only cleanup**

If no cleanup was needed, run:

```bash
git status
```
Expected: clean working tree

If there were navigation-only fixes from Step 2, commit them with:

```bash
git add frontend/safety_detection/monitor.html frontend/safety_detection/records.html frontend/safety_detection/settings.html frontend/safety_detection/shared.js frontend/safety_detection/styles/glass-clay.css
git commit -m "fix: polish sidebar navigation behavior"
```

## Self-Review

- Spec coverage: covered sidebar replacement, first-level items, expandable 设置 parent, second-level items, tab query sync, default `cameras` behavior, child highlighting, and preservation of existing multi-page structure.
- Placeholder scan: removed TBD-style language and provided exact files, exact helper names, exact CSS classes, exact verification commands.
- Type consistency: settings tab values are consistently `cameras | detection | system` in CSS/HTML/JS helper names and watch logic.
