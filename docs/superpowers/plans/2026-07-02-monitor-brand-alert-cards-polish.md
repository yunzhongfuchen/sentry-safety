# Monitor Brand and Alert Cards Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Polish `monitor.html` so the left/top brand text is fully Chinese and the recent-alert cards area shows 5 cards with a tighter layout that avoids horizontal scrolling at common desktop widths while still allowing scrolling on smaller windows.

**Architecture:** Keep the existing monitor page structure, data flow, and glass-clay / Cool Slate visual system. Make a surgical monitor-only change: update brand copy in `monitor.html`, reduce the recent alert feed from 6 to 5 items, and tighten the alert-card rail layout through the monitor page’s local CSS rather than changing the shared design system globally.

**Tech Stack:** Static HTML, Vue 3 global build, existing shared.js helpers, monitor page local CSS, shared glass-clay CSS tokens.

## Global Constraints

- Only optimize `frontend/safety_detection/monitor.html` brand copy and the bottom recent-alert area layout.
- Do not modify `records.html` or `settings.html`.
- Do not change monitor video area, right camera list, top status logic, or business behavior.
- Keep the recent-alert presentation as cards; do not convert it to a list or table.
- Change recent-alert display count from 6 to 5.
- Use Chinese brand copy: main title `安全哨兵`, subtitle `安全检测平台`.
- If monitor still shows English `Sentry` in the top header title, change that to Chinese as well.
- At common desktop widths, the bottom recent-alert area should not show a horizontal scrollbar.
- On smaller windows, a horizontal scrollbar is allowed.
- Do not add pagination, carousel, “more” button, or extra interactions.
- Keep the current glass-clay / Cool Slate material and shadow language.

---

## File Structure

- `frontend/safety_detection/monitor.html`
  - Update the Chinese brand text in the sidebar and header.
  - Change the recent alert data slicing from 6 to 5.
  - Tighten the local recent-alert rail/card CSS so 5 cards fit at common desktop widths without breaking the rest of monitor.

## Task 1: Update monitor brand copy and recent-alert layout

**Files:**
- Modify: `frontend/safety_detection/monitor.html`
- Test: browser verification on `frontend/safety_detection/monitor.html`

**Interfaces:**
- Consumes:
  - existing `recentAlerts` reactive list in `monitor.html`
  - existing `data.recent_records` payload from `/status`
  - existing monitor-local CSS classes for the recent-alert area
- Produces:
  - sidebar brand main title `安全哨兵`
  - top header title `安全哨兵`
  - `recentAlerts.value = data.recent_records.slice(0, 5)...`
  - tightened recent-alert rail/card layout that keeps horizontal scroll available only as a small-window fallback

- [ ] **Step 1: Read the current monitor brand and recent-alert implementation**

Run: `python - <<'PY'
from pathlib import Path
p = Path('frontend/safety_detection/monitor.html')
text = p.read_text(encoding='utf-8').splitlines()
for i in range(140, 220):
    if i <= len(text):
        print(f"{i}:{text[i-1]}")
print('---RECENT ALERTS---')
for i in range(340, 490):
    if i <= len(text):
        print(f"{i}:{text[i-1]}")
PY`
Expected: see the current sidebar/header brand copy, local recent-alert CSS, card markup, and `slice(0, 6)` data assignment.

- [ ] **Step 2: Update the Chinese brand copy in monitor.html**

Change the two visible brand titles in `frontend/safety_detection/monitor.html`:

```html
<div class="sidebar-brand-title">安全哨兵</div>
```

and

```html
<span>安全哨兵</span>
```

Keep the subtitle as:

```html
<div class="sidebar-brand-subtitle">安全检测平台</div>
```

Do not rename unrelated page titles, API names, or shared CSS comments.

- [ ] **Step 3: Reduce the recent-alert feed from 6 cards to 5 cards**

In the code path that maps `data.recent_records`, change:

```javascript
recentAlerts.value = data.recent_records.slice(0, 6).map(r => {
```

to:

```javascript
recentAlerts.value = data.recent_records.slice(0, 5).map(r => {
```

No other business logic should change in that block.

- [ ] **Step 4: Tighten the recent-alert rail layout but keep small-window scrolling available**

In the monitor page’s local `<style>` block, update the recent-alert rail/card CSS so it behaves like this:

```css
.alerts-strip {
    display: grid;
    grid-template-columns: repeat(5, minmax(220px, 1fr));
    gap: 14px;
    overflow-x: auto;
    padding-bottom: 6px;
}

.alert-card {
    min-width: 220px;
    padding: 18px;
}

.alert-card .time {
    font-size: 12px;
}

.alert-card .meta {
    font-size: 13px;
}
```

If the current class names differ, apply these values to the existing monitor-local classes instead of inventing a second alert rail. The intended behavior is:
- 5 cards fit cleanly at normal desktop widths
- the area still uses `overflow-x: auto` as the fallback for small windows
- no forced wrap behavior is introduced

- [ ] **Step 5: Run a focused structural/sanity check**

Run: `python - <<'PY'
from pathlib import Path
text = Path('frontend/safety_detection/monitor.html').read_text(encoding='utf-8')
assert '安全哨兵' in text
assert 'slice(0, 5)' in text
assert 'overflow-x: auto' in text
assert 'repeat(5,' in text or 'grid-template-columns: repeat(5' in text
print('monitor brand and alert-card changes present')
PY`
Expected: `monitor brand and alert-card changes present`

- [ ] **Step 6: Verify in browser**

Run the app and manually verify on `monitor.html`:
- left sidebar brand no longer shows English `Sentry`
- top header title no longer shows English `Sentry`
- bottom recent-alert area shows 5 cards, not 6
- at common desktop width, the bottom section does not show a horizontal scrollbar
- when the window is made narrower, horizontal scrolling is still allowed instead of forcing wrap
- monitor video, camera list, and top status pills still behave as before

Expected: monitor remains visually consistent with the current glass-clay / Cool Slate design.

- [ ] **Step 7: Commit**

```bash
git add frontend/safety_detection/monitor.html
git commit -m "feat: polish monitor brand and recent alert cards"
```

## Self-Review

- Spec coverage: covers Chinese brand copy, 5-card limit, tighter recent-alert layout, common-width no-scroll requirement, and small-window scroll allowance.
- Placeholder scan: no TBD/TODO placeholders remain; all edits target the exact file and exact code areas.
- Type consistency: `recentAlerts` remains the same reactive list and still consumes `data.recent_records`; only the slice count and layout parameters change.
