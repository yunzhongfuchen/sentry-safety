# 模型管理页颜色增强 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为模型管理页卡片内的关键标签、引用数字和操作按钮分别着色，使页面视觉层次更丰富，各元素颜色互不重复。

**Architecture:** 纯前端样式调整。在 `frontend/safety_detection/models.html` 的局部 `<style>` 中新增几个工具类，分别控制“文件:”标签（青绿）、“策略:”标签（橙）、引用次数数字（紫）、更换权重按钮（蓝）、删除按钮（红）。编辑按钮保持默认 `.clay-button` 中性灰，不额外着色。

**Tech Stack:** Vue 3 (global build), CSS, glass-clay design system

## Global Constraints
- 不修改全局 `glass-clay.css`
- 不修改数据模型或 API
- 保持 clay-button 的按压/浮起交互效果
- 弹窗内按钮不受模型卡片按钮新样式影响

---

### Task 1: Add color utility classes and apply to model card

**Files:**
- Modify: `frontend/safety_detection/models.html`

**Interfaces:**
- No external interfaces; pure presentational changes

- [ ] **Step 1: Add CSS utility classes**

In `frontend/safety_detection/models.html`, inside the existing `<style>` block (after `.class-chip`), add:

```css
.meta-label-file { color: #0d9488; font-weight: 600; }
.meta-label-post { color: var(--warning); font-weight: 600; }
.used-count { color: #7c3aed; font-weight: 700; }

.clay-button.btn-replace {
    background: var(--accent);
    color: #fff;
    box-shadow:
        4px 4px 10px rgba(37, 99, 235, 0.25),
        -4px -4px 10px rgba(255, 255, 255, 0.5);
}
.clay-button.btn-replace:hover { background: var(--accent-hover); }
.clay-button.btn-replace:active {
    transform: translateY(1px);
    box-shadow: var(--clay-shadow-pressed);
}

.clay-button.btn-delete {
    background: var(--danger);
    color: #fff;
    box-shadow:
        4px 4px 10px rgba(220, 38, 38, 0.25),
        -4px -4px 10px rgba(255, 255, 255, 0.5);
}
.clay-button.btn-delete:hover { background: #b91c1c; }
.clay-button.btn-delete:active {
    transform: translateY(1px);
    box-shadow: var(--clay-shadow-pressed);
}
```

- [ ] **Step 2: Apply classes to card labels and buttons**

Modify the model card template (around line 38-52) from:

```html
<div class="model-card-meta">文件: {{ m.file }}<span v-if="m.file_size">（{{ (m.file_size / 1048576).toFixed(1) }} MB）</span></div>
<div class="model-card-meta">策略: {{ m.post_process }} · 被 {{ m.used_by }} 个算法引用</div>
...
<button class="clay-button" @click="openEdit(m)">编辑</button>
<button class="clay-button" @click="openReplace(m)">更换权重</button>
<button class="clay-button" @click="deleteModel(m)">删除</button>
```

To:

```html
<div class="model-card-meta"><span class="meta-label-file">文件:</span> {{ m.file }}<span v-if="m.file_size">（{{ (m.file_size / 1048576).toFixed(1) }} MB）</span></div>
<div class="model-card-meta"><span class="meta-label-post">策略:</span> {{ m.post_process }} · 被 <span class="used-count">{{ m.used_by }}</span> 个算法引用</div>
...
<button class="clay-button" @click="openEdit(m)">编辑</button>
<button class="clay-button btn-replace" @click="openReplace(m)">更换权重</button>
<button class="clay-button btn-delete" @click="deleteModel(m)">删除</button>
```

- [ ] **Step 3: Verify in browser**

1. Open `http://127.0.0.1:8000/models.html`.
2. Confirm:
   - “文件:” text is teal/green.
   - “策略:” text is orange.
   - The number in “被 X 个算法引用” is purple and bold.
   - “编辑” button remains default gray.
   - “更换权重” button is blue with white text.
   - “删除” button is red with white text.
3. Confirm hover and active states still work.
4. Confirm buttons inside upload/replace/edit modals are NOT affected (they do not have `.btn-replace` / `.btn-delete` classes).

- [ ] **Step 4: Commit**

```bash
git add frontend/safety_detection/models.html
git commit -m "feat: add color accents to model management card labels and actions"
```

---

### Task 2: Responsive and edge-case check

**Files:**
- Modify: `frontend/safety_detection/models.html` (if needed)

- [ ] **Step 1: Check long model names / many classes**

1. Verify colored labels remain readable when model name or class list wraps.
2. Verify `.used-count` bolding does not break layout.

- [ ] **Step 2: Dark mode / contrast check (if project supports dark mode)**

Check `glass-clay.css` for dark mode media query. If present, verify the new colors still have acceptable contrast. The existing CSS variables already handle dark mode; hardcoded `#0d9488` and `#7c3aed` are bright enough on both light and dark backgrounds.

- [ ] **Step 3: Final commit if any tweaks**

```bash
git add frontend/safety_detection/models.html
git commit -m "fix: model card color adjustments"
```

---

## Self-Review

- Spec coverage:
  - “文件:” 青绿 → `.meta-label-file`
  - “策略:” 橙 → `.meta-label-post`
  - 引用次数 X 紫 → `.used-count`
  - 编辑按钮灰 → no class added
  - 更换权重蓝 → `.btn-replace`
  - 删除红 → `.btn-delete`
- No placeholders.
- No data/API changes, no type consistency issues.
