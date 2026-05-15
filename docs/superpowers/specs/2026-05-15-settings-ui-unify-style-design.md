# settings.html 样式统一设计

## 背景
安全检测前端设置页（`frontend/safety_detection/settings.html`）中：
1. "全局配置"里的"启用 VLM 复核"使用 checkbox，样式与其他 input/select 不一致。
2. "检测类型默认值"区域使用 `.type-row` 紧凑样式，和 `.form-field` 的统一样式不一致。

## 改动范围
仅修改 `frontend/safety_detection/settings.html` 的 HTML 结构与 CSS。

## 详细设计

### 1. "启用 VLM 复核"改下拉框
**位置**：`#global` 区域 `.form-field`（"启用 VLM 复核"）。

**当前代码**：
```html
<input type="checkbox" v-model="settings.use_vlm" />
```

**改为**：
```html
<select v-model="settings.use_vlm">
    <option :value="true">是</option>
    <option :value="false">否</option>
</select>
```

**理由**：`select` 自动继承 `.form-field select` 的统一样式（padding 8px 12px、圆角、聚焦边框高亮），与相邻的"VLM 最大并发"等输入框视觉一致。

**行为兼容**：`v-model` 绑定值仍为布尔值，不影响保存逻辑。

### 2. 检测类型默认值样式统一
**位置**：`.type-row` 及其内部 `input/select`。

**布局不变**：保留横向 grid 行布局，不改卡片式网格，确保一屏仍能看全所有检测类型。

**CSS 调整**：
```css
.type-row input, .type-row select {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);   /* 原为 4px */
    color: var(--text);
    padding: 8px 12px;                 /* 原为 4px 6px */
    font-size: 14px;                   /* 原为 12px */
    font-family: inherit;
    transition: border-color .2s;      /* 新增 */
}

.type-row input:focus, .type-row select:focus {
    outline: none;
    border-color: var(--accent);       /* 新增，与 .form-field 一致 */
}
```

**列宽微调**：为适应变大的输入框，将 grid 模板列做如下调整：
```css
.type-row {
    grid-template-columns: 80px 60px 70px 80px 80px 60px;
    /* 原：80px 50px 70px 80px 80px 50px */
}
```

## 无行为变更
- 所有 `v-model` 绑定字段不变。
- 保存接口参数不变。
- 不涉及 JavaScript 逻辑修改。
