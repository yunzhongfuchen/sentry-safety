# Glass-Clay 精修：Cool Slate 冷静蓝灰配色

**日期**：2026-07-02
**状态**：已确认，待实现
**范围**：前端视觉精修（配色 + 阴影质感），不改布局与组件结构

## 背景

前端 Sentry 安全检测系统近期统一到一套 "glass-clay"（玻璃 + 黏土拟态）设计系统，暖白底 + 橙色主色。本次在**保留 glass-clay 方向**的前提下做精修，只针对两个痛点：**配色**与**阴影质感**。

设计系统集中在 [glass-clay.css](../../../frontend/safety_detection/styles/glass-clay.css) 的 `:root` 令牌，所有页面（monitor / records / settings / index）通过 `var(--...)` 继承，因此改动高度集中。

## 目标

- 将暖橙配色替换为 **Cool Slate 冷静蓝灰**：冷调中性底 + 蓝色主色，更贴近专业监控后台气质。
- 将拟态阴影调整为 **平衡档（Balanced）**：保留玻璃卡与内凹黏土质感，但投影收薄、转冷。
- 不改字体、间距、布局、组件结构。

## 非目标

- 不改任何页面布局或组件 DOM 结构。
- 不改字体（保留 Space Grotesk + Noto Sans SC）。
- 不动视频/图片区的近黑底 `#0f1210`。
- 不做深色模式、不新增主题切换。

## 设计决策

用户在可视化对比中确认：
- 配色方向 = **B · 冷静蓝灰 Cool Slate**
- 阴影质感 = **② 平衡玻璃黏土 Balanced**

## 详细改动

### 1. 配色令牌（`glass-clay.css` `:root`）

| 令牌 | 现值 | 新值 |
|---|---|---|
| `--bg-base` | `#f0f2ee` | `#eef1f4` |
| `--bg-soft` | `#f7f8f6` | `#f4f6f9` |
| `--bg-elevated` | `#ffffff` | 不变 |
| `--text-primary` | `#1a211c` | `#1e293b` |
| `--text-secondary` | `#5f6b63` | `#64748b` |
| `--text-muted` | `#949f96` | `#94a3b8` |
| `--accent` | `#e05a18` | `#2563eb` |
| `--accent-soft` | `rgba(224,90,24,.10)` | `rgba(37,99,235,.10)` |
| `--accent-hover` | `#c44c12` | `#1d4ed8` |
| `--success` | `#0d9f6e` | `#0d9488` |
| `--warning` | `#e05a18` | `#d97706` |
| `--danger` | `#dc2626` | 不变 |
| `--info` | `#2563eb` | `#0284c7` |

### 2. 阴影令牌（`glass-clay.css` `:root`）

| 令牌 | 新值 |
|---|---|
| `--glass-edge` | `rgba(30,41,59,.06)` |
| `--glass-shadow` | `0 1px 2px rgba(30,41,59,.05), 0 10px 22px rgba(30,41,59,.07), inset 0 1px 0 rgba(255,255,255,.95)` |
| `--clay-shadow` | `3px 3px 8px rgba(148,163,184,.4), -3px -3px 8px rgba(255,255,255,.9)` |
| `--clay-shadow-inset` | `inset 2px 2px 5px rgba(148,163,184,.35), inset -2px -2px 5px rgba(255,255,255,.95)` |
| `--clay-shadow-pressed` | `inset 3px 3px 6px rgba(148,163,184,.4), inset -3px -3px 6px rgba(255,255,255,.9)` |

`--glass-bg` / `--glass-border` 维持现值（白玻璃在冷底上仍适用）。

### 3. 硬编码收尾

- [index.html](../../../frontend/safety_detection/index.html) 第 14 行 `#e94560` 链接颜色/边框 → 改用 `var(--accent)`（`#2563eb`）。
- monitor.html / records.html 的 `#0f1210` 视频/图片底 → **保留不动**（近黑底适配任意配色）。

## 验证标准

1. 逐页目视 monitor / records / settings / index 四个页面：
   - 无残留暖橙色（除 `#0f1210` 近黑底外，无 `#e05a18` / `#e94560` 出现）。
   - 主色统一为蓝色 `#2563eb`。
   - 告警红 `#dc2626` 在冷底上仍醒目、对比充足。
   - 文字与背景对比度可读（正文 `#1e293b` on `#eef1f4`）。
2. 交互态（hover / pressed）阴影表现正常，无过重或缺失。

## 影响范围

- 主要改动：[glass-clay.css](../../../frontend/safety_detection/styles/glass-clay.css) 的 `:root`（约 20 行令牌）。
- 次要改动：[index.html](../../../frontend/safety_detection/index.html) 1 行硬编码颜色。
- 无 JS 改动，无布局改动，无新增依赖。
