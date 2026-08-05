# 模型管理页颜色增强设计

## 背景
当前模型管理页（`frontend/safety_detection/models.html`）的模型卡片整体使用 glass-clay 默认色调，视觉层次较弱。用户希望对卡片内的关键标签、数字和操作按钮进行颜色区分，使页面更生动、信息更易扫读。

## 目标
为模型卡片内的以下 6 个元素分别赋予不同颜色，且互不重复：
1. “文件:” 标签文字 → 青绿
2. “策略:” 标签文字 → 橙色
3. “被 X 个算法引用” 中的 X 数字 → 紫色
4. 编辑按钮 → 保持默认灰/中性，不额外着色
5. 更换权重按钮 → 蓝色
6. 删除按钮 → 红色

## 设计细节

### 颜色映射
| 元素 | 颜色 | 色值 / CSS 变量 |
|------|------|----------------|
| “文件:” 标签 | 青绿 | `#0d9488`（使用现有 success 色值） |
| “策略:” 标签 | 橙色 | `var(--warning)` `#d97706` |
| 引用次数 X | 紫色 | `#7c3aed`（新增局部色） |
| 编辑按钮 | 中性灰 | 默认 `.clay-button` |
| 更换权重按钮 | 蓝色 | `var(--accent)` `#2563eb` |
| 删除按钮 | 红色 | `var(--danger)` `#dc2626` |

### 实现方式
- 不修改全局 `glass-clay.css`，仅在 `models.html` 的局部 `<style>` 中扩展类：
  - `.model-card-meta .meta-label-file { color: #0d9488; }`
  - `.model-card-meta .meta-label-post { color: var(--warning); }`
  - `.model-card-meta .used-count { color: #7c3aed; font-weight: 600; }`
  - `.clay-button.btn-replace { background: var(--accent); color: #fff; ... }`
  - `.clay-button.btn-delete { background: var(--danger); color: #fff; ... }`
- 按钮着色采用与 `.clay-button.primary` 类似的微立体阴影，保持 glass-clay 风格。

### 兼容性
- 不影响其他页面。
- 不修改数据模型或 API。
- 纯前端样式调整，无需后端配合。

## 测试要点
1. 模型卡片内各元素颜色按上述规则显示。
2. 按钮 hover/active 状态保持 clay 按钮的按下/浮起效果。
3. 弹窗内按钮不受模型卡片按钮样式影响。
