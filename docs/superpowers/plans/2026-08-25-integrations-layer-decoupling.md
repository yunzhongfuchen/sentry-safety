# 接口层解耦重构实现计划 (Integrations Layer)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建 `backend/integrations/` 统一适配层，将国经公司主动推送（原 alarm_push）与西艾氟公司被动 API（原 cvapi）归纳到外层按公司分包管理，主系统与外部接口彻底解耦。

**Architecture:**
- `backend/integrations/base.py`：推送通道基类
- `backend/integrations/manager.py`：适配器管理器
- `backend/integrations/router.py`：路由统一入口
- `backend/integrations/guojing/`：国经公司适配器
- `backend/integrations/xilu/`：西艾氟公司适配器
- `backend/integrations/__init__.py`：统一入口与初始化工厂
- 兼容旧模块 `backend/alarm_push` 和 `backend/cvapi`

**Spec:** `docs/superpowers/specs/2026-08-25-integrations-layer-decoupling-design.md`

---

### Task 1: 搭建 `integrations` 核心框架与国经公司适配器

**Files:**
- Create: `backend/integrations/__init__.py`
- Create: `backend/integrations/base.py`
- Create: `backend/integrations/manager.py`
- Create: `backend/integrations/guojing/__init__.py`
- Create: `backend/integrations/guojing/channel.py`
- Test: `tests/test_integrations_core.py`

- [ ] **Step 1: Write test for integrations core & guojing channel**
- [ ] **Step 2: Implement base.py, manager.py, guojing/channel.py**
- [ ] **Step 3: Run test to ensure PASS**

---

### Task 2: 迁移西艾氟适配器至 `integrations/xilu/` 并聚合统一路由

**Files:**
- Create: `backend/integrations/xilu/__init__.py`
- Move/Refactor: `backend/cvapi/*` -> `backend/integrations/xilu/*`
- Create: `backend/integrations/router.py`
- Test: `tests/test_integrations_xilu.py`

- [ ] **Step 1: Write test for xilu integration and unified router**
- [ ] **Step 2: Move & implement xilu module under integrations**
- [ ] **Step 3: Create integrations/router.py**
- [ ] **Step 4: Run test to ensure PASS**

---

### Task 3: 兼容旧模块导入 & 主应用 `main_multi.py` 切换解耦

**Files:**
- Modify: `backend/alarm_push/__init__.py` (向后兼容转发)
- Modify: `backend/cvapi/__init__.py` (向后兼容转发)
- Modify: `backend/main_multi.py` (使用 integrations 门面)
- Test: `tests/test_main_multi_cvapi.py`, `tests/test_alarm_push.py`

- [ ] **Step 1: Update backward compatibility forwarders**
- [ ] **Step 2: Update `backend/main_multi.py` with `init_integration_manager` and `integrations_router`**
- [ ] **Step 3: Run full regression test suite (344+ tests)**
