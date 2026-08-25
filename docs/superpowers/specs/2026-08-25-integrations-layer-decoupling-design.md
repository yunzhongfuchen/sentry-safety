# 外部系统接口适配层 (Integrations Layer) 设计规范

## 1. 架构目标

将所有外部对接系统（主动推送与被动开放 API）从系统核心检测引擎中剥离，形成统一、高内聚、低耦合的 `backend/integrations/` 架构。
未来对接任何新公司/新系统时，只需在该目录下新增对应的公司适配子包，主系统业务与核心检测引擎无需做任何侵入式修改。

---

## 2. 目录架构

```
backend/integrations/
├── __init__.py                   # 统一暴露 init_integrations, IntegrationManager, integrations_router
├── base.py                       # 基础抽象：AlarmPushChannel 基类
├── manager.py                    # IntegrationManager 适配器管理器（异步分发、容错隔离）
├── router.py                     # 统一 API 路由聚合入口
│
├── guojing/                      # 【国经公司】专用适配（主动推送）
│   ├── __init__.py
│   └── channel.py                # WebhookChannel 报文组装与 HTTP 推送
│
└── xilu/ (cvapi)                 # 【西艾氟公司】专用适配（被动查询 OpenAPI）
    ├── __init__.py
    ├── router.py                 # /cvApi/open/api/cv/* 路由
    ├── service.py                # 数据查询与协议转换
    ├── schemas.py                # 请求与精简响应模型
    ├── mappings.py               # 类型与报警编码映射表
    └── auth.py                   # 鉴权
```

---

## 3. 核心接口与职责划分

### 3.1 抽象与门面 (`base.py` & `manager.py`)
- `AlarmPushChannel(ABC)`:
  - `send_created(record: dict, snapshot_b64: Optional[str], frames_b64: list) -> bool`
  - `send_reviewed(record: dict) -> bool`
- `IntegrationManager`:
  - 维护启用的推送通道列表
  - `push_created(record, snapshot_b64, frames_b64)`: 后台广播
  - `push_reviewed(record)`: 后台广播

### 3.2 国经公司适配 (`integrations/guojing/`)
- 承接原有 `backend/alarm_push/webhook.py` 的职责。
- 封装 `GuojingWebhookChannel`，生成符合国经规范的 `alarm.created` / `alarm.reviewed` JSON 报文并发送。

### 3.3 西艾氟公司适配 (`integrations/xilu/`)
- 承接 `backend/cvapi/` 的全部功能。
- 提供 `/cvApi/open/api/cv/findModelPage`、`/cvApi/open/api/cv/findCvWarningPage`、`/cvApi/open/api/cv/findCvWarningNumber` 和 `/cvApi/open/api/cv/warning/image/{id}` 接口。

---

## 4. 主程序 (`main_multi.py`) 解耦调用方式

1. **路由挂载**：
   ```python
   from backend.integrations import integrations_router
   app.include_router(integrations_router)
   ```
2. **推送初始化**：
   ```python
   from backend.integrations import init_integration_manager
   push_manager = init_integration_manager(_global_settings, log=log_message)
   ```

---

## 5. 向后兼容机制
- `backend/alarm_push/__init__.py` 转发引用至 `backend/integrations`，确保老代码和老单测平滑过渡。
- `backend/cvapi/__init__.py` 转发引用至 `backend/integrations/xilu`。
