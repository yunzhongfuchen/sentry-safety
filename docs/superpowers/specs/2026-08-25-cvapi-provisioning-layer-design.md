# 西艾氟 (CV_OpenAPI) 接口提供层设计规范

## 1. 目标与范围

为外部系统（如西艾氟公司）提供标准的 OpenAPI 接口层，严格按《视频综合平台&视频智能应用OpenAPI对接文档2.3.9》规范实现以下 3 个核心接口，仅返回调用方业务真正消费的必要字段，剥离一切冗余字段：

1. `POST /cvApi/open/api/cv/findModelPage` - 查询模型列表
2. `POST /cvApi/open/api/cv/findCvWarningPage` - 查询报警列表
3. `POST /cvApi/open/api/cv/findCvWarningNumber` - 查询报警数量
4. `GET /cvApi/open/api/cv/warning/image/{record_id}` - 报警快照图片下载端点

---

## 2. 架构设计

### 2.1 目录结构
```
backend/cvapi/
├── __init__.py
├── auth.py          # 静态 Token / 鉴权依赖（支持 Bearer 或 Query token）
├── mappings.py      # 西艾氟算法类型与报警编码映射表
├── schemas.py       # Pydantic 请求参数与精简响应模型
├── service.py       # 业务服务层：读取 performance_storage 与 detection_registry 并做数据转换
└── router.py        # FastAPI APIRouter 路由挂载
```

### 2.2 统一响应信封
```json
{
  "requestId": "<uuid4>",
  "code": 0,
  "state": 200,
  "msg": null,
  "timestamp": "<epoch_ms_string>",
  "data": {},
  "success": true
}
```

---

## 3. 接口详细规范与精简字段定义

### 3.1 查询模型列表 `POST /cvApi/open/api/cv/findModelPage`
- **入参（Form / JSON）**：
  - `size` (int, default=10, -1表示全部)
  - `current` (int, default=1)
- **返回 `data` 精简字段**：
  ```json
  {
    "total": 6,
    "size": 10,
    "current": 1,
    "pages": 1,
    "orders": [],
    "searchCount": true,
    "records": [
      {
        "modelId": "fire",
        "modelName": "明火检测",
        "modelDes": "检测画面中的明火现象",
        "modelType": "fire_recog",
        "modelColour": "#ef4444",
        "modelState": "1",
        "number": 0,
        "modelUrl": ""
      }
    ]
  }
  ```

### 3.2 查询报警列表 `POST /cvApi/open/api/cv/findCvWarningPage`
- **入参（Form / JSON）**：
  - `size` (int, default=10, -1表示全部)
  - `current` (int, default=1)
  - `beginTime` / `endTime` (str)
  - `clearBeginTime` / `clearEndTime` (str)
  - `warningState` (str, "1"=未销警, "0"=已销警)
  - `cameraName` (str, 模糊查询)
  - `cameraIdList` (str 或 list[str])
  - `warningTypeList` (str 或 list[str])
- **返回 `data` 精简字段**：
  ```json
  {
    "total": 1,
    "size": 10,
    "current": 1,
    "pages": 1,
    "orders": [],
    "searchCount": true,
    "records": [
      {
        "id": "cam01_fire_1711697008000",
        "cameraId": "cam01",
        "cameraCode": "cam01",
        "cameraName": "一号车间东门",
        "warningType": "fire_recog",
        "warningContent": "检测到明火现象",
        "warningTime": "2024-03-25 10:08:18",
        "warningTimeEnd": "2024-03-25 10:08:18",
        "warningState": "1",
        "clearTime": null,
        "imgUrl": "/cvApi/open/api/cv/warning/image/cam01_fire_1711697008000",
        "policeType": "SP008",
        "policeLeave": "2",
        "warningValue": "1",
        "warningNumber": 1,
        "warningRange": "[[[100,200,300,400]],[0.85]]",
        "warningPatrolType": "0"
      }
    ]
  }
  ```

### 3.3 查询报警数量 `POST /cvApi/open/api/cv/findCvWarningNumber`
- **入参（Form / JSON）**：
  - `treeId` (str, 可选)
- **返回 `data` 字段**：
  ```json
  {
    "todayWarningNumber": 0,
    "weekWarningNumber": 0,
    "monthWarningNumber": 0,
    "quarterWarningNumber": 0,
    "yearWarningNumber": 0
  }
  ```

### 3.4 报警快照图片 `GET /cvApi/open/api/cv/warning/image/{record_id}`
- 返回 JPEG 二进制流 (`image/jpeg`)。

---

## 4. 路由双重兼容
同时支持：
1. `/cvApi/open/api/cv/...` (指定完整路径)
2. `/open/api/cv/...` (PDF Path 栏兼容路径)

---

## 5. 鉴权机制
- 读取环境变量 `CVAPI_TOKEN`。
- 如果配置了 `CVAPI_TOKEN`，检查 `Authorization: Bearer <token>` 或 Header `X-Token` 或 Query `token`。
- 如果未配置 `CVAPI_TOKEN`，允许直接免鉴权访问（便于内网快速对接调试）。
