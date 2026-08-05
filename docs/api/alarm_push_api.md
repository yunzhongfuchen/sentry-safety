# 报警推送接口文档

视频诊断系统在报警事件发生时，向外部系统提供的接收接口主动推送报文。

推送方式为**事件推送**：每条报警产生 **1~2 条**报文——创建时一条；若该算法启用了 VLM 复核，复核完成后再推一条。

报文内嵌图片：创建报文携带**标注快照 + 检测关键帧**（base64 JPEG），接收方无需回源拉取。

---

## 1. 接收接口约定（由接收方实现）

| 项       | 约定                                                                                                 |
| -------- | ---------------------------------------------------------------------------------------------------- |
| 方法     | `POST`                                                                                             |
| 地址     | 由接收方提供，如`https://<接收方>/api/alarm`                                                       |
| 请求头   | `Content-Type: application/json; charset=utf-8Authorization: Bearer <token>`（token 双方线下约定） |
| 成功响应 | HTTP 200，body`{"code": 0}`                                                                        |
| 失败处理 | 其余响应视为失败；推送方按 5s / 30s / 5min 间隔重试 3 次（复用同一`event_id`），仍失败记死信日志   |
| 幂等     | 接收方以`event_id` 去重；同一 `data.id` 的报文按后者覆盖前者（upsert）                           |

## 2. 报文结构

信封 + 数据两层：

```json
{
  "event": "alarm.created",
  "event_id": "3_安全背心_05dc45_1784538005159_created",
  "sent_at": "2026-07-30 14:23:16",
  "data": { ...完整报警记录... }
}
```

| 字段         | 类型   | 说明                                                           |
| ------------ | ------ | -------------------------------------------------------------- |
| `event`    | string | 事件类型：`alarm.created` / `alarm.reviewed`，见第 3 节    |
| `event_id` | string | 事件唯一 ID，格式`{记录id}_{created\|reviewed}`，用于幂等去重 |
| `sent_at`  | string | 发送时间，`YYYY-MM-DD HH:MM:SS`，部署机本地时间              |
| `data`     | object | 完整报警记录，字段见第 4 节；每条报文都带全量字段              |

## 3. 事件类型

| event              | 触发时机                          | 图片                          | 说明                                                                       |
| ------------------ | --------------------------------- | ----------------------------- | -------------------------------------------------------------------------- |
| `alarm.created`  | 小模型检测到异常                  | 携带`snapshot` + `frames` | `vlm_review=null`，`level=small_model_alarm`，`status=pending`       |
| `alarm.reviewed` | VLM 复核完成（创建后数秒~数十秒） | 不带                          | `level`/`reason`/`vlm_review` 已更新为复核结果，其余字段与创建时一致 |

注意：

- 只有**启用了 VLM 复核的算法**才会产生 `alarm.reviewed`；未启用的算法只有 `alarm.created` 一条
- 复核结论可能是确认（`vlm_alarm`）也可能是排除（`vlm_ignore`），两种都会推送
- `alarm.reviewed` 一定晚于同记录的 `alarm.created` 到达（推送方按序发送），接收方以 `data.id` 关联两条报文

## 4. data 字段说明

| 字段                | 类型          | 可为空 | 说明                                                                                                         |
| ------------------- | ------------- | ------ | ------------------------------------------------------------------------------------------------------------ |
| `id`              | string        | 否     | 记录唯一 ID：`{camera_id}_{算法key}_{毫秒时间戳}`                                                          |
| `camera_id`       | string        | 否     | 摄像头唯一标识                                                                                               |
| `detection_type`  | string        | 否     | 算法 key，如`fire`、`安全帽_925834`（唯一标识，非显示名）                                                |
| `detection_label` | string        | 否     | 算法显示名称，如`明火`、`安全帽`，可直接用于展示                                                         |
| `level`           | string        | 否     | 告警级别：`small_model_alarm` / `vlm_alarm` / `vlm_ignore`                                             |
| `status`          | string        | 否     | 处理状态：`pending` / `confirmed` / `false_positive`，推送时恒为 `pending`（人工操作不推送）         |
| `time`            | string        | 否     | 触发时间，`YYYY-MM-DD HH:MM:SS`                                                                            |
| `confidence`      | float         | 否     | 小模型置信度，0~1                                                                                            |
| `reason`          | string        | 否     | 报警说明文本：算法自定义说明或`检测到{算法显示名}异常`；复核后为 `[VLM 确认] ...` / `[VLM 已排除] ...` |
| `small_model`     | object        | 否     | 小模型结果：`{detected, confidence, boxes}`，`boxes` 为 `[x1,y1,x2,y2]` 像素坐标数组                   |
| `vlm_review`      | object\| null | 是     | VLM 复核结果：`{confirmed, confidence, reason}`；created 报文中为 `null`                                 |
| `source`          | string        | 否     | 固定`"small_model"`                                                                                        |
| `frame_count`     | int           | 否     | `frames` 数组长度                                                                                          |
| `snapshot`        | string        | 否     | 标注快照 base64 JPEG，图上已绘制检测框和时间戳水印；仅 created 报文携带                                      |
| `frames`          | array         | 否     | 检测关键帧 base64 JPEG 数组，按采集时间排序；仅 created 报文携带，可为空数组                                 |

## 5. 完整示例

### 5.1 alarm.created

```json
{
  "event": "alarm.created",
  "event_id": "3_安全背心_05dc45_1784538005159_created",
  "sent_at": "2026-07-30 14:23:16",
  "data": {
    "id": "3_安全背心_05dc45_1784538005159",
    "camera_id": "3",
    "detection_type": "安全背心_05dc45",
    "detection_label": "安全背心",
    "level": "small_model_alarm",
    "status": "pending",
    "time": "2026-07-30 14:23:15",
    "confidence": 0.876,
    "reason": "检测到安全背心异常",
    "small_model": {
      "detected": true,
      "confidence": 0.876,
      "boxes": [[120, 45, 380, 420]]
    },
    "vlm_review": null,
    "source": "small_model",
    "frame_count": 3,
    "snapshot": "/9j/4AAQSkZJRgABAQAA...(base64 JPEG)",
    "frames": [
      "/9j/4AAQSkZJRgABAQAA...(第1帧)",
      "/9j/4AAQSkZJRgABAQAA...(第2帧)",
      "/9j/4AAQSkZJRgABAQAA...(第3帧)"
    ]
  }
}
```

### 5.2 alarm.reviewed（复核确认）

```json
{
  "event": "alarm.reviewed",
  "event_id": "3_安全背心_05dc45_1784538005159_reviewed",
  "sent_at": "2026-07-30 14:23:41",
  "data": {
    "id": "3_安全背心_05dc45_1784538005159",
    "camera_id": "3",
    "detection_type": "安全背心_05dc45",
    "detection_label": "安全背心",
    "level": "vlm_alarm",
    "status": "pending",
    "time": "2026-07-30 14:23:15",
    "confidence": 0.876,
    "reason": "[VLM 确认] 画面中人员未穿安全背心",
    "small_model": {
      "detected": true,
      "confidence": 0.876,
      "boxes": [[120, 45, 380, 420]]
    },
    "vlm_review": {
      "confirmed": true,
      "confidence": 0.92,
      "reason": "画面中人员未穿安全背心"
    },
    "source": "small_model",
    "frame_count": 3
  }
}
```

### 5.3 alarm.reviewed（复核排除，仅示意关键字段）

```json
{
  "event": "alarm.reviewed",
  "data": {
    "id": "3_安全背心_05dc45_1784538005159",
    "level": "vlm_ignore",
    "reason": "[VLM 已排除] 人员穿着符合要求，光影误判",
    "vlm_review": { "confirmed": false, "confidence": 0.88, "reason": "人员穿着符合要求，光影误判" }
  }
}
```

## 6. 报文大小参考

| 内容                               | 典型大小（base64 后） |
| ---------------------------------- | --------------------- |
| 快照 1 张                          | 约 70~200 KB          |
| 关键帧每张                         | 约 30~100 KB          |
| created 报文整体（快照 + 3~10 帧） | 通常 < 1.5 MB         |
| reviewed 报文                      | < 5 KB                |

## 7. 备注

- 时间均为字符串格式 `YYYY-MM-DD HH:MM:SS`（部署机本地时间），`data.id` 尾部含毫秒时间戳可用于排序
- 检测框坐标基于摄像头原始分辨率，接收方绘制时需按显示尺寸等比缩放
- 人工处理（确认/误报）不推送；如后续需要可增加 `alarm.status_changed` 事件，结构不变
