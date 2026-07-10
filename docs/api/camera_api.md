# 摄像头管理接口文档

服务默认监听 `0.0.0.0:8000`，内网其他机器可通过部署机器的内网 IP 访问。

基础地址：`http://<部署机IP>:8000`

---

## 1. 添加摄像头

### 接口

```
POST /cameras/add
```

用于新增一个摄像头。如果 `camera_id` 已存在，返回 `400` 错误。

### 请求参数

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `camera_id` | string | **是** | 摄像头唯一标识，不可重复 |
| `source` | string | **是** | 视频源地址，如 RTSP 地址、本地摄像头索引、本地视频文件路径 |
| `name` | string | 否 | 显示名称，默认空字符串 |
| `enabled` | bool | 否 | 是否立即启用，默认 `true` |
| `source_type` | string | 否 | 源类型：`"auto"` / `"camera"` / `"rtsp"`，默认 `"auto"` |
| `width` | int | 否 | 分辨率宽，默认 `640` |
| `height` | int | 否 | 分辨率高，默认 `480` |
| `fps` | int | 否 | 帧率，默认 `15` |
| `detection_types` | object | 否 | 检测类型配置，详见下方「检测类型配置」 |

### 请求示例

```json
{
  "camera_id": "cam_01",
  "source": "rtsp://192.168.1.100/stream",
  "name": "车间入口",
  "enabled": true,
  "source_type": "rtsp",
  "width": 1280,
  "height": 720,
  "detection_types": {
    "fire": {
      "enabled": true,
      "threshold": 0.6
    },
    "uniform": {
      "enabled": true,
      "threshold": 0.5
    }
  }
}
```

### 响应示例

成功：

```json
{
  "success": true,
  "camera_id": "cam_01"
}
```

失败：

```json
{
  "error": "Camera ID already exists"
}
```

---

## 2. 修改摄像头配置

### 接口

```
POST /cameras/{camera_id}/config
```

用于修改已有摄像头的配置。只传需要修改的字段即可，未传字段保持不变。

### 路径参数

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `camera_id` | string | **是** | 要修改的摄像头 ID |

### 请求参数

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 否 | 显示名称 |
| `source` | string | 否 | 视频源地址（修改时需要同时传 `source_type`） |
| `source_type` | string | 否 | 源类型，与 `source` 配合使用 |
| `width` | int | 否 | 分辨率宽 |
| `height` | int | 否 | 分辨率高 |
| `enabled` | bool | 否 | 启用/禁用摄像头 |
| `detection_types` | object | 否 | 检测类型配置，详见下方「检测类型配置」 |

### 请求示例

```json
{
  "name": "车间入口-改名",
  "source": "rtsp://192.168.1.101/stream",
  "source_type": "rtsp",
  "enabled": true,
  "detection_types": {
    "fire": {
      "enabled": true,
      "threshold": 0.6,
      "cooldown": 10
    },
    "smoke": {
      "enabled": false
    }
  }
}
```

### 响应示例

成功：

```json
{
  "success": true
}
```

失败：

```json
{
  "error": "Not initialized"
}
```

---

## 3. 删除摄像头

### 接口

```
DELETE /cameras/{camera_id}
```

删除指定摄像头，同时停止该摄像头的视频流和检测任务。

### 路径参数

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `camera_id` | string | **是** | 要删除的摄像头 ID |

### 响应示例

成功：

```json
{
  "success": true
}
```

失败（摄像头不存在）：

```json
{
  "error": "Camera not found"
}
```

---

## 4. 检测类型配置

`detection_types` 是一个对象，key 为检测类型，value 为该类型的配置。

### 支持的检测类型

| 类型 | 说明 |
|------|------|
| `fire` | 火焰检测 |
| `smoke` | 烟雾检测 |
| `uniform` | 工服/反光背心检测 |
| `mask` | 口罩佩戴检测 |
| `cigarette` | 吸烟检测 |
| `sleep` | 睡岗检测 |

### 单类型配置字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `enabled` | bool | 否 | 是否启用该类型检测，默认 `false` |
| `interval` | float | 否 | 检测间隔（秒），默认按类型不同 |
| `threshold` | float | 否 | 置信度阈值，默认按类型不同 |
| `consecutive_required` | int | 否 | 连续检测到多少次才触发告警，默认按类型不同 |
| `cooldown` | float | 否 | 两次告警之间的冷却时间（秒），默认按类型不同 |
| `use_vlm` | bool | 否 | 是否使用 VLM 复核，默认 `false` |

### 各类型默认值

```json
{
  "fire": {
    "enabled": false,
    "interval": 1,
    "threshold": 0.6,
    "consecutive_required": 2,
    "cooldown": 10,
    "use_vlm": false
  },
  "smoke": {
    "enabled": false,
    "interval": 1,
    "threshold": 0.55,
    "consecutive_required": 2,
    "cooldown": 10,
    "use_vlm": false
  },
  "uniform": {
    "enabled": false,
    "interval": 1,
    "threshold": 0.5,
    "consecutive_required": 2,
    "cooldown": 3,
    "use_vlm": false
  },
  "mask": {
    "enabled": false,
    "interval": 1,
    "threshold": 0.5,
    "consecutive_required": 1,
    "cooldown": 3,
    "use_vlm": false
  },
  "cigarette": {
    "enabled": false,
    "interval": 1,
    "threshold": 0.5,
    "consecutive_required": 1,
    "cooldown": 3,
    "use_vlm": false
  },
  "sleep": {
    "enabled": false,
    "interval": 60,
    "threshold": 0.7,
    "consecutive_required": 3,
    "cooldown": 30,
    "use_vlm": false
  }
}
```

---

## 5. 其他相关接口

| 接口 | 说明 |
|------|------|
| `GET /cameras` | 列出所有摄像头及其状态 |
| `POST /cameras/{camera_id}/enable` | 启用指定摄像头 |
| `POST /cameras/{camera_id}/disable` | 禁用指定摄像头 |
| `POST /cameras/{camera_id}/source` | 只切换视频源 |
| `POST /cameras/batch-config` | 批量修改多个摄像头的检测类型 |
| `POST /cameras/{camera_id}/reset-config` | 恢复单个摄像头到全局默认配置 |

---

## 6. 联调注意

1. 服务默认监听 `0.0.0.0:8000`，内网可访问。
2. 已开启 CORS（`allow_origins=["*"]`），浏览器页面可直接调用。
3. Windows 防火墙需放行 `8000` 端口，否则外部无法访问。
4. 如需改端口，启动前设置环境变量：

```bash
set API_PORT=8080
```
