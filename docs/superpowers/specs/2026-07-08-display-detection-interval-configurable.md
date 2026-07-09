# 主画面显示检测刷新频率可配置设计

## 背景
当前主画面显示检测固定每 1 秒执行一次。调试过程中发现，有时希望把刷新频率调快或调慢以适应不同场景。因此需要让主画面显示检测的刷新频率变成可配置项，并在监控页“显示检测类型”面板提供设置入口。

## 目标
1. 主画面显示检测刷新频率可在 **0.1 ~ 10 秒** 范围内设置。
2. 前端监控页在“显示检测类型”面板增加一个刷新频率输入框。
3. 刷新频率与显示检测类型开关一起持久化到 `config/global.json`。
4. 后端 `SelectedCameraDisplay` 在运行时能够响应刷新频率变更。
5. 刷新频率为**全局统一**，不区分检测类型。

## 非目标
1. 不需要为每个检测类型单独设置刷新频率。
2. 不需要把这个频率应用到全局检测链路。
3. 不需要在 settings.html 页面设置该频率。

## 推荐方案：扩展 `/display-types` 接口
复用现有的 `/display-types` GET/POST 接口，在返回和接收的数据中加入 `display_detection_interval` 字段。

### 后端 API 变更
- `GET /display-types` 返回：
  ```json
  {
    "fire": true,
    "smoke": true,
    ...,
    "display_detection_interval": 1.0
  }
  ```
- `POST /display-types` 接收：
  ```json
  {
    "display_detection_types": {"fire": true, ...},
    "display_detection_interval": 0.5
  }
  ```
- 后端校验：`< 0.1` 截断为 `0.1`，`> 10` 截断为 `10`。

### 后端模块变更
- `backend/config.py`
  - 在 `DEFAULT_GLOBAL_SETTINGS` 中新增 `"display_detection_interval": 1.0`。
- `backend/main_multi.py`
  - `SelectedCameraDisplay` 新增 `self._display_interval: float = 1.0`。
  - 增加 `set_display_config(display_types, display_interval=None)` 方法，替代或扩展 `set_display_types()`。
  - `detect_loop` 中 `time.sleep(1.0)` 改为 `time.sleep(self._display_interval)`。
  - 初始化时从 `_global_settings` 读取默认值。
  - `/display-types` GET 和 POST 处理新字段。

### 前端变更
- `frontend/safety_detection/monitor.html`
  - 在“显示检测类型”面板新增数字输入框：
    ```html
    <div style="margin-top: 12px;">
      <label>刷新频率（秒）</label>
      <input type="number" step="0.1" min="0.1" max="10"
             v-model.number="displayInterval"
             @change="updateDisplayConfig" />
    </div>
    ```
  - `displayInterval` 初始从 `/display-types` 接口读取 `display_detection_interval`。
  - 切换类型或修改频率时，统一调用 `/display-types` POST。
  - 如果后端返回了实际生效值（截断后），前端同步更新显示。

## 数据流
1. 后端启动：从 `global.json` 读取 `display_detection_interval`，默认值 1.0。
2. 前端加载：调用 `GET /display-types`，同时拿到类型开关和刷新频率。
3. 用户修改频率：前端调用 `POST /display-types`，后端保存配置并通知 `SelectedCameraDisplay` 更新。
4. `SelectedCameraDisplay.detect_loop` 立即使用新的间隔。

## 边界与校验
- 前端输入框限制 `min="0.1" max="10" step="0.1"`。
- 后端再做一次范围截断，防止 API 直接传非法值。
- 非法值或非数字值不抛出异常，而是静默截断到范围内。
- 配置保存失败时，后端返回错误，前端回退到上一次成功值。

## 受影响文件
- `backend/config.py`
- `backend/main_multi.py`
- `frontend/safety_detection/monitor.html`
- `tests/test_select_main_camera_api.py`（如有必要更新 API 测试）
- `tests/test_selected_camera_display.py`（新增刷新频率变更测试）

## 验证标准
1. 默认启动时主画面显示检测间隔为 1 秒。
2. 在监控页把频率改为 0.5 秒后，主画面检测间隔变为 0.5 秒。
3. 输入 0.05 时，实际生效为 0.1；输入 20 时，实际生效为 10。
4. 修改频率后刷新页面，仍显示上次保存的值。
5. 所有类型关闭时，频率设置保留，但不进行检测。
6. 全量 pytest 通过。
