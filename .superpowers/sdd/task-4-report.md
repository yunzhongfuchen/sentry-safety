# Task 4 Report: 主画面切换 promote/demote 与 overlay 只处理主画面

## What Was Implemented

### 1. CameraManager.set_main_camera() / get_main_camera()
- **File**: `backend/camera_manager.py`
- Added `self._main_camera_id: Optional[str]` to `CameraManager.__init__()`
- Added `set_main_camera(camera_id)`:
  - 旧主画面降级为 `DecodeMode.SCHEDULED`
  - 新主画面提升为 `DecodeMode.CONTINUOUS`
  - 返回旧主画面 `camera_id`
  - 未知 camera_id 时清空主画面并记录 warning
- Added `get_main_camera()` -> `Optional[str]`

### 2. main_multi.py set_main_camera() + _overlay_loop rewrite
- **File**: `backend/main_multi.py`
- Added `set_main_camera(camera_id)` function:
  - 注销旧主画面流缓冲 (`stream_server.unregister_camera`)
  - 调用 `camera_manager.set_main_camera(camera_id)`
  - 注册新主画面流缓冲 (`stream_server.register_camera`)
- Rewrote `_overlay_loop`:
  - 只处理 `camera_manager.get_main_camera()` 返回的摄像头
  - 使用 `request_frame()` 获取帧（触发 SCHEDULED 模式解码）
  - 只推送主画面的原始帧和标注帧
  - 移除了遍历所有摄像头的循环

### 3. MultiDetector strategies fallback to request_frame
- **File**: `backend/safety_detection/detector_core.py`
- Modified `MultiDetector._process_camera()`:
  - `get_frame()` 返回 None 时，fallback 到 `request_frame(camera_id, timeout=1.0, store_history=True)`
  - 这确保 `SCHEDULED` 模式下的摄像头在检测到期时仍能获取帧

## TDD Evidence

### RED (before implementation)
```
python -m pytest tests/test_camera_manager_decode_modes.py::test_camera_manager_set_main_camera -v
FAILED tests/test_camera_manager_decode_modes.py::test_camera_manager_set_main_camera
AttributeError: 'CameraManager' object has no attribute 'set_main_camera'
```
Expected: `set_main_camera` did not exist yet.

### GREEN (after implementation)
```
python -m pytest tests/test_camera_manager_decode_modes.py -v
14 passed in 0.83s

python -m pytest tests/ -v
40 passed in 3.41s
```

## Files Changed
- `backend/camera_manager.py` — added `_main_camera_id`, `set_main_camera()`, `get_main_camera()`
- `backend/main_multi.py` — added `set_main_camera()`, rewrote `_overlay_loop()`
- `backend/safety_detection/detector_core.py` — added `request_frame` fallback in `_process_camera()`
- `tests/test_camera_manager_decode_modes.py` — added 4 tests for set_main_camera behavior

## Self-Review Findings
- No issues found. All tests pass.
- The implementation follows the spec exactly.
- No speculative code added.
- Existing patterns in the codebase are preserved.

## Issues or Concerns
- None.
