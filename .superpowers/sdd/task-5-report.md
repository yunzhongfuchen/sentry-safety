## Task 5 Report: GPUDynamicScheduler 丢弃策略

### What Was Implemented

Modified `backend/gpu_scheduler.py` to add frame drop policy preventing backlog under heavy inference load:

1. **`__init__` additions:**
   - `self._busy = False` — flag to skip new collection while previous batch is still inferring
   - `self.MAX_FRAME_AGE = 0.5` — frames older than 0.5 seconds are dropped

2. **Extracted `_collect_due_frames(self, now)` method:**
   - Collects due detection tasks per camera
   - Calls `camera_manager.request_frame(cam_id, timeout=1.0, store_history=True)` instead of `get_frame`
   - Measures frame acquisition time and drops frames where `frame_age > MAX_FRAME_AGE`
   - Filters by detection interval (unchanged logic, moved from `run()`)
   - Returns `Dict[str, List[Tuple[str, np.ndarray]]]` mapping detection_type to list of (cam_id, frame)

3. **Modified `run()` method:**
   - Added `_busy` check at loop start: if busy, sleeps 0.05s and skips collection
   - Sets `_busy = True` before collection, `_busy = False` in `finally` block
   - Calls `_collect_due_frames(now)` instead of inline collection
   - Tracks `collected_keys` list of (cam_id, dtype) tuples for all submitted tasks
   - Updates `last_infer` timestamps **after inference completes** (`completed_at = time.time()`) instead of at collection time

### Files Changed

- `backend/gpu_scheduler.py` — main implementation
- `tests/test_gpu_scheduler_drop.py` — new test file (5 tests)

### Test Results

```
$ python -m pytest tests/ -q
45 passed in 3.87s
```

All 40 existing tests pass (no regressions). 5 new tests added and passing:
- `test_scheduler_busy_flag_initialized_false` — verifies `_busy` starts False and `MAX_FRAME_AGE` is 0.5
- `test_collect_due_frames_uses_request_frame` — verifies `request_frame` is called with correct args
- `test_collect_due_frames_drops_none_frames` — skips cameras returning None
- `test_collect_due_frames_drops_old_frames` — drops frames older than MAX_FRAME_AGE
- `test_collect_due_frames_respects_interval` — does not collect when interval has not elapsed

### Self-Review Findings

- **Completeness:** All 4 requirements from the task brief are implemented: `_busy` flag, `MAX_FRAME_AGE` filtering, `last_infer` updated after inference, `request_frame` usage.
- **Quality:** `_collect_due_frames` is extracted as a testable unit method, matching the brief's adjusted design.
- **Discipline:** No speculative changes. Only touched what the task required.
- **Edge cases:** The `finally` block ensures `_busy` is always cleared even if an exception occurs during collection or inference. The frame age check uses the time between `request_frame` call and return, which is a reasonable proxy for frame staleness in the SCHEDULED decode mode context.

### Concerns

None. The implementation is straightforward and all tests pass.
