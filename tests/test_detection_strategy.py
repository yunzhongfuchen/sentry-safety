"""检测调度策略选择测试（detection_strategy: auto/parallel/serial）"""

import pytest
from backend.safety_detection import select_detection_strategy
from backend.safety_detection.detector_core import (
    CorePinnedStrategy, SerialStrategy,
)
from backend import config


class TestSelectDetectionStrategy:
    def test_auto_uses_corepinned_with_enough_cores(self):
        strategy = select_detection_strategy("auto", npu_cores=2)
        assert isinstance(strategy, CorePinnedStrategy)

    def test_auto_uses_serial_with_few_cores(self):
        strategy = select_detection_strategy("auto", npu_cores=1)
        assert isinstance(strategy, SerialStrategy)
        strategy0 = select_detection_strategy("auto", npu_cores=0)
        assert isinstance(strategy0, SerialStrategy)

    def test_serial_forced(self):
        strategy = select_detection_strategy("serial", npu_cores=2)
        assert isinstance(strategy, SerialStrategy)
        assert isinstance(select_detection_strategy("Serial", npu_cores=2),
                          SerialStrategy)

    def test_parallel_forced(self):
        strategy = select_detection_strategy("parallel", npu_cores=0)
        assert isinstance(strategy, CorePinnedStrategy)

    def test_invalid_falls_back_to_auto(self):
        assert isinstance(select_detection_strategy("bad", npu_cores=2), CorePinnedStrategy)
        assert isinstance(select_detection_strategy("", npu_cores=1), SerialStrategy)
        assert isinstance(select_detection_strategy(None, npu_cores=2), CorePinnedStrategy)


class TestDefaultConfig:
    def test_default_strategy_is_auto(self):
        assert config.get_default_global_settings()["detection_strategy"] == "auto"
        assert config.DEFAULT_GLOBAL_SETTINGS["detection_strategy"] == "auto"