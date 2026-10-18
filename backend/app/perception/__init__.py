"""感知层包。

后续阶段将在 :mod:`app.perception.emotion`、:mod:`app.perception.event`、
:mod:`app.perception.intensity` 中接入 BERT/RoBERTa 等本地推理模型。
当前阶段仅定义接口契约，具体 Mock 实现位于 :mod:`app.services`。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.perception import PerceptionResult


class BasePerception(ABC):
    """感知层接口：输入文本，输出结构化感知结果。"""

    @abstractmethod
    def analyze(self, text: str) -> PerceptionResult:
        """对单条文本进行情绪 / 强度 / 事件识别。"""
        raise NotImplementedError
