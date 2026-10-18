"""决策层包。

后续阶段将在 :mod:`app.decision.rules` 中实现规则引擎，
在 :mod:`app.decision.strategy` 中实现 Strategy Manager。
当前阶段仅定义接口契约。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.decision import DecisionResult
from app.models.perception import PerceptionResult


class BaseDecision(ABC):
    """决策层接口：输入感知结果，输出决策结果。"""

    @abstractmethod
    def decide(self, perception: PerceptionResult) -> DecisionResult:
        """根据感知结果选择回复策略与动作。"""
        raise NotImplementedError
