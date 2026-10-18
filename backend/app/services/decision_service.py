"""决策服务。

委托给独立的规则决策引擎（:mod:`app.decision.rules`），根据
「情绪 + 事件 + 强度」产出结构化决策结果，供回复层消费。

本类保留原有薄封装，便于后续替换为更复杂的策略管理器而不影响上层调用链。
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.decision import rules
from app.models import DecisionResult, PerceptionResult

logger = get_logger(__name__)


class DecisionService:
    """决策服务：PerceptionResult -> DecisionResult。"""

    def decide(self, perception: PerceptionResult) -> DecisionResult:
        """根据感知结果选择策略与动作（规则决策引擎）。"""

        result = rules.evaluate(perception)
        logger.info(
            "决策结果: strategy=%s urgency=%s escalation=%s actions=%s rule=%s",
            result.strategy,
            result.urgency.value,
            result.needs_escalation,
            result.actions,
            result.extras.get("matched_rule"),
        )
        return result
