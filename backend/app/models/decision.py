"""决策层输出模型。"""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from app.models.base import BaseSchema
from app.models.enums import UrgencyLevel


class DecisionResult(BaseSchema):
    """规则决策层的输出。

    Attributes:
        strategy: 选定的回复策略（如 "共情安抚"、"紧急处理"）。
        actions: 待执行动作列表，供回复层与后续执行层消费。
        urgency: 决策确认的紧急程度。
        needs_escalation: 是否需要升级处理（如人工介入、上报）。
        rationale: 决策理由，便于调试与审计。
        extras: 扩展字段。
    """

    strategy: str = Field(..., description="选定的回复策略")
    actions: list[str] = Field(default_factory=list, description="待执行动作列表")
    urgency: UrgencyLevel = Field(..., description="决策确认的紧急程度")
    needs_escalation: bool = Field(default=False, description="是否需要升级处理")
    rationale: Optional[str] = Field(default=None, description="决策理由")
    extras: dict = Field(default_factory=dict, description="扩展字段")
