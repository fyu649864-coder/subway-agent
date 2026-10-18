"""对话状态模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from app.models.base import BaseSchema
from app.models.enums import UrgencyLevel
from app.models.perception import PerceptionResult


class ConversationState(BaseSchema):
    """单个会话的状态快照。

    本阶段仅在内存中维护，后续阶段可替换为持久化存储。
    """

    session_id: str = Field(..., description="会话 ID")
    turn_count: int = Field(default=0, description="已处理轮次")
    created_at: datetime = Field(default_factory=datetime.now, description="会话创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="最近更新时间")
    current_urgency: Optional[UrgencyLevel] = Field(default=None, description="当前紧急程度")
    history: list[PerceptionResult] = Field(default_factory=list, description="历史感知结果")
    context: dict = Field(default_factory=dict, description="会话上下文，预留字段")
