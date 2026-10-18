"""数据模型（Pydantic schemas）。

本模块定义系统内部统一的数据结构，供感知层、决策层、回复层与对话管理共享。
所有模型均继承 :class:`BaseSchema`，默认允许额外字段（extra="allow"），
以便在不破坏既有结构的前提下平滑扩展。
"""

from app.models.base import BaseSchema
from app.models.enums import UrgencyLevel
from app.models.user import UserInput
from app.models.perception import (
    EmotionResult,
    IntensityResult,
    EventResult,
    PerceptionResult,
)
from app.models.decision import DecisionResult
from app.models.response import ResponseResult
from app.models.conversation import ConversationState

__all__ = [
    "BaseSchema",
    "UrgencyLevel",
    "UserInput",
    "EmotionResult",
    "IntensityResult",
    "EventResult",
    "PerceptionResult",
    "DecisionResult",
    "ResponseResult",
    "ConversationState",
]
