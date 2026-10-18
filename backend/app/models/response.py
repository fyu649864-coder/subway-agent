"""回复层输出模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from app.models.base import BaseSchema
from app.models.decision import DecisionResult
from app.models.enums import UrgencyLevel
from app.models.perception import PerceptionResult


class ResponseResult(BaseSchema):
    """最终返回给用户的结构化结果。

    顶层扁平字段与感知结果对齐，方便前端直接消费；同时内嵌
    ``perception`` 与 ``decision`` 保留完整链路，便于调试与审计。
    """

    reply: str = Field(..., description="回复给用户的文本")
    template_id: Optional[str] = Field(default=None, description="使用的回复模板 ID")
    # 以下扁平字段与感知结果对齐，匹配 Mock 示例输出
    text: str = Field(..., description="原始输入文本")
    emotion: str = Field(..., description="主情绪标签")
    emotion_score: float = Field(..., ge=0.0, le=1.0, description="主情绪置信度")
    emotion_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="主情绪置信度")
    intensity: float = Field(..., ge=0.0, le=1.0, description="情绪强度")
    emotion_intensity: Optional[str] = Field(default=None, description="情绪强度档位：低/中/高")
    intensity_evidence: list[str] = Field(default_factory=list, description="强度判断依据")
    event: str = Field(..., description="主事件标签")
    event_score: float = Field(..., ge=0.0, le=1.0, description="主事件置信度")
    event_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="主事件置信度")
    event_evidence: list[str] = Field(default_factory=list, description="事件匹配依据")
    urgency: UrgencyLevel = Field(..., description="紧急程度")
    perception: PerceptionResult = Field(..., description="感知层完整结果")
    decision: DecisionResult = Field(..., description="决策层完整结果")
    reply_time: datetime = Field(default_factory=datetime.now, description="回复生成时间")
    extras: dict = Field(default_factory=dict, description="扩展字段")
