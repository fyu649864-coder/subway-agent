"""感知层输出模型：情绪、强度、事件及聚合结果。"""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from app.models.base import BaseSchema
from app.models.enums import UrgencyLevel


class EmotionResult(BaseSchema):
    """情绪识别结果。

    Attributes:
        emotion: 主情绪标签（开放式字符串，如 "焦虑"、"愤怒"）。
        emotion_score: 主情绪置信度，取值 [0, 1]。
        emotions: 全量情绪分布，预留字段（如 {"焦虑": 0.91, "愤怒": 0.05}）。
        model_name: 产出该结果的模型标识，预留字段。
    """

    emotion: str = Field(..., description="主情绪标签")
    emotion_score: float = Field(..., ge=0.0, le=1.0, description="主情绪置信度")
    emotions: Optional[dict[str, float]] = Field(default=None, description="全量情绪分布")
    model_name: Optional[str] = Field(default=None, description="模型标识")


class IntensityResult(BaseSchema):
    """情绪强度识别结果。"""

    intensity: float = Field(..., ge=0.0, le=1.0, description="情绪强度，取值 [0, 1]")
    level: Optional[str] = Field(default=None, description="强度档位：低/中/高")
    evidence: list[str] = Field(default_factory=list, description="强度判断依据")
    model_name: Optional[str] = Field(default=None, description="模型标识")


class EventResult(BaseSchema):
    """事件识别结果。

    Attributes:
        event: 主事件标签（开放式字符串，如 "地铁延误"、"突发事故"）。
        event_score: 主事件置信度，取值 [0, 1]。
        events: 全量事件分布，预留字段。
        model_name: 产出该结果的模型标识，预留字段。
    """

    event: str = Field(..., description="主事件标签")
    event_score: float = Field(..., ge=0.0, le=1.0, description="主事件置信度")
    events: Optional[dict[str, float]] = Field(default=None, description="全量事件分布")
    evidence: list[str] = Field(default_factory=list, description="事件匹配依据（命中的关键词）")
    all_events: Optional[list[dict]] = Field(default=None, description="所有命中事件明细")
    model_name: Optional[str] = Field(default=None, description="模型标识")


class PerceptionResult(BaseSchema):
    """感知层聚合输出，是决策层的输入。

    顶层字段保持扁平，便于与示例 Mock 数据对齐；``*_detail`` 字段保存
    各子任务的完整结果，供后续阶段使用。
    """

    text: str = Field(..., description="原始输入文本")
    emotion: str = Field(..., description="主情绪标签")
    emotion_score: float = Field(..., ge=0.0, le=1.0, description="主情绪置信度")
    emotion_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="主情绪置信度（emotion_score 别名）")
    intensity: float = Field(..., ge=0.0, le=1.0, description="情绪强度")
    emotion_intensity: Optional[str] = Field(default=None, description="情绪强度档位：低/中/高")
    intensity_evidence: list[str] = Field(default_factory=list, description="强度判断依据")
    event: str = Field(..., description="主事件标签")
    event_score: float = Field(..., ge=0.0, le=1.0, description="主事件置信度")
    event_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="主事件置信度（event_score 别名）")
    event_evidence: list[str] = Field(default_factory=list, description="事件匹配依据")
    urgency: UrgencyLevel = Field(..., description="紧急程度")
    emotion_detail: Optional[EmotionResult] = Field(default=None, description="情绪识别完整结果")
    event_detail: Optional[EventResult] = Field(default=None, description="事件识别完整结果")
    intensity_detail: Optional[IntensityResult] = Field(default=None, description="强度识别完整结果")
    extras: dict = Field(default_factory=dict, description="扩展字段")
