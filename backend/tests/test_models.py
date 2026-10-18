"""数据模型（Pydantic）基础测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models import (
    ConversationState,
    DecisionResult,
    EmotionResult,
    EventResult,
    IntensityResult,
    PerceptionResult,
    ResponseResult,
    UrgencyLevel,
    UserInput,
)


def test_all_models_importable() -> None:
    """所有核心模型均可正常导入。"""
    for cls in (
        UserInput,
        EmotionResult,
        IntensityResult,
        EventResult,
        PerceptionResult,
        DecisionResult,
        ResponseResult,
        ConversationState,
    ):
        assert cls is not None


def test_user_input_requires_text() -> None:
    with pytest.raises(ValidationError):
        UserInput()  # type: ignore[call-arg]


def test_user_input_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        UserInput(text="")


def test_user_input_extra_fields_allowed() -> None:
    """结构可扩展：允许额外字段。"""
    ui = UserInput(text="你好", extra_field="value")
    assert ui.text == "你好"


def test_perception_result_score_bounds() -> None:
    with pytest.raises(ValidationError):
        PerceptionResult(
            text="x",
            emotion="焦虑",
            emotion_score=1.5,  # 越界
            intensity=0.5,
            event="地铁延误",
            event_score=0.9,
            urgency=UrgencyLevel.high,
        )


def test_urgency_enum_values() -> None:
    assert UrgencyLevel.low.value == "low"
    assert UrgencyLevel.medium.value == "medium"
    assert UrgencyLevel.high.value == "high"


def test_perception_result_construction() -> None:
    result = PerceptionResult(
        text="地铁怎么还不来",
        emotion="焦虑",
        emotion_score=0.91,
        intensity=0.87,
        event="地铁延误",
        event_score=0.94,
        urgency=UrgencyLevel.high,
    )
    assert result.emotion == "焦虑"
    assert result.urgency is UrgencyLevel.high
