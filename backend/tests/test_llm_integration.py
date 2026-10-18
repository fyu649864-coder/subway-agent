"""本地大模型接入行为测试（不依赖真实模型，全部 mock）。"""

from __future__ import annotations

from unittest.mock import patch

from app.decision.rules import STRATEGY_ACTION, STRATEGY_EMERGENCY
from app.models import DecisionResult, PerceptionResult, UrgencyLevel
from app.services.response_service import ResponseService


def _perception(text="车怎么还不来，我要赶不上考试了", emotion="焦虑", event="赶时间"):
    return PerceptionResult(
        text=text,
        emotion=emotion,
        emotion_score=0.8,
        emotion_confidence=0.8,
        intensity=0.7,
        emotion_intensity="高",
        intensity_evidence=["急"],
        event=event,
        event_score=0.85,
        event_confidence=0.85,
        event_evidence=[],
        urgency=UrgencyLevel.high,
    )


def _decision(strategy=STRATEGY_ACTION):
    return DecisionResult(
        strategy=strategy,
        actions=["建议"],
        urgency=UrgencyLevel.high,
        needs_escalation=False,
        rationale="t",
    )


def test_llm_reply_used_when_available():
    svc = ResponseService(seed=1)
    with patch("app.services.response_service.llm_generator.generate_reply", return_value="这是大模型生成的回复") as m:
        res = svc.generate(_perception(), _decision())
    assert m.called
    assert res.reply == "这是大模型生成的回复"
    assert res.template_id == "llm_local"
    assert res.extras.get("llm_used") is True


def test_fallback_to_template_when_llm_none():
    svc = ResponseService(seed=1)
    with patch("app.services.response_service.llm_generator.generate_reply", return_value=None):
        res = svc.generate(_perception(), _decision())
    assert res.reply  # 非空
    assert res.template_id != "llm_local"
    assert res.extras.get("llm_used") is False


def test_emergency_never_calls_llm():
    svc = ResponseService(seed=1)
    with patch("app.services.response_service.llm_generator.generate_reply") as m:
        res = svc.generate(_perception(emotion="恐惧", event="突发事件"), _decision(STRATEGY_EMERGENCY))
    assert not m.called
    assert res.template_id.startswith("tpl_emergency")


def test_disabled_llm_skips(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_enabled", False, raising=False)
    from app.response import llm_generator
    llm_generator.reset_for_tests()
    assert llm_generator.generate_reply(_perception(), _decision()) is None
