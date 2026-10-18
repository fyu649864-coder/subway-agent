"""回复系统测试。"""

from __future__ import annotations

from app.decision.rules import (
    STRATEGY_ACTION,
    STRATEGY_COMFORT,
    STRATEGY_COMFORT_ACTION,
    STRATEGY_CONSERVATIVE,
    STRATEGY_EMERGENCY,
)
from app.models import DecisionResult, PerceptionResult, UrgencyLevel
from app.response import templates
from app.services.response_service import ResponseService


def _perception(emotion="焦虑", event="地铁延误", intensity="高") -> PerceptionResult:
    return PerceptionResult(
        text="地铁又延误了，我要迟到了",
        emotion=emotion,
        emotion_score=0.85,
        emotion_confidence=0.85,
        intensity=0.8,
        emotion_intensity=intensity,
        intensity_evidence=["情绪词: 急"],
        event=event,
        event_score=0.9,
        event_confidence=0.9,
        event_evidence=["延误"],
        urgency=UrgencyLevel.high,
    )


def _decision(strategy=STRATEGY_COMFORT_ACTION) -> DecisionResult:
    return DecisionResult(
        strategy=strategy,
        actions=["共情安抚", "提供信息"],
        urgency=UrgencyLevel.high,
        needs_escalation=False,
        rationale="test",
    )


def test_render_all_strategies() -> None:
    for s in templates.available_strategies():
        tid, reply = templates.render(s)
        assert tid.startswith("tpl_")
        assert isinstance(reply, str) and len(reply) > 10


def test_unknown_strategy_fallback_conservative() -> None:
    tid, reply = templates.render("不存在的策略")
    assert tid.startswith("tpl_conservative")
    assert reply


def test_response_generation_fields() -> None:
    svc = ResponseService(seed=42)
    p = _perception()
    d = _decision(STRATEGY_COMFORT_ACTION)
    r = svc.generate(p, d)
    assert r.reply
    assert r.template_id.startswith("tpl_comfort_action")
    assert r.emotion == p.emotion
    assert r.emotion_confidence == p.emotion_confidence
    assert r.emotion_intensity == p.emotion_intensity
    assert r.event == p.event
    assert r.event_confidence == p.event_confidence
    assert r.urgency == d.urgency
    assert r.decision.strategy == d.strategy


def test_emergency_reply_mentions_safety() -> None:
    svc = ResponseService(seed=1)
    r = svc.generate(_perception(event="突发事件"), _decision(STRATEGY_EMERGENCY))
    assert "安全" in r.reply or "工作人员" in r.reply


def test_conservative_does_not_invent_facts() -> None:
    # 保守策略不应编造具体实时信息/线路编号
    svc = ResponseService(seed=7)
    for _ in range(5):
        r = svc.generate(_perception(event="其他/未知事件", emotion="平静"),
                         _decision(STRATEGY_CONSERVATIVE))
        assert "1号线" not in r.reply and "2号线" not in r.reply
        assert "预计" not in r.reply  # 不编造预计时间


def test_reply_is_chinese_and_empathetic() -> None:
    svc = ResponseService(seed=3)
    r = svc.generate(_perception(), _decision(STRATEGY_COMFORT))
    # 含共情相关中文表达（新版模板避免机械套话，使用更自然的共情词）
    assert any(w in r.reply for w in ("理解", "抱歉", "心情", "感受", "难熬", "不好受", "糟心", "急人", "磨人", "不安", "发愁", "无奈"))


def test_seed_reproducible() -> None:
    p, d = _perception(), _decision()
    a = ResponseService(seed=123).generate(p, d).reply
    b = ResponseService(seed=123).generate(p, d).reply
    assert a == b
