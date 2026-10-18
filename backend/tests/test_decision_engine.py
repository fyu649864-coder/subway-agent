"""规则决策引擎测试。

直接构造 PerceptionResult 验证规则匹配，不依赖情绪模型推理。
"""

from __future__ import annotations

from app.decision import rules
from app.decision.rules import (
    STRATEGY_ACTION,
    STRATEGY_CONSERVATIVE,
    STRATEGY_COMFORT,
    STRATEGY_COMFORT_ACTION,
    STRATEGY_EMERGENCY,
)
from app.models import PerceptionResult, UrgencyLevel
from app.services.decision_service import DecisionService


def _perception(
    emotion: str = "平静",
    event: str = "其他/未知事件",
    intensity: str = "中",
    urgency: UrgencyLevel = UrgencyLevel.low,
) -> PerceptionResult:
    return PerceptionResult(
        text="测试文本",
        emotion=emotion,
        emotion_score=0.8,
        emotion_confidence=0.8,
        intensity=0.6,
        emotion_intensity=intensity,
        intensity_evidence=[],
        event=event,
        event_score=0.9,
        event_confidence=0.9,
        event_evidence=[],
        urgency=urgency,
    )


def test_emergency_event() -> None:
    p = _perception(emotion="恐惧", event="突发事件", intensity="高")
    d = DecisionService().decide(p)
    assert d.strategy == STRATEGY_EMERGENCY
    assert d.urgency == UrgencyLevel.high
    assert d.needs_escalation is True
    assert "安全" in "".join(d.actions) or "工作人员" in "".join(d.actions)


def test_delay_high_intensity_comfort_plus_action() -> None:
    p = _perception(emotion="焦虑", event="地铁延误", intensity="高")
    d = rules.evaluate(p)
    assert d.strategy == STRATEGY_COMFORT_ACTION
    assert d.urgency == UrgencyLevel.high


def test_delay_low_intensity_comfort_action() -> None:
    # 策略调整：延误不再自动进入纯客服建议模式，安抚优先
    p = _perception(emotion="平静", event="地铁延误", intensity="低")
    d = rules.evaluate(p)
    assert d.strategy == STRATEGY_COMFORT_ACTION


def test_crowd_medium_comfort_action() -> None:
    p = _perception(emotion="烦躁", event="车厢拥挤", intensity="中")
    d = rules.evaluate(p)
    # 策略调整：拥挤中低强度以安抚为主（comfort），高强度才 comfort+action
    assert d.strategy in {STRATEGY_COMFORT, STRATEGY_COMFORT_ACTION}
    assert d.actions


def test_last_train_comfort_action() -> None:
    p = _perception(emotion="无助", event="末班车与错过列车", intensity="中")
    d = rules.evaluate(p)
    assert d.strategy == STRATEGY_COMFORT_ACTION
    assert any("备选" in a or "打车" in a or "公交" in a for a in d.actions)


def test_rush_time_comfort_action() -> None:
    # 策略调整：赶时间先安抚情绪，建议按需给
    p = _perception(emotion="焦虑", event="赶时间", intensity="高")
    d = rules.evaluate(p)
    assert d.strategy == STRATEGY_COMFORT_ACTION
    assert d.urgency == UrgencyLevel.high


def test_transfer_difficulty_action() -> None:
    p = _perception(emotion="烦躁", event="换乘困难", intensity="中")
    d = rules.evaluate(p)
    assert d.strategy == STRATEGY_ACTION


def test_complaint_comfort_and_escalation() -> None:
    p = _perception(emotion="愤怒", event="服务投诉", intensity="高")
    d = rules.evaluate(p)
    assert d.strategy == STRATEGY_COMFORT
    assert d.needs_escalation is True
    assert d.urgency == UrgencyLevel.high


def test_evidence_insufficient_conservative() -> None:
    # 事件未知 + 情绪平静 → 保守
    p = _perception(emotion="平静", event="其他/未知事件", intensity="中")
    d = rules.evaluate(p)
    assert d.strategy == STRATEGY_CONSERVATIVE
    assert d.urgency == UrgencyLevel.low
    assert d.needs_escalation is False


def test_intensity_only_comfort_when_no_event() -> None:
    # 无明确事件但高强度 → 安抚
    p = _perception(emotion="烦躁", event="其他/未知事件", intensity="高")
    d = rules.evaluate(p)
    assert d.strategy == STRATEGY_COMFORT


def test_decision_result_has_rationale_and_rule() -> None:
    p = _perception(emotion="焦虑", event="地铁延误", intensity="高")
    d = rules.evaluate(p)
    assert d.rationale
    assert d.extras.get("matched_rule")


def test_all_scenarios_produce_valid_decision() -> None:
    """覆盖主要场景均能产出合法决策结果（无异常、字段齐全）。"""
    scenarios = [
        ("焦虑", "地铁延误", "高"),
        ("烦躁", "车厢拥挤", "中"),
        ("无助", "末班车与错过列车", "高"),
        ("焦虑", "赶时间", "高"),
        ("恐惧", "突发事件", "高"),
        ("愤怒", "服务投诉", "高"),
        ("平静", "其他/未知事件", "中"),
        ("烦躁", "换乘困难", "低"),
    ]
    for emo, evt, inten in scenarios:
        d = DecisionService().decide(_perception(emotion=emo, event=evt, intensity=inten))
        assert d.strategy
        assert isinstance(d.actions, list) and d.actions
        assert d.urgency in {UrgencyLevel.low, UrgencyLevel.medium, UrgencyLevel.high}
