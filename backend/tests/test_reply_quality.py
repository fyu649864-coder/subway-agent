"""回复质量专项测试。

不只检查 HTTP 状态码，而是检查：
- 回复是否回应用户的具体诉求（赶考试 / 家人等待 / 车厢拥挤 / 末班车）；
- 多轮对话是否避免连续重复套话；
- 是否编造线路编号、预计时间等未经确认的事实；
- 紧急情况安全规则是否保留。
"""

from __future__ import annotations

from app.decision.rules import (
    STRATEGY_ACTION,
    STRATEGY_COMFORT,
    STRATEGY_COMFORT_ACTION,
    STRATEGY_CONSERVATIVE,
    STRATEGY_EMERGENCY,
)
from app.models import DecisionResult, PerceptionResult, UrgencyLevel
from app.response.templates import render
from app.services.response_service import ResponseService


def _perception(text: str, emotion: str = "焦虑", event: str = "地铁延误",
                intensity: str = "中", event_ev: list[str] | None = None) -> PerceptionResult:
    return PerceptionResult(
        text=text,
        emotion=emotion,
        emotion_score=0.8,
        emotion_confidence=0.8,
        intensity=0.6,
        emotion_intensity=intensity,
        intensity_evidence=["情绪词: 急"],
        event=event,
        event_score=0.85,
        event_confidence=0.85,
        event_evidence=event_ev or [event],
        urgency=UrgencyLevel.medium,
    )


def _decision(strategy: str) -> DecisionResult:
    return DecisionResult(
        strategy=strategy,
        actions=["x"],
        urgency=UrgencyLevel.medium,
        rationale="test",
    )


def _reply(text: str, strategy: str, **kw) -> str:
    svc = ResponseService(seed=2026)
    return svc.generate(_perception(text, **kw), _decision(strategy)).reply


_FORBIDDEN = ("1号线", "2号线", "3号线", "4号线", "5号线", "10号线", "预计", "马上恢复", "五分钟")


def _assert_no_invented_facts(reply: str) -> None:
    for w in _FORBIDDEN:
        assert w not in reply, f"回复编造了未经确认的信息: {w!r} in {reply!r}"


def test_reply_acknowledges_exam_target() -> None:
    """赶考试场景：回复必须回应“考试”这一具体诉求，而非泛泛安慰。"""
    r = _reply("地铁怎么还不来啊，我九点要赶考试快迟到了！", STRATEGY_COMFORT_ACTION, event="赶时间")
    assert "考试" in r
    _assert_no_invented_facts(r)


def test_reply_acknowledges_family_waiting() -> None:
    """家人在等待：回复需呼应“家人/家里人在等”。"""
    r = _reply("车一直不来，家里人还在等我回去吃饭，急死了", STRATEGY_COMFORT_ACTION, event="地铁延误")
    assert any(w in r for w in ("家人", "家里人"))
    _assert_no_invented_facts(r)


def test_reply_acknowledges_crowd() -> None:
    """车厢拥挤挤不上：回复需回应拥挤处境。"""
    r = _reply("这班也太挤了根本挤不上，烦死了", STRATEGY_ACTION, event="车厢拥挤", intensity="高")
    assert any(w in r for w in ("挤", "车厢"))
    _assert_no_invented_facts(r)


def test_reply_acknowledges_last_train() -> None:
    """末班车没赶上：回复需回应末班车/回家处境，并给出备选建议。"""
    r = _reply("末班车都走了我没赶上，回不去了怎么办", STRATEGY_COMFORT_ACTION, event="末班车与错过列车")
    assert any(w in r for w in ("末班车", "回家", "回不去"))
    assert any(w in r for w in ("打车", "公交", "备选"))
    _assert_no_invented_facts(r)


def test_multi_turn_no_identical_replies() -> None:
    """同一会话连续两轮相同输入，回复不应完全相同（避免机械重复）。"""
    svc = ResponseService(seed=7)
    p1 = _perception("地铁延误了我要迟到了", event="地铁延误")
    d = _decision(STRATEGY_COMFORT_ACTION)
    r1 = svc.generate(p1, d, history=[])
    p2 = _perception("地铁延误了我要迟到了", event="地铁延误")
    r2 = svc.generate(p2, d, history=[p1])
    assert r1.reply != r2.reply, f"两轮回复完全相同: {r1.reply!r}"


def test_multi_turn_continuation_acknowledges_prior() -> None:
    """多轮补充信息：第二轮在同事件下补充细节，回复仍需回应新信息。"""
    svc = ResponseService(seed=11)
    p1 = _perception("地铁延误了", event="地铁延误")
    d = _decision(STRATEGY_COMFORT_ACTION)
    r1 = svc.generate(p1, d, history=[])
    p2 = _perception("我还要赶考试快迟到了", event="赶时间")
    r2 = svc.generate(p2, d, history=[p1])
    assert "考试" in r2.reply
    assert r1.reply != r2.reply


def test_diverse_inputs_diverse_openings() -> None:
    """不同情境的回复开头不应千篇一律（避免所有回复都用同一句套话开头）。"""
    cases = [
        ("我要赶考试来不及了", STRATEGY_COMFORT_ACTION, "赶时间"),
        ("末班车走了我回不去了", STRATEGY_COMFORT_ACTION, "末班车与错过列车"),
        ("车厢挤不上去烦死了", STRATEGY_ACTION, "车厢拥挤"),
        ("这服务态度太差我要投诉", STRATEGY_COMFORT, "服务投诉"),
        ("今天天气不错", STRATEGY_CONSERVATIVE, "其他/未知事件"),
    ]
    openings = []
    for text, strat, event in cases:
        r = _reply(text, strat, event=event)
        openings.append(r[:4])
        _assert_no_invented_facts(r)
    assert len(set(openings)) >= 3, f"回复开头过于单一: {openings}"


def test_emergency_keeps_safety_rule() -> None:
    """紧急情况策略必须保留安全提示，不被自然度优化削弱。"""
    for _ in range(5):
        _, r = render(STRATEGY_EMERGENCY, perception=_perception("冒烟了", event="突发事件"))
        assert "安全" in r or "工作人员" in r


def test_low_intensity_no_forced_comfort() -> None:
    """低强度、无负面情绪词的信息型输入，不应强行塞入过度安慰。"""
    r = _reply("请问末班车几点", STRATEGY_ACTION, event="末班车与错过列车",
               emotion="平静", intensity="低")
    _assert_no_invented_facts(r)
    # 不出现过度安抚套话
    assert "深表歉意" not in r
