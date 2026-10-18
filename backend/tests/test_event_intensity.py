"""事件识别 + 情绪强度 + 模型整合测试。

分别验证：
- 事件识别规则（确定性，不依赖模型）
- 情绪强度规则（确定性，不依赖模型）
- 本地情绪模型可正常加载与调用（不假定预测出指定情绪）
- 感知服务端到端整合
"""

from __future__ import annotations

import pytest

from app.perception.event_recognizer import EVENT_OTHER, recognize_events
from app.perception.intensity_analyzer import (
    INTENSITY_HIGH,
    INTENSITY_LOW,
    INTENSITY_MID,
    analyze_intensity,
)
from app.services.perception_service import PerceptionService

VALID_EMOTIONS = {"平静", "焦虑", "愤怒", "悲伤", "恐惧", "烦躁", "无助"}


def _events(text: str) -> list[str]:
    return [h.event for h in recognize_events(text)]


# ---------------------------------------------------------------------------
# 1-7. 事件识别规则（不依赖模型）
# ---------------------------------------------------------------------------
def test_event_delay_anxiety() -> None:
    text = "地铁又延误了，我等了半小时还不来，真的要迟到了"
    events = _events(text)
    assert "地铁延误" in events
    assert "赶时间" in events
    primary = recognize_events(text)[0]
    assert primary.evidence  # 有匹配依据
    assert 0.0 < primary.confidence <= 1.0


def test_event_crowd() -> None:
    text = "这节车厢也太挤了，完全上不去，人太多了"
    assert "车厢拥挤" in _events(text)


def test_event_last_train() -> None:
    text = "末班车都走了，我没赶上，今天回不去了"
    events = _events(text)
    assert "末班车与错过列车" in events
    # 末班车语境下易伴随无助感，强度至少中等
    assert analyze_intensity(text).level in {INTENSITY_MID, INTENSITY_HIGH}


def test_event_rush_exam() -> None:
    text = "我要赶考试，这班车再不来就来不及了"
    assert "赶时间" in _events(text)


def test_event_emergency() -> None:
    text = "前面车厢好像冒烟了，是不是出事故了"
    assert "突发事件" in _events(text)


def test_event_complaint_anger() -> None:
    text = "这服务态度也太差了，我要投诉，什么破地铁"
    assert "服务投诉" in _events(text)


def test_event_unknown_plain_text() -> None:
    text = "今天天气不错，我坐地铁去公司"
    hits = recognize_events(text)
    assert hits[0].event == EVENT_OTHER
    assert hits[0].confidence < 0.5  # 证据不足，低置信度


# ---------------------------------------------------------------------------
# 8. 多事件同时出现
# ---------------------------------------------------------------------------
def test_multiple_events() -> None:
    text = "地铁延误了半个小时，车厢又挤，我还赶不上飞机，真的崩溃"
    events = _events(text)
    for expected in ("地铁延误", "车厢拥挤", "赶时间"):
        assert expected in events


# ---------------------------------------------------------------------------
# 9. 空文本 / 异常输入
# ---------------------------------------------------------------------------
def test_empty_and_invalid_input() -> None:
    assert analyze_intensity("").level == INTENSITY_LOW
    assert analyze_intensity("   ").level == INTENSITY_LOW
    assert analyze_intensity(None).level == INTENSITY_LOW  # type: ignore[arg-type]
    assert analyze_intensity(123).level == INTENSITY_LOW  # type: ignore[arg-type]

    assert recognize_events("")[0].event == EVENT_OTHER
    assert recognize_events(None)[0].event == EVENT_OTHER  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 10. 情绪类别与强度不一致：平静陈述但高强度表达，或强烈情绪词但缓和语气
# ---------------------------------------------------------------------------
def test_intensity_independent_of_emotion_keywords() -> None:
    # 没有明显情绪关键词，但用了高程度词 + 重复标点 → 高强度（不依赖情绪词）
    r1 = analyze_intensity("这也太慢了吧！！！怎么回事啊！！！")
    assert r1.level == INTENSITY_HIGH
    assert r1.evidence  # 有依据

    # 有情绪词但被缓和语气弱化 → 不应判高
    r2 = analyze_intensity("我有点生气，还好问题不大")
    assert r2.level in {INTENSITY_LOW, INTENSITY_MID}

    # 普通陈述无任何信号 → 保守中等
    r3 = analyze_intensity("我坐二号线去人民广场")
    assert r3.level == INTENSITY_MID


def test_intensity_not_only_exclamation() -> None:
    # 单个感叹号不应直接判高
    assert analyze_intensity("知道了。").level in {INTENSITY_LOW, INTENSITY_MID}
    assert analyze_intensity("好的！").level in {INTENSITY_LOW, INTENSITY_MID}


# ---------------------------------------------------------------------------
# 模型整合：确认本地模型可加载并返回合法结构（不校验具体情绪）
# ---------------------------------------------------------------------------
def test_emotion_model_loads_and_predicts() -> None:
    service = PerceptionService()
    result = service.analyze("地铁怎么还不来，我马上要迟到了！")
    assert result.emotion in VALID_EMOTIONS
    assert 0.0 < result.emotion_score <= 1.0
    assert result.emotion_detail is not None
    assert result.emotion_detail.emotions is not None
    assert set(result.emotion_detail.emotions.keys()) == VALID_EMOTIONS
    assert result.emotion_intensity in {INTENSITY_LOW, INTENSITY_MID, INTENSITY_HIGH}
    assert result.event
    assert result.urgency.value in {"low", "medium", "high"}


def test_perception_service_fields_populated() -> None:
    service = PerceptionService()
    result = service.analyze("这车厢太挤了，气死我了！")
    # 新字段均已填充
    assert result.emotion_confidence == result.emotion_score
    assert result.event_confidence == result.event_score
    assert isinstance(result.intensity_evidence, list)
    assert isinstance(result.event_evidence, list)
    assert result.emotion_intensity in {INTENSITY_LOW, INTENSITY_MID, INTENSITY_HIGH}
    # 事件明细存在
    assert result.event_detail is not None
    assert result.event_detail.all_events  # list，非空


def test_model_load_failure_is_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    """模型加载失败时应明确抛错，不得静默返回固定/随机情绪。"""
    from app.perception import emotion_classifier as ec

    broken = ec.EmotionClassifier(model_path="/nonexistent/path/that/does/not/exist")
    service = PerceptionService(emotion_classifier=broken)
    with pytest.raises(ec.EmotionModelLoadError):
        service.analyze("测试文本")
