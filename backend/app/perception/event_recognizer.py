"""事件识别（本地规则 + 关键词匹配）。

不调用任何云端大模型或第三方 AI API，纯本地规则。
识别 8 类事件，支持多事件同时命中，无充分证据时返回「其他/未知事件」。
置信度基于匹配关键词的数量与强度，是规则匹配的确定程度，不是训练得到的概率。
"""

from __future__ import annotations

from dataclasses import dataclass

# 8 类事件及其关键词/同义词（按从强到弱、从具体到宽泛排列）
_EVENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "地铁延误": (
        "延误", "晚点", "列车延误", "车晚了", "还不来", "怎么还不来", "等了好久",
        "等了半小时", "等了一个小时", "一直不来", "迟迟不来", "龟速", "开得慢",
    ),
    "车厢拥挤": (
        "拥挤", "好挤", "太挤", "挤死", "挤不上", "上不去", "爆满", "人太多",
        "人满为患", "塞满", "沙丁鱼", "挤不上去", "挤不下", "又挤", "挤",
    ),
    "末班车与错过列车": (
        "末班车", "最后一班", "错过车", "没赶上", "没坐上", "赶不上车", "车走了",
        "关门了", "门关了", "就差一点", "差一步", "追不上",
    ),
    "换乘困难": (
        "换乘", "转乘", "换线", "找不到出口", "找不到换乘", "绕路", "走了很远",
        "通道", "换乘通道", "换乘站",
    ),
    "赶时间": (
        "赶考试", "赶考", "赶飞机", "赶高铁", "赶火车", "赶时间", "要迟到",
        "快迟到", "来不及", "马上要迟到", "要误机", "要误车", "急着", "赶着",
        "赶不上", "误机", "误车",
    ),
    "突发事件": (
        "事故", "故障", "停运", "冒烟", "着火", "急刹", "紧急刹车", "晕倒",
        "受伤", "打架", "骚乱", "有人", "突发事件", "警报", "广播说",
    ),
    "服务投诉": (
        "投诉", "服务差", "态度差", "垃圾", "什么破", "差评", "举报", "反馈",
        "没人管", "工作人员", "形同虚设",
    ),
}

EVENT_OTHER = "其他/未知事件"

# 强信号词：命中后显著提升置信度
_STRONG_HINTS = ("末班车", "延误", "投诉", "事故", "故障", "着火", "冒烟", "赶飞机", "赶考试")


@dataclass
class EventHit:
    """单个事件类别的命中结果。"""

    event: str
    confidence: float  # [0,1]，规则匹配确定度，非概率
    evidence: list[str]  # 命中的关键词


def _match_category(text: str, keywords: tuple[str, ...]) -> list[str]:
    return [kw for kw in keywords if kw in text]


def recognize_events(text: str) -> list[EventHit]:
    """识别文本中的事件，返回命中的事件列表（按置信度降序）。

    - 多事件可同时返回。
    - 没有任何关键词命中时返回 ``[EventHit(其他/未知事件, 低置信度, [])]``。
    """
    if not isinstance(text, str):
        text = ""
    text = text.strip()

    hits: list[EventHit] = []
    for event, keywords in _EVENT_KEYWORDS.items():
        matched = _match_category(text, keywords)
        if not matched:
            continue
        # 置信度：基础 0.6 + 每个匹配词 0.05（上限 0.95），强信号额外 +0.1
        score = 0.6 + min(len(matched), 6) * 0.05
        if any(kw in matched for kw in _STRONG_HINTS):
            score += 0.1
        score = min(score, 0.95)
        hits.append(EventHit(event=event, confidence=round(score, 2), evidence=matched))

    if not hits:
        # 证据不足，保守返回未知事件
        hits.append(EventHit(event=EVENT_OTHER, confidence=0.3, evidence=[]))

    hits.sort(key=lambda h: h.confidence, reverse=True)
    return hits


def primary_event(text: str) -> EventHit:
    """返回主事件（置信度最高的事件，或未知事件）。"""
    return recognize_events(text)[0]
