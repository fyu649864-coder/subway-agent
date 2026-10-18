"""规则决策引擎。

根据「情绪 + 事件 + 强度」匹配本地规则，输出结构化决策结果：
strategy（回复策略）、actions（动作列表）、urgency、needs_escalation、rationale。

设计原则：
- 独立模块，纯规则，不调用模型/云端。
- 规则按优先级匹配：突发事件安全类 > 明确事件+情绪+强度 > 仅事件 > 仅情绪 > 保守默认。
- 证据不足（事件=其他/未知 且 情绪=平静/无明显信号）时采用保守策略。
- 可扩展：新增规则只需在 ``RULES`` 中追加，不改动匹配逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.models.decision import DecisionResult
from app.models.enums import UrgencyLevel
from app.models.perception import PerceptionResult

# 策略常量
STRATEGY_EMERGENCY = "紧急情况提示"
STRATEGY_COMFORT = "情绪安抚"
STRATEGY_ACTION = "行动建议"
STRATEGY_COMFORT_ACTION = "安抚与建议结合"
STRATEGY_CONSERVATIVE = "保守常规回复"

# 需升级人工介入的事件
ESCALATION_EVENTS = {"突发事件", "服务投诉"}

# 高强度档位
HIGH_INTENSITY = {"高"}
MID_INTENSITY = {"中", "高"}


@dataclass
class Rule:
    """单条决策规则。

    匹配字段为 None 表示「通配」；数值越小优先级越高（越具体的规则越靠前）。
    """

    name: str
    strategy: str
    actions: list[str]
    rationale: str
    urgency: UrgencyLevel = UrgencyLevel.medium
    needs_escalation: bool = False
    # 匹配条件（None = 通配）
    event: str | None = None
    emotion: str | None = None
    intensity: str | None = None
    priority: int = 100


# ---------------------------------------------------------------------------
# 规则库：按 priority 升序匹配（越具体越优先）
# ---------------------------------------------------------------------------
RULES: list[Rule] = [
    # ---------- 最高优先级：突发事件，安全优先 ----------
    Rule(
        name="突发事件-安全提示",
        event="突发事件",
        strategy=STRATEGY_EMERGENCY,
        actions=["安全提示", "引导听从现场工作人员指引", "升级人工介入"],
        rationale="识别到车厢/站台突发事件，优先保障乘客安全并升级处理。",
        urgency=UrgencyLevel.high,
        needs_escalation=True,
        priority=1,
    ),
    # ---------- 末班车：安抚为主，建议其次（不强制推销方案） ----------
    Rule(
        name="末班车-安抚+建议",
        event="末班车与错过列车",
        emotion=None,
        intensity=None,
        strategy=STRATEGY_COMFORT_ACTION,
        actions=["共情安抚", "按需提供备选出行方案", "致歉"],
        rationale="错过末班车易伴随无助感，先安抚情绪，建议只在对方需要时给。",
        urgency=UrgencyLevel.medium,
        priority=10,
    ),
    # ---------- 赶时间：先安抚情绪，再按需建议 ----------
    Rule(
        name="赶时间-安抚+建议",
        event="赶时间",
        intensity=None,
        strategy=STRATEGY_COMFORT_ACTION,
        actions=["共情安抚", "按需提供出行建议", "不编造实时信息"],
        rationale="赶时间易焦虑，先回应情绪；建议仅在对方明确求助时给出。",
        urgency=UrgencyLevel.high,
        priority=11,
    ),
    # ---------- 地铁延误：按强度分层，均以安抚优先 ----------
    Rule(
        name="地铁延误-高强度",
        event="地铁延误",
        intensity="高",
        strategy=STRATEGY_COMFORT_ACTION,
        actions=["共情安抚（致歉）", "按需提供信息与建议"],
        rationale="延误伴随高强度负面情绪，以安抚为主，不推销解决方案。",
        urgency=UrgencyLevel.high,
        priority=20,
    ),
    Rule(
        name="地铁延误-中低强度",
        event="地铁延误",
        intensity=None,
        strategy=STRATEGY_COMFORT_ACTION,
        actions=["共情回应", "按需告知关注官方信息"],
        rationale="延误情境先回应感受，建议从简。",
        urgency=UrgencyLevel.medium,
        priority=21,
    ),
    # ---------- 车厢拥挤：按强度分层，均以安抚优先 ----------
    Rule(
        name="车厢拥挤-高强度",
        event="车厢拥挤",
        intensity="高",
        strategy=STRATEGY_COMFORT_ACTION,
        actions=["共情安抚", "按需建议缓解方式"],
        rationale="拥挤引发明显烦躁，先共情，建议从简。",
        urgency=UrgencyLevel.medium,
        priority=30,
    ),
    Rule(
        name="车厢拥挤-中低强度",
        event="车厢拥挤",
        intensity=None,
        strategy=STRATEGY_COMFORT,
        actions=["共情回应"],
        rationale="拥挤时先回应感受，不强行给建议。",
        urgency=UrgencyLevel.low,
        priority=31,
    ),
    # ---------- 换乘困难：建议为主（纯信息需求） ----------
    Rule(
        name="换乘困难-建议",
        event="换乘困难",
        intensity=None,
        strategy=STRATEGY_ACTION,
        actions=["提供换乘路线指引", "建议查看站内导向标识/APP"],
        rationale="换乘困难以信息与路线建议为主。",
        urgency=UrgencyLevel.low,
        priority=40,
    ),
    # ---------- 服务投诉：安抚 + 升级 ----------
    Rule(
        name="服务投诉-安抚+升级",
        event="服务投诉",
        intensity=None,
        strategy=STRATEGY_COMFORT,
        actions=["共情安抚并致歉", "记录反馈并转交相关部门", "升级人工跟进"],
        rationale="服务投诉需先安抚，再记录并升级人工处理。",
        urgency=UrgencyLevel.high,
        needs_escalation=True,
        priority=50,
    ),
    # ---------- 兜底：仅按情绪判断 ----------
    Rule(
        name="高强度负面情绪-安抚",
        event=None,
        intensity="高",
        strategy=STRATEGY_COMFORT,
        actions=["共情安抚", "温和询问具体情况"],
        rationale="检测到高强度情绪但无明确事件，以安抚为主并温和询问。",
        urgency=UrgencyLevel.medium,
        priority=80,
    ),
    Rule(
        name="中强度情绪-安抚+询问",
        event=None,
        intensity="中",
        strategy=STRATEGY_COMFORT_ACTION,
        actions=["共情回应", "温和询问需求"],
        rationale="中等强度情绪，共情并询问以补充信息。",
        urgency=UrgencyLevel.low,
        priority=81,
    ),
]

# 保守默认规则（证据不足时使用）
DEFAULT_RULE = Rule(
    name="默认-保守常规",
    strategy=STRATEGY_CONSERVATIVE,
    actions=["礼貌回应", "邀请乘客补充说明具体情况"],
    rationale="事件与情绪证据不足，采用保守策略，避免过度承诺。",
    urgency=UrgencyLevel.low,
    priority=999,
)

UNKNOWN_EVENT = "其他/未知事件"
CALM_EMOTIONS = {"平静"}


def _match(rule: Rule, perception: PerceptionResult) -> bool:
    if rule.event is not None and rule.event != perception.event:
        return False
    if rule.emotion is not None and rule.emotion != perception.emotion:
        return False
    if rule.intensity is not None and rule.intensity != perception.emotion_intensity:
        return False
    return True


def evaluate(perception: PerceptionResult) -> DecisionResult:
    """对感知结果执行规则匹配，返回决策结果。

    - 按 priority 升序遍历，返回第一条匹配的规则。
    - 事件为未知且情绪平静（无明显证据）时，直接走保守默认规则。
    """
    # 证据不足 → 保守
    if perception.event == UNKNOWN_EVENT and perception.emotion in CALM_EMOTIONS:
        rule = DEFAULT_RULE
    else:
        matched = next((r for r in sorted(RULES, key=lambda x: x.priority) if _match(r, perception)), None)
        rule = matched or DEFAULT_RULE

    # 决策层确认的紧急程度：规则给定，但突发事件/投诉强制 high
    urgency = rule.urgency
    if perception.event in ESCALATION_EVENTS:
        urgency = UrgencyLevel.high

    return DecisionResult(
        strategy=rule.strategy,
        actions=list(rule.actions),
        urgency=urgency,
        needs_escalation=rule.needs_escalation or (perception.event in ESCALATION_EVENTS),
        rationale=f"[规则:{rule.name}] {rule.rationale}",
        extras={"matched_rule": rule.name, "rule_priority": rule.priority},
    )
