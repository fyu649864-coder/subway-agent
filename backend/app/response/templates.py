"""回复模板库（情境感知版）。

设计要点：
1. 不重新训练模型、不调用云端，纯本地规则与模板。
2. 紧急情况策略保持原有安全提示不变。
3. 其余策略不再“按策略随机选一句通用套话”，而是：
   - 先回应用户的**具体处境**（赶考试 / 接孩子 / 挤不上 / 末班车走了 …）；
   - 仅在情绪强度中高或存在负面情绪时加入共情，避免每句都强行安慰；
   - 仅在行动类策略下给出**通用且可执行**的建议，不编造线路、实时信息或预计时间；
   - 参考最近一轮历史，避免连续输出完全相同的句子（多轮连续性）。
"""

from __future__ import annotations

import random

from app.decision.rules import (
    STRATEGY_ACTION,
    STRATEGY_COMFORT,
    STRATEGY_COMFORT_ACTION,
    STRATEGY_CONSERVATIVE,
    STRATEGY_EMERGENCY,
)
from app.models import PerceptionResult
from app.response.context import ReplyContext, extract_context, summarize_for_log

# 紧急情况：安全提示模板保持不变（安全规则优先于自然度）
_EMERGENCY_TEMPLATES = [
    "请您先保持冷静，务必注意自身安全，听从现场工作人员的指引。"
    "如遇紧急情况，请立即联系站内工作人员或拨打服务热线，我们会优先保障您的安全。",
    "非常理解您此刻的紧张心情，请您第一时间注意安全并配合现场工作人员的处置。"
    "如需帮助，请立即向周边工作人员求助，我们会尽快响应。",
]

_TEMPLATE_ID_PREFIX = {
    STRATEGY_EMERGENCY: "tpl_emergency",
    STRATEGY_COMFORT: "tpl_comfort",
    STRATEGY_ACTION: "tpl_action",
    STRATEGY_COMFORT_ACTION: "tpl_comfort_action",
    STRATEGY_CONSERVATIVE: "tpl_conservative",
}

# 通用、安全、不含任何具体线路/实时信息的建议池
_ADVICE_TIME_PRESSURE = [
    "建议您留意站内广播和官方 APP 的动态，若情况允许也可考虑打车、公交等备选出行方式。",
    "可以多关注站台广播与官方 APP 信息，必要时改换其他交通方式，尽量减少耽误。",
]
_ADVICE_CROWD = [
    "您可以往车厢两端走一走，或者耐心等下一班，错峰出行也会宽松不少。",
    "建议尽量靠近车厢两端，或选择下一班列车，错峰时体验会好一些。",
]
_ADVICE_LAST_TRAIN = [
    "别着急，可以看看附近的夜班公交，或用打车软件叫车，安全到家最重要。",
    "末班车走了的话，建议优先考虑打车或夜班公交，务必注意安全。",
]
_ADVICE_TRANSFER = [
    "您可以留意站内导向标识，或在官方 APP 里查看换乘路线，应该会更顺畅。",
    "建议看一下站台导向标识，或用官方 APP 查一下换乘路径。",
]
_ADVICE_DELAY_GENERIC = [
    "建议您留意站内广播与官方 APP 的实时动态，以便及时调整行程。",
    "可以关注官方渠道的最新信息，灵活安排接下来的行程。",
]


def _acknowledgements(ctx: ReplyContext) -> list[str]:
    """针对用户具体处境的回应句池（不编造事实，语气自然不客服腔）。"""
    out: list[str] = []

    if ctx.urgency_target == "考试":
        out += [
            "考试快迟到了还被卡在这儿，确实急人。",
            "赶着去考试偏偏走不动，换谁都得慌。",
            "要赶考试还遇上这种事，心里肯定七上八下的。",
        ]
    elif ctx.urgency_target in ("飞机", "高铁", "火车"):
        t = ctx.urgency_target
        out += [
            f"要赶{t}还遇上这种情况，确实让人揪心。",
            f"赶着去坐{t}却被耽搁，那股着急劲隔着屏幕都能感觉到。",
        ]
    elif ctx.urgency_target == "上班打卡":
        out += [
            "赶着上班打卡却走不动，确实让人焦虑。",
            "上班要迟到的滋味不好受，干等着最磨人。",
        ]
    elif ctx.urgency_target == "开会":
        out += [
            "赶着开会却被卡住，确实让人担心。",
            "会议要迟到了，干等着确实煎熬。",
        ]
    elif ctx.urgency_target == "回家":
        if ctx.last_train_gone:
            out += [
                "末班车走了回不去，确实让人发愁。",
                "错过末班车回不了家，那种无力感不好受。",
            ]
        else:
            out += [
                "想早点回家却被耽搁，确实让人心里不快。",
            ]
    elif ctx.urgency_target == "接人":
        out += [
            "赶着去接人却走不动，对方还在等着，确实让人着急。",
        ]

    if ctx.person_waiting and out:
        p = ctx.person_waiting
        out = [s + f"而且{p}还在等着，更是让人挂心。" for s in out]
    elif ctx.person_waiting:
        p = ctx.person_waiting
        out += [
            f"{p}还在等着，您却被卡在这儿，确实让人着急。",
            f"那边{p}在等，这边又走不了，心情肯定不好受。",
        ]

    if ctx.crowd_detail and ctx.urgency_target is None and ctx.person_waiting is None:
        if ctx.crowd_detail == "挤不上":
            out += [
                "连着几班都挤不上去，确实让人又急又无奈。",
                "挤不上车的滋味不好受，越等越烦。",
            ]
        else:
            out += [
                "车厢这么挤，确实让人不舒服。",
                "人多到这个程度，体验确实很差。",
            ]

    if ctx.last_train_gone and not out:
        out += [
            "末班车没赶上，回不去确实让人发愁。",
            "错过末班车的滋味不好受。",
        ]

    if ctx.has_delay and not out:
        if ctx.waited_long:
            out += [
                "等了这么久车还不来，确实让人心里没底。",
                "等了半天还没动静，换谁都会着急。",
            ]
        else:
            out += [
                "列车迟迟不来，确实让人不安。",
                "遇到延误确实影响出行，干等着最磨人。",
            ]

    if ctx.has_complaint and not out:
        out += [
            "遇到这样的服务体验，让您不满意了，抱歉。",
            "给您带来不好的感受，我们很抱歉。",
        ]

    if not out:
        out += [
            "嗯，我在听着呢，你说。",
            "好的，你说吧，我在。",
        ]
    return out


def _empathy(ctx: ReplyContext, intensity: str | None) -> list[str]:
    """共情句池：仅在中高情绪或存在负面情绪词时返回非空，避免套话。"""
    if intensity not in ("中", "高") and not ctx.negative_emotion:
        return []
    if intensity == "高":
        return [
            "这种时候确实难熬，别急坏了自己。",
            "搁谁身上都不好受，你已经在尽力应对了。",
        ]
    return [
        "这种体验确实不好受。",
        "让你遇到这种事，挺糟心的。",
    ]


def _advices(strategy: str, ctx: ReplyContext) -> list[str]:
    """根据情境与策略返回可执行、不编造事实的建议。"""
    if strategy not in (STRATEGY_ACTION, STRATEGY_COMFORT_ACTION):
        return []
    if ctx.urgency_target in ("考试", "飞机", "高铁", "火车", "上班打卡", "开会", "接人") or ctx.has_time_pressure:
        return list(_ADVICE_TIME_PRESSURE)
    if ctx.last_train_gone:
        return list(_ADVICE_LAST_TRAIN)
    if ctx.crowd_detail:
        return list(_ADVICE_CROWD)
    if ctx.has_transfer:
        return list(_ADVICE_TRANSFER)
    if ctx.has_delay:
        return list(_ADVICE_DELAY_GENERIC)
    return list(_ADVICE_DELAY_GENERIC)


def _conservative_replies(ctx: ReplyContext, text: str = "") -> list[str]:
    """保守策略：自然回应，不强行安慰、不编造、不客服腔。"""
    # 普通闲聊/日常需求：自然接话，不要追问"具体问题"
    normal_chat = any(k in text for k in ("饿了", "笑话", "无聊", "困了", "累", "说说话", "聊聊", "陪"))
    if normal_chat:
        return [
            "嗯，我在呢，想聊什么都行，你说。",
            "好啊，你说吧，我听着呢。",
        ]
    if ctx.urgency_target or ctx.crowd_detail or ctx.last_train_gone or ctx.has_delay or ctx.has_transfer:
        return [
            "你说的情况我大概了解了，先别急，慢慢说。",
            "嗯，我在听着呢，你继续说。",
        ]
    return [
        "嗯，我在听着呢，你说吧。",
        "好的，你说吧，我在听。",
    ]


def _join(parts: list[str]) -> str:
    return "".join(p for p in parts if p)


def _pick(rng: random.Random, pool: list[str], exclude: set[str]) -> str:
    fresh = [s for s in pool if s not in exclude]
    if not fresh:
        fresh = pool
    return rng.choice(fresh)


def render(
    strategy: str,
    perception: PerceptionResult | None = None,
    history: list[PerceptionResult] | None = None,
    rng: random.Random | None = None,
) -> tuple[str, str]:
    """根据策略与上下文渲染一条回复，返回 ``(template_id, reply)``。

    - 未提供 perception 时退化为旧的“按策略随机选模板”，保证向后兼容。
    - 紧急策略始终使用安全提示模板。
    - 其余策略按情境组合：回应具体处境 →（按需）共情 →（按需）通用建议。
    - 参考最近一轮历史，避免与上一条回复完全相同。
    """
    rng = rng or random

    # 向后兼容：无 perception 时走简单模板
    if perception is None:
        if strategy == STRATEGY_EMERGENCY:
            idx = rng.randrange(len(_EMERGENCY_TEMPLATES))
            return f"tpl_emergency_{idx}", _EMERGENCY_TEMPLATES[idx]
        # 其它策略无旧模板池时使用保守兜底
        idx = rng.randrange(len(_CONSERVATIVE_FALLBACK))
        return f"tpl_conservative_{idx}", _CONSERVATIVE_FALLBACK[idx]

    ctx = extract_context(
        perception.text,
        history_texts=[h.text for h in (history or []) if getattr(h, "text", None)],
    )
    intensity = perception.emotion_intensity
    # 上一轮的完整回复文本（若调用方通过 extras 回填）用于去重，避免连续重复
    used_replies: set[str] = set()
    for h in (history or [])[-3:]:
        prev = h.extras.get("last_reply") if isinstance(h.extras, dict) else None
        if isinstance(prev, str) and prev:
            used_replies.add(prev)

    prefix = _TEMPLATE_ID_PREFIX.get(strategy, "tpl_default")

    # ---- 紧急：安全优先，措辞不变 ----
    if strategy == STRATEGY_EMERGENCY:
        idx = rng.randrange(len(_EMERGENCY_TEMPLATES))
        reply = _EMERGENCY_TEMPLATES[idx]
        return f"{prefix}_{idx}", reply

    # ---- 保守 ----
    if strategy == STRATEGY_CONSERVATIVE:
        pool = _conservative_replies(ctx, perception.text or "")
        reply = _pick(rng, pool, used_replies)
        return f"{prefix}_{abs(hash(reply)) % 1000}", reply

    # ---- 其它策略：情境组合 ----
    acks = _acknowledgements(ctx)
    empathies = _empathy(ctx, intensity)
    advices = _advices(strategy, ctx)

    # 组合：先具体处境，再共情（按需），再建议（按需）
    ack = _pick(rng, acks, used_replies)
    empathy = _pick(rng, empathies, set()) if empathies else ""

    # 多轮承接：只用极轻量的承接，不机械复述历史信息
    continuity = ""
    if history and len(perception.text or "") < 10:
        prev = history[-1]
        if prev.event == perception.event and perception.event not in ("其他/未知事件",):
            continuity = "嗯，还在呢。"

    reply = _join([continuity, ack, empathy])

    if advices:
        advice = _pick(rng, advices, used_replies)
        if reply:
            reply = reply + advice
        else:
            reply = advice

    if not reply:
        pool = _conservative_replies(ctx, perception.text or "")
        reply = _pick(rng, pool, used_replies)
        return f"{prefix}_fallback_{abs(hash(reply)) % 1000}", reply

    tid = f"{prefix}_{abs(hash(reply)) % 100000}"
    return tid, reply


# 仅用于无 perception 时的向后兼容兜底
_CONSERVATIVE_FALLBACK = [
    "感谢您的反馈，我们已收到您的信息。为了更好地帮助您，可以请您再描述一下具体情况吗？",
    "您好，感谢您与我们沟通。如果方便的话，请补充说明一下具体遇到的问题。",
]


def available_strategies() -> list[str]:
    return [
        STRATEGY_EMERGENCY,
        STRATEGY_COMFORT,
        STRATEGY_ACTION,
        STRATEGY_COMFORT_ACTION,
        STRATEGY_CONSERVATIVE,
    ]
