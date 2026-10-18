"""轻量情境上下文提取（纯规则，零依赖、零模型）。

当前链路中，事件识别与强度分析已经把用户原文里的关键信号提取出来了，
但回复层只拿到一个 ``strategy`` 字符串就去选通用模板，导致：

- 用户的具体诉求（赶考试 / 赶飞机 / 接孩子 / 回家）被丢弃；
- 时间压力（迟到 / 来不及）只影响决策优先级，不影响回复措辞；
- 关键情境（家人在等、车厢挤不上、末班车已走、已经等了很久）从未被回应；
- 多轮历史虽然存进了 ConversationState，但从未参与回复生成。

本模块从原始文本中抽取这些“回复真正需要”的信号，供模板层组合出
有针对性的回复。所有规则均为关键词匹配，不调用模型、不新增依赖、
不访问网络。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReplyContext:
    """回复层可用的情境信号集合。"""

    text: str
    # 具体诉求目标：考试 / 飞机 / 高铁 / 火车 / 上班 / 打卡 / 开会 / 面试 / 回家 / 接人 / 约会 …
    urgency_target: str | None = None
    has_time_pressure: bool = False          # 迟到 / 来不及 / 马上要
    person_waiting: str | None = None        # 家人 / 朋友 / 孩子 / 老婆 / 老公 …
    crowd_detail: str | None = None          # 挤不上 / 太挤 / 人多 / 爆满
    last_train_gone: bool = False            # 末班车已走 / 没赶上
    waited_long: bool = False                # 等了半小时 / 等了好久
    has_delay: bool = False
    has_transfer: bool = False
    has_complaint: bool = False
    negative_emotion: bool = False           # 出现明显负面情绪词
    # 标记：目标/等待的人是否继承自历史轮（本轮文本未直接提及）
    inherited_target: bool = False
    inherited_person: bool = False


# 具体诉求目标（按匹配优先级排列，越具体越靠前）
_URGENCY_TARGETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("考试", ("赶考试", "考试", "赶考", "进考场", "笔试", "面试")),
    ("飞机", ("赶飞机", "误机", "要误机", "飞机", "航班", "登机", "机场")),
    ("高铁", ("赶高铁", "高铁", "动车")),
    ("火车", ("赶火车", "误车", "火车", "火车站")),
    ("上班打卡", ("上班", "打卡", "迟到", "晨会", "早会")),
    ("开会", ("开会", "会议", "参会")),
    ("面试", ("面试", "笔试")),
    ("回家", ("回家", "回去", "回不去", "到家")),
    ("接人", ("接人", "接站", "接孩子", "接老婆", "接老公", "接机", "接站")),
    ("约会", ("约会", "约了人", "等人", "碰面")),
)

_TIME_PRESSURE = ("迟到", "来不及", "快迟到", "马上要迟到", "要迟到", "赶不上", "急着", "赶着", "快要")
_PERSON_WAITING = (
    ("家人", ("家人", "家里人", "父母", "爸妈", "妈妈", "爸爸")),
    ("孩子", ("孩子", "儿子", "女儿", "小孩")),
    ("老婆", ("老婆", "媳妇", "爱人")),
    ("老公", ("老公", "先生")),
    ("朋友", ("朋友", "同学", "同事")),
    ("客人", ("客人", "亲戚")),
)
_CROWD_DETAIL = (
    ("挤不上", ("挤不上", "上不去", "挤不上去", "挤不下")),
    ("太挤", ("太挤", "好挤", "挤死", "拥挤", "爆满", "塞满", "沙丁鱼", "人满为患")),
    ("人多", ("人太多", "人好多", "人满为患")),
)
_LAST_TRAIN_GONE = ("末班车", "最后一班", "没赶上", "没坐上", "车走了", "关门了", "门关了", "错过")
_WAITED_LONG = ("等了半小时", "等了一个小时", "等了好久", "等了很久", "半天不来", "一直不来")
_DELAY = ("延误", "晚点", "还不来", "怎么还不来", "迟迟不来", "龟速", "开得慢")
_TRANSFER = ("换乘", "转乘", "换线", "找不到出口", "绕路")
_COMPLAINT = ("投诉", "服务差", "态度差", "差评", "举报", "垃圾", "没人管")
_NEGATIVE = (
    "生气", "愤怒", "烦", "烦躁", "焦虑", "着急", "害怕", "恐惧", "难过",
    "伤心", "无助", "崩溃", "担心", "怕", "急", "气", "郁闷", "憋屈", "气死",
    "急死", "吓死", "哭了", "绝望", "要疯了", "受不了",
)


def _find_first(text: str, keywords: tuple[str, ...]) -> str | None:
    for kw in keywords:
        if kw in text:
            return kw
    return None


def extract_context(text: str, history_texts: list[str] | None = None) -> ReplyContext:
    """从原始文本抽取回复层情境信号。

    ``history_texts`` 为最近几轮的用户原文（按时间顺序）。当本轮文本较短、
    未直接提及诉求目标或等待的人时，从历史中继承，保证多轮对话的连续性。
    继承的信号会标记 ``inherited_*``，供回复层措辞时区分（如“您要赶考试的事我还记着”）。
    """
    if not isinstance(text, str):
        text = ""
    t = text.strip()

    ctx = ReplyContext(text=t)

    # 先在本轮文本中抽取
    # 1. 具体诉求目标（取第一个命中的，避免“上班打卡”覆盖“考试”）
    for target, kws in _URGENCY_TARGETS:
        if _find_first(t, kws):
            ctx.urgency_target = target
            break

    # 2. 时间压力
    if _find_first(t, _TIME_PRESSURE):
        ctx.has_time_pressure = True

    # 3. 等待的人
    for person, kws in _PERSON_WAITING:
        if _find_first(t, kws):
            ctx.person_waiting = person
            break

    # 4. 拥挤细节
    for detail, kws in _CROWD_DETAIL:
        if _find_first(t, kws):
            ctx.crowd_detail = detail
            break

    # 5. 末班车已走
    if _find_first(t, _LAST_TRAIN_GONE):
        ctx.last_train_gone = True

    # 6. 等待已久
    if _find_first(t, _WAITED_LONG):
        ctx.waited_long = True

    if _find_first(t, _DELAY):
        ctx.has_delay = True
    if _find_first(t, _TRANSFER):
        ctx.has_transfer = True
    if _find_first(t, _COMPLAINT):
        ctx.has_complaint = True
    if _find_first(t, _NEGATIVE):
        ctx.negative_emotion = True

    # 7. 从最近历史继承本轮缺失的核心信号（倒序，取最近的先命中）
    if history_texts:
        for prev in reversed(history_texts[-5:]):
            if ctx.urgency_target is None:
                for target, kws in _URGENCY_TARGETS:
                    if _find_first(prev, kws):
                        ctx.urgency_target = target
                        ctx.inherited_target = True
                        break
            if ctx.person_waiting is None:
                for person, kws in _PERSON_WAITING:
                    if _find_first(prev, kws):
                        ctx.person_waiting = person
                        ctx.inherited_person = True
                        break
            if not ctx.has_time_pressure and _find_first(prev, _TIME_PRESSURE):
                ctx.has_time_pressure = True
            if not ctx.last_train_gone and _find_first(prev, _LAST_TRAIN_GONE):
                ctx.last_train_gone = True
            # 一旦核心信号都补齐就停止回溯
            if ctx.urgency_target is not None and ctx.person_waiting is not None:
                break

    return ctx


def summarize_for_log(ctx: ReplyContext) -> str:
    """便于日志打印的一行摘要。"""
    flags = []
    if ctx.urgency_target:
        flags.append(f"target={ctx.urgency_target}")
    if ctx.has_time_pressure:
        flags.append("time_pressure")
    if ctx.person_waiting:
        flags.append(f"waiting={ctx.person_waiting}")
    if ctx.crowd_detail:
        flags.append(f"crowd={ctx.crowd_detail}")
    if ctx.last_train_gone:
        flags.append("last_train_gone")
    if ctx.waited_long:
        flags.append("waited_long")
    return ",".join(flags) or "no_specific_signal"
