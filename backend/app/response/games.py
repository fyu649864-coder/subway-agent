"""轻量游戏状态管理：为猜谜语、成语接龙等需要可靠状态的游戏提供状态锚点。

设计原则：
- 纯内存、零依赖、不调用任何模型或网络。
- 谜语题库、成语词库与规则校验均由 Python 侧持有，保证判断对错与游戏推进
  不依赖 3B 大模型的"记性"。
- 状态按 session_id 隔离，会话结束或用户明确退出时清理。
"""

from __future__ import annotations

import random
import random as _rnd
from dataclasses import dataclass, field


@dataclass
class Riddle:
    q: str
    a: str
    hints: list[str]


# 谜语题库：题目简短、答案常见，适合 3B 模型场景下的轻松互动
_RIDDLE_BANK: list[Riddle] = [
    Riddle("什么东西越洗越脏？", "水", ["它是液体", "人和动植物都离不开它"]),
    Riddle("什么门永远关不上？", "球门", ["它在运动场上", "和足球有关"]),
    Riddle("什么东西有头没有脚，却能走遍天下？", "硬币", ["它是圆的", "买东西会用到它"]),
    Riddle("白天出现晚上消失，越黑越看不见的是什么？", "影子", ["它跟着你", "有光才有它"]),
    Riddle("什么东西你左手能拿，右手却拿不到？", "右手", ["它长在你身上", "是你的一只手"]),
    Riddle("越热它越往上爬，是什么？", "温度计", ["里面有水银或酒精", "看病时会量它"]),
    Riddle("什么布剪不断？", "瀑布", ["它是一种自然景观", "和水有关"]),
    Riddle("什么东西不破的时候人们用它，破了之后人们更喜欢它？", "蛋", ["它是一种食物", "母鸡会下"]),
    Riddle("一个人从十层楼跳下来却没事，为什么？", "他在拍电影", ["不是真的跳", "有保护措施"]),
    Riddle("什么车寸步难行？", "风车", ["它靠风转", "不是用来坐的"]),
    Riddle("什么东西明明是你的，别人却用得比你多？", "名字", ["它是一个称呼", "别人叫你时用"]),
    Riddle("什么东西越擦越小？", "橡皮擦", ["它是文具", "写错字时用"]),
]

# 内存态：session_id -> 游戏状态
_states: dict[str, dict] = {}


def active_game(session_id: str | None) -> dict | None:
    """返回当前会话的活跃游戏状态，无则返回 None。"""
    if not session_id:
        return None
    return _states.get(session_id)


def start_riddle(session_id: str | None) -> Riddle | None:
    """为会话开启一局猜谜语，返回选中的谜语（题目+答案+提示）。"""
    if not session_id:
        return None
    r = random.choice(_RIDDLE_BANK)
    _states[session_id] = {
        "game": "riddle",
        "q": r.q,
        "a": r.a,
        "hints": list(r.hints),
        "hints_given": 0,
    }
    return r


def submit_guess(session_id: str | None, guess: str) -> dict:
    """提交一次猜谜答案，返回判断结果。

    返回 dict 包含：
      correct: bool
      hint: str | None   （答错且还有提示时给出下一条提示）
      answer: str        （始终附带真实答案，供大模型参考，但模型不应直接说破）
      done: bool         （答对或提示用尽后置 True）
    """
    st = active_game(session_id)
    if not st or st.get("game") != "riddle":
        return {"correct": False, "hint": None, "answer": "", "done": True}
    answer = st["a"]
    g = (guess or "").strip()
    correct = bool(g) and (g == answer or answer in g or g in answer)
    hint = None
    done = correct
    if not correct:
        idx = st["hints_given"]
        if idx < len(st["hints"]):
            hint = st["hints"][idx]
            st["hints_given"] = idx + 1
        else:
            # 提示用尽，本局结束
            done = True
    if done:
        _states.pop(session_id, None)
    return {"correct": correct, "hint": hint, "answer": answer, "done": done}


def end_game(session_id: str | None) -> None:
    """结束当前会话的游戏状态（用户拒绝、切换话题等情况）。"""
    if session_id:
        _states.pop(session_id, None)


# ============================================================================
# 成语接龙
# ============================================================================

# 常见四字成语词库（纯手写，零依赖、不下载）。用于 AI 侧选取合规接龙成语，
# 不用于校验用户成语是否"真实存在"（用户只要首字合规、形式像成语即可放行，
# 避免词库不全导致误杀用户的合理回答）。
_IDIOM_BANK: list[str] = [
    "一心一意", "意气风发", "发扬光大", "大显身手", "手到擒来",
    "来日方长", "长年累月", "月下老人", "人山人海", "海阔天空",
    "空前绝后", "后来居上", "上下同心", "心想事成", "成千上万",
    "万事如意", "异想天开", "开门见山", "山高水长", "长篇大论",
    "论功行赏", "赏心悦目", "目中无人", "人定胜天", "天翻地覆",
    "覆水难收", "收回成命", "命中注定", "脚踏实地", "地久天长",
    "长驱直入", "入木三分", "分秒必争", "争先恐后", "上行下效",
    "笑逐颜开", "开卷有益", "益寿延年", "年富力强", "强词夺理",
    "理直气壮", "壮志凌云", "云开见日", "日新月异", "异口同声",
    "声东击西", "西风落叶", "叶公好龙", "龙马精神", "神采飞扬",
    "扬眉吐气", "气象万千", "千军万马", "马到成功", "功德无量",
    "量力而行", "行云流水", "水落石出", "出人头地", "地动山摇",
    "摇头摆尾", "尾大不掉", "掉以轻心", "心花怒放", "放任自流",
    "流连忘返", "返璞归真", "真知灼见", "见义勇为", "为所欲为",
    "惟妙惟肖", "笑里藏刀", "刀山火海", "海枯石烂", "烂熟于心",
]

# 首字 -> 成语列表 索引，便于 AI 快速选取合规下一个成语
_idiom_index: dict[str, list[str]] = {}
for _idy in _IDIOM_BANK:
    _idiom_index.setdefault(_idy[0], []).append(_idy)


def _looks_like_idiom(text: str) -> bool:
    """粗判用户输入是否像一个成语（纯中文、长度 3-8）。"""
    t = (text or "").strip()
    if not (3 <= len(t) <= 8):
        return False
    return all("\u4e00" <= ch <= "\u9fff" for ch in t)


def _pick_idiom_for(char: str) -> str | None:
    """从词库中选一个以指定字开头的成语。"""
    pool = _idiom_index.get(char)
    if not pool:
        return None
    return random.choice(pool)


def start_chain(session_id: str | None) -> dict | None:
    """开启一局成语接龙，返回 {idiom, required_char}。"""
    if not session_id:
        return None
    idiom = _pick_idiom_for(random.choice([d[0] for d in _IDIOM_BANK[:20]])) or _IDIOM_BANK[0]
    _states[session_id] = {
        "game": "chain",
        "last_idiom": idiom,
        "required_char": idiom[-1],
        "turn": 0,
    }
    return {"idiom": idiom, "required_char": idiom[-1]}


def submit_chain(session_id: str | None, text: str) -> dict:
    """校验用户本轮接龙输入，推进状态。

    返回 dict：
      valid: bool            用户输入是否合规（首字匹配、形式像成语）
      reason: str            不合规时的原因（合规时为空）
      user_idiom: str        用户的成语
      ai_idiom: str | None   AI 接的下一个成语（合规时给出）
      required_char: str     用户下一步需要接的字
      topic_switch: bool     用户明显在切换话题（应退出游戏，不报错）
    只有 valid=True 才更新状态。
    """
    st = active_game(session_id)
    if not st or st.get("game") != "chain":
        return {"valid": False, "reason": "", "user_idiom": "", "ai_idiom": None,
                "required_char": "", "topic_switch": False}

    raw = (text or "").strip()
    required = st["required_char"]

    # 用户明显切换话题（长句、包含正常聊天/需求关键词）：退出游戏，不报错
    normal_chat_kw = ("我饿", "我困", "聊聊", "说说话", "地铁", "几点", "多久", "谢谢", "你好")
    if len(raw) > 10 or any(k in raw for k in normal_chat_kw):
        end_game(session_id)
        return {"valid": False, "reason": "", "user_idiom": raw, "ai_idiom": None,
                "required_char": "", "topic_switch": True}

    if not _looks_like_idiom(raw):
        return {"valid": False, "reason": "看起来不像一个成语哦",
                "user_idiom": raw, "ai_idiom": None, "required_char": required,
                "topic_switch": False}

    if raw[0] != required:
        return {"valid": False, "reason": f"第一个字应该是「{required}」哦",
                "user_idiom": raw, "ai_idiom": None, "required_char": required,
                "topic_switch": False}

    # 合规：AI 接下一个
    ai_idiom = _pick_idiom_for(raw[-1])
    if ai_idiom is None:
        # 词库里接不上，用一个万能收尾并结束本局
        end_game(session_id)
        return {"valid": True, "reason": "", "user_idiom": raw, "ai_idiom": None,
                "required_char": "", "topic_switch": False, "no_more": True}

    st["last_idiom"] = ai_idiom
    st["required_char"] = ai_idiom[-1]
    st["turn"] += 1
    return {"valid": True, "reason": "", "user_idiom": raw, "ai_idiom": ai_idiom,
            "required_char": ai_idiom[-1], "topic_switch": False}


# ---- 接龙回复组合模板池（保证自然且规则明确，不依赖模型生成） ----
_CHAIN_PRAISE = ["接得好！", "漂亮！", "没错！", "棒！", "可以啊！"]
_CHAIN_START = [
    "好，我们来玩成语接龙！规则很简单：下一个成语的第一个字要和上一个成语的最后一个字相同。我先来：{idiom}。请你用「{char}」字开头接一个成语。",
    "行，成语接龙开始！规则：接上一个成语的最后一个字。我出第一个：{idiom}。请接「{char}」字开头的成语。",
    "没问题，玩成语接龙！第一个字接最后一个字就行。我先来：{idiom}。该你接「{char}」字啦。",
]
_CHAIN_OK = [
    "{praise}「{user}」接对了！我接「{ai}」。请你用「{char}」字开头。",
    "{praise}「{user}」没错！我来接：{ai}。接下来请接「{char}」字。",
    "{praise}「{user}」可以！我接「{ai}」，现在轮到你接「{char}」字开头的成语。",
]
_CHAIN_RETRY = [
    "{reason}，我们要接「{char}」字开头的成语，再试试？",
    "{reason}。规则是接「{char}」字开头，你再想想？",
]


def compose_chain_start(idiom: str, char: str) -> str:
    return random.choice(_CHAIN_START).format(idiom=idiom, char=char)


def compose_chain_reply(result: dict) -> str:
    if result.get("topic_switch"):
        return ""  # 切换话题：交由大模型正常回应
    if result.get("no_more"):
        return f"「{result['user_idiom']}」接得好！我一时接不上来了，这局算你赢～还要玩别的吗？"
    if result["valid"]:
        return random.choice(_CHAIN_OK).format(
            praise=random.choice(_CHAIN_PRAISE),
            user=result["user_idiom"],
            ai=result["ai_idiom"],
            char=result["required_char"],
        )
    return random.choice(_CHAIN_RETRY).format(reason=result["reason"], char=result["required_char"])


# 猜谜开局的自然开场白池（与题库题目组合，保证出题可靠、语气自然有变化）
_RIDDLE_OPENERS = [
    "好，那我出题啦：",
    "行，听好这道题：",
    "没问题，你来猜猜这个：",
    "好啊，那猜猜看：",
    "来吧，这道题：",
]


def compose_riddle_start(r: Riddle) -> str:
    """组合一条自然的猜谜开局回复：开场白 + 题目，不泄露答案。"""
    opener = _rnd.choice(_RIDDLE_OPENERS)
    return f"{opener}{r.q}"


# 表示"接受邀请"的简短表达
_ACCEPT_PATTERNS = ("好", "行", "来", "玩", "可以", "嗯", "行啊", "好啊", "来吧", "没问题", "要")
# 表示"拒绝/退出"的表达（注意："无聊"不算拒绝，用户说无聊往往是想找点事做）
_REJECT_PATTERNS = ("不玩", "不要玩", "算了", "不想玩", "别玩了", "没意思", "不猜了", "不玩了", "换个话题", "没心情", "不要了")


def is_acceptance(text: str) -> bool:
    t = (text or "").strip()
    if len(t) > 6:
        return False
    return any(p in t for p in _ACCEPT_PATTERNS)


def is_rejection(text: str) -> bool:
    t = (text or "").strip()
    return any(p in t for p in _REJECT_PATTERNS)


def last_assistant_reply(history: list) -> str:
    """从历史中取上一轮助手的回复文本。"""
    for h in reversed(history or []):
        ex = getattr(h, "extras", None)
        if isinstance(ex, dict):
            r = ex.get("last_reply")
            if isinstance(r, str) and r:
                return r
    return ""
