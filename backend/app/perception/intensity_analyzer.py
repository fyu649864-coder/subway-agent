"""情绪强度识别（本地规则）。

输出三档：低 / 中 / 高。与情绪类别分开判断。
依据：程度副词、重复表达、标点符号、明显情绪表达；证据不足时保守判断。
不只看感叹号数量。
"""

from __future__ import annotations

from dataclasses import dataclass, field

INTENSITY_LOW = "低"
INTENSITY_MID = "中"
INTENSITY_HIGH = "高"

# 高程度词 / 强烈情绪表达
_HIGH_DEGREE = (
    "非常", "极其", "极度", "超级", "太", "真的", "简直", "快要", "崩溃",
    "受不了", "忍不了", "气死", "急死", "吓死", "哭了", "绝望", "要疯了",
    "受不了了", "撑不住",
)
# 中等程度词
_MID_DEGREE = ("有点", "有些", "比较", "挺", "蛮", "稍微", "不太", "好", "真是")
# 明显情绪词（出现即至少中等）
_EMOTION_WORDS = (
    "生气", "愤怒", "烦", "烦躁", "焦虑", "着急", "害怕", "恐惧", "难过",
    "伤心", "无助", "崩溃", "担心", "怕", "急", "气", "郁闷", "憋屈",
)
# 弱化/缓和信号（出现则倾向低强度）
_SOFTENERS = ("还好", "还行", "一般", "只是", "随便", "无所谓", "没事")


@dataclass
class IntensityResult:
    level: str  # 低/中/高
    evidence: list[str] = field(default_factory=list)
    # 数值映射，便于与现有 intensity 字段（[0,1]）兼容
    score: float = 0.5


def _count_exclamation(text: str) -> int:
    return text.count("!") + text.count("！")


def _has_repeated(text: str) -> bool:
    """检测重复表达，如 '啊啊啊'、'！！！'、叠词 AABB。"""
    if "！！" in text or "!!" in text or "？？" in text or "??" in text:
        return True
    for ch in set(text):
        if text.count(ch) >= 3 and "\u4e00" <= ch <= "\u9fff":
            return True
    return False


def analyze_intensity(text: str) -> IntensityResult:
    """分析情绪强度。

    - 文本无明显情绪信号时，保守返回「中」（而非强行判低/高）。
    - 空文本返回「低」，证据为空。
    """
    if not isinstance(text, str):
        text = ""
    text = text.strip()

    evidence: list[str] = []

    if not text:
        return IntensityResult(level=INTENSITY_LOW, evidence=[], score=0.2)

    high_hits = [w for w in _HIGH_DEGREE if w in text]
    mid_hits = [w for w in _MID_DEGREE if w in text]
    emo_hits = [w for w in _EMOTION_WORDS if w in text]
    soft_hits = [w for w in _SOFTENERS if w in text]
    excl = _count_exclamation(text)
    repeated = _has_repeated(text)

    # 打分：基础 0.5（中），各类信号加权
    score = 0.5
    if high_hits:
        score += 0.25 + min(len(high_hits) - 1, 3) * 0.05
        evidence.append(f"高程度词/强烈表达: {','.join(high_hits[:3])}")
    if mid_hits:
        score += 0.1
        evidence.append(f"中程度词: {','.join(mid_hits[:3])}")
    if emo_hits:
        score += 0.1
        evidence.append(f"情绪词: {','.join(emo_hits[:3])}")
    if excl >= 2:
        score += 0.1
        evidence.append(f"感叹号 x{excl}")
    elif excl == 1:
        score += 0.03
    if repeated:
        score += 0.1
        evidence.append("重复表达/叠词/多标点")
    if soft_hits:
        score -= 0.15
        evidence.append(f"缓和语气: {','.join(soft_hits[:3])}")

    score = max(0.0, min(1.0, round(score, 2)))

    # 证据不足（无任何情绪/程度信号）时保守判断：默认归为「中」
    if not (high_hits or mid_hits or emo_hits or excl or repeated):
        level = INTENSITY_MID
        evidence.append("无明显强度信号，保守判为中等")
    elif score >= 0.75 or (high_hits and (excl >= 2 or repeated)):
        level = INTENSITY_HIGH
    elif score <= 0.45 and soft_hits and not (high_hits or emo_hits):
        level = INTENSITY_LOW
    elif score < 0.6 and not high_hits:
        level = INTENSITY_LOW
    else:
        level = INTENSITY_MID

    return IntensityResult(level=level, evidence=evidence, score=score)
