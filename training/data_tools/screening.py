# -*- coding: utf-8 -*-
"""候选公开数据的导入与质量筛选（纯函数，便于单测，不修改任何正式数据集）。

流程：
1. 读取候选文件（CSV / XLSX），来源信息由 manifest 登记，不硬编码。
2. 字段规范化：要求 text 与原始标签列；缺失/格式错误直接标记剔除。
3. 文本规范化后做：精确重复、近似重复（字符 n-gram Jaccard + difflib）、
   异常短文本、疑似无关内容（纯标点/纯链接/纯数字等）。
4. 原始标签 -> 项目 7 类（平静/焦虑/愤怒/悲伤/恐惧/烦躁/无助）映射；
   无法可靠映射的标记为「待复核」，不强行分类。
5. 与现有 train/val/test 做精确与近似重复比对，防止泄漏；
   命中 test 集合的样本绝对不得进入训练。
6. 疑似错标识别：规则证据（关键词与标签冲突）为主，本地 BERT 预测仅作
   辅助佐证；两者同时指向冲突才标记「疑似错标-待复核」。
7. 每条输出：来源 / 原始标签 / 映射标签 / 筛选状态 / 筛选原因 / 模型辅助标签。

设计约束（与项目记忆一致）：
- 不修改原始训练集、验证集、测试集；候选结果只写入 data/candidate/screened/。
- 不抓取需登录或禁止采集的内容；数据许可由 manifest 登记，由人确认。
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

# 项目目标 7 类
CLASSES_7 = ["平静", "焦虑", "愤怒", "悲伤", "恐惧", "烦躁", "无助"]
REVIEW = "待复核"
KEEP = "保留"
DROP = "剔除"

# 筛选状态
STATUS_VALUES = (KEEP, REVIEW, DROP)

# 通用标签映射表（公开数据集常见标签 -> 7 类）。
# 未列出的标签一律走「待复核」，绝不猜测。
LABEL_MAP: dict[str, str] = {
    # 平静 / 中性 / 正向
    "neutral": "平静", "中性": "平静", "平静": "平静", "calm": "平静",
    "happy": "平静", "开心": "平静", "喜悦": "平静", "高兴": "平静",
    "surprise": "平静", "惊讶": "平静", "惊奇": "平静",
    # 焦虑
    "anxiety": "焦虑", "anxious": "焦虑", "焦虑": "焦虑", "着急": "焦虑",
    "焦急": "焦虑", "担心": "焦虑", "担忧": "焦虑", "worry": "焦虑",
    "worried": "焦虑", "紧张": "焦虑",
    # 愤怒
    "anger": "愤怒", "angry": "愤怒", "愤怒": "愤怒", "生气": "愤怒",
    "恼火": "愤怒", "不满": "愤怒", "irritated": "愤怒", "mad": "愤怒",
    # 悲伤
    "sad": "悲伤", "sadness": "悲伤", "悲伤": "悲伤", "难过": "悲伤",
    "伤心": "悲伤", "失望": "悲伤", "depressed": "悲伤", "sorrow": "悲伤",
    # 恐惧
    "fear": "恐惧", "fearful": "恐惧", "afraid": "恐惧", "恐惧": "恐惧",
    "害怕": "恐惧", "恐慌": "恐惧", "惊吓": "恐惧", "scared": "恐惧",
    # 烦躁
    "烦躁": "烦躁", "irritation": "烦躁", "frustrated": "烦躁", "annoyed": "烦躁",
    "烦": "烦躁",
    # 无助
    "helpless": "无助", "无助": "无助", "无奈": "无助", "despair": "无助",
    "hopeless": "无助",
    # 常见二分类/其它：无法可靠映射
    "positive": REVIEW, "negative": REVIEW, "pos": REVIEW, "neg": REVIEW,
    "正向": REVIEW, "负向": REVIEW, "objective": REVIEW, "其他": REVIEW,
    "none": REVIEW, "no-emotion": REVIEW, "无": REVIEW, "": REVIEW,
}

# 文本清洗用：匹配空白与各类中英文标点符号（字符类内多数标点无需转义）
_PUNCT_RE = re.compile(r"""[\s\u3000，。！？、；："'（）《》【】,.!?;:()\[\]<>\"'`~@#$%^&*_=+|\\/-]+""")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MIN_CHARS = 2  # 规范化后少于该字符数视为异常短


@dataclass
class ScreenResult:
    source: str
    raw_id: str
    text: str
    original_label: str
    mapped_label: str
    status: str
    reason: str
    model_aux_label: str | None = None
    model_aux_score: float | None = None
    extras: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 文本规范化与相似度
# --------------------------------------------------------------------------- #
def normalize_text(text: object) -> str:
    """去 URL、空白、标点，转小写，用于重复/泄漏比对。"""
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = _URL_RE.sub("", s)
    s = _PUNCT_RE.sub("", s)
    return s


def _ngrams(s: str, n: int = 3) -> set[str]:
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def jaccard_ngram(a: str, b: str, n: int = 3) -> float:
    sa, sb = _ngrams(a, n), _ngrams(b, n)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_near_duplicate(a: str, b: str, jaccard_th: float = 0.9, ratio_th: float = 0.92) -> bool:
    """近似重复判定：n-gram Jaccard 或 difflib 相似度超阈值。"""
    if a == b:
        return True
    if not a or not b:
        return False
    if jaccard_ngram(a, b) >= jaccard_th:
        return True
    # 仅对较短文本做 difflib，避免长文本 O(n^2) 爆炸
    if max(len(a), len(b)) <= 200 and difflib.SequenceMatcher(None, a, b).ratio() >= ratio_th:
        return True
    return False


# --------------------------------------------------------------------------- #
# 现有数据集引用（只读，用于泄漏检查）
# --------------------------------------------------------------------------- #
def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_existing_texts() -> dict[str, list[str]]:
    """返回 {split: [normalize(text)]}，覆盖 train/val/test 全部现有数据。

    只读 data/processed 下的正式划分文件，不做任何修改。
    """
    root = _project_root()
    proc = root / "data" / "processed"
    out: dict[str, list[str]] = {"train": [], "val": [], "test": [], "all": []}
    files = {
        "train": proc / "emotion_train.csv",
        "val": proc / "emotion_val.csv",
        "test": proc / "emotion_test.csv",
    }
    text_cols = ["乘客说的话", "text"]
    for split, fp in files.items():
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        col = next((c for c in text_cols if c in df.columns), None)
        if col is None:
            continue
        for t in df[col].tolist():
            nt = normalize_text(t)
            if nt:
                out[split].append(nt)
                out["all"].append(nt)
    # 3000 xlsx（v2 模型实际使用的数据）同样纳入比对
    xlsx = proc / "metro_emotion_dataset_3000.xlsx"
    if xlsx.exists():
        df = pd.read_excel(xlsx)
        col = "text" if "text" in df.columns else None
        split_col = "split" if "split" in df.columns else None
        if col:
            for _, row in df.iterrows():
                nt = normalize_text(row[col])
                if not nt:
                    continue
                out["all"].append(nt)
                sp = str(row[split_col]) if split_col else "train"
                if sp in out:
                    out[sp].append(nt)
    return out


# --------------------------------------------------------------------------- #
# 标签映射
# --------------------------------------------------------------------------- #
def map_label(raw_label: object, extra_map: dict | None = None) -> tuple[str, str]:
    """返回 (mapped_label, note)。无法映射 -> (REVIEW, reason)。"""
    if raw_label is None or (isinstance(raw_label, float) and pd.isna(raw_label)):
        return REVIEW, "原始标签缺失"
    key = str(raw_label).strip()
    table = {**LABEL_MAP, **(extra_map or {})}
    if key == "":
        return REVIEW, "原始标签为空"
    # 大小写不敏感匹配英文标签
    low = key.lower()
    if key in table:
        mapped = table[key]
    elif low in table:
        mapped = table[low]
    else:
        return REVIEW, f"未登记的原始标签 {key!r}，无法可靠映射"
    if mapped == REVIEW:
        return REVIEW, f"标签 {key!r} 属二分类/中性等，不强行映射到 7 类"
    return mapped, "ok"


# --------------------------------------------------------------------------- #
# 规则错标检测（纯规则，不依赖模型）
# --------------------------------------------------------------------------- #
# 每个目标类的强提示词；原始标签与提示词明显冲突时记为规则证据
_CLASS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "恐惧": ("害怕", "恐惧", "恐慌", "吓死", "惊悚", "可怕", "好怕", "吓人"),
    "愤怒": ("气死", "愤怒", "生气", "恼火", "骂", "滚", "垃圾", "投诉", "差评"),
    "悲伤": ("难过", "伤心", "哭了", "心碎", "失落", "沮丧"),
    "焦虑": ("焦虑", "着急", "担心", "紧张", "来不及", "迟到"),
    "烦躁": ("烦躁", "烦死", "好烦", "真烦", "挤死", "吵死"),
    "无助": ("无助", "没办法", "无可奈何", "回不去", "绝望"),
    "平静": ("还好", "还行", "没事", "平静", "正常"),
}


def rule_label_conflict(text: str, mapped_label: str) -> str | None:
    """若文本强提示另一类别，返回冲突类别名，否则 None。"""
    if mapped_label not in _CLASS_KEYWORDS:
        return None
    norm = text or ""
    for cls, kws in _CLASS_KEYWORDS.items():
        if cls == mapped_label:
            continue
        if any(kw in norm for kw in kws):
            # 文本也包含本类强提示词时不判冲突
            if any(kw in norm for kw in _CLASS_KEYWORDS[mapped_label]):
                continue
            return cls
    return None


def looks_offtopic(text: str) -> bool:
    """疑似无关：规范化后极短、纯数字/纯链接/纯标点。"""
    nt = normalize_text(text)
    if len(nt) < _MIN_CHARS:
        return True
    if re.fullmatch(r"[\d]+", nt):
        return True
    return False


# --------------------------------------------------------------------------- #
# 主筛选流程
# --------------------------------------------------------------------------- #
def _load_candidate_file(fp: Path) -> pd.DataFrame:
    if fp.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(fp)
    return pd.read_csv(fp)


def screen_dataframe(
    df: pd.DataFrame,
    source: str,
    text_col: str = "text",
    label_col: str = "label",
    id_col: str | None = None,
    extra_label_map: dict | None = None,
    existing: dict[str, list[str]] | None = None,
    model_predict: Callable[[str], tuple[str, float]] | None = None,
    near_dup_th: tuple[float, float] = (0.9, 0.92),
) -> list[ScreenResult]:
    """对单个候选 DataFrame 执行完整筛选，返回逐条结果列表。"""
    results: list[ScreenResult] = []
    if existing is None:
        existing = load_existing_texts()
    existing_norm = {s: set(v) for s, v in existing.items()}
    seen_norm: set[str] = set()  # 候选集内部去重

    text_col = text_col if text_col in df.columns else next(
        (c for c in ("text", "乘客说的话", "content", "sentence") if c in df.columns), None
    )
    label_col = label_col if label_col in df.columns else next(
        (c for c in ("label", "情绪", "emotion", "tag") if c in df.columns), None
    )

    for i, row in df.iterrows():
        raw_id = str(row[id_col]) if id_col and id_col in df.columns and pd.notna(row.get(id_col)) else f"{source}#{i}"
        reasons: list[str] = []

        # 1) 字段完整性
        if text_col is None or label_col is None:
            results.append(ScreenResult(source, raw_id, "", "", REVIEW, REVIEW,
                                        f"缺少必要列(text/label)，候选表头={list(df.columns)[:8]}"))
            break
        raw_text = row.get(text_col)
        if raw_text is None or (isinstance(raw_text, float) and pd.isna(raw_text)) or str(raw_text).strip() == "":
            results.append(ScreenResult(source, raw_id, "", str(row.get(label_col, "")),
                                        "", DROP, "文本缺失"))
            continue
        text = str(raw_text).strip()
        original_label = str(row.get(label_col, "")).strip()

        # 2) 异常短 / 无关
        if looks_offtopic(text):
            results.append(ScreenResult(source, raw_id, text, original_label, "", DROP,
                                        "异常短文本或疑似无关内容"))
            continue

        nt = normalize_text(text)

        # 3) 候选集内部精确重复
        if nt in seen_norm:
            results.append(ScreenResult(source, raw_id, text, original_label, "", DROP,
                                        "候选集内精确重复"))
            continue
        seen_norm.add(nt)

        # 4) 与现有数据泄漏检查（精确）
        leak_split = None
        for sp in ("test", "val", "train"):
            if nt in existing_norm.get(sp, set()):
                leak_split = sp
                break
        if leak_split:
            reason = f"与现有{leak_split}集精确重复，禁止进入训练" if leak_split == "test" \
                else f"与现有{leak_split}集重复"
            results.append(ScreenResult(source, raw_id, text, original_label, "", DROP, reason))
            continue

        # 5) 近似重复：先与 test（最严格），再 val/train，再候选内部
        near_hit = _find_near(nt, existing_norm.get("test", []), *near_dup_th)
        if near_hit:
            results.append(ScreenResult(source, raw_id, text, original_label, "", DROP,
                                        f"与现有测试集近似重复，禁止进入训练"))
            continue
        near_hit = _find_near(nt, list(existing_norm.get("val", [])) + list(existing_norm.get("train", [])),
                              *near_dup_th)
        if near_hit:
            results.append(ScreenResult(source, raw_id, text, original_label, "", DROP,
                                        "与现有训练/验证集近似重复"))
            continue
        # 与已保留候选的近似重复
        kept_norms = {normalize_text(r.text) for r in results if r.status == KEEP}
        if any(is_near_duplicate(nt, k, *near_dup_th) for k in kept_norms):
            results.append(ScreenResult(source, raw_id, text, original_label, "", DROP,
                                        "候选集内近似重复"))
            continue

        # 6) 标签映射
        mapped, map_note = map_label(original_label, extra_label_map)
        if mapped == REVIEW:
            results.append(ScreenResult(source, raw_id, text, original_label, REVIEW, REVIEW,
                                        f"标签映射: {map_note}"))
            continue

        # 7) 疑似错标：规则为主，模型仅辅助
        conflict_cls = rule_label_conflict(text, mapped)
        model_label: str | None = None
        model_score: float | None = None
        if model_predict is not None:
            try:
                model_label, model_score = model_predict(text)
            except Exception:
                model_label, model_score = None, None
        if conflict_cls and model_label and model_label == conflict_cls and model_score and model_score >= 0.6:
            results.append(ScreenResult(source, raw_id, text, original_label, REVIEW, REVIEW,
                                        f"疑似错标：规则提示={conflict_cls}，模型辅助={model_label}({model_score:.2f})，与映射标签{mapped}冲突",
                                        model_aux_label=model_label, model_aux_score=model_score))
            continue
        if conflict_cls and not model_label:
            # 仅有规则冲突、无模型时，也标记待复核（规则证据独立成立）
            results.append(ScreenResult(source, raw_id, text, original_label, REVIEW, REVIEW,
                                        f"疑似错标：规则提示文本更接近{conflict_cls}，与映射标签{mapped}冲突，建议人工复核",
                                        model_aux_label=model_label, model_aux_score=model_score))
            continue

        results.append(ScreenResult(
            source, raw_id, text, original_label, mapped, KEEP, "通过筛选",
            model_aux_label=model_label, model_aux_score=model_score,
        ))
    return results


def _find_near(nt: str, pool: Iterable[str], jt: float, rt: float) -> bool:
    for other in pool:
        if is_near_duplicate(nt, other, jt, rt):
            return True
    return False


def results_to_frame(results: list[ScreenResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "source": r.source, "raw_id": r.raw_id, "text": r.text,
            "original_label": r.original_label, "mapped_label": r.mapped_label,
            "status": r.status, "reason": r.reason,
            "model_aux_label": r.model_aux_label, "model_aux_score": r.model_aux_score,
        })
    return pd.DataFrame(rows)


def build_report(results: list[ScreenResult]) -> dict:
    df = results_to_frame(results)
    rep = {
        "total": len(df),
        "by_status": df["status"].value_counts().to_dict() if len(df) else {},
        "by_source": df.groupby("source")["status"].value_counts().unstack(fill_value=0).to_dict()
        if len(df) else {},
        "by_mapped_label_keep": df[df["status"] == KEEP]["mapped_label"].value_counts().to_dict()
        if len(df) else {},
        "top_reasons": df[df["status"] != KEEP]["reason"].value_counts().head(20).to_dict()
        if len(df) else {},
    }
    return rep
