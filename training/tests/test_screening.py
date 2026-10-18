# -*- coding: utf-8 -*-
"""候选数据筛选模块单元测试。

覆盖：文本规范化、近似重复检测、标签映射、规则错标识别、
完整筛选流程（缺失/重复/泄漏/标签映射/错标），以及「不修改正式数据集」约束。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from data_tools.screening import (  # noqa: E402
    CLASSES_7, DROP, KEEP, REVIEW, build_report, is_near_duplicate,
    jaccard_ngram, load_existing_texts, map_label, normalize_text,
    results_to_frame, rule_label_conflict, screen_dataframe,
)


# --------------------------------------------------------------------------- #
# 文本规范化与相似度
# --------------------------------------------------------------------------- #
def test_normalize_text_strips_url_punct_whitespace() -> None:
    assert normalize_text("Hello, World! http://x.com  ") == "helloworld"
    assert normalize_text("你好，世界！") == "你好世界"
    assert normalize_text(None) == ""
    assert normalize_text("   ") == ""


def test_jaccard_and_near_duplicate() -> None:
    a = normalize_text("车厢里突然冒烟了我好害怕")
    b = normalize_text("车厢里突然冒烟了，我好害怕啊")
    assert jaccard_ngram(a, b) > 0.8
    assert is_near_duplicate(a, b)
    # 完全不同的文本不应判近似重复
    assert not is_near_duplicate("今天天气不错适合出去玩", "地铁延误了我要迟到了")


# --------------------------------------------------------------------------- #
# 标签映射
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    ("fear", "恐惧"),
    ("Fear", "恐惧"),   # 大小写不敏感
    ("焦虑", "焦虑"),
    ("positive", REVIEW),   # 二分类不强行映射
    ("unknown_xyz", REVIEW),
    ("", REVIEW),
    (None, REVIEW),
])
def test_map_label(raw, expected) -> None:
    mapped, note = map_label(raw)
    assert mapped == expected
    if expected == REVIEW:
        assert note != "ok"


def test_map_label_values_are_within_7_classes_or_review() -> None:
    for raw in ["fear", "anxiety", "anger", "sad", "helpless", "neutral", "happy"]:
        mapped, _ = map_label(raw)
        assert mapped in CLASSES_7 or mapped == REVIEW


# --------------------------------------------------------------------------- #
# 规则错标识别
# --------------------------------------------------------------------------- #
def test_rule_label_conflict_detects_mismatch() -> None:
    # 文本强提示恐惧，但标签映射为平静 -> 应检出冲突
    assert rule_label_conflict("我好害怕车要翻了", "平静") == "恐惧"
    # 文本与标签一致 -> None
    assert rule_label_conflict("我好害怕车要翻了", "恐惧") is None
    # 文本包含多类强提示词时不轻易判冲突
    assert rule_label_conflict("我又害怕又生气", "恐惧") is None


# --------------------------------------------------------------------------- #
# 完整筛选流程（使用合成数据，不依赖外部模型）
# --------------------------------------------------------------------------- #
def _make_df(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["text", "label"])


def test_screen_handles_missing_text_and_offtopic() -> None:
    df = _make_df([
        ("", "fear"),                       # 空文本
        ("12345", "anger"),                 # 纯数字
        ("a", "fear"),                      # 异常短
        ("正常的一句完整句子", "neutral"),    # 正常
    ])
    existing = {"train": [], "val": [], "test": [], "all": []}
    results = screen_dataframe(df, source="unit", existing=existing)
    statuses = {r.status for r in results}
    assert KEEP in statuses
    assert DROP in statuses
    # 空文本与纯数字均应剔除
    drop_reasons = [r.reason for r in results if r.status == DROP]
    assert any("文本缺失" in r for r in drop_reasons)
    assert any("异常短" in r for r in drop_reasons)


def test_screen_dedup_within_candidate() -> None:
    df = _make_df([
        ("车厢里冒烟了我好害怕", "fear"),
        ("车厢里冒烟了，我好害怕！", "fear"),  # 规范化后精确重复
    ])
    existing = {"train": [], "val": [], "test": [], "all": []}
    results = screen_dataframe(df, source="unit", existing=existing)
    assert len([r for r in results if r.status == DROP]) == 1


def test_screen_blocks_test_leakage() -> None:
    test_text = "好不容易挤上车车子一直不走半小时路程硬生生走一小时"
    existing = {
        "train": [], "val": [],
        "test": [normalize_text(test_text)],
        "all": [normalize_text(test_text)],
    }
    df = _make_df([(test_text, "fear")])
    results = screen_dataframe(df, source="unit", existing=existing)
    assert results[0].status == DROP
    assert "test" in results[0].reason


def test_screen_unmappable_label_goes_to_review() -> None:
    df = _make_df([("这个视频太好笑了哈哈哈", "positive")])
    existing = {"train": [], "val": [], "test": [], "all": []}
    results = screen_dataframe(df, source="unit", existing=existing)
    assert results[0].status == REVIEW
    assert "标签映射" in results[0].reason


def test_screen_rule_conflict_without_model_marks_review() -> None:
    df = _make_df([("我好害怕车是不是要翻了", "happy")])
    existing = {"train": [], "val": [], "test": [], "all": []}
    results = screen_dataframe(df, source="unit", existing=existing)
    assert results[0].status == REVIEW
    assert "疑似错标" in results[0].reason


def test_screen_results_contain_required_fields() -> None:
    df = _make_df([("车厢里人还行一切正常", "neutral")])
    existing = {"train": [], "val": [], "test": [], "all": []}
    results = screen_dataframe(df, source="unit", existing=existing)
    r = results[0]
    assert r.source == "unit"
    assert r.original_label == "neutral"
    assert r.mapped_label == "平静"
    assert r.status == KEEP
    assert r.reason
    frame = results_to_frame(results)
    for col in ("source", "raw_id", "text", "original_label", "mapped_label", "status", "reason"):
        assert col in frame.columns
    rep = build_report(results)
    assert rep["total"] == 1
    assert rep["by_status"].get(KEEP) == 1


# --------------------------------------------------------------------------- #
# 「不修改正式数据集」约束：load_existing_texts 只读不写
# --------------------------------------------------------------------------- #
def test_load_existing_texts_is_readonly() -> None:
    """调用 load_existing_texts 不应产生任何对 data/processed 的写入。"""
    proc = ROOT / "data" / "processed"
    import os
    before = {f.name: os.path.getmtime(f) for f in proc.glob("*") if f.is_file()}
    data = load_existing_texts()
    after = {f.name: os.path.getmtime(f) for f in proc.glob("*") if f.is_file()}
    assert before == after
    assert set(data.keys()) == {"train", "val", "test", "all"}
