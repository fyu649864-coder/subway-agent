# -*- coding: utf-8 -*-
"""构建 v3 训练数据集（约 6000 条真实域内数据）。

- 数据源：C:/Users/64513/Desktop/augmented_metro.xlsx（只读，不修改原始文件）
- 57 类原始标签 → 7 类（复用既有映射，与 build_emotion_dataset_3000.py 一致）
- 通过与 112 条原始种子文本的相似度，为每条增强样本还原所属 group_id，
  保证同一原始种子的全部变体只进入 train/val/test 中的一个集合，杜绝跨集合泄漏。
- 分层分组划分：按类别内的 group 做 70/15/15 划分，固定随机种子 42，
  优先保证每个类别在 test 中至少有 1 个 group（即多条真实样本）。
- 输出：
    data/processed/metro_emotion_dataset_v3.xlsx
    data/processed/emotion_v3_split_report.json
  不覆盖 v2 数据集与既有 emotion_*.csv。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_tools.screening import jaccard_ngram, normalize_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW_XLSX = Path(r"C:\Users\64513\Desktop\augmented_metro.xlsx")
OUT_DIR = ROOT / "data" / "processed"
SEED = 42
CLASS_7 = ["平静", "焦虑", "愤怒", "悲伤", "恐惧", "烦躁", "无助"]

LABEL_MAP_57_TO_7 = {
    "平静": "平静", "平和": "平静", "平复": "平静", "克制": "平静", "轻松": "平静",
    "礼貌": "平静", "歉意": "平静", "理解": "平静", "建议": "平静", "满意": "平静",
    "舒心": "平静", "认可": "平静", "开心": "平静", "愉悦": "平静", "庆幸": "平静",
    "感激": "平静", "感动": "平静", "暖心": "平静", "敬佩": "平静", "欣慰": "平静",
    "善意": "平静", "激动": "平静",
    "焦虑": "焦虑", "焦急": "焦虑", "着急": "焦虑", "担心": "焦虑", "担忧": "焦虑",
    "紧张": "焦虑", "疑惑": "焦虑", "困惑": "焦虑", "窘迫": "焦虑",
    "愤怒": "愤怒", "生气": "愤怒", "恼火": "愤怒", "不满": "愤怒", "反感": "愤怒",
    "厌恶": "愤怒", "厌烦": "愤怒", "嘲讽": "愤怒", "抱怨": "愤怒", "吐槽": "愤怒",
    "失望": "悲伤", "沮丧": "悲伤", "委屈": "悲伤", "懊悔": "悲伤", "懊恼": "悲伤",
    "恐慌": "恐惧", "惊吓": "恐惧",
    "烦躁": "烦躁", "难受": "烦躁", "不适": "烦躁", "疲惫": "烦躁",
    "反胃": "烦躁", "无语": "烦躁",
    "无奈": "无助", "压抑": "无助", "不便": "无助",
}


def main() -> None:
    if not RAW_XLSX.exists():
        raise FileNotFoundError(f"原始数据不存在（只读引用）：{RAW_XLSX}")
    df = pd.read_excel(RAW_XLSX)
    df["text"] = df["乘客说的话"].astype(str).str.strip()
    df = df[(df["text"] != "") & df["情绪"].notna() & (df["text"].str.len() >= 4)].copy()
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["emotion"] = df["情绪"].map(LABEL_MAP_57_TO_7)
    unmapped = df[df["emotion"].isna()]["情绪"].unique().tolist()
    assert not unmapped, f"存在未映射标签：{unmapped}"
    df = df[df["emotion"].notna()].copy()

    # 112 条原始种子（有序号），其余为增强变体
    seeds = df[df["序号"].notna()].copy()
    seeds["gid"] = seeds["序号"].astype(int)
    seed_norm = {int(g): normalize_text(t) for g, t in zip(seeds["序号"], seeds["text"])}

    def assign_group(text: str) -> int:
        nt = normalize_text(text)
        best_g, best_s = -1, -1.0
        for g, sn in seed_norm.items():
            s = jaccard_ngram(nt, sn)
            if s > best_s:
                best_s, best_g = s, g
        return best_g

    df["group_id"] = -1
    df.loc[df["序号"].notna(), "group_id"] = seeds.set_index(df[df["序号"].notna()].index)["gid"].values
    aug_mask = df["序号"].isna()
    df.loc[aug_mask, "group_id"] = [assign_group(t) for t in df.loc[aug_mask, "text"]]
    df["group_id"] = df["group_id"].astype(int)
    assert (df["group_id"] > 0).all()

    rng = np.random.RandomState(SEED)
    df["split"] = ""
    group2emo = df.groupby("group_id")["emotion"].first().to_dict()
    test_counts, val_counts = {}, {}
    for emo in CLASS_7:
        gids = [g for g, e in group2emo.items() if e == emo]
        rng.shuffle(gids)
        n = len(gids)
        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.10)) if n - n_test > 1 else 0
        if n_test + n_val >= n:
            n_val = 0
            n_test = max(1, n - 1)
        test_g, val_g = set(gids[:n_test]), set(gids[n_test:n_test + n_val])
        df.loc[df["group_id"].isin(test_g), "split"] = "test"
        df.loc[df["group_id"].isin(val_g), "split"] = "validation"
        test_counts[emo] = len(test_g)
        val_counts[emo] = len(val_g)
    df.loc[df["split"] == "", "split"] = "train"

    # 泄漏自检：同一 group 不得跨 split
    leak = df.groupby("group_id")["split"].nunique()
    n_leak = int((leak > 1).sum())

    out = pd.DataFrame({
        "text": df["text"], "emotion": df["emotion"], "group_id": df["group_id"],
        "source": np.where(df["序号"].notna(), "original", "augmentation"),
        "is_augmented": (df["序号"].isna()).astype(int),
        "split": df["split"], "scene": df["场景"],
        "original_label": df["情绪"],
    }).reset_index(drop=True)

    report = {
        "source_file": str(RAW_XLSX),
        "label_mapping": "57->7 (reused from build_emotion_dataset_3000.py)",
        "total": int(len(out)),
        "num_groups": int(out["group_id"].nunique()),
        "group_size": {"min": int(out.groupby("group_id").size().min()),
                       "max": int(out.groupby("group_id").size().max()),
                       "mean": float(out.groupby("group_id").size().mean())},
        "by_split": {k: int(v) for k, v in Counter(out["split"]).items()},
        "by_emotion": {k: int(v) for k, v in Counter(out["emotion"]).items()},
        "split_x_emotion": out.groupby(["split", "emotion"]).size().unstack(fill_value=0).astype(int).to_dict(),
        "groups_per_class_in_test": test_counts,
        "groups_per_class_in_val": val_counts,
        "groups_cross_split": n_leak,
        "seed": SEED,
        "split_method": "stratified grouped split 70/15/15 by group_id within each class",
    }

    out_path = OUT_DIR / "metro_emotion_dataset_v3.xlsx"
    out.to_excel(out_path, index=False)
    with open(OUT_DIR / "emotion_v3_split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[OK] 已保存: {out_path}")


if __name__ == "__main__":
    main()
