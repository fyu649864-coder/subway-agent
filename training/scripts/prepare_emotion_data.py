# -*- coding: utf-8 -*-
"""
情绪分类数据预处理脚本（第三阶段 - 分组划分修正版）

依据 docs/dataset_analysis.md 与训练后发现的数据泄漏问题：
- 6179 条中 6067 条为 112 条原始样本（有「序号」）的增强改写。
- 随机划分会导致同一原始样本的不同改写分散到训练/验证/测试集，造成泄漏。
- 本脚本按「原始样本」分组，确保同组（同一原始句 + 其所有增强）只能落入一个子集。

字段（来自数据集分析报告，不猜测）：
- 输入：乘客说的话
- 标签：情绪（57 类，自动检测）
- 分组键：推断的原始样本 group_id（1..112）

不修改原始 augmented_metro.xlsx，结果另存于 data/processed/。
"""
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_XLSX = r"C:\Users\64513\Desktop\augmented_metro.xlsx"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "乘客说的话"
LABEL_COL = "情绪"
SEED = 42


def assign_group_ids(df: pd.DataFrame) -> pd.DataFrame:
    """为每行推断其原始样本 group_id（1..112）。
    原始行（序号非空）group_id = 序号；
    增强行：优先在同(场景,情绪)的原始样本中选文本最相似者，否则全局最相似。"""
    df = df.copy().reset_index(drop=True)
    orig_mask = df["序号"].notna()
    orig = df[orig_mask].reset_index(drop=True)
    ot = orig["乘客说的话"].tolist()
    osc = orig["场景"].tolist()
    oe = orig["情绪"].tolist()
    oid = [int(x) for x in orig["序号"].tolist()]

    def best_idx(text: str, scene: str, emo: str):
        cands = [j for j in range(len(orig)) if osc[j] == scene and oe[j] == emo]
        pool = cands if cands else range(len(orig))
        return max(pool, key=lambda j: SequenceMatcher(None, text, ot[j]).ratio())

    group_ids = np.empty(len(df), dtype=int)
    group_ids[orig_mask.to_numpy()] = df.loc[orig_mask, "序号"].to_numpy().astype(int)
    aug_idx = np.where(~orig_mask.to_numpy())[0]
    for i in aug_idx:
        row = df.iloc[i]
        j = best_idx(row[TEXT_COL], row["场景"], row[LABEL_COL])
        group_ids[i] = oid[j]
    df["group_id"] = group_ids
    return df


def grouped_stratified_split(df: pd.DataFrame, seed: int):
    """按 group 划分 8:1:1，确保：
    1) 同一 group（原始样本 + 其所有增强）只落入一个子集（无泄漏）；
    2) 每个情绪标签在训练集中至少出现 1 个 group（避免训练时缺类）。
    做法：每个标签先取 1 组入训练集；剩余组再按 8:1:1 随机分配。"""
    rng = np.random.RandomState(seed)
    g = df.groupby("group_id")[LABEL_COL].first()
    by_label: dict[str, list] = {}
    for gid, lab in g.items():
        by_label.setdefault(lab, []).append(gid)

    s_train, s_val, s_test = set(), set(), set()
    leftover = []
    for lab, gids in by_label.items():
        gids = list(gids)
        rng.shuffle(gids)
        s_train.add(gids[0])            # 每标签至少 1 组进训练集
        leftover.extend(gids[1:])

    rng.shuffle(leftover)
    n = len(leftover)
    n_train = round(n * 0.8)
    n_val = round(n * 0.1)
    s_train.update(leftover[:n_train])
    s_val.update(leftover[n_train:n_train + n_val])
    s_test.update(leftover[n_train + n_val:])

    df = df.copy()
    df["split"] = np.where(df["group_id"].isin(s_train), "train",
                           np.where(df["group_id"].isin(s_val), "val", "test"))
    return df


def main():
    df = pd.read_excel(RAW_XLSX)
    print(f"[INFO] 读取原始数据: {df.shape}")

    df = df[[TEXT_COL, LABEL_COL, "场景", "序号"]].copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df = df[(df[TEXT_COL] != "") & (df[LABEL_COL].notna())]
    df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)
    print(f"[INFO] 清洗后样本数: {len(df)}")

    # 分组
    df = assign_group_ids(df)
    ng = df["group_id"].nunique()
    sizes = df.groupby("group_id").size()
    print(f"[INFO] 分组完成: {ng} 组，组大小 min/mean/max = {sizes.min()}/{sizes.mean():.1f}/{sizes.max()}")

    # 校验：无 group 跨子集（split 前先划分）
    df = grouped_stratified_split(df, SEED)

    # 泄漏自检：同一 group_id 不应出现在多个 split
    cross = df.groupby("group_id")["split"].nunique()
    leak = (cross > 1).sum()
    print(f"[CHECK] 跨子集泄漏的 group 数: {leak}（应为 0）")

    labels = sorted(df[LABEL_COL].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    print(f"[INFO] 情绪类别数（自动检测）: {len(labels)}")

    train_df = df[df["split"] == "train"][[TEXT_COL, LABEL_COL, "group_id"]]
    val_df = df[df["split"] == "val"][[TEXT_COL, LABEL_COL, "group_id"]]
    test_df = df[df["split"] == "test"][[TEXT_COL, LABEL_COL, "group_id"]]
    print(f"[INFO] 分组划分 -> 训练:{len(train_df)} 验证:{len(val_df)} 测试:{len(test_df)}")
    print(f"[INFO] group 数 -> 训练:{train_df['group_id'].nunique()} "
          f"验证:{val_df['group_id'].nunique()} 测试:{test_df['group_id'].nunique()}")

    train_df.to_csv(PROCESSED_DIR / "emotion_train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(PROCESSED_DIR / "emotion_val.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(PROCESSED_DIR / "emotion_test.csv", index=False, encoding="utf-8-sig")

    with open(PROCESSED_DIR / "emotion_label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label,
                   "num_labels": len(labels), "labels": labels}, f, ensure_ascii=False, indent=2)

    summary = {
        "raw_source": RAW_XLSX,
        "total_clean": len(df),
        "text_col": TEXT_COL,
        "label_col": LABEL_COL,
        "num_labels": len(labels),
        "labels": labels,
        "split_method": "grouped_stratified (group_id 分层 8:1:1)",
        "num_groups": int(ng),
        "group_size": {"min": int(sizes.min()), "mean": float(round(sizes.mean(), 2)), "max": int(sizes.max())},
        "groups_with_cross_split_leak": int(leak),
        "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "split_group_count": {"train": int(train_df["group_id"].nunique()),
                              "val": int(val_df["group_id"].nunique()),
                              "test": int(test_df["group_id"].nunique())},
        "seed": SEED,
    }
    with open(PROCESSED_DIR / "emotion_dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 保存 group 映射（可审计）
    df[["group_id", TEXT_COL, LABEL_COL, "split"]].to_csv(
        PROCESSED_DIR / "emotion_group_membership.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] 已保存到: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
