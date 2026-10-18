# -*- coding: utf-8 -*-
"""情绪数据诊断脚本（不训练，只读分析）。
输出到 data/processed/diagnosis.json，并打印关键信息。"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RAW_XLSX = r"C:\Users\64513\Desktop\augmented_metro.xlsx"
PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT = PROC_DIR / "diagnosis.json"


def main():
    df = pd.read_excel(RAW_XLSX)
    df = df[["序号", "场景", "乘客说的话", "情绪", "情绪类别"]].copy()
    df["乘客说的话"] = df["乘客说的话"].astype(str).str.strip()
    df = df[(df["乘客说的话"] != "") & (df["情绪"].notna())].reset_index(drop=True)

    orig_mask = df["序号"].notna()
    orig = df[orig_mask].copy()
    aug = df[~orig_mask].copy()

    # 1. 原始样本类别分布
    orig_label_cnt = Counter(orig["情绪"].tolist())
    orig_scene_label = orig.groupby(["场景", "情绪"]).size().reset_index(name="n")

    # 2. 加载现有 group 划分
    mem = pd.read_csv(PROC_DIR / "emotion_group_membership.csv")
    train_df = pd.read_csv(PROC_DIR / "emotion_train.csv")
    val_df = pd.read_csv(PROC_DIR / "emotion_val.csv")
    test_df = pd.read_csv(PROC_DIR / "emotion_test.csv")

    # 校验 group_id 是否与「序号」一致（原始行）
    merged = mem.merge(df[["乘客说的话", "序号", "情绪", "场景"]],
                       on="乘客说的话", how="left", suffixes=("", "_raw"))
    orig_rows = merged[merged["序号"].notna()].copy()
    orig_rows["序号"] = orig_rows["序号"].astype(int)
    gid_match = (orig_rows["group_id"] == orig_rows["序号"]).mean()
    gid_mismatch = (orig_rows["group_id"] != orig_rows["序号"]).sum()

    # 同 group 是否可能含两种情绪（异常）
    g_label = merged.groupby("group_id")["情绪"].nunique()
    groups_multi_label = int((g_label > 1).sum())
    multi_label_detail = g_label[g_label > 1].to_dict()

    # 同 group 是否含多个场景
    g_scene = merged.groupby("group_id")["场景"].nunique()
    groups_multi_scene = int((g_scene > 1).sum())

    # 3. 各子集标签分布
    def dist(d):
        c = Counter(d["情绪"].tolist())
        return dict(c.most_common())

    # 4. 增强行匹配准确率：增强行的(场景,情绪)是否与所属 group 原始行一致
    aug_rows = merged[merged["序号"].isna()].copy()
    orig_meta = orig.set_index("序号")[["场景", "情绪"]]
    aug_rows["orig_scene"] = aug_rows["group_id"].map(orig_meta["场景"])
    aug_rows["orig_emo"] = aug_rows["group_id"].map(orig_meta["情绪"])
    scene_match = (aug_rows["场景"] == aug_rows["orig_scene"]).mean()
    emo_match = (aug_rows["情绪"] == aug_rows["orig_emo"]).mean()

    # 5. 类别是否集中在少数原始样本：每类 orig 样本数 / 每类总行数
    label_stats = []
    for lab in sorted(df["情绪"].unique()):
        n_orig = int((orig["情绪"] == lab).sum())
        n_rows = int((df["情绪"] == lab).sum())
        n_groups = int(merged[merged["情绪"] == lab]["group_id"].nunique())
        label_stats.append({"label": lab, "orig_samples": n_orig,
                            "total_rows": n_rows, "groups": n_groups})

    # 6. 只出现 1 个原始样本的类别数
    single_orig = [s for s in label_stats if s["orig_samples"] == 1]

    result = {
        "total_rows": len(df),
        "orig_rows": len(orig),
        "aug_rows": len(aug),
        "orig_label_distribution": dict(orig_label_cnt.most_common()),
        "orig_scene_label_table": orig_scene_label.to_dict(orient="records"),
        "group_id_match_on_orig_rows": float(gid_match),
        "group_id_mismatch_count": int(gid_mismatch),
        "groups_with_multiple_labels": groups_multi_label,
        "multi_label_groups_detail": {int(k): int(v) for k, v in multi_label_detail.items()},
        "groups_with_multiple_scenes": groups_multi_scene,
        "aug_scene_match_rate": float(scene_match),
        "aug_emotion_match_rate": float(emo_match),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_labels": dist(train_df),
        "val_labels": dist(val_df),
        "test_labels": dist(test_df),
        "label_stats": label_stats,
        "labels_with_only_one_original_sample": [s["label"] for s in single_orig],
        "num_labels_with_one_original": len(single_orig),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
