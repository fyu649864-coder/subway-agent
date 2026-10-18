# -*- coding: utf-8 -*-
"""
重新划分现有 metro_emotion_dataset_3000.xlsx：
- Train 70 group / Validation 21 group / Test 21 group（共 112，无跨集合）。
- 仅从 train 移出 9 个最小的「平静」group 到 val/test，行损失最小，总量仍约 2900。
- val/test 仅保留真实原始文本（is_augmented=0），不保留增强。
- 不生成任何新数据，不训练，不改原始 xlsx，不动 models/。
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "processed"
XLSX = DATA / "metro_emotion_dataset_3000.xlsx"
MAP_JSON = DATA / "emotion_label_mapping.json"

df = pd.read_excel(XLSX)
print("重划分前:", dict(Counter(df["split"])), "| 总行:", len(df))

# 每个 group 的元信息
ginfo = df.groupby("group_id").agg(
    emo=("emotion", "first"), rows=("text", "size"),
    split=("split", "first")).reset_index()

# 目标：平静 test 10 / val 11 / train 22（其余类维持原划分）
# 当前平静：test 6, val 6, train 31 → 需从 train 移 4 到 test、5 到 val
move = {"test": 4, "validation": 5}
calm_train = ginfo[(ginfo["emo"] == "平静") & (ginfo["split"] == "train")]
calm_train = calm_train.sort_values("rows")  # 移最小的，行损失最少
to_test = calm_train.head(move["test"])["group_id"].tolist()
to_val = calm_train.iloc[move["test"]:move["test"] + move["validation"]]["group_id"].tolist()

df.loc[df["group_id"].isin(to_test), "split"] = "test"
df.loc[df["group_id"].isin(to_val), "split"] = "validation"

# val/test 只保留原始文本
keep_mask = ((df["split"] == "train") | (df["is_augmented"] == 0))
df = df[keep_mask].reset_index(drop=True)

# ── 质量检查 ──
leak = df.groupby("group_id")["split"].nunique()
n_leak = int((leak > 1).sum())
n_dup = int(df["text"].duplicated().sum())
n_empty = int((df["text"].astype(str).str.strip() == "").sum())
val_aug = int(((df["split"] == "validation") & (df["is_augmented"] == 1)).sum())
test_aug = int(((df["split"] == "test") & (df["is_augmented"] == 1)).sum())

report = {
    "total": int(len(df)),
    "by_emotion": dict(Counter(df["emotion"])),
    "by_split": dict(Counter(df["split"])),
    "groups_by_split": df.groupby("split")["group_id"].nunique().to_dict(),
    "split_x_emotion": df.groupby(["split", "emotion"]).size().unstack(fill_value=0).to_dict(),
    "groups_cross_split": n_leak,
    "duplicate_texts": n_dup,
    "empty_texts": n_empty,
    "augmented_in_validation": val_aug,
    "augmented_in_test": test_aug,
    "val_test_all_original": (val_aug == 0 and test_aug == 0),
}

df.to_excel(XLSX, index=False)
with open(MAP_JSON, encoding="utf-8") as f:
    meta = json.load(f)
meta["quality_report"] = report
with open(MAP_JSON, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(json.dumps(report, ensure_ascii=False, indent=2))
