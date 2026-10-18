# -*- coding: utf-8 -*-
"""
地铁情绪 7 分类模型训练 v2
- 数据：data/processed/metro_emotion_dataset_3000.xlsx（按 split 字段划分，不重新划分）
- 骨干：本地 iic/nlp_bert_backbone_base_std（= bert-base-chinese），离线加载
- 严格防泄漏：val/test 不参与训练，test 只在最后评估一次
- 输出：models/emotion/metro_emotion_v2/
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
DATA_XLSX = ROOT / "data" / "processed" / "metro_emotion_dataset_3000.xlsx"
PRETRAINED = str(ROOT / "models" / "pretrained" / "models" /
                 "iic--nlp_bert_backbone_base_std" / "snapshots" / "master")
OUT_DIR = ROOT / "models" / "emotion" / "metro_emotion_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOP_PATIENCE = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] device={DEVICE}  预训练模型={PRETRAINED}")

CLASSES_7 = ["平静", "焦虑", "愤怒", "悲伤", "恐惧", "烦躁", "无助"]
label2id = {c: i for i, c in enumerate(CLASSES_7)}
id2label = {i: c for c, i in label2id.items()}


class EmoDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels, self.tok, self.max_len = texts, labels, tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, padding="max_length",
                       max_length=self.max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[i], dtype=torch.long)}


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            loss = criterion(logits, b["labels"])
            total_loss += loss.item() * b["labels"].size(0)
            n += b["labels"].size(0)
            y_true.extend(b["labels"].cpu().numpy())
            y_pred.extend(logits.argmax(-1).cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pw, rw, f1w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"loss": total_loss / max(n, 1), "accuracy": acc,
            "precision_macro": p, "recall_macro": r, "f1_macro": f1,
            "precision_weighted": pw, "recall_weighted": rw, "f1_weighted": f1w}, y_true, y_pred


def main():
    df = pd.read_excel(DATA_XLSX)
    # 严格按 split 字段，不重新划分
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()

    # 泄漏自检：group 不跨集合
    leak = df.groupby("group_id")["split"].nunique()
    n_leak = int((leak > 1).sum())
    # val/test 不得含增强
    assert (val_df["is_augmented"] == 0).all(), "validation 含增强数据！"
    assert (test_df["is_augmented"] == 0).all(), "test 含增强数据！"
    print(f"[INFO] train/val/test = {len(train_df)}/{len(val_df)}/{len(test_df)}  "
          f"| 跨集合 group 数 = {n_leak}（应为 0）")

    tokenizer = BertTokenizer.from_pretrained(PRETRAINED)

    def to_arrays(d):
        return d["text"].astype(str).tolist(), [label2id[e] for e in d["emotion"].tolist()]

    tr_x, tr_y = to_arrays(train_df)
    va_x, va_y = to_arrays(val_df)
    te_x, te_y = to_arrays(test_df)

    counts = np.bincount(tr_y, minlength=7).astype(float)
    counts[counts == 0] = 1.0
    cw = len(tr_y) / (7 * counts)
    cw = cw / cw.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float).to(DEVICE),
                                    label_smoothing=0.05)

    loaders = {
        "train": DataLoader(EmoDataset(tr_x, tr_y, tokenizer, MAX_LEN), BATCH_SIZE, shuffle=True),
        "val": DataLoader(EmoDataset(va_x, va_y, tokenizer, MAX_LEN), BATCH_SIZE),
        "test": DataLoader(EmoDataset(te_x, te_y, tokenizer, MAX_LEN), BATCH_SIZE),
    }

    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED, num_labels=7, id2label=id2label, label2id=label2id).to(DEVICE)

    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=LR)
    total_steps = len(loaders["train"]) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(WARMUP_RATIO * total_steps), total_steps)

    history, best_f1, best_state, bad = [], -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss, tr_n = 0.0, 0
        for step, b in enumerate(loaders["train"], 1):
            b = {k: v.to(DEVICE) for k, v in b.items()}
            optimizer.zero_grad()
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            loss = criterion(logits, b["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            tr_loss += loss.item() * b["labels"].size(0)
            tr_n += b["labels"].size(0)
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}/{len(loaders['train'])} loss={loss.item():.4f}")
        vm, _, _ = evaluate(model, loaders["val"], criterion)
        rec = {"epoch": epoch, "train_loss": tr_loss / max(tr_n, 1), **vm}
        history.append(rec)
        print(f"[EPOCH {epoch}] train_loss={rec['train_loss']:.4f} "
              f"val_loss={vm['loss']:.4f} val_acc={vm['accuracy']:.4f} "
              f"val_f1_macro={vm['f1_macro']:.4f} val_f1_weighted={vm['f1_weighted']:.4f}")
        if vm["f1_macro"] > best_f1:
            best_f1, best_state, bad = vm["f1_macro"], {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= EARLY_STOP_PATIENCE:
                print(f"[INFO] 早停于 epoch {epoch}")
                break

    # 加载最佳，仅一次 test 评估
    model.load_state_dict(best_state)
    tm, y_true, y_pred = evaluate(model, loaders["test"], criterion)
    print("\n[TEST] 最终测试集指标:")
    for k, v in tm.items():
        print(f"  {k}: {v:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(7)))
    cls_report = classification_report(y_true, y_pred, labels=list(range(7)),
                                       target_names=CLASSES_7, zero_division=0, digits=4)

    # 保存
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    with open(OUT_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label,
                   "num_labels": 7, "labels": CLASSES_7}, f, ensure_ascii=False, indent=2)
    np.save(OUT_DIR / "test_confusion_matrix.npy", cm)

    results = {
        "pretrained": PRETRAINED, "data": str(DATA_XLSX),
        "max_len": MAX_LEN, "batch_size": BATCH_SIZE, "epochs_run": len(history),
        "lr": LR, "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
        "seed": SEED, "device": str(DEVICE), "labels": CLASSES_7,
        "split": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "groups_cross_split": n_leak,
        "history": history, "best_val_f1_macro": best_f1,
        "test_metrics": tm, "test_classification_report": cls_report,
        "test_confusion_matrix": cm.tolist(),
    }
    with open(OUT_DIR / "training_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(7)); ax.set_yticks(range(7))
        ax.set_xticklabels(CLASSES_7, rotation=45); ax.set_yticklabels(CLASSES_7)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (test, 21 samples)")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(OUT_DIR / "test_confusion_matrix.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] 混淆矩阵图跳过: {e}")

    print(f"\n[OK] 模型已保存到: {OUT_DIR}")


if __name__ == "__main__":
    main()
