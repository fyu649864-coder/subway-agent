# -*- coding: utf-8 -*-
"""
情绪分类模型训练脚本（第三阶段）

严格依据 docs/dataset_analysis.md：
- 输入字段：乘客说的话
- 标签字段：情绪（自动检测类别数，不写死）
- 骨干：bert-base-chinese（从 ModelScope 拉取的本地副本，离线加载，不接云端大模型）
- 类别不平衡：使用加权交叉熵（class weights），不删除数据
- 输出：模型 / Tokenizer / 标签映射 / 训练参数与指标 -> models/emotion/
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

# 离线模式：禁止任何联网下载
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------- 配置 ----------
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "emotion"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 本地预训练模型（ModelScope: iic/nlp_bert_backbone_base_std = bert-base-chinese）
PRETRAINED = str(Path(__file__).resolve().parents[2] /
                 "models" / "pretrained" / "models" /
                 "iic--nlp_bert_backbone_base_std" / "snapshots" / "master")
if not Path(PRETRAINED, "pytorch_model.bin").exists():
    raise FileNotFoundError(f"本地预训练模型不存在: {PRETRAINED}")
print(f"[INFO] 使用本地预训练模型: {PRETRAINED}")
MAX_LEN = 64                        # 报告：文本最长 34 字，64 足够
BATCH_SIZE = 32
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOP_PATIENCE = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] device = {DEVICE}")


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, padding="max_length",
                       max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
        }


def evaluate(model, loader, criterion, id2label):
    model.eval()
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
            loss = criterion(logits, batch["labels"])
            total_loss += loss.item() * batch["labels"].size(0)
            n += batch["labels"].size(0)
            preds = logits.argmax(dim=-1).cpu().numpy()
            y_true.extend(batch["labels"].cpu().numpy())
            y_pred.extend(preds)
    avg_loss = total_loss / max(n, 1)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pw, rw, f1w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics = {"loss": avg_loss, "accuracy": acc,
               "precision_macro": p, "recall_macro": r, "f1_macro": f1,
               "precision_weighted": pw, "recall_weighted": rw, "f1_weighted": f1w}
    return metrics, y_true, y_pred


def main():
    # 1. 加载数据与标签映射（自动检测类别）
    with open(PROCESSED_DIR / "emotion_label_map.json", encoding="utf-8") as f:
        lm = json.load(f)
    label2id, id2label = lm["label2id"], {int(k): v for k, v in lm["id2label"].items()}
    num_labels = lm["num_labels"]
    labels_list = lm["labels"]
    print(f"[INFO] 情绪类别数: {num_labels}")

    train_df = pd.read_csv(PROCESSED_DIR / "emotion_train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "emotion_val.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "emotion_test.csv")
    print(f"[INFO] 训练/验证/测试 = {len(train_df)}/{len(val_df)}/{len(test_df)}")

    tokenizer = BertTokenizer.from_pretrained(PRETRAINED)

    def to_arrays(df):
        texts = df["乘客说的话"].astype(str).tolist()
        labels = [label2id[x] for x in df["情绪"].tolist()]
        return texts, labels

    tr_x, tr_y = to_arrays(train_df)
    va_x, va_y = to_arrays(val_df)
    te_x, te_y = to_arrays(test_df)

    # 2. 类别权重（处理不平衡，不删数据）
    counts = np.bincount(tr_y, minlength=num_labels).astype(float)
    counts[counts == 0] = 1.0
    class_weights = len(tr_y) / (num_labels * counts)
    class_weights = class_weights / class_weights.mean()  # 归一化，稳定训练
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # 3. 数据加载器
    train_ds = EmotionDataset(tr_x, tr_y, tokenizer, MAX_LEN)
    val_ds = EmotionDataset(va_x, va_y, tokenizer, MAX_LEN)
    test_ds = EmotionDataset(te_x, te_y, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # 4. 模型
    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED, num_labels=num_labels,
        id2label=id2label, label2id=label2id).to(DEVICE)

    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps)

    # 5. 训练循环
    history = []
    best_f1 = -1.0
    best_state = None
    bad_epochs = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss, tr_n = 0.0, 0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
            loss = criterion(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            tr_loss += loss.item() * batch["labels"].size(0)
            tr_n += batch["labels"].size(0)
            if step % 30 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}")
        val_metrics, _, _ = evaluate(model, val_loader, criterion, id2label)
        rec = {"epoch": epoch, "train_loss": tr_loss / max(tr_n, 1), **val_metrics}
        history.append(rec)
        print(f"[EPOCH {epoch}] train_loss={rec['train_loss']:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
              f"val_f1_macro={val_metrics['f1_macro']:.4f} val_f1_weighted={val_metrics['f1_weighted']:.4f}")
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOP_PATIENCE:
                print(f"[INFO] 早停：连续 {EARLY_STOP_PATIENCE} 轮 val_f1 未提升")
                break

    # 6. 加载最佳权重，测试集评估
    model.load_state_dict(best_state)
    test_metrics, y_true, y_pred = evaluate(model, test_loader, criterion, id2label)
    print("\n[TEST] 测试集指标:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 7. 混淆矩阵 + 分类报告
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_labels)))
    report = classification_report(y_true, y_pred, labels=list(range(num_labels)),
                                   target_names=labels_list, zero_division=0)

    # 8. 保存模型、tokenizer、标签映射
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    with open(MODEL_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label,
                   "num_labels": num_labels, "labels": labels_list}, f, ensure_ascii=False, indent=2)
    np.save(MODEL_DIR / "test_confusion_matrix.npy", cm)

    # 9. 保存训练结果与参数
    results = {
        "pretrained": PRETRAINED,
        "max_len": MAX_LEN, "batch_size": BATCH_SIZE, "epochs_run": len(history),
        "lr": LR, "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
        "seed": SEED, "device": str(DEVICE),
        "text_col": "乘客说的话", "label_col": "情绪",
        "num_labels": num_labels, "labels": labels_list,
        "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "history": history,
        "best_val_f1_macro": best_f1,
        "test_metrics": test_metrics,
        "test_classification_report": report,
    }
    with open(MODEL_DIR / "training_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 10. 混淆矩阵图（可选保存）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(16, 14))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(num_labels)); ax.set_yticks(range(num_labels))
        ax.set_xticklabels(labels_list, rotation=90, fontsize=7)
        ax.set_yticklabels(labels_list, fontsize=7)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Emotion Classification - Confusion Matrix (test)")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(MODEL_DIR / "test_confusion_matrix.png", dpi=150)
        plt.close(fig)
        print("[OK] 混淆矩阵图已保存")
    except Exception as e:
        print(f"[WARN] 混淆矩阵图跳过: {e}")

    print(f"\n[OK] 模型与结果已保存到: {MODEL_DIR}")


if __name__ == "__main__":
    main()
