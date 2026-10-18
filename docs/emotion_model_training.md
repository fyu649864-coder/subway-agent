# 情绪分类模型训练报告（第三阶段）

> 生成时间：2026-10-08
> 训练脚本：`training/scripts/train_emotion.py`
> 数据预处理脚本：`training/scripts/prepare_emotion_data.py`
> 依据：`docs/dataset_analysis.md`（第二阶段数据集分析结论）

---

## 一、使用的数据字段与标签

严格按照第二阶段分析报告，未自行猜测字段：

| 项目 | 内容 | 来源 |
|------|------|------|
| 输入字段 | `乘客说的话` | 数据集分析报告 §三 |
| 标签字段 | `情绪` | 数据集分析报告 §三 |
| 标签类别数 | **57**（自动从数据检测，未写死） | 预处理脚本自动统计 |
| 原始数据源 | `C:\Users\64513\Desktop\augmented_metro.xlsx` | 未修改 |
| 清洗后总样本 | 6179（去空、去重后不变） | — |

57 个情绪类别（自动检测）：
不便、不满、不适、克制、压抑、厌恶、厌烦、反感、反胃、吐槽、善意、嘲讽、困惑、失望、委屈、平和、平复、平静、庆幸、建议、开心、恐慌、恼火、惊吓、愉悦、感动、感激、愤怒、懊恼、懊悔、抱怨、担心、担忧、敬佩、无奈、无语、暖心、欣慰、歉意、沮丧、满意、激动、烦躁、焦急、焦虑、理解、生气、疑惑、疲惫、着急、礼貌、窘迫、紧张、舒心、认可、轻松、难受。

---

## 二、数据集划分（已做分组划分修正）

> 修正原因：6179 条中 6067 条是 112 条原始样本的增强改写。随机划分会让同一句的不同改写分散到训练/测试集，造成数据泄漏（首次训练 F1=0.9968 即为泄漏所致）。现已改为按「原始样本」分组划分。

**分组方法**：
- 以「序号」非空的 112 行为原始样本，每条为一个 group（group_id=1..112）。
- 增强行通过文本相似度匹配到其原始样本（优先同 `场景`+`情绪`，否则全局最相似），归入同一 group。
- 共 **112 组**，组大小 20–152（均值 55.2）。
- 划分：每个标签先取 1 组入训练集（保证 57 类全部在训练集中可见），剩余 55 组再按 8:1:1 随机分配。
- **泄漏自检：跨子集的 group 数 = 0**（同一原始样本及其所有增强只出现在一个子集）。

| 子集 | 样本数（行） | group 数 | 覆盖标签数 |
|------|------------|----------|-----------|
| 训练集 | 5720 | 101 | 57（全部） |
| 验证集 | 233 | 6 | 6 |
| 测试集 | 226 | 5 | 5 |

> 说明：受限于「组不可拆分」，验证/测试集的行数与标签数较少（6 / 5 个 group），这是消除泄漏的必然代价。行级比例约为 92.6 : 3.8 : 3.6。

处理后的数据另存于 `data/processed/`（`emotion_train.csv` / `emotion_val.csv` / `emotion_test.csv` / `emotion_label_map.json` / `emotion_dataset_summary.json` / `emotion_group_membership.csv`），**未修改原始 xlsx**。

---

## 三、使用的模型

| 项目 | 内容 |
|------|------|
| 模型架构 | `BertForSequenceClassification`（BERT-base，12 层，hidden=768，102M 参数） |
| 预训练权重 | `iic/nlp_bert_backbone_base_std`（ModelScope，等价于 `bert-base-chinese`，vocab=21128） |
| 权重来源 | 从 ModelScope 拉取至本地 `models/pretrained/`，**全程离线加载**，未接入云端大模型 |
| 分类头 | 随机初始化，num_labels=57（自动） |
| Tokenizer | `BertTokenizer`（vocab.txt），max_length=64（文本最长 34 字，64 足够） |

> 备注：最初尝试从 HuggingFace 下载 `bert-base-chinese`，因网络对 400MB 大文件链路不稳定（WinError 10054）多次失败；经确认本地无权重缓存后，按用户指示改用 ModelScope 拉取等价模型并离线加载。

---

## 四、训练参数

| 参数 | 值 |
|------|-----|
| 设备 | CPU（本机无 CUDA） |
| Batch size | 32 |
| 最大 epoch | 5（实际跑满 5 轮，早停触发） |
| 学习率 | 2e-5 |
| 优化器 | AdamW（weight_decay=0.01，bias/LayerNorm 无衰减） |
| 调度器 | linear warmup（warmup_ratio=0.1） |
| 损失函数 | 加权交叉熵 CrossEntropyLoss（含 label_smoothing=0.05） |
| 类别不平衡处理 | 按训练集各类频次计算 class weights（反比加权），**不删除任何数据** |
| 梯度裁剪 | max_norm=1.0 |
| 早停 | 监控 val_f1_macro，patience=2 |
| 随机种子 | 42 |

---

## 五、训练过程（验证集，分组划分后）

| Epoch | train_loss | val_loss | val_acc | val_f1_macro | val_f1_weighted |
|-------|-----------|----------|---------|--------------|-----------------|
| 1 | 3.3867 | 3.9064 | 0.0258 | 0.0070 | 0.0337 |
| 2 | 1.3047 | 3.9315 | 0.0300 | 0.0084 | 0.0390 |
| 3 | 0.6279 | 4.1062 | 0.0300 | **0.0096** | 0.0418 |
| 4 | 0.5204 | 4.2520 | 0.0258 | 0.0060 | 0.0340 |
| 5 | 0.5022 | 4.3178 | 0.0258 | 0.0063 | 0.0340 |

最佳验证 epoch = 3（val_f1_macro=0.0096），后 2 轮未提升，触发早停，加载 epoch 3 权重在测试集评估。

> 对比：随机划分（泄漏）时同期 val_acc 已达 0.95+；分组划分后 val_acc 仅约 3%，直观验证了此前的高指标确由数据泄漏造成。

---

## 六、测试集指标（分组划分后，无泄漏）

| 指标 | 值 |
|------|-----|
| **Accuracy** | **0.1637** |
| Precision（macro） | 0.1429 |
| Recall（macro） | 0.0734 |
| **F1（macro）** | **0.0753** |
| Precision（weighted） | 0.3186 |
| Recall（weighted） | 0.1637 |
| F1（weighted） | 0.1679 |

> 注：测试集仅含 5 个 group（226 行、5 个标签）。macro 指标在 57 类上平均，未出现的类贡献为 0，故 macro 值偏低；weighted 与 Accuracy 反映这 226 条上的实际表现，Accuracy ≈ 16.4%（随机基线约 1/57≈1.75%，模型仍显著高于随机）。

- 混淆矩阵：已保存 `models/emotion/test_confusion_matrix.png`（图）与 `test_confusion_matrix.npy`（数组）。
- 逐类分类报告：保存在 `models/emotion/training_results.json` 的 `test_classification_report` 字段。

---

## 七、当前模型效果（分组划分后的真实结论）

- 消除数据泄漏后，模型在**未见过的原始样本（及其增强）**上泛化能力很弱：
  - 测试集 Accuracy ≈ **16.4%**，macro-F1 ≈ **0.075**，weighted-F1 ≈ **0.168**。
  - 训练损失持续下降（epoch 5 train_loss=0.50）而验证损失上升（3.9→4.3），呈现典型**过拟合**：模型记住了训练集 101 个 group 的文本模式，却无法迁移到新的 group。
- 对比：随机划分（泄漏）时 F1=0.9968；分组划分后 F1≈0.075。**两者之差即数据泄漏带来的虚高**。
- 仍高于随机基线（≈1.75%），说明模型学到了部分可迁移的情绪语义，但远不足以支持可靠的细粒度情绪识别。

**根因分析**：
1. 数据由 112 个模板句改写而来，「情绪」与「模板文本」高度绑定，模型容易学到模板特征而非通用情绪表达。
2. 57 类粒度过细，且部分标签语义重叠，跨模板时同类情绪的表述差异大、异类表述反而相似。
3. 真正不同的原始样本只有 112 条，独立样本量严重不足。

---

## 八、后续需要改进的问题

1. **增加真实独立样本**：当前唯一根本解法是补充**大量未参与增强的真实乘客原话**（建议数千条），重新分组划分后训练，才能获得可信模型。
2. **标签归并**：将 57 类按语义合并为 15–25 类（如 焦虑/焦急/着急→焦虑，愤怒/恼火/生气→愤怒），降低任务难度、改善小样本鲁棒性。
3. **降低过拟合**：增大 dropout、降低学习率、减少 epoch、加重 weight decay；或改用更强的中文预训练骨干（如 RoBERTa-wwm）。
4. **先做粗粒度**：可先用 `情绪类别`（负面/中性/正面，需补全增强数据的极性）做 3 分类，验证流程与泛化，再上细粒度。
5. **工程接入暂缓**：在泛化能力改善前，不建议直接替换 `PerceptionService` 的 mock 规则，避免线上效果不可控。
6. **算力**：本次为 CPU 训练（每 epoch 约 12–15 分钟）；后续迭代建议使用 GPU 加速。

---

## 九、产出文件清单

| 文件 | 位置 |
|------|------|
| 预处理脚本 | `training/scripts/prepare_emotion_data.py` |
| 训练脚本 | `training/scripts/train_emotion.py` |
| 处理后数据 | `data/processed/emotion_{train,val,test}.csv`、`emotion_label_map.json`、`emotion_dataset_summary.json`、`emotion_group_membership.csv` |
| 模型权重 | `models/emotion/model.safetensors` |
| 模型配置 | `models/emotion/config.json` |
| Tokenizer | `models/emotion/tokenizer.json`、`tokenizer_config.json` |
| 标签映射 | `models/emotion/label_map.json` |
| 训练结果与参数 | `models/emotion/training_results.json` |
| 混淆矩阵 | `models/emotion/test_confusion_matrix.png`、`test_confusion_matrix.npy` |
| 预训练骨干（本地） | `models/pretrained/models/iic--nlp_bert_backbone_base_std/snapshots/master/` |
| 本报告 | `docs/emotion_model_training.md` |

> 本阶段未训练事件模型、未训练情绪强度模型、未修改原始数据、未接入云端大模型、未开发前端。
