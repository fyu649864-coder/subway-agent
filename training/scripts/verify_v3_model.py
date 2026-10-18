# -*- coding: utf-8 -*-
"""训练后验证：v3 模型可被现有 Agent 接口正常加载并输出 7 类分布。"""
import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.perception.emotion_classifier import EmotionClassifier

CLS = ["平静", "焦虑", "愤怒", "悲伤", "恐惧", "烦躁", "无助"]
model_path = str(ROOT / "models" / "emotion" / "metro_emotion_v3")
clf = EmotionClassifier(model_path=model_path, device="cpu")
labels = clf.labels
assert labels == CLS, f"标签不匹配: {labels}"
samples = [
    "车一直不来我上班要迟到了真的急死了",
    "车厢里突然冒烟了我好害怕",
    "这服务态度太差了气死我了要投诉",
    "末班车走了我回不去了好无助",
    "今天地铁人还行一切正常",
]
for s in samples:
    emo, conf, dist = clf.predict(s)
    assert set(dist.keys()) == set(CLS)
    assert abs(sum(dist.values()) - 1.0) < 1e-4
    print(f"{emo:4s} ({conf:.3f})  {s}")
print("[OK] v3 模型接口兼容，7 类输出正常")
