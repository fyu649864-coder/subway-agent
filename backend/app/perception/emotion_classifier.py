"""本地情绪分类模型加载与推理。

封装 ``models/emotion/metro_emotion_v2/`` 下的 BERT 7 分类模型。
- 懒加载单例：首次调用时加载，加载失败会明确抛出异常，不会静默退化为固定/随机结果。
- 离线模式：强制 ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE``，禁止任何下载行为。
- 线程内复用同一模型实例，避免重复加载。
"""

from __future__ import annotations

import os
from typing import Optional

# 强制离线，禁止 transformers 访问网络
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from transformers import BertForSequenceClassification, BertTokenizer  # noqa: E402

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmotionModelLoadError(RuntimeError):
    """情绪模型加载失败时抛出，调用方可据此明确报告错误。"""


class EmotionClassifier:
    """本地 BERT 情绪分类器（7 类：平静/焦虑/愤怒/悲伤/恐惧/烦躁/无助）。"""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None) -> None:
        self.model_path = model_path or settings.emotion_model_path
        self.device = device or settings.device
        self._tokenizer: Optional[BertTokenizer] = None
        self._model: Optional[BertForSequenceClassification] = None
        self._id2label: dict[int, str] = {}
        self._loaded = False
        self._load_error: Optional[BaseException] = None

    # ------------------------------------------------------------------ load
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._load_error is not None:
            # 已经失败过，直接重抛，避免反复尝试加载
            raise EmotionModelLoadError(
                f"情绪模型加载失败（已尝试过）：{self._load_error}"
            ) from self._load_error

        if not self.model_path or not os.path.isdir(self.model_path):
            err = EmotionModelLoadError(
                f"情绪模型路径不存在或不是目录：{self.model_path!r}"
            )
            self._load_error = err
            raise err

        try:
            logger.info("加载本地情绪模型: path=%s device=%s", self.model_path, self.device)
            self._tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self._model = BertForSequenceClassification.from_pretrained(self.model_path)
            self._model.to(self.device)
            self._model.eval()
            self._id2label = {int(k): v for k, v in self._model.config.id2label.items()}
            self._loaded = True
            logger.info("情绪模型加载成功: labels=%s", self._id2label)
        except Exception as exc:  # 明确捕获，不静默吞掉
            self._load_error = exc
            logger.exception("情绪模型加载失败: path=%s", self.model_path)
            raise EmotionModelLoadError(f"情绪模型加载失败：{exc}") from exc

    @property
    def labels(self) -> list[str]:
        self._ensure_loaded()
        return [self._id2label[i] for i in sorted(self._id2label)]

    # ---------------------------------------------------------------- infer
    @torch.no_grad()
    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        """对单条文本推理。

        Returns:
            (emotion, confidence, emotions)：主情绪标签、置信度（softmax 最大值）、
            全量 7 类概率分布。
        """
        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None

        enc = self._tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=settings.emotion_max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self._model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().tolist()

        emotions = {self._id2label[i]: float(p) for i, p in enumerate(probs)}
        best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        emotion = self._id2label[best_idx]
        confidence = float(probs[best_idx])
        return emotion, confidence, emotions


# ---------------------------------------------------------------------------
# 进程级单例
# ---------------------------------------------------------------------------
_classifier: Optional[EmotionClassifier] = None


def get_emotion_classifier() -> EmotionClassifier:
    """获取情绪分类器单例。"""
    global _classifier
    if _classifier is None:
        _classifier = EmotionClassifier()
    return _classifier
