"""感知服务。

整合三部分能力：
1. 情绪分类：调用本地 BERT 模型（``models/emotion/metro_emotion_v2/``），7 类。
   模型加载失败会明确抛出异常，不会静默退化为固定/随机结果。
2. 事件识别：本地规则 + 关键词匹配（8 类），支持多事件。
3. 情绪强度：本地规则（低/中/高），与情绪类别独立判断。
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.models import (
    EmotionResult,
    EventResult,
    IntensityResult,
    PerceptionResult,
    UrgencyLevel,
)
from app.perception.emotion_classifier import EmotionModelLoadError, get_emotion_classifier
from app.perception.event_recognizer import EVENT_OTHER, recognize_events
from app.perception.intensity_analyzer import analyze_intensity

logger = get_logger(__name__)

# 需升级处理的事件类别（与决策服务保持一致）
_ESCALATION_EVENTS = {"突发事件", "服务投诉"}


class PerceptionService:
    """感知服务：文本 -> PerceptionResult。"""

    def __init__(self, emotion_classifier=None) -> None:
        # 允许注入（便于测试 mock），默认使用单例
        self._emotion_classifier = emotion_classifier

    @property
    def emotion_classifier(self):
        if self._emotion_classifier is None:
            self._emotion_classifier = get_emotion_classifier()
        return self._emotion_classifier

    def analyze(self, text: str, context_text: str | None = None) -> PerceptionResult:
        """识别情绪、强度、事件与紧急程度。

        ``context_text`` 为最近几轮对话拼接出的上下文文本（可选）：
        - 事件识别与强度分析可使用它，避免多轮中短句丢失已确立的情境；
        - 情绪分类（BERT）始终只看当前 ``text``，不改变模型行为。

        Raises:
            EmotionModelLoadError: 情绪模型不可用时明确抛出，不静默降级。
        """
        text = text or ""
        # 事件/强度使用“上下文 + 当前文本”，但对外仍以当前文本为主
        merged = f"{context_text} {text}".strip() if context_text else text

        # 1. 事件识别（规则，多事件）
        hits = recognize_events(merged)
        primary = hits[0]
        events_dist = {h.event: h.confidence for h in hits}
        all_events = [
            {"event": h.event, "confidence": h.confidence, "evidence": h.evidence}
            for h in hits
        ]

        # 2. 情绪强度（规则）
        intensity = analyze_intensity(merged)

        # 3. 情绪分类（本地模型）—— 仅用当前文本，加载失败则明确抛错
        try:
            emotion, emotion_score, emotions = self.emotion_classifier.predict(text)
        except EmotionModelLoadError:
            logger.exception("情绪模型不可用，无法完成感知: text_len=%d", len(text))
            raise
        except Exception:
            logger.exception("情绪模型推理异常: text_len=%d", len(text))
            raise

        # 紧急程度：突发事件 / 高投诉 高；强度高 或 负面情绪明显 中；否则 low
        urgency = UrgencyLevel.low
        if primary.event in _ESCALATION_EVENTS and primary.event != EVENT_OTHER:
            urgency = UrgencyLevel.high
        elif intensity.level == "高" or primary.confidence >= 0.8:
            urgency = UrgencyLevel.high
        elif intensity.level == "中" or primary.event != EVENT_OTHER:
            urgency = UrgencyLevel.medium

        result = PerceptionResult(
            text=text,
            emotion=emotion,
            emotion_score=emotion_score,
            emotion_confidence=emotion_score,
            intensity=intensity.score,
            emotion_intensity=intensity.level,
            intensity_evidence=intensity.evidence,
            event=primary.event,
            event_score=primary.confidence,
            event_confidence=primary.confidence,
            event_evidence=primary.evidence,
            urgency=urgency,
            emotion_detail=EmotionResult(
                emotion=emotion,
                emotion_score=emotion_score,
                emotions=emotions,
                model_name="metro_emotion_v2",
            ),
            event_detail=EventResult(
                event=primary.event,
                event_score=primary.confidence,
                events=events_dist,
                evidence=primary.evidence,
                all_events=all_events,
                model_name="rule-event-v1",
            ),
            intensity_detail=IntensityResult(
                intensity=intensity.score,
                level=intensity.level,
                evidence=intensity.evidence,
                model_name="rule-intensity-v1",
            ),
        )
        logger.info(
            "感知结果: emotion=%s(%.2f) intensity=%s(%.2f) event=%s(%.2f) urgency=%s",
            result.emotion,
            result.emotion_score,
            result.emotion_intensity,
            result.intensity,
            result.event,
            result.event_score,
            result.urgency.value,
        )
        return result
