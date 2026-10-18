"""回复服务。

根据决策结果（strategy）从本地模板库生成自然、共情且实用的中文回复。
纯本地模板与规则，不调用云端 API；信息不足时使用保守模板，不编造实时路况或路线。
"""

from __future__ import annotations

import random

from app.core.logging import get_logger
from app.models import DecisionResult, PerceptionResult, ResponseResult
from app.decision.rules import STRATEGY_EMERGENCY
from app.response import templates
from app.response import llm_generator

logger = get_logger(__name__)


class ResponseService:
    """回复服务：(perception, decision) -> ResponseResult。"""

    def __init__(self, seed: int | None = None) -> None:
        # 可选固定随机种子，便于测试复现
        self._rng = random.Random(seed)

    def generate(
        self,
        perception: PerceptionResult,
        decision: DecisionResult,
        history: list[PerceptionResult] | None = None,
        session_id: str | None = None,
    ) -> ResponseResult:
        """根据感知、决策与最近历史生成最终回复。

        ``history`` 用于：(1) 避免与上一轮回复完全重复；(2) 同事件延续时给出简短承接。
        """

        template_id, reply = templates.render(
            decision.strategy, perception=perception, history=history, rng=self._rng,
        )

        # 回复来源标签（写日志与 extras，便于排查实际走了哪条路径）：
        #   llm_local          —— 本地大模型生成成功并采用
        #   template_fallback  —— 尝试了大模型但失败/超时/不可用，回退模板
        #   emergency_template —— 紧急安全策略，直接使用模板（规则优先，不调用大模型）
        reply_source = "template_fallback"
        llm_used = False

        if decision.strategy == STRATEGY_EMERGENCY:
            # 紧急策略：安全优先，不调用大模型
            reply_source = "emergency_template"
            logger.info("[回复] 紧急策略，直接使用安全模板 | source=emergency_template")
        else:
            llm_reply = llm_generator.generate_reply(perception, decision, history=history, session_id=session_id)
            if llm_reply:
                reply = llm_reply
                template_id = "llm_local"
                llm_used = True
                reply_source = "llm_local"
            else:
                # generate_reply 内部已记录具体失败原因，这里只记录最终来源
                logger.info("[回复] 大模型未产出，使用模板回退 | source=template_fallback")

        # 回填到 perception.extras，供下一轮去重使用（不影响对外字段）
        perception.extras["last_reply"] = reply

        result = ResponseResult(
            reply=reply,
            template_id=template_id,
            text=perception.text,
            emotion=perception.emotion,
            emotion_score=perception.emotion_score,
            emotion_confidence=perception.emotion_confidence,
            intensity=perception.intensity,
            emotion_intensity=perception.emotion_intensity,
            intensity_evidence=perception.intensity_evidence,
            event=perception.event,
            event_score=perception.event_score,
            event_confidence=perception.event_confidence,
            event_evidence=perception.event_evidence,
            urgency=decision.urgency,
            perception=perception,
            decision=decision,
        )
        result.extras["llm_used"] = llm_used
        result.extras["reply_source"] = reply_source
        logger.info(
            "[回复] 最终结果 source=%s strategy=%s template_id=%s reply_len=%d",
            reply_source,
            decision.strategy,
            result.template_id,
            len(result.reply),
        )
        return result
