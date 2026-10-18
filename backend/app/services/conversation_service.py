"""对话服务：编排感知 -> 决策 -> 回复的完整链路。

API 层只依赖本服务，不直接接触感知/决策/回复的具体实现，
从而保证各层低耦合、可独立替换。
"""

from __future__ import annotations

from datetime import datetime

from app.core.config import settings
from app.core.logging import get_logger
from app.models import ConversationState, ResponseResult, UserInput
from app.services.decision_service import DecisionService
from app.services.perception_service import PerceptionService
from app.services.response_service import ResponseService

logger = get_logger(__name__)


class ConversationService:
    """对话管理服务，维护内存态会话状态并编排处理链路。"""

    def __init__(
        self,
        perception: PerceptionService | None = None,
        decision: DecisionService | None = None,
        response: ResponseService | None = None,
    ) -> None:
        self.perception = perception or PerceptionService()
        self.decision = decision or DecisionService()
        self.response = response or ResponseService()
        # 内存态会话表：session_id -> ConversationState
        self._states: dict[str, ConversationState] = {}

    def _get_state(self, session_id: str) -> ConversationState:
        state = self._states.get(session_id)
        if state is None:
            state = ConversationState(session_id=session_id)
            self._states[session_id] = state
        return state

    def _get_state(self, session_id: str) -> ConversationState:
        state = self._states.get(session_id)
        if state is None:
            state = ConversationState(session_id=session_id)
            self._states[session_id] = state
        return state

    def handle(self, user_input: UserInput) -> ResponseResult:
        """处理一次用户输入，返回最终回复结果。"""

        # 未提供 session_id 时按单次请求处理：使用临时状态，不与其它请求共享历史，
        # 避免匿名请求互相污染上下文。
        has_session = bool(user_input.session_id)
        session_id = user_input.session_id or "default"
        request_time = user_input.request_time or datetime.now()

        # 记录请求（不记录用户敏感标识）
        logger.info("收到请求: session=%s time=%s text_len=%d", session_id, request_time.isoformat(), len(user_input.text))

        state = self._get_state(session_id) if has_session else ConversationState(session_id=session_id)
        # 最近几轮的用户原文拼接为上下文，供事件/强度识别与回复层使用
        history_texts = [h.text for h in state.history if getattr(h, "text", None)]
        context_text = " ".join(history_texts[-5:]) if history_texts else None
        try:
            # 1. 感知层
            perception = self.perception.analyze(user_input.text, context_text=context_text)
            # 2. 决策层
            decision = self.decision.decide(perception)
            # 3. 回复层（传入最近历史与会话 id，支持多轮承接与游戏状态）
            response = self.response.generate(
                perception, decision, history=list(state.history), session_id=session_id if has_session else None,
            )
        except Exception:
            logger.exception("处理链路发生异常: session=%s", session_id)
            raise

        # 更新会话状态
        state.turn_count += 1
        state.updated_at = datetime.now()
        state.current_urgency = perception.urgency
        state.history.append(perception)
        # 控制历史长度
        if len(state.history) > settings.max_history:
            state.history = state.history[-settings.max_history :]

        logger.info("完成响应: session=%s turn=%d", session_id, state.turn_count)
        return response
