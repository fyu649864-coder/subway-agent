"""API 依赖注入。"""

from __future__ import annotations

from app.services.conversation_service import ConversationService

# 进程级单例：对话服务（含其依赖的感知/决策/回复服务）
_conversation_service: ConversationService | None = None


def get_conversation_service() -> ConversationService:
    """获取对话服务单例，供路由注入使用。"""

    global _conversation_service
    if _conversation_service is None:
        _conversation_service = ConversationService()
    return _conversation_service
