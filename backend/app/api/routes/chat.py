"""对话路由：POST /api/v1/chat。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_conversation_service
from app.models import ResponseResult, UserInput
from app.services.conversation_service import ConversationService

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ResponseResult, summary="对话接口")
def chat(
    payload: UserInput,
    service: ConversationService = Depends(get_conversation_service),
) -> ResponseResult:
    """接收用户文本，返回感知/决策/回复的结构化结果。

    本阶段为 Mock 数据，后续可在 Service 内部无缝替换为真实模型。
    """

    return service.handle(payload)
