"""用户输入模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import Field

from app.models.base import BaseSchema


class UserInput(BaseSchema):
    """终端用户的一次输入。

    Attributes:
        text: 用户输入的原始文本，必填且非空。
        session_id: 会话 ID，用于多轮对话关联；本阶段可为空。
        request_time: 客户端请求时间，缺省由服务端填充。
        metadata: 扩展元数据，预留字段，不参与核心逻辑。
    """

    text: str = Field(..., min_length=1, description="用户输入的原始文本")
    session_id: Optional[str] = Field(default=None, description="会话 ID，用于多轮对话")
    request_time: Optional[datetime] = Field(default=None, description="请求时间，缺省由服务端填充")
    metadata: dict[str, Any] = Field(default_factory=dict, description="扩展元数据，预留字段")
