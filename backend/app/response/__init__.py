"""回复执行层包。

后续阶段将在 :mod:`app.response.templates` 中管理回复模板，
并考虑加入小模型进行润色。当前阶段仅定义接口契约。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.decision import DecisionResult
from app.models.perception import PerceptionResult
from app.models.response import ResponseResult


class BaseResponse(ABC):
    """回复层接口：输入感知与决策结果，输出最终回复。"""

    @abstractmethod
    def generate(self, perception: PerceptionResult, decision: DecisionResult) -> ResponseResult:
        """生成面向用户的最终回复。"""
        raise NotImplementedError
