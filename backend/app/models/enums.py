"""系统内部使用的枚举类型。"""

from __future__ import annotations

from enum import Enum


class UrgencyLevel(str, Enum):
    """紧急程度分级。

    作为闭合枚举使用；情绪类别、事件类别等开放式集合则使用普通字符串，
    以避免把类别写死、阻碍后续扩展。
    """

    low = "low"
    medium = "medium"
    high = "high"
