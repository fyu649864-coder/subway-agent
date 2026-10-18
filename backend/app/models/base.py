"""模型基类与通用配置。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    """所有业务数据模型的基类。

    - ``extra="allow"``：允许传入未声明字段，保证结构可向前扩展。
    - ``populate_by_name=True``：允许同时使用字段名与别名赋值。
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True, use_enum_values=False)
