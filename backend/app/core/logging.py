"""基础日志系统。

提供统一的日志格式与 ``get_logger`` 工厂函数。
注意：日志中不记录敏感个人信息（如用户 ID、手机号等）；
请求文本属于业务必要内容，会被记录用于调试感知结果。
"""

from __future__ import annotations

import logging
import sys

from app.core.config import settings

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(level: str | None = None) -> None:
    """初始化根日志配置。重复调用安全（仅首次生效）。"""

    global _configured
    if _configured:
        return

    log_level = (level or settings.log_level).upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    root = logging.getLogger()
    # 清空默认 handler，避免重复输出
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # 降低第三方库日志噪音
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """获取一个已配置好的命名 logger。"""

    if not _configured:
        setup_logging()
    return logging.getLogger(name)
