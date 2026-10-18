"""pytest 配置与通用 fixtures。"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保 backend 目录在 sys.path 中，使 `app` 包可被导入
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def _disable_llm_for_deterministic_tests(monkeypatch):
    """默认禁用本地大模型，保证模板层测试确定性、可复现。

    LLM 接入行为由 test_llm_integration.py 通过 mock 单独覆盖。
    """
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_enabled", False, raising=False)
    from app.response import llm_generator
    llm_generator.reset_for_tests()
    yield


@pytest.fixture
def client() -> TestClient:
    """提供 FastAPI 测试客户端。"""
    return TestClient(app)
