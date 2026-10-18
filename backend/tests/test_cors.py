"""CORS 配置测试：仅允许配置的前端来源。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

ALLOWED = "http://127.0.0.1:8080"
DISALLOWED = "http://evil.example.com"


def test_preflight_allowed_origin() -> None:
    client = TestClient(app)
    resp = client.options(
        "/api/v1/chat",
        headers={
            "Origin": ALLOWED,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == ALLOWED


def test_preflight_disallowed_origin() -> None:
    client = TestClient(app)
    resp = client.options(
        "/api/v1/chat",
        headers={
            "Origin": DISALLOWED,
            "Access-Control-Request-Method": "POST",
        },
    )
    # 非允许来源：不回显该来源，即不放行跨域
    acao = resp.headers.get("access-control-allow-origin")
    assert acao != DISALLOWED


def test_health_still_works() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
