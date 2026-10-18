"""/api/v1/chat 与 /health 接口测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_returns_structured_result(client: TestClient) -> None:
    payload = {"text": "地铁怎么还不来，我马上要迟到了！"}
    resp = client.post("/api/v1/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    # 结构字段齐全（不假定模型一定预测出某种情绪）
    assert data["text"] == payload["text"]
    assert data["emotion"] in {"平静", "焦虑", "愤怒", "悲伤", "恐惧", "烦躁", "无助"}
    assert 0.0 <= data["emotion_score"] <= 1.0
    assert data["emotion_confidence"] == data["emotion_score"]
    assert data["emotion_intensity"] in {"低", "中", "高"}
    assert isinstance(data["intensity_evidence"], list)
    assert data["event"]  # 非空
    assert 0.0 <= data["event_confidence"] <= 1.0
    assert isinstance(data["event_evidence"], list)
    assert data["urgency"] in {"low", "medium", "high"}
    # 完整链路字段存在
    assert data["reply"]
    assert data["template_id"]
    assert data["perception"]["emotion"] == data["emotion"]
    assert data["decision"]["strategy"]


def test_chat_with_session_id(client: TestClient) -> None:
    payload = {"text": "这班车人太多了挤不上去", "session_id": "s-001"}
    resp = client.post("/api/v1/chat", json=payload)
    assert resp.status_code == 200
    assert resp.json()["reply"]


def test_chat_validation_error_on_empty_text(client: TestClient) -> None:
    resp = client.post("/api/v1/chat", json={"text": ""})
    assert resp.status_code == 422


def test_chat_validation_error_on_missing_text(client: TestClient) -> None:
    resp = client.post("/api/v1/chat", json={})
    assert resp.status_code == 422
