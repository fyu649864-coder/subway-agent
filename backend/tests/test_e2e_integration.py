"""端到端整合测试：通过 /api/v1/chat 验证完整处理链路。

覆盖：正常对话、不同情绪、不同地铁事件、紧急情况、证据不足、
以及情绪模型加载失败时的明确报错（不静默降级）。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_conversation_service
from app.main import app
from app.models import UrgencyLevel
from app.perception import emotion_classifier as ec
from app.services.conversation_service import ConversationService
from app.services.perception_service import PerceptionService

VALID_EMOTIONS = {"平静", "焦虑", "愤怒", "悲伤", "恐惧", "烦躁", "无助"}
VALID_INTENSITIES = {"低", "中", "高"}
VALID_STRATEGIES = {
    "紧急情况提示", "情绪安抚", "行动建议", "安抚与建议结合", "保守常规回复",
}


def _post(client: TestClient, text: str, session: str | None = None):
    payload = {"text": text}
    if session:
        payload["session_id"] = session
    return client.post("/api/v1/chat", json=payload)


def _assert_common(data: dict) -> None:
    assert data["emotion"] in VALID_EMOTIONS
    assert 0.0 <= data["emotion_score"] <= 1.0
    assert data["emotion_confidence"] == data["emotion_score"]
    assert data["emotion_intensity"] in VALID_INTENSITIES
    assert isinstance(data["intensity_evidence"], list)
    assert data["event"]
    assert 0.0 <= data["event_confidence"] <= 1.0
    assert isinstance(data["event_evidence"], list)
    assert data["urgency"] in {"low", "medium", "high"}
    assert data["decision"]["strategy"] in VALID_STRATEGIES
    assert isinstance(data["decision"]["actions"], list) and data["decision"]["actions"]
    assert data["reply"] and len(data["reply"]) > 10
    assert data["template_id"].startswith("tpl_")
    # 内嵌完整链路一致
    assert data["perception"]["emotion"] == data["emotion"]
    assert data["perception"]["event"] == data["event"]


def test_e2e_normal_dialogue(client: TestClient) -> None:
    resp = _post(client, "地铁怎么还不来，我马上要迟到了！")
    assert resp.status_code == 200
    data = resp.json()
    _assert_common(data)
    assert data["event"] in {"地铁延误", "赶时间"}


def test_e2e_different_events(client: TestClient) -> None:
    cases = [
        ("这节车厢太挤了完全上不去", {"车厢拥挤"}),
        ("末班车都走了我没赶上回不去了", {"末班车与错过列车"}),
        ("换乘绕了好大一圈找不到出口", {"换乘困难"}),
        ("我要赶考试快来不及了", {"赶时间"}),
    ]
    for text, expected in cases:
        data = _post(client, text).json()
        _assert_common(data)
        assert data["event"] in expected or data["event"] == "其他/未知事件"


def test_e2e_emergency_escalation(client: TestClient) -> None:
    resp = _post(client, "前面车厢好像冒烟了，是不是出事故了，我好害怕！")
    assert resp.status_code == 200
    data = resp.json()
    _assert_common(data)
    assert data["event"] == "突发事件"
    assert data["urgency"] == "high"
    assert data["decision"]["strategy"] == "紧急情况提示"
    assert data["decision"]["needs_escalation"] is True
    assert "安全" in data["reply"] or "工作人员" in data["reply"]


def test_e2e_evidence_insufficient_conservative(client: TestClient) -> None:
    resp = _post(client, "今天天气不错")
    assert resp.status_code == 200
    data = resp.json()
    _assert_common(data)
    assert data["event"] == "其他/未知事件"
    assert data["decision"]["strategy"] == "保守常规回复"
    # 不编造具体线路/预计时间
    assert "1号线" not in data["reply"] and "2号线" not in data["reply"]
    assert "预计" not in data["reply"]


def test_e2e_multi_turn_session(client: TestClient) -> None:
    s = "sess-e2e-1"
    r1 = _post(client, "地铁延误了我要迟到了", session=s)
    r2 = _post(client, "这车厢也太挤了吧", session=s)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["reply"] and r2.json()["reply"]


def test_e2e_empty_text_rejected(client: TestClient) -> None:
    resp = client.post("/api/v1/chat", json={"text": ""})
    assert resp.status_code == 422


def test_e2e_model_load_failure_is_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    """模型不可用时，端到端链路应明确报错（HTTP 500），不得返回固定/随机情绪。"""
    broken = ec.EmotionClassifier(model_path="/nonexistent/path/that/does/not/exist")
    svc = ConversationService(perception=PerceptionService(emotion_classifier=broken))

    def _override():
        return svc

    app.dependency_overrides[get_conversation_service] = _override
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/api/v1/chat", json={"text": "测试"})
        assert resp.status_code == 500
    finally:
        app.dependency_overrides.pop(get_conversation_service, None)


def test_e2e_full_python_chain() -> None:
    """不经 HTTP，直接验证 Python 服务链各字段正确传递。"""
    svc = ConversationService()
    result = svc.handle(__import__("app.models", fromlist=["UserInput"]).UserInput(
        text="这服务态度太差了我要投诉，气死我了！"
    ))
    assert result.emotion in VALID_EMOTIONS
    assert result.event == "服务投诉"
    assert result.decision.needs_escalation is True
    assert result.decision.urgency == UrgencyLevel.high
    assert result.reply
    assert result.emotion_confidence == result.emotion_score
    assert result.event_confidence == result.event_score
