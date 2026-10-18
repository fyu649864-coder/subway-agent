"""主动娱乐陪伴能力回归测试（不加载真实模型，覆盖游戏状态与意图判定）。"""

from __future__ import annotations

from app.response import games
from app.response import llm_generator


def setup_function(_):
    games._states.clear()
    llm_generator.reset_for_tests()


# ---------- games.py 单元测试 ----------

def test_riddle_bank_has_questions_and_answers():
    from app.response.games import _RIDDLE_BANK
    assert len(_RIDDLE_BANK) >= 10
    for r in _RIDDLE_BANK:
        assert r.q and r.a
        assert "？" in r.q or "?" in r.q
        assert r.hints


def test_start_and_guess_correct():
    sid = "t1"
    r = games.start_riddle(sid)
    assert r is not None
    st = games.active_game(sid)
    assert st and st["game"] == "riddle"
    res = games.submit_guess(sid, r.a)
    assert res["correct"] is True
    assert res["done"] is True
    # 答对后状态清除
    assert games.active_game(sid) is None


def test_guess_wrong_then_hint_then_correct():
    sid = "t2"
    r = games.start_riddle(sid)
    res = games.submit_guess(sid, "乱猜的")
    assert res["correct"] is False
    assert res["hint"] == r.hints[0]
    assert res["done"] is False
    # 第二次错给第二条提示
    res2 = games.submit_guess(sid, "还是不对")
    assert res2["hint"] == r.hints[1]
    # 猜对
    res3 = games.submit_guess(sid, r.a)
    assert res3["correct"] is True


def test_hints_exhausted_ends_game():
    sid = "t3"
    r = games.start_riddle(sid)
    for _ in range(len(r.hints) + 1):
        games.submit_guess(sid, "不对")
    assert games.active_game(sid) is None


def test_rejection_clears_game():
    sid = "t4"
    games.start_riddle(sid)
    assert games.active_game(sid)
    games.end_game(sid)
    assert games.active_game(sid) is None


def test_is_acceptance_and_rejection():
    assert games.is_acceptance("好啊")
    assert games.is_acceptance("来吧")
    assert not games.is_acceptance("我今天好难过啊")  # 太长不算简短接受
    assert games.is_rejection("不玩了")
    assert games.is_rejection("没心情玩")
    assert not games.is_rejection("好无聊啊")  # 无聊不是拒绝


def test_compose_riddle_start_contains_question():
    from app.response.games import _RIDDLE_BANK, compose_riddle_start
    r = _RIDDLE_BANK[0]
    out = compose_riddle_start(r)
    assert r.q in out
    assert r.a not in out  # 不泄露答案


# ---------- _prepare_game_context 逻辑测试 ----------

def _hist_with_last_reply(text: str):
    """构造一条带 last_reply 的假历史，供邀请检测使用。"""
    from app.models import PerceptionResult, UrgencyLevel
    p = PerceptionResult(
        text="x", emotion="平静", emotion_score=0.5, emotion_confidence=0.5,
        intensity=0.5, emotion_intensity="低", intensity_evidence=[],
        event="其他/未知事件", event_score=0.5, event_confidence=0.5,
        event_evidence=[], urgency=UrgencyLevel.low,
    )
    p.extras["last_reply"] = text
    return [p]


def test_explicit_riddle_starts_game():
    sid = "t5"
    ctx = llm_generator._prepare_game_context(sid, "玩猜谜语吧", [])
    assert ctx is not None
    assert "direct_reply" in ctx
    assert games.active_game(sid)  # 状态已建立


def test_acceptance_after_invitation_starts_riddle():
    sid = "t6"
    hist = _hist_with_last_reply("要不咱们猜个谜语？")
    ctx = llm_generator._prepare_game_context(sid, "好啊", hist)
    assert ctx is not None and "direct_reply" in ctx


def test_rejection_returns_none_and_clears():
    sid = "t7"
    games.start_riddle(sid)
    ctx = llm_generator._prepare_game_context(sid, "不玩了", [])
    assert ctx is None
    assert games.active_game(sid) is None


def test_active_riddle_guess_gets_instruction():
    sid = "t8"
    r = games.start_riddle(sid)
    ctx = llm_generator._prepare_game_context(sid, "我猜是空气", [])
    assert ctx is not None
    assert "instruction" in ctx
    assert r.q in ctx["instruction"]  # 题目应出现在指引中


def test_venting_text_does_not_start_game():
    # 倾诉/悲伤文本不应触发游戏
    sid = "t9"
    ctx = llm_generator._prepare_game_context(sid, "我今天好难过，就想找人说说话", [])
    assert ctx is None


def test_no_session_id_returns_none():
    ctx = llm_generator._prepare_game_context(None, "玩猜谜", [])
    assert ctx is None


# ---------- 成语接龙状态管理测试 ----------

def test_chain_start_sets_state_and_required_char():
    sid = "c1"
    init = games.start_chain(sid)
    assert init is not None
    assert len(init["idiom"]) == 4
    assert init["required_char"] == init["idiom"][-1]
    st = games.active_game(sid)
    assert st["game"] == "chain"
    assert st["required_char"] == init["idiom"][-1]


def test_chain_correct_move_advances_state():
    sid = "c2"
    init = games.start_chain(sid)
    # 手动构造一个一定合规的接龙：找词库里以 required_char 开头的成语
    req = init["required_char"]
    user_idiom = next((d for d in games._IDIOM_BANK if d[0] == req), None)
    assert user_idiom, f"测试词库应覆盖起始字 {req}"
    res = games.submit_chain(sid, user_idiom)
    assert res["valid"] is True
    assert res["topic_switch"] is False
    assert res["ai_idiom"][0] == user_idiom[-1], "AI 成语首字应等于用户成语尾字"
    assert res["required_char"] == res["ai_idiom"][-1]
    # 状态已推进
    assert games.active_game(sid)["required_char"] == res["required_char"]


def test_chain_wrong_first_char_rejected_no_advance():
    sid = "c3"
    init = games.start_chain(sid)
    req = init["required_char"]
    # 用一个首字不对的"伪成语"
    wrong = next(ch for ch in "一二三四五六七八九十" if ch != req) + "紫千红"
    before = games.active_game(sid)["required_char"]
    res = games.submit_chain(sid, wrong)
    assert res["valid"] is False
    assert "应该是" in res["reason"] or req in res["reason"]
    assert games.active_game(sid)["required_char"] == before  # 状态未推进


def test_chain_non_idiom_sentence_topic_switches():
    sid = "c4"
    games.start_chain(sid)
    res = games.submit_chain(sid, "我今天有点累了不想玩了")
    assert res["topic_switch"] is True
    assert games.active_game(sid) is None  # 已退出


def test_chain_short_non_idiom_rejected():
    sid = "c5"
    init = games.start_chain(sid)
    res = games.submit_chain(sid, "好的")
    assert res["valid"] is False
    assert res["topic_switch"] is False
    assert "不像" in res["reason"]
    assert games.active_game(sid) is not None  # 未退出，允许重试


def test_chain_reply_composition_mentions_required_char():
    from app.response.games import compose_chain_start, compose_chain_reply
    start = compose_chain_start("一心一意", "意")
    assert "一心一意" in start and "意" in start
    ok = compose_chain_reply({"valid": True, "user_idiom": "意气风发", "ai_idiom": "发扬光大", "required_char": "大"})
    assert "意气风发" in ok and "发扬光大" in ok and "大" in ok
    retry = compose_chain_reply({"valid": False, "reason": "第一个字应该是「意」哦", "required_char": "意"})
    assert "意" in retry


def test_chain_idiom_bank_connectivity():
    # 词库中随机抽取的起始成语，其尾字应能在词库中找到下一个（保证 AI 能接）
    covered = set()
    for d in games._IDIOM_BANK:
        covered.add(d[0])
    misses = [d for d in games._IDIOM_BANK if d[-1] not in covered]
    # 允许极少数尾字无法接续（会触发 no_more 收尾），但不应超过 20%
    assert len(misses) < len(games._IDIOM_BANK) * 0.2, f"词库接续覆盖率过低: {misses}"


def test_prepare_context_starts_chain():
    sid = "c6"
    ctx = llm_generator._prepare_game_context(sid, "我们玩成语接龙吧", [])
    assert ctx is not None and "direct_reply" in ctx
    assert games.active_game(sid) and games.active_game(sid)["game"] == "chain"


def test_prepare_context_chain_move_returns_direct_reply():
    sid = "c7"
    games.start_chain(sid)
    req = games.active_game(sid)["required_char"]
    user_idiom = next((d for d in games._IDIOM_BANK if d[0] == req), None)
    ctx = llm_generator._prepare_game_context(sid, user_idiom, [])
    assert ctx is not None and "direct_reply" in ctx
    assert user_idiom in ctx["direct_reply"]


def test_idiom_not_misdetected_as_acceptance():
    # 回归：进行中接龙时，"来日方长"含"来"字但不应被当作接受邀请而重开游戏
    sid = "c8"
    games.start_chain(sid)
    games.active_game(sid)["required_char"] = "来"
    games.active_game(sid)["last_idiom"] = "手到擒来"
    ctx = llm_generator._prepare_game_context(sid, "来日方长", _hist_with_last_reply("我们玩成语接龙吧"))
    assert "direct_reply" in ctx
    # 应是接龙回合回复（含"接"），而非重开（含"规则"）
    assert "规则" not in ctx["direct_reply"] or "接" in ctx["direct_reply"]
    assert games.active_game(sid)["game"] == "chain"  # 仍在游戏中
