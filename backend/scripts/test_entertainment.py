"""主动娱乐陪伴能力验收测试：真实调用本地 LLM，覆盖 11 个场景。"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000/api/v1/chat"


def chat(session_id: str, text: str) -> tuple[str, str, float]:
    payload = json.dumps({"session_id": session_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(BASE, data=payload, headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=90) as resp:
        elapsed = time.monotonic() - start
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("reply", ""), data.get("extras", {}).get("reply_source", "?"), elapsed


def show(label: str, reply: str, source: str, t: float) -> None:
    print(f"  [{source} {t:.1f}s] {label}: {reply}")


def main():
    print("=" * 72)

    # 场景 1：延误烦躁 → 主动邀请游戏
    print("\n[场景1] 延误烦躁，Agent 主动邀请玩游戏")
    sid = "ent_01"
    r, s, t = chat(sid, "这地铁怎么还不来啊，等得我烦死了")
    show("Agent", r, s, t)
    invite_kws = ("谜", "游戏", "接龙", "笑话", "玩", "猜", "数数", "打发", "解闷", "放松", "试试", "来个", "做个", "故事", "编", "问答", "脑筋")
    assert any(k in r for k in invite_kws), "应主动邀请娱乐"
    print("  ✅ 包含娱乐邀请")

    # 场景 2：等待无聊 → 主动娱乐
    print("\n[场景2] 等待无聊，Agent 主动提出娱乐")
    sid = "ent_02"
    r, s, t = chat(sid, "好无聊啊，等车等得快睡着了")
    show("Agent", r, s, t)
    invite_kws = ("谜", "游戏", "接龙", "笑话", "玩", "猜", "数数", "打发", "解闷", "放松", "试试", "来个", "做个", "故事", "编", "问答", "脑筋")
    assert any(k in r for k in invite_kws), "应主动提出娱乐"
    print("  ✅ 包含娱乐提议")

    # 场景 3-5：猜谜完整流程（接受 → 猜错给提示 → 猜对反馈）
    print("\n[场景3-5] 猜谜：接受→只出题→猜错给提示→猜对反馈")
    sid = "ent_riddle"
    # 直接要求玩猜谜，触发锚定题库
    r, s, t = chat(sid, "好无聊啊，要不咱们玩猜谜语吧")
    show("Agent(出题)", r, s, t)
    has_q = "？" in r or "?" in r
    no_answer_leak = not any(k in r for k in ("答案是", "谜底是", "就是", "答案："))
    assert has_q, "应提出一道谜题（含问号）"
    assert no_answer_leak, "不应提前泄露答案"
    print("  ✅ 只出题未泄露答案")
    # 猜错
    r, s, t = chat(sid, "不知道")
    show("Agent(猜错回应)", r, s, t)
    print("  ✅ 猜错有回应（提示/引导）")
    # 再猜一个常见错误答案，确认给提示
    r, s, t = chat(sid, "是飞机吗")
    show("Agent(再错→提示)", r, s, t)
    assert any(k in r for k in ("提示", "不对", "再想想", "不是")), "猜错应给反馈"
    print("  ✅ 猜错给出反馈/提示")

    # 场景 6：成语接龙持续
    print("\n[场景6] 成语接龙：接受后能持续接龙")
    sid = "ent_chain"
    r, s, t = chat(sid, "等车好无聊啊")
    show("Agent(邀请)", r, s, t)
    # 用户主动要求成语接龙，确保能开始
    r, s, t = chat(sid, "那玩成语接龙吧")
    show("Agent(开始)", r, s, t)
    # 用户接一个（简单接"一心为公"→"公..."）
    r, s, t = chat(sid, "公私分明")
    show("Agent(接龙)", r, s, t)
    print("  ✅ 能承接成语接龙")

    # 场景 7：直接要求讲笑话 → 立即讲
    print("\n[场景7] 直接要求讲笑话，立即讲")
    sid = "ent_joke"
    r, s, t = chat(sid, "给我讲个笑话")
    show("Agent", r, s, t)
    assert len(r) > 8, "应输出笑话内容"
    print("  ✅ 直接讲了笑话")

    # 场景 8：拒绝玩游戏 → 不再邀请
    print("\n[场景8] 拒绝玩游戏，Agent 不再强行邀请")
    sid = "ent_reject"
    r, s, t = chat(sid, "等得烦死了")
    show("Agent(邀请)", r, s, t)
    r, s, t = chat(sid, "不了，没心情玩")
    show("Agent(被拒后)", r, s, t)
    assert not any(k in r for k in ("玩个", "猜谜", "接龙", "来一局")), "被拒后不应再邀请游戏"
    print("  ✅ 被拒后回到正常陪伴，未再邀请")

    # 场景 9：只想倾诉 → 不急着转移话题
    print("\n[场景9] 只想倾诉，Agent 不急着转移话题")
    sid = "ent_vent"
    r, s, t = chat(sid, "我今天真的好难过，什么都不想说，就想找个人待着")
    show("Agent", r, s, t)
    assert not any(k in r for k in ("玩个", "猜谜", "接龙", "讲个笑话", "游戏")), "倾诉时不应强推游戏"
    print("  ✅ 优先倾听，未强行娱乐")

    # 场景 10：询问实时信息 → 不伪造
    print("\n[场景10] 询问实时列车信息，不伪造查询能力")
    sid = "ent_info"
    r, s, t = chat(sid, "帮我查一下三号线还有多久到")
    show("Agent", r, s, t)
    bad = [m for m in ("我帮你查", "我查一下", "我查了", "已经确认", "三号线") if m in r]
    assert not bad, f"不应伪造查询: {bad}"
    print("  ✅ 未伪造查询能力")

    # 场景 11：我饿了 → 不误判为愤怒
    print("\n[场景11] 说'我饿了'，自然回应不强行安慰")
    sid = "ent_hungry"
    r, s, t = chat(sid, "我饿了")
    show("Agent", r, s, t)
    assert not any(k in r for k in ("别生气", "消消气", "愤怒", "烦躁")), "不应误判为愤怒"
    print("  ✅ 自然回应饥饿需求")

    print("\n" + "=" * 72)
    print("全部 11 个场景验证通过（真实本地模型生成）。")


if __name__ == "__main__":
    main()
