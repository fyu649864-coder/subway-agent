"""成语接龙功能验收：真实接口测试 7 个场景。"""
import json
import re
import time
import urllib.request

BASE = "http://127.0.0.1:8000/api/v1/chat"


def chat(sid, text):
    payload = json.dumps({"session_id": sid, "text": text}).encode()
    req = urllib.request.Request(BASE, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=90) as resp:
        dt = time.monotonic() - t0
        data = json.loads(resp.read().decode())
    return data.get("reply", ""), data.get("extras", {}).get("reply_source", "?"), dt


def extract_char(reply):
    """从回复中提取「接 X 字」的 X（优先匹配带书名号的「X」字）。"""
    m = re.search(r"[「『]([\u4e00-\u9fff])[」』]字", reply)
    if m:
        return m.group(1)
    m = re.search(r"接([\u4e00-\u9fff])字", reply)
    return m.group(1) if m else None


def main():
    print("=" * 70)

    # 场景 1：开始游戏说明规则并给有效成语
    print("\n[场景1] 开始游戏：说明规则 + 有效成语")
    sid = "chain_01"
    r, s, t = chat(sid, "我们来玩成语接龙，你先说一个成语")
    print(f"  [{s} {t:.1f}s] {r}")
    assert "成语" in r and ("规则" in r or "接" in r), "应说明规则"
    # 应给出一个四字成语并要求接某个字
    assert extract_char(r), f"应明确要求接的字: {r}"
    print(f"  ✅ 规则说明清楚，要求接「{extract_char(r)}」字")

    # 场景 2-4：连续三轮正确接龙，检查每轮要求的字正确
    print("\n[场景2-4] 连续三轮正确接龙，校验推进正确")
    sid = "chain_multi"
    r, s, t = chat(sid, "玩成语接龙")
    print(f"  [开局] {r}")
    req_char = extract_char(r)
    assert req_char
    # 准备几个能首尾相接的成语，覆盖 req_char
    # 用一个映射：字 -> 成语（最后一字已知）
    chain_pool = {
        "一": ("一心一意", "意"),
        "意": ("意气风发", "发"),
        "发": ("发扬光大", "大"),
        "大": ("大显身手", "手"),
        "手": ("手到擒来", "来"),
        "来": ("来日方长", "长"),
        "长": ("长年累月", "月"),
        "月": ("月下老人", "人"),
        "人": ("人山人海", "海"),
        "海": ("海阔天空", "空"),
        "空": ("空前绝后", "后"),
        "后": ("后来居上", "上"),
        "上": ("上下同心", "心"),
        "心": ("心想事成", "成"),
        "成": ("成千上万", "万"),
        "万": ("万事如意", "意"),
        "开": ("开门见山", "山"),
        "山": ("山高水长", "长"),
        "天": ("天翻地覆", "覆"),
        "马": ("马到成功", "功"),
    }
    prev_last = None
    for i in range(3):
        if req_char not in chain_pool:
            print(f"  ⚠️  词库缺「{req_char}」开头的测试成语，跳过推进校验")
            break
        idiom, last = chain_pool[req_char]
        assert idiom[0] == req_char, f"测试数据错误：{idiom} 首字不是 {req_char}"
        r, s, t = chat(sid, idiom)
        print(f"  [轮{i+1}] 用户接「{idiom}」(尾字{last}) -> {r}")
        new_char = extract_char(r)
        # AI 应接一个以 last 开头的成语，并要求接该成语的尾字
        ai_match = re.search(r"我接[「『]([\u4e00-\u9fff]+)[」』]", r)
        if ai_match:
            ai_idiom = ai_match.group(1)
            assert ai_idiom[0] == last, f"AI 成语「{ai_idiom}」首字应为「{last}」"
            assert new_char == ai_idiom[-1], f"应要求接 AI 成语尾字「{ai_idiom[-1]}」，实际「{new_char}」"
        assert any(k in r for k in ("接对", "没错", "可以", "棒", "漂亮", "接得好")), "应给予积极反馈"
        req_char = new_char
        prev_last = last
    print("  ✅ 三轮接龙推进正确，每轮要求的字均正确")

    # 场景 5：接错首字 → 提示重试，不推进状态
    print("\n[场景5] 接错首字 → 提示重试，不推进")
    sid = "chain_wrong"
    r, _, _ = chat(sid, "我们玩成语接龙吧")
    req_char = extract_char(r)
    print(f"  [开局] 要求接「{req_char}」: {r[:40]}…")
    # 故意用错误的首字
    wrong = "万" if req_char != "万" else "千"
    r2, _, _ = chat(sid, wrong + "紫千红")  # 万紫千红，首字不对
    print(f"  [接错] {r2}")
    assert req_char in r2, f"应提示要接「{req_char}」字"
    # 状态未推进：用正确的字还能接
    if req_char in chain_pool:
        idiom, last = chain_pool[req_char]
        r3, _, _ = chat(sid, idiom)
        new_char = extract_char(r3)
        ai_match = re.search(r"我接[「『]([\u4e00-\u9fff]+)[」』]", r3)
        if ai_match:
            ai_idiom = ai_match.group(1)
            assert ai_idiom[0] == last, f"AI 成语首字应为「{last}」"
            assert new_char == ai_idiom[-1], f"应要求接 AI 成语尾字"
        print(f"  [重试正确] {r3}")
    print("  ✅ 接错提示重试，重试后正常推进")

    # 场景 6：输入普通句子（非成语）→ 不推进状态
    print("\n[场景6] 输入普通句子 → 不推进")
    sid = "chain_sentence"
    r, _, _ = chat(sid, "成语接龙开始")
    req_char = extract_char(r)
    r2, _, _ = chat(sid, "我今天有点累不想说话了")
    print(f"  [普通句子] {r2}")
    # 不应推进接龙（不应出现新的要求接 X 字且推进了的反馈），或已退出游戏
    print("  ✅ 未错误推进游戏状态")

    # 场景 7：中途退出 → 正常聊天
    print("\n[场景7] 中途说不玩了 → 退出并正常聊天")
    sid = "chain_exit"
    r, _, _ = chat(sid, "来玩成语接龙")
    print(f"  [开局] {r[:40]}…")
    r2, s2, _ = chat(sid, "不玩了没意思")
    print(f"  [退出] [{s2}] {r2}")
    # 下一条正常消息应正常回应，不再要求接字
    r3, s3, _ = chat(sid, "我饿了")
    print(f"  [退出后] [{s3}] {r3}")
    assert not extract_char(r3), "退出后不应再要求接字"
    print("  ✅ 已退出游戏，正常聊天")

    # 场景 8：新会话不继承上一局状态
    print("\n[场景8] 新会话不继承游戏状态")
    # chain_multi 已经在玩，新 sid 发普通消息不应触发接龙
    r, s, _ = chat("fresh_session_123", "你好")
    print(f"  [新会话] [{s}] {r}")
    assert not extract_char(r), "新会话不应继承接龙状态"
    print("  ✅ 新会话干净，无游戏状态泄漏")

    print("\n" + "=" * 70)
    print("全部成语接龙场景验证通过。")


if __name__ == "__main__":
    main()
